import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse

from llm_attacks import get_embedding_matrix, get_embeddings

def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):
    """
    Computes gradients of the loss with respect to the coordinates.
    """
    # Use the device from the input tensor (should be the same as our model device)
    device = input_ids.device

    embed_weights = model.get_input_embeddings().weight.detach()  # [vocab_size, hidden_dim]
    embeds = model.get_input_embeddings()(input_ids.unsqueeze(0)).detach()
    embeds = embeds.to(embed_weights.dtype)

    # Create a one-hot tensor on the correct device
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # Stitch together embeddings: before control tokens, new control, after control tokens
    full_embeds = torch.cat(
        [
            embeds[:, :input_slice.start, :].detach(),
            input_embeds,
            embeds[:, input_slice.stop:, :].detach()
        ],
        dim=1
    )
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice].detach()
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)
    loss.backward()
    
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    
    return grad  # shape: (number of tokens in input_slice, vocab size)

def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)
    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        torch.randint(0, topk, (batch_size, 1), device=grad.device)
    )

    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
    return new_control_toks

def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
    return cands

def forward(*, model, input_ids, attention_mask, batch_size=512):
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size] if attention_mask is not None else None
        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)
        gc.collect()
    return torch.cat(logits, dim=0)

def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=input_ids.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape (n, {control_slice.stop - control_slice.start}), "
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(input_ids.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(input_ids.device),
        1,
        locs,
        test_ids
    )
    attn_mask = (ids != pad_tok).type(ids.dtype) if pad_tok >= 0 else None

    if return_ids:
        del locs, test_ids; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids; gc.collect()
        return logits

def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    loss = crit(logits[:, loss_slice, :].transpose(1, 2), ids[:, target_slice])
    return loss.mean(dim=-1)

def load_model_and_tokenizer(model_path, tokenizer_path=None, device_str='cuda', **kwargs):
    # Define an explicit device variable
    device = torch.device(device_str)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **kwargs
    ).eval()

    # If multiple GPUs are available, wrap the model in DataParallel for simplicity
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Some token adjustments for specific models
    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/root/internlm2-chat")
    parser.add_argument('--device', type=str, default="cuda")
    # add other args as needed
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print("Using device:", device)
    
    # Load model and tokenizer with the explicit device setting
    model, tokenizer = load_model_and_tokenizer(args.model_path, device_str=args.device)
    
    # Proceed with the rest of your code...
    print("Model and tokenizer loaded successfully.")

if __name__ == "__main__":
    main()