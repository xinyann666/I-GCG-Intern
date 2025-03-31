import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
import time
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
    # print(f"input_ids[input_slice].shape: {input_ids[input_slice].shape}")  # e.g., [3]

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

    # Compute the logits slice for the loss
    logits_slice = logits[0, loss_slice, :]
    
    # If the logits slice has one more token than targets, drop the last prediction.
    if logits_slice.shape[0] != targets.shape[0]:
        logits_slice = logits_slice[:-1]

    loss = nn.CrossEntropyLoss()(logits_slice, targets)
    loss.backward()
    
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    
    return grad  # shape: (number of tokens in input_slice, vocab size)



def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.inf

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

'''
def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):  # batch_size
        # Decode the current candidate tokens
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        
        # Check original length vs re-encoded length
        original_len = len(control_cand[i])
        reencoded_ids = tokenizer(decoded_str, add_special_tokens=False).input_ids
        reencoded_len = len(reencoded_ids)
        
        # If lengths don't match, try to fix the candidate
        if reencoded_len != original_len and filter_cand:
            print(f"Length mismatch for candidate {i}: original={original_len}, reencoded={reencoded_len}")
            
            # Option 1: Try to fix by adjusting token representation
            if reencoded_len < original_len:
                # Too short: Pad with repeated tokens from the end
                padding_needed = original_len - reencoded_len
                if len(reencoded_ids) > 0:
                    # Repeat last token
                    reencoded_ids = reencoded_ids + [reencoded_ids[-1]] * padding_needed
                else:
                    # In case of empty list, use a safe padding token
                    pad_token = 1  # Usually a safe token (adjust as needed)
                    reencoded_ids = [pad_token] * original_len
                
                # Re-decode with the fixed tokens
                fixed_str = tokenizer.decode(reencoded_ids, skip_special_tokens=True)
                
                # Verify the fix worked
                fixed_ids = tokenizer(fixed_str, add_special_tokens=False).input_ids
                if len(fixed_ids) == original_len and fixed_str != curr_control:
                    cands.append(fixed_str)
                    print(f"✓ Successfully fixed candidate {i} by padding")
                    continue
                
            elif reencoded_len > original_len:
                # Too long: Truncate to the correct length
                truncated_ids = reencoded_ids[:original_len]
                fixed_str = tokenizer.decode(truncated_ids, skip_special_tokens=True)
                
                # Verify the fix worked
                fixed_ids = tokenizer(fixed_str, add_special_tokens=False).input_ids
                if len(fixed_ids) == original_len and fixed_str != curr_control:
                    cands.append(fixed_str)
                    print(f"✓ Successfully fixed candidate {i} by truncating")
                    continue
            
            # If we got here, both fix attempts failed
            count += 1
            print(f"✗ Failed to fix candidate {i}")
            
        # Regular filtering case
        elif filter_cand:
            if decoded_str != curr_control and reencoded_len == original_len:
                cands.append(decoded_str)
            else:
                count += 1
        else:
            # No filtering case
            cands.append(decoded_str)

    print(f"Filtered out {count}/{control_cand.shape[0]} candidates")
    
    # Handling empty cands list
    if filter_cand and len(cands) == 0 and control_cand.shape[0] > 0:
        print("WARNING: All candidates were filtered out. Keeping the first one anyway.")
        cands.append(tokenizer.decode(control_cand[0], skip_special_tokens=True))
    
    if filter_cand:
        # Ensure we have enough candidates by duplicating the last one
        if cands:  # Check if cands is not empty
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
    
    return cands
'''

# try 1
def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None, debug=False):
   
    cands = []
    
    for i in range(control_cand.shape[0]):
        # Decode the candidate tokens into a string
        decoded_str = tokenizer.decode(
            control_cand[i], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        # Optionally print debugging info
        if debug:
            original_length = len(control_cand[i])
            reencoded_length = len(tokenizer(decoded_str, add_special_tokens=False).input_ids)
            print(f"[DEBUG] Candidate {i}: Original length: {original_length}, Re-encoded length: {reencoded_length}")
        
        # If filtering is enabled, check and adjust the candidate string
        if filter_cand:
            tokenized_ids = tokenizer(decoded_str, add_special_tokens=False).input_ids
            if decoded_str != curr_control and len(tokenized_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                # Adjust decoded_str until the tokenized length matches the target length
                target_len = len(control_cand[i])
                # print(f"target_len: {target_len}")
                max_iter = 500 # Safeguard to prevent infinite loops
                iter_count = 0
                
                while iter_count < max_iter:
                    current_ids = tokenizer(decoded_str, add_special_tokens=False).input_ids
                    current_len = len(current_ids)
                    
                    if current_len > target_len:
                        # Remove the last character to reduce token count
                        decoded_str = decoded_str[:-1]
                    elif current_len < target_len:
                        # Append the last character to increase token count (if possible)
                        if decoded_str:
                            decoded_str = decoded_str + decoded_str[-1]
                        else:
                            break  # If decoded_str becomes empty, exit the loop
                    else:
                        # Length matches; exit the loop
                        break
                    
                    iter_count += 1
                
                if iter_count >= max_iter:
                    # print(f"Warning: Maximum iterations reached for candidate {i} without matching target length.")
                    pass
                if decoded_str:
                    # print(f"newly added string after tokenzier length: {len(tokenizer(decoded_str, add_special_tokens=False).input_ids)}")
                    cands.append(decoded_str)
                else:
                    print(f"Warning: Candidate {i} is empty after all iterations.")
        else:
            cands.append(decoded_str)
    
    # If filtering was enabled but some candidates were adjusted/omitted,
    # extend the candidate list to match the original count by duplicating the last valid candidate.
    if filter_cand and len(cands) < control_cand.shape[0]:
        cands.extend([cands[-1]] * (control_cand.shape[0] - len(cands)))
    return cands


'''
# try 2 : set to False
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
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands
'''


def forward(*, model, input_ids, attention_mask, batch_size=512):
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size] if attention_mask is not None else None
        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)
        gc.collect()
    return torch.cat(logits, dim=0)


from torch.nn.utils.rnn import pad_sequence
import torch
import gc
'''
def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    # Convert test_controls (list of strings) to a padded tensor of token ids.
    if isinstance(test_controls[0], str):
        # Determine expected length from control_slice, but note that actual tokenization may yield fewer tokens.
        expected_len = control_slice.stop - control_slice.start
        test_ids_list = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:expected_len],
                         device=input_ids.device)
            for control in test_controls
        ]
        # Select a safe pad token.
        pad_tok = 0
        while pad_tok in input_ids or any(pad_tok in t for t in test_ids_list):
            pad_tok += 1

        # Convert the list to a padded tensor.
        test_ids = pad_sequence(test_ids_list, batch_first=True, padding_value=pad_tok)
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    # Check the actual tokenized length.
    actual_len = test_ids.shape[1]
    if actual_len != expected_len:
        print(f"Warning: expected control length {expected_len} but got {actual_len}. Adjusting control_slice accordingly.")
        # Adjust the control slice to match the actual tokenized length.
        control_slice = slice(control_slice.start, control_slice.start + actual_len)

    # Create a repeated copy of input_ids.
    repeated_input = input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1)

    # Ensure that the control slice is within the bounds of repeated_input.
    if control_slice.stop > repeated_input.size(1):
        raise ValueError(
            f"control_slice.stop ({control_slice.stop}) is out-of-bounds for input_ids of length {repeated_input.size(1)}"
        )

    # Use a loop to insert the control tokens into the designated slice.
    ids = repeated_input.clone()
    for i in range(test_ids.shape[0]):
        ids[i, control_slice] = test_ids[i]

    # Create an attention mask that ignores the pad token.
    attn_mask = (ids != pad_tok).type(ids.dtype) if pad_tok >= 0 else None

    # Forward the modified ids through the model.
    if return_ids:
        del test_ids
        gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids
        gc.collect()
        return logits
'''

def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    # print(f"shape of input_ids: {input_ids.shape}") # [121]
    
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
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
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), " 
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    # print(f"shape of locs: {locs.shape}") # [256,20]
    input_ids = input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device)
    # print(f"shape of input_ids: {input_ids.shape}") # [256,121]
    # print(f"shape of test_ids: {test_ids.shape}") # [256,20]
    # print(f"control slice start: {control_slice.start}") # 111
    # print(f"control slice stop: {control_slice.stop}") # 131
   
    ids = torch.scatter(
        input_ids,
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None
        
    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
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
        device_map="auto",
        local_files_only=True,
        **kwargs
    ).eval()

    
    
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
