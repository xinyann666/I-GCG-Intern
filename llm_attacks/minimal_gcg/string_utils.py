import torch

class InternLMConvTemplate:
    def __init__(self,
                 system='<|System|>:',
                 meta_instruction=(
                     "You are an AI assistant whose name is InternLM (书生·浦语).\n"
                     "- InternLM (书生·浦语) is a conversational language model developed by Shanghai AI Laboratory (上海人工智能实验室). "
                     "It is designed to be helpful, honest, and harmless.\n"
                     "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."
                 ),
                 eosys='\n',
                 user='<|User|>:',
                 eoh='\n',
                 assistant='<|Bot|>:',
                 eoa='<eoa>',
                 separator='\n',
                 stop_words=['<eoa>']):
        self.name = "internlm2_5-7b-chat"
        self.system = system
        self.meta_instruction = meta_instruction
        self.eosys = eosys
        self.user = user
        self.eoh = eoh
        self.assistant = assistant
        self.eoa = eoa
        self.separator = separator
        self.stop_words = stop_words
        self.messages = []  # to store conversation history

    def append_message(self, role, message):
        """Append a message for the given role to the conversation."""
        if message is not None:
            self.messages.append(f"{role}{self.separator}{message}{self.separator}")
        else:
            self.messages.append(f"{role}{self.separator}")

    def update_last_message(self, message):
        """Replace the last message with a new one, preserving the role prefix."""
        if self.messages:
            role_prefix = self.messages[-1].split(self.separator)[0]
            self.messages[-1] = f"{role_prefix}{self.separator}{message}{self.separator}"

    def get_prompt(self):
        """Compose the full prompt from system info, meta instruction, and messages."""
        prompt = f"{self.system}{self.separator}{self.meta_instruction}{self.separator}{self.separator}"
        prompt += "".join(self.messages)
        return prompt

class SuffixManager:
    def __init__(self, tokenizer, conv_template, instruction, target, adv_string):
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction or ""
        self.target = target or ""
        self.adv_string = adv_string or ""

        self._control_slice = None
        self._target_slice = None
        self._loss_slice = None
        self._assistant_role_slice = None

    def get_prompt(self, adv_string=None):
        if adv_string is not None:
            self.adv_string = adv_string
        # Clear previous messages.
        self.conv_template.messages = []
        # Build initial prompt (system + meta_instruction)
        prompt_text = self.conv_template.get_prompt()
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        # Append user message: instruction + adv_string.
        space = " " if self.instruction else ""
        user_message = f"{self.instruction}{space}{self.adv_string}"
        self.conv_template.append_message(self.conv_template.user, user_message)
        user_prompt_text = self.conv_template.get_prompt()
        user_prompt_ids = self.tokenizer(user_prompt_text, add_special_tokens=False).input_ids
        # Compute _control_slice.
        control_start = len(prompt_ids) + len(
            self.tokenizer(f"{self.conv_template.user}{self.conv_template.separator}{self.instruction}{space}",
                           add_special_tokens=False).input_ids
        ) - 1
        control_end = len(user_prompt_ids)
        self._control_slice = slice(control_start, control_end)
        # Get assistant marker tokens and calculate assistant role slice.
        # Remove trailing separator so that the target starts immediately after the marker.
        assistant_marker = f"{self.conv_template.separator}{self.conv_template.assistant}"
        assistant_marker_ids = self.tokenizer(assistant_marker, add_special_tokens=False).input_ids
        assistant_role_start = len(user_prompt_ids)
        assistant_role_end = assistant_role_start + len(assistant_marker_ids)
        self._assistant_role_slice = slice(assistant_role_start, assistant_role_end)
        # Append assistant message: target.
        self.conv_template.append_message(self.conv_template.assistant, self.target)
        full_prompt_text = self.conv_template.get_prompt()
        full_prompt_ids = self.tokenizer(full_prompt_text, add_special_tokens=False).input_ids
        # Compute _target_slice and _loss_slice.
        target_start = assistant_role_end
        target_end = len(full_prompt_ids)
        self._target_slice = slice(target_start, target_end)
        self._loss_slice = slice(target_start - 1, target_end - 1)
        # Clear messages for the next call.
        self.conv_template.messages = []
        return full_prompt_text

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string)
        full_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        return torch.tensor(full_ids[:self._target_slice.stop])



       