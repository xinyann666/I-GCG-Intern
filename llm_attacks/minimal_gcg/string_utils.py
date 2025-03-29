import torch
import fastchat 


def load_conversation_template(template_name):
    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    return conv_template

'''
class InternLMConvTemplate:
    def __init__(self):
        self.name = "internlm2_5-7b-chat"
        self.system = "<|System|>:"
        self.meta_instruction = (
            "You are an AI assistant whose name is InternLM (书生·浦语).\n"
            "- InternLM (书生·浦语) is a conversational language model developed by Shanghai AI Laboratory (上海人工智能实验室). "
            "It is designed to be helpful, honest, and harmless.\n"
            "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."
        )
        self.eosys = "\n"
        self.user = "<|User|>:"
        self.eoh = "\n"
        self.assistant = "<|Bot|>:"
        self.eoa = "<eoa>"
        self.separator = "\n"
        self.stop_words = ["<eoa>"]
        self.messages = []  # to store conversation history

    def append_message(self, role, message):
        """Append a message for a given role to the conversation."""
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
'''

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

'''
class SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
    
    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction} {self.adv_string}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        # if self.conv_template.name == 'llama-2':
        if self.conv_template.name == 'internlm2_5-7b-chat':
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2)
            self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)

        else:
            python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
            try:
                encoding.char_to_token(len(prompt)-1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                # This is specific to the vicuna and pythia tokenizer and conversation prompt.
                # It will not work with other tokenizers or prompts.
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.instruction}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)-1))

                separator = ' ' if self.instruction else ''
                self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._control_slice = slice(self._goal_slice.stop, len(toks)-1)

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-1)
                self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-2)
            else:
                self._system_slice = slice(
                    None, 
                    encoding.char_to_token(len(self.conv_template.system))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(self.instruction)),
                    encoding.char_to_token(prompt.find(self.instruction) + len(self.instruction))
                )
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(self.adv_string)),
                    encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
                )

        self.conv_template.messages = []

        return prompt
'''

class SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
    
    def get_prompt(self, adv_string=None):
        if adv_string is not None:
            self.adv_string = adv_string

        # Use the template's user and assistant attributes.
        self.conv_template.append_message(self.conv_template.user, f"{self.instruction} {self.adv_string}")
        self.conv_template.append_message(self.conv_template.assistant, f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        if self.conv_template.name == 'internlm2_5-7b-chat':
            # For internlm, we build the prompt in stages and compute slice indices.
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.user, None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.assistant, None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2)
            self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)
        else:
            # Since we only care about internlm2_5-7b-chat, we raise an error for other templates.
            raise ValueError("Unsupported conversation template for SuffixManager.")

        self.conv_template.messages = []  # clear the messages after computing slices
        return prompt
    
    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])
        return input_ids
    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids

