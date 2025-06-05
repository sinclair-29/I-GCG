import torch
import fastchat.model
from transformers import AutoTokenizer

def load_conversation_template(template_name):
    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.system_message = "You are a helpful assistant."  # 添加默认系统提示
        conv_template.sep2 = conv_template.sep2.strip()
    
    return conv_template


"""
# llama2 template
# reference: https://huggingface.co/blog/codellama#conversational-instructions
# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)
"""

class SuffixManager:
    # 这个类的作用作为储存 suffix的buffer
    # 存在问题，fastchat
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

        if self.conv_template.name == 'llama-2':

            self.conv_template.messages = []
            test = "a"
            self.conv_template.append_message(self.conv_template.roles[0], test)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            test_toks = self.tokenizer(test).input_ids
            self._user_role_slice = slice(None, len(toks) - len(test_toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt(), add_special_tokens=False ).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt(), add_special_tokens=False ).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.roles[1], test)
            toks = self.tokenizer(self.conv_template.get_prompt(), add_special_tokens=False).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks) - len(test_toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt(), add_special_tokens=False ).input_ids
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
        print("===========")
        print(prompt)
        print("===========")
        return prompt
    
    def get_input_ids(self, adv_string=None, is_add_special_tokens = True):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt, add_special_tokens=is_add_special_tokens).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids

def main():
    def decode_slice(slice_obj):
        return tokenizer.decode(input_ids[slice_obj], skip_special_tokens=True)

    model_path = "../LLMJailbreak/models/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    conv_template = load_conversation_template("llama-2")

    manager = SuffixManager(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction="Write a positive movie review.",
        target="This movie was amazing!",
        adv_string="!!!TEST ADVERSARIAL STRING!!!"
    )

    # prompt_str = manager.get_prompt()
    # print("\n【生成的完整 prompt】\n", prompt_str)

    input_ids = manager.get_input_ids(is_add_special_tokens= True)
    print("\n【input_ids 形状】:", input_ids.shape)
    print("【input_ids 内容】:", input_ids.tolist())

    print("\n【各 slice 范围】:")
    print("User Role Slice:", manager._user_role_slice)
    print("Goal Slice:", manager._goal_slice)
    print("Control Slice:", manager._control_slice)
    print("Assistant Role Slice:", manager._assistant_role_slice)
    print("Target Slice:", manager._target_slice)
    print("Loss Slice:", manager._loss_slice)

    print("User Role Slice内容:", decode_slice(manager._user_role_slice))
    print("Goal Slice内容:", decode_slice(manager._goal_slice))
    print("Control Slice内容:", decode_slice(manager._control_slice))
    print("Assistant Role Slice内容:", decode_slice(manager._assistant_role_slice))
    print("Target Slice内容:", decode_slice(manager._target_slice))
    print("Loss Slice内容:", decode_slice(manager._loss_slice))
    print("==============\n")
    input_ids = manager.get_input_ids(is_add_special_tokens= False)
    print("\n【input_ids 形状】:", input_ids.shape)
    print("【input_ids 内容】:", input_ids.tolist())

    print("\n【各 slice 范围】:")
    print("User Role Slice:", manager._user_role_slice)
    print("Goal Slice:", manager._goal_slice)
    print("Control Slice:", manager._control_slice)
    print("Assistant Role Slice:", manager._assistant_role_slice)
    print("Target Slice:", manager._target_slice)
    print("Loss Slice:", manager._loss_slice)

    print("User Role Slice内容:", decode_slice(manager._user_role_slice))
    print("Goal Slice内容:", decode_slice(manager._goal_slice))
    print("Control Slice内容:", decode_slice(manager._control_slice))
    print("Assistant Role Slice内容:", decode_slice(manager._assistant_role_slice))
    print("Target Slice内容:", decode_slice(manager._target_slice))
    print("Loss Slice内容:", decode_slice(manager._loss_slice))


if __name__ == "__main__":
    main()