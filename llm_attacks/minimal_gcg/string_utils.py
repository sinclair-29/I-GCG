import torch
import fastchat.model
from transformers import AutoTokenizer

def load_conversation_template(template_name: str):
    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'llama-2':
        conv_template.system_message = "You are a helpful assistant."
    return conv_template


class SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction: str, target: str, adv_string: str):
        self.tokenizer = tokenizer
        self.conv_template = conv_template  # Store the original template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string

        self._user_role_slice = None
        self._goal_slice = None
        self._control_slice = None
        self._assistant_role_slice = None
        self._target_slice = None
        self._loss_slice = None

        self._full_prompt_str = None


    def get_prompt(self, adv_string: str = None) -> str:
        final_prompt_str = None

        if adv_string is not None:
            self.adv_string = adv_string

        if self.conv_template.name == "llama-2":
            self.conv_template.messages = []
            test = "a"
            self.conv_template.append_message(self.conv_template.roles[0], test)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks) - 2)

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, len(toks) - 1)

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks) - 1)

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            final_prompt_str = self.conv_template.get_prompt()
            toks = self.tokenizer(final_prompt_str).input_ids

            # skip </s><s>
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        self.conv_template.messages = []
        return final_prompt_str
    
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

    input_ids = manager.get_input_ids(is_add_special_tokens= True)
    print("\n【input_ids 形状】:", input_ids.shape)
    print("【input_ids 内容】:", input_ids.tolist())

    print("\n【各 slice 范围】:")
    print("User Role Slice:", manager._user_role_slice)
    print("Goal Slice:", manager._goal_slice)
    print("Control Slice:", manager._adv_slice)
    print("Assistant Role Slice:", manager._assistant_role_slice)
    print("Target Slice:", manager._target_slice)
    print("Loss Slice:", manager._loss_slice)

    print("User Role Slice内容:", decode_slice(manager._user_role_slice))
    print("Goal Slice内容:", decode_slice(manager._goal_slice))
    print("Control Slice内容:", decode_slice(manager._adv_slice))
    print("Assistant Role Slice内容:", decode_slice(manager._assistant_role_slice))
    print("Target Slice内容:", decode_slice(manager._target_slice))
    print("Loss Slice内容:", decode_slice(manager._loss_slice))


if __name__ == "__main__":
    main()