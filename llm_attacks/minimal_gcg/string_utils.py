import torch
import fastchat.model
from transformers import AutoTokenizer

def load_conversation_template(template_name: str):
    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'llama-2':
        conv_template.system_message = "You are a helpful assistant."
    return conv_template


class PromptManager:
    def __init__(self, *, tokenizer, conv_template, instruction: str, target: str, adv_string: str):
        self.tokenizer = tokenizer
        self.conv_template = conv_template  # Store the original template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string

        self._user_role_slice = None
        self._goal_slice = None
        self._adv_slice = None
        self._assistant_role_slice = None
        self._target_slice = None
        self._loss_slice = None

        self._full_prompt_str = None

    def get_prompt(self, adv_string: str = None) -> str:
        if adv_string is not None:
            self.adv_string = adv_string

        self.conv_template.message = []

        str_instruction_segment = self.instruction
        str_user_role_start_segment = self.conv_template.system_template.format(
            system_message=self.conv_template.system_message)

        separator = ' ' if self.instruction else ''
        str_adv_segment_with_separator = separator + self.adv_string

        user_content_full = str_instruction_segment + str_adv_segment_with_separator

        str_assistant_role_segment = self.conv_template.roles[1] + " "
        str_target_segment = self.target
        str_eos_segment = " </s>"

        self.conv_template.append_message(self.conv_template.roles[0], user_content_full)
        self.conv_template.append_message(self.conv_template.roles[1], str_target_segment)
        self._full_prompt_str = self.conv_template.get_prompt()

        tok_user_role_start = self.tokenizer(str_user_role_start_segment, add_special_tokens=False).input_ids
        tok_instruction = self.tokenizer(str_instruction_segment, add_special_tokens=False).input_ids
        tok_adv_with_sep = self.tokenizer(str_adv_segment_with_separator, add_special_tokens=False).input_ids
        tok_assistant_role = self.tokenizer(str_assistant_role_segment, add_special_tokens=False).input_ids
        tok_target = self.tokenizer(str_target_segment, add_special_tokens=False).input_ids

        current_idx = 0
        self._user_role_slice = slice(current_idx, current_idx + len(tok_user_role_start))
        current_idx += len(tok_user_role_start)

        self._goal_slice = slice(current_idx, current_idx + len(tok_instruction))
        current_idx += len(tok_instruction)

        self._adv_slice = slice(current_idx, current_idx + len(tok_adv_with_sep))
        current_idx += len(tok_adv_with_sep)

        self._assistant_role_slice = slice(current_idx, current_idx + len(tok_assistant_role))
        current_idx += len(tok_assistant_role)

        self._target_slice = slice(current_idx, current_idx + len(tok_target))

        self._loss_slice = slice(
            self._assistant_role_slice.stop - 1,
            self._target_slice.stop - 1 if self.target else self._assistant_role_slice.stop - 1
        )

        return self._full_prompt_str

    def get_input_ids(self, adv_string: str = None) -> torch.Tensor:
        prompt_str = self.get_prompt(adv_string=adv_string)
        input_ids_list = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
        return torch.tensor(input_ids_list[:self._target_slice.stop])
    
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