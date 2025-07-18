import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

system_prompt = "You are a helpful agent that decode the brain activity of a person looking at an image."


def apply_chat_template(**kwargs):
    name_or_path = kwargs["tokenizer"].name_or_path
    if "vicuna" in name_or_path:
        func = apply_chat_template_vicuna
    else:
        raise NotImplementedError

    return func(**kwargs, max_length=800)


def apply_chat_template_vicuna(
    chat,
    tokenizer,
    add_generation_prompt: bool = False,
    return_assistant_tokens_mask: bool = False,
    max_length=800,
):
    assistant_tokens_mask = []
    tokens = tokenizer.tokenize(tokenizer.bos_token + system_prompt + "\n\n")
    assert tokenizer.eos_token == "</s>"

    for i, message in enumerate(chat):
        role = message["role"]
        assert role in ["user", "assistant"]
        content = message["content"]
        if role == "assistant":
            content += "</s>"

        tokens.extend(tokenizer.tokenize(f"</s>{role.upper()}: ")[1:])
        assistant_tokens_mask.extend([0] * (len(tokens) - len(assistant_tokens_mask)))
        tokens.extend(tokenizer.tokenize(content + "\n"))
        assistant_tokens_mask.extend(
            [1 if role == "assistant" else 0]
            * (len(tokens) - len(assistant_tokens_mask))
        )

    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        assistant_tokens_mask = assistant_tokens_mask[:max_length]

    if add_generation_prompt:
        tokens.extend(tokenizer.tokenize("ASSISTANT: "))
        assistant_tokens_mask.extend([0] * (len(tokens) - len(assistant_tokens_mask)))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    assert len(tokens) == len(assistant_tokens_mask)
    if return_assistant_tokens_mask:
        return {
            "input_ids": input_ids,
            "assistant_masks": assistant_tokens_mask,
        }
    else:
        return {
            "input_ids": input_ids,
        }
