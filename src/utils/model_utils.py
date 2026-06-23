from pathlib import Path

import torch


def save_model_and_tokenizer(model, tokenizer, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[model_utils] Saved -> {output_dir}")


def load_model_for_inference(
    model_path: str,
    base_model_name: str = "unsloth/Qwen3-4B-Base",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )
    model.load_adapter(model_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    do_sample: bool = True,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)
