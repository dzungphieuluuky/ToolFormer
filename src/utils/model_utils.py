"""model_utils.py — Helpers for saving, loading, and merging LoRA adapters."""

from __future__ import annotations
from pathlib import Path
import torch


def save_model_and_tokenizer(model, tokenizer, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[model_utils] Saved → {output_dir}")


def load_model_from_path(
    model_path: str,
    base_model_name: str = "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """Load a fine-tuned LoRA adapter on top of the base model."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )
    # Load the LoRA adapter
    model.load_adapter(model_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def merge_and_export(
    model,
    tokenizer,
    output_dir: str,
    quantize_to: str = "q4_k_m",  # gguf quantisation type
) -> None:
    """Merge LoRA weights into the base model and export."""
    merged_dir = str(Path(output_dir) / "merged")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"[model_utils] Merged model → {merged_dir}")


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    do_sample: bool = True,
) -> str:
    """Single-sample inference helper."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the new tokens
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)
