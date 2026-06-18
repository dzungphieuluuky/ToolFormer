#!/usr/bin/env python
"""
run_sft.py
──────────
SFT warm-start + optional RCTP-FT (RC-GRPO Phase 1).

Standard SFT:  python scripts/run_sft.py
RCTP-FT mode:  python scripts/run_sft.py --rctp

Conversation format used:
    <|im_start|>system    ← SYSTEM_PROMPT (+ reward token prefix for RCTP)
    <|im_start|>user      ← raw query
    <|im_start|>retriever ← functions + argument values
    <|im_start|>assistant ← ground-truth reasoning + tool_call
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import jsonlines

from src.algorithms.base_trainer import (
    load_model,
    patch_tokenizer_for_custom_roles,
    SYSTEM_PROMPT,
    build_retriever_block,
    build_messages_for_sft,
    load_sft_dataset,
    _deserialise_arg_values,
)
from src.reward.rc_grpo_reward import HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN
from src.algorithms.rc_grpo_trainer import RCGRPOTrainer
from src.utils.logging_utils import get_logger

logger = get_logger("run_sft")


# ──────────────────────────────────────────────────────────────────────────────
# RCTP-FT formatter
# ──────────────────────────────────────────────────────────────────────────────

def format_rctp_sample(
    sample:           dict,
    function_library: dict,
    tokenizer,
) -> dict:
    """
    RCTP-FT format (RC-GRPO Phase 1).

    Identical to standard SFT except the reward-conditioning token is
    prepended to the system prompt content:
        <|im_start|>system
        <|high_reward|>                  ← or <|low_reward|>
        You are a telecom assistant...
        <|im_end|>
        <|im_start|>user
        ...
        <|im_end|>
        <|im_start|>retriever
        ...
        <|im_end|>
        <|im_start|>assistant
        <reasoning>...</reasoning>
        <tool_call>...</tool_call>
        <|im_end|>

    High-quality trajectories (non-abstention, has function + arguments)
    receive <|high_reward|>; everything else receives <|low_reward|>.
    """
    query     = sample["query"]
    retrieved = sample.get("retrieved_functions", [])
    gt        = sample.get("ground_truth", {})

    is_high = (
        gt.get("workflow") != "abstention"
        and gt.get("function") is not None
        and bool(gt.get("arguments"))
    )
    reward_token = HIGH_REWARD_TOKEN if is_high else LOW_REWARD_TOKEN

    arg_vals = _deserialise_arg_values(sample.get("retrieved_argument_values"))

    # Build retriever block content
    retriever_content = build_retriever_block(retrieved, function_library, arg_vals)

    # Build ground-truth assistant response
    reasoning  = gt.get("reasoning", "Selecting the correct function based on the query.")
    func_name  = gt.get("function")
    arguments  = gt.get("arguments", {})
    call_json  = (
        "null"
        if func_name is None
        else json.dumps({"function": func_name, "arguments": arguments},
                        indent=2, ensure_ascii=False)
    )
    assistant_content = (
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<tool_call>\n{call_json}\n</tool_call>"
    )

    # Inject reward token into the system prompt content (not a separate message)
    rctp_system = f"{reward_token}\n{SYSTEM_PROMPT}"

    messages = [
        {"role": "system",    "content": rctp_system},
        {"role": "user",      "content": query},
        {"role": "retriever", "content": retriever_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = False,
        )
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument(
        "--rctp", action="store_true",
        help="Run RCTP-FT (RC-GRPO Phase 1) instead of standard SFT",
    )
    args = parser.parse_args()

    cfg       = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    data_cfg  = cfg["data"]
    sft_cfg   = cfg.get("sft", {})

    # ── 1. Load model (patch_tokenizer_for_custom_roles is called inside) ─────
    model, tokenizer = load_model(cfg)

    # ── 2. Register RC reward tokens if RCTP mode ─────────────────────────────
    if args.rctp:
        logger.info("[RCTP-FT] Registering reward tokens...")
        RCGRPOTrainer.register_reward_tokens(tokenizer)
        model.resize_token_embeddings(len(tokenizer))

    # ── 3. Load function library ───────────────────────────────────────────────
    with open(data_cfg["function_library_path"], encoding="utf-8") as fh:
        function_library = json.load(fh)

    # ── 4. Build dataset ───────────────────────────────────────────────────────
    if args.rctp:
        # Format all samples with reward-token conditioning
        raw_samples: list[dict] = []
        with jsonlines.open(data_cfg["train_path"]) as reader:
            for obj in reader:
                raw_samples.append(obj)

        formatted = [
            format_rctp_sample(s, function_library, tokenizer)
            for s in raw_samples
        ]
        dataset    = Dataset.from_list(formatted)
        output_dir = sft_cfg.get("rctp_output_dir", "outputs/rctp_sft_model")
        logger.info(
            f"[RCTP-FT] {len(formatted)} samples formatted with reward tokens."
        )
    else:
        dataset    = load_sft_dataset(
            jsonl_path       = data_cfg["train_path"],
            function_library = function_library,
            tokenizer        = tokenizer,
        )
        output_dir = sft_cfg.get("output_dir", "outputs/sft_model")

    # ── 5. Print one sample for visual verification ────────────────────────────
    print("\n" + "=" * 80)
    mode_label = "RCTP-FT" if args.rctp else "Standard SFT"
    print(f"  {mode_label} — SAMPLE (first 2000 chars):")
    print("=" * 80)
    text = dataset[0]["text"]
    print(text[:2000])
    if len(text) > 2000:
        print(f"\n... ({len(text)} chars total)")
    print("=" * 80 + "\n")

    # ── 6. Train ───────────────────────────────────────────────────────────────
    sft_args = SFTConfig(
        output_dir                  = output_dir,
        num_train_epochs            = sft_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size = sft_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps = sft_cfg.get("gradient_accumulation_steps", 4),
        learning_rate               = sft_cfg.get("learning_rate", 2e-4),
        warmup_ratio                = 0.1,
        logging_steps               = 10,
        save_steps                  = 200,
        max_seq_length              = (
            data_cfg.get("max_prompt_length",    1024)
            + data_cfg.get("max_completion_length", 512)
        ),
        report_to          = "none",
        dataset_text_field = "text",
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = dataset,
        args          = sft_args,
    )

    logger.info(f"[SFT] Starting {mode_label}...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"[SFT] Model saved → {output_dir}")


if __name__ == "__main__":
    main()