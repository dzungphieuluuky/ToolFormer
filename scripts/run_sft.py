#!/usr/bin/env python
"""
run_sft.py
──────────
SFT warm-start with optional RC-GRPO Phase-1 (RCTP-FT) mode.

Standard mode:  python scripts/run_sft.py
RCTP-FT mode:   python scripts/run_sft.py --rctp

RCTP-FT (Reward-Conditioned Trajectory Policy Fine-Tuning) differences
from standard SFT:
  1. Loads MIXED trajectories (both R=1 and R=0 rollouts)
  2. Prepends <|high_reward|> to R=1 trajectories in the prompt
  3. Prepends <|low_reward|>  to R=0 trajectories in the prompt
  4. Objective:  min_θ Σ -log π_θ(a | h, r)
     (standard NLL, but conditioned on reward token)
  5. Registers the special tokens in the tokenizer before training

After RCTP-FT, the model learns to produce high-quality outputs when
it sees <|high_reward|> and low-quality outputs for <|low_reward|>.
This is the prerequisite for RC-GRPO Phase-2 to work.
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

from src.algorithms.base_trainer import load_model, SYSTEM_PROMPT, build_prompt
from src.reward.rc_grpo_reward import HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN
from src.algorithms.rc_grpo_trainer import RCGRPOTrainer
from src.utils.logging_utils import get_logger

logger = get_logger("run_sft")


def format_standard_sft(sample: dict, function_library: dict, tokenizer) -> dict:
    """Standard SFT format — no reward conditioning."""
    query = sample["query"]
    retrieved = sample.get("retrieved_functions", [])
    gt = sample.get("ground_truth", {})

    user_content = build_prompt(query, retrieved, function_library)
    assistant_content = _build_assistant_response(gt)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


def format_rctp_sft(sample: dict, function_library: dict, tokenizer) -> dict:
    """
    RCTP-FT format — prepends reward token to system prompt.
    Reward token is determined by whether ground truth has a valid function call.
    """
    query = sample["query"]
    retrieved = sample.get("retrieved_functions", [])
    gt = sample.get("ground_truth", {})

    # Determine reward token based on trajectory quality
    # R=1 (high reward) if:  non-abstention AND has function AND has arguments
    is_high_reward = (
        gt.get("workflow") != "abstention"
        and gt.get("function") is not None
        and bool(gt.get("arguments"))
    )
    reward_token = HIGH_REWARD_TOKEN if is_high_reward else LOW_REWARD_TOKEN

    user_content = build_prompt(query, retrieved, function_library)
    assistant_content = _build_assistant_response(gt)

    # Inject reward token at the START of the system prompt
    rctp_system = f"{reward_token}\n{SYSTEM_PROMPT}"

    messages = [
        {"role": "system", "content": rctp_system},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


def _build_assistant_response(gt: dict) -> str:
    """Build the expected assistant response string from ground truth."""
    call_json = json.dumps(
        {
            "function": gt.get("function"),
            "arguments": gt.get("arguments", {}),
        },
        indent=2,
    )
    reasoning = gt.get(
        "reasoning", "Selecting the appropriate function based on the query."
    )
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<call>\n{call_json}\n</call>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument(
        "--rctp",
        action="store_true",
        help="Run RCTP-FT (RC-GRPO Phase 1) instead of standard SFT",
    )
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    data_cfg = cfg["data"]
    sft_cfg = cfg.get("sft", {})
    rc_cfg = cfg.get("rc_grpo", {})

    model, tokenizer = load_model(cfg)

    # Register reward tokens BEFORE training if RCTP mode
    if args.rctp:
        logger.info("[RCTP-FT] Registering reward tokens...")
        RCGRPOTrainer.register_reward_tokens(tokenizer)
        model.resize_token_embeddings(len(tokenizer))

    with open(data_cfg["function_library_path"]) as fh:
        function_library = json.load(fh)

    # Load dataset
    raw_samples = []
    with jsonlines.open(data_cfg["train_path"]) as reader:
        for obj in reader:
            raw_samples.append(obj)

    # Format samples
    if args.rctp:
        formatted = [
            format_rctp_sft(s, function_library, tokenizer) for s in raw_samples
        ]
        output_dir = rc_cfg.get("sft_output_dir", "outputs/rctp_sft_model")
        logger.info(f"[RCTP-FT] Formatted {len(formatted)} samples with reward tokens.")
    else:
        formatted = [
            format_standard_sft(s, function_library, tokenizer) for s in raw_samples
        ]
        output_dir = sft_cfg.get("output_dir", "outputs/sft_model")

    dataset = Dataset.from_list(formatted)

    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=sft_cfg.get("learning_rate", 2e-4),
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=200,
        max_seq_length=(
            data_cfg.get("max_prompt_length", 1024)
            + data_cfg.get("max_completion_length", 512)
        ),
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_args,
    )

    mode = "RCTP-FT (Phase 1 for RC-GRPO)" if args.rctp else "standard SFT"
    logger.info(f"[SFT] Starting {mode}...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"[SFT] Model saved → {output_dir}")


if __name__ == "__main__":
    main()
