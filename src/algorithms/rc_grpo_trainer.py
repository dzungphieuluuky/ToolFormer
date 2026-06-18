"""
rc_grpo_trainer.py
───────────────────
RC-GRPO Trainer  (arxiv 2602.03025)
"Reward-Conditioned GRPO for Multi-Turn Tool Calling Agents"

Two-phase implementation
────────────────────────
Phase 1 — RCTP-FT  (Reward-Conditioned Trajectory Policy Fine-Tuning)
    Handled in run_sft.py with rctp=True flag.
    Objective: min_θ Σ -log π_θ(a | h, r)
    - Takes MIXED trajectories (both good and bad rollouts)
    - Prepends reward goal token to every prompt:
        <|high_reward|> → for trajectories with R=1
        <|low_reward|>  → for trajectories with R=0
    - Trains the model to CONDITION on reward signal

Phase 2 — RC-GRPO RL  (this file)
    Key modification over vanilla GRPO:
    Within each rollout group of size G, sample reward tokens
    with a CONTROLLED MIX:
        p_high = proportion of successful trajectories in RCTP dataset
        ~p_high rollouts conditioned on <|high_reward|>
        ~1-p_high rollouts conditioned on <|low_reward|>

    This is the core innovation: it GUARANTEES within-group diversity
    even when the base policy has collapsed to near-deterministic behaviour
    (the vanishing advantage problem).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from trl import GRPOTrainer

from .base_trainer import (
    load_model,
    build_grpo_config,
    load_grpo_dataset,
    SYSTEM_PROMPT,
)
from src.reward.rc_grpo_reward import (
    HIGH_REWARD_TOKEN,
    LOW_REWARD_TOKEN,
    rc_grpo_reward_func,
    rc_grpo_format_func,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Reward-token injection (Phase 2: diverse token sampling within each group)
# ──────────────────────────────────────────────────────────────────────────────


def inject_diverse_reward_tokens(
    dataset,
    num_generations: int = 8,
    high_token: str = HIGH_REWARD_TOKEN,
    low_token: str = LOW_REWARD_TOKEN,
    high_fraction: float = 0.5,
) -> Any:
    """
    Expand each dataset sample into `num_generations` copies and assign
    reward-conditioning tokens so that approximately `high_fraction` of
    the copies use the high reward token and the rest the low token.

    We return a flattened dataset with G copies per original sample.
    Within each expanded group the order is shuffled deterministically
    (seeded by the original sample index) to avoid position bias.
    """
    import random

    n_high = max(1, round(num_generations * high_fraction))

    def _expand(example, idx):
        out = []
        for g in range(num_generations):
            ex = dict(example)
            if g < n_high:
                token = high_token
            else:
                token = low_token
            if not ex["prompt"].startswith(token):
                ex["prompt"] = token + "\n" + ex["prompt"]
            ex["reward_token"] = token
            out.append(ex)
        # Deterministic shuffle per-sample to avoid position bias
        rnd = random.Random(int(idx))
        rnd.shuffle(out)
        return out

    # Use flat_map to expand each example into `num_generations` entries
    return dataset.flat_map(_expand, with_indices=True)


def compute_high_fraction_from_rctp_dataset(rctp_dataset_path: str) -> float:
    """
    Compute the proportion of high-reward trajectories in the RCTP dataset.
    This is used as the sampling probability p_high in Phase 2.
    """
    import jsonlines

    total = 0
    high = 0
    with jsonlines.open(rctp_dataset_path) as reader:
        for obj in reader:
            total += 1
            # Expect a field 'reward' or 'ground_truth' indicating success
            # We'll assume the dataset has a 'reward' key (1 for high, 0 for low)
            reward = obj.get("reward", 0)
            if reward == 1:
                high += 1
    return high / max(total, 1)


# ──────────────────────────────────────────────────────────────────────────────
# RC-GRPO Trainer
# ──────────────────────────────────────────────────────────────────────────────


class RCGRPOTrainer(GRPOTrainer):
    """
    Wraps TRL GRPOTrainer with:
    1. Special token registration (<|high_reward|>, <|low_reward|>)
    2. Reward-token injection into prompts for within-group diversity
    3. Standard GRPO advantage computation (works correctly post-injection
       because group rewards now vary due to conditioning)

    No changes to the loss function — the innovation is entirely in the
    sampling strategy that restores non-zero advantage signals.
    """

    @classmethod
    def register_reward_tokens(cls, tokenizer) -> None:
        """Register <|high_reward|> and <|low_reward|> as special tokens."""
        new_tokens = [HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN]
        existing = set(tokenizer.additional_special_tokens)
        to_add = [t for t in new_tokens if t not in existing]
        if to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": to_add})
            logger.info(f"[RC-GRPO] Registered special tokens: {to_add}")
        else:
            logger.info("[RC-GRPO] Reward tokens already registered.")


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────


def train_rc_grpo(config: dict) -> None:
    """
    Full RC-GRPO Phase-2 training.

    Assumes Phase-1 RCTP-FT has already been run (run_sft.py --rctp).
    If sft_model_path is set in config, loads the RCTP checkpoint.
    Otherwise starts from base model (not recommended for best results).
    """
    rc_cfg = config.get("rc_grpo", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    # ── 1. Load model (from RCTP-FT checkpoint if available) ─────────────────
    sft_path = train_cfg.get("sft_model_path")
    if sft_path and Path(sft_path).exists():
        logger.info(f"[RC-GRPO] Loading RCTP-FT checkpoint: {sft_path}")
        from src.utils.model_utils import load_model_from_path

        model, tokenizer = load_model_from_path(
            sft_path,
            base_model_name=config["model"]["name"],
            max_seq_length=config["model"]["max_seq_length"],
        )
    else:
        logger.warning("[RC-GRPO] No RCTP-FT checkpoint found — using base model.")
        model, tokenizer = load_model(config)

    # ── 2. Register reward tokens ─────────────────────────────────────────────
    RCGRPOTrainer.register_reward_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    # ── 3. Load function library + dataset ────────────────────────────────────
    with open(data_cfg["function_library_path"], "r") as fh:
        function_library = json.load(fh)

    dataset = load_grpo_dataset(data_cfg["train_path"], function_library, tokenizer)

    # ── 4. Compute high_fraction from RCTP dataset ───────────────────────────
    rctp_dataset_path = data_cfg.get("rctp_dataset_path")
    if rctp_dataset_path and Path(rctp_dataset_path).exists():
        high_fraction = compute_high_fraction_from_rctp_dataset(rctp_dataset_path)
        logger.info(
            f"[RC-GRPO] Computed high_fraction={high_fraction:.3f} from RCTP dataset."
        )
    else:
        high_fraction = rc_cfg.get("high_fraction", 0.5)
        logger.warning(
            f"[RC-GRPO] No RCTP dataset found; using fixed high_fraction={high_fraction}."
        )

    # ── 5. Inject diverse reward tokens (Phase-2 sampling strategy) ──────────
    num_gen = config.get("training", {}).get("num_generations", 8)
    dataset = inject_diverse_reward_tokens(
        dataset,
        num_generations=num_gen,
        high_fraction=high_fraction,
    )
    logger.info(f"[RC-GRPO] Reward tokens injected (high_fraction={high_fraction}).")

    # ── 6. Build GRPOConfig ───────────────────────────────────────────────────
    grpo_args = build_grpo_config(
        config,
        output_dir=train_cfg.get("output_dir", "outputs/rc_grpo_model"),
    )

    # ── 7. Trainer ────────────────────────────────────────────────────────────
    trainer = RCGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            rc_grpo_reward_func,  # binary verifiable outcome reward
            rc_grpo_format_func,  # format reward (XML tags)
        ],
        args=grpo_args,
        train_dataset=dataset,
    )

    logger.info("[RC-GRPO] Starting RL training (Phase 2)...")
    trainer.train()

    out_dir = grpo_args.output_dir
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info(f"[RC-GRPO] Model saved → {out_dir}")
