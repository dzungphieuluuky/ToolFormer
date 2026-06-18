"""
awpo_trainer.py
────────────────
AWPO Trainer  (arxiv 2512.19126)
"Enhancing Tool-Use of LLMs through Adaptive Integration of Reasoning Rewards"

Paper Implementation (Eq. 20-24)
───────────────────────────────────
For each group g:
  1. Compute outcome variance σ²_out and mixed variance σ²_mix
  2. Continuous mixing weight: ρ_g = σ_mix / (σ_out + σ_mix + ε)   [Eq. 20]
  3. Difficulty weight: w_g = 4·μ_out·(1−μ_out)                    [Eq. 22]
  4. Mixed reward: r_mix_i = (1-ρ_g)·r_out_i + ρ_g·r_reason_i
  5. Mixed advantage: A_i = w_g · (r_mix_i − μ_mix) / (σ_mix + ε)
  6. Dynamic clipping: ε_clip = original_eps / (1 + η·ρ_g)        [Eq. 24]
     where η is a hyperparameter (default 0.2, shrinks clip when reasoning is used)

We store ρ_g as `self._last_mixing_weight` for use in compute_loss.
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch
from trl import GRPOTrainer

from .base_trainer import load_model, build_grpo_config, load_grpo_dataset
from src.reward.awpo_reward import (
    awpo_reward_func,
    reasoning_quality,
)
from src.reward.base_reward import format_reward
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────


class AWPOTrainer(GRPOTrainer):
    """
    GRPO trainer with AWPO advantage computation.

    Key overrides
    ─────────────
    1. __init__          — stores AWPO hyperparams
    2. _generate_and_score_completions — caches reasoning rewards
    3. compute_advantages — replaces GRPO normalisation with AWPO algorithm
    4. compute_loss — shrinks clip radius when reasoning mixing weight is high
    """

    def __init__(
        self,
        *args,
        awpo_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        cfg = awpo_config or {}

        # η: clip shrink coefficient (Eq. 24)
        self.clip_shrink_coeff = cfg.get("clip_shrink_coeff", 0.2)

        # Outcome reward weights
        self.outcome_weights = cfg.get(
            "outcome_weights",
            {
                "func_selection": 0.4,
                "args_accuracy": 0.3,
                "execution": 0.3,
            },
        )

        # Cache for reasoning rewards
        self._reasoning_cache: dict[str, float] = {}

        # Last computed mixing weight ρ (average across groups)
        self._last_mixing_weight: float = 0.0

        logger.info(f"[AWPO] clip_shrink_coeff η={self.clip_shrink_coeff}")

    # ── Populate reasoning cache during rollout generation ────────────────────

    def _generate_and_score_completions(self, prompts, **kwargs):
        completions, rewards = super()._generate_and_score_completions(
            prompts, **kwargs
        )
        for comp in completions:
            if comp not in self._reasoning_cache:
                self._reasoning_cache[comp] = reasoning_quality(comp)
        return completions, rewards

    # ── AWPO advantage computation ────────────────────────────────────────────

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_indices: torch.Tensor,
        completions: list[str] | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        AWPO advantage with continuous variance gate and difficulty weighting.

        Implements Eq.20-24 more faithfully by iteratively solving for
        ρ = σ_mix / (σ_out + σ_mix + eps).  Uses a small fixed-point loop
        to compute ρ per group, then forms the mixed reward and advantages.
        """
        B = rewards.shape[0]
        advantages = torch.zeros(B, dtype=torch.float32)
        mixing_weights: list[float] = []

        unique_groups = group_indices.unique().tolist()
        eps = 1e-8

        for gid in unique_groups:
            mask = (group_indices == gid).nonzero(as_tuple=True)[0]
            g_idx = mask.tolist()

            # Outcome rewards
            out_r = [float(rewards[i]) for i in g_idx]

            # Reasoning rewards
            if completions is not None:
                reason_r = [
                    self._reasoning_cache.get(completions[i], 0.0) for i in g_idx
                ]
            else:
                reason_r = [0.0] * len(g_idx)

            # Group statistics
            n = len(out_r)
            mu_out = sum(out_r) / n
            sigma2_out = sum((r - mu_out) ** 2 for r in out_r) / n
            sigma_out = math.sqrt(sigma2_out + eps)

            # Iteratively solve for rho = sigma_mix / (sigma_out + sigma_mix + eps)
            # Start with rho = 0.5*(1 - sigma_out) clipped to [0,1]
            rho = max(0.0, min(1.0, 0.5 * (1.0 - sigma_out)))
            for _ in range(10):
                mixed = [
                    (1 - rho) * r_out + rho * r_rea for r_out, r_rea in zip(out_r, reason_r)
                ]
                mu_mix = sum(mixed) / n
                sigma_mix = math.sqrt(sum((r - mu_mix) ** 2 for r in mixed) / n + eps)
                new_rho = sigma_mix / (sigma_out + sigma_mix + eps)
                if abs(new_rho - rho) < 1e-4:
                    rho = new_rho
                    break
                rho = new_rho

            # Difficulty weight
            w = 4.0 * mu_out * (1.0 - mu_out)

            # Final mixed rewards and normalisation
            mixed = [
                (1 - rho) * r_out + rho * r_rea for r_out, r_rea in zip(out_r, reason_r)
            ]
            mu_mix = sum(mixed) / n
            sigma_mix = math.sqrt(sum((r - mu_mix) ** 2 for r in mixed) / n + eps)
            sigma_mix = max(sigma_mix, eps)

            # AWPO advantages
            group_advs = [w * (r - mu_mix) / sigma_mix for r in mixed]

            for local_idx, global_idx in enumerate(g_idx):
                advantages[global_idx] = group_advs[local_idx]

            mixing_weights.append(rho)

        # Store average mixing weight for clip shrinking
        self._last_mixing_weight = sum(mixing_weights) / max(len(mixing_weights), 1)
        return advantages, self._last_mixing_weight

    # ── Adaptive clipping (shrink when mixing weight is high) ─────────────────

    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """
        Shrink the clip radius when reasoning mixing weight is high.
        Eq. 24: ε_clip = ε / (1 + η·ρ_avg)
        """
        rho = getattr(self, "_last_mixing_weight", 0.0)
        original_eps = self.args.epsilon

        if rho > 0.0:
            # Shrink clip radius
            self.args.epsilon = original_eps / (1.0 + self.clip_shrink_coeff * rho)

        loss_output = super().compute_loss(
            model, inputs, return_outputs, num_items_in_batch
        )

        # Restore
        self.args.epsilon = original_eps
        return loss_output


# ──────────────────────────────────────────────────────────────────────────────


def train_awpo(config: dict) -> None:
    awpo_cfg = config.get("awpo", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    model, tokenizer = load_model(config)

    with open(data_cfg["function_library_path"], "r") as fh:
        function_library = json.load(fh)

    dataset = load_grpo_dataset(data_cfg["train_path"], function_library, tokenizer)

    grpo_args = build_grpo_config(
        config,
        output_dir=train_cfg.get("output_dir", "outputs/awpo_model"),
    )

    trainer = AWPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            awpo_reward_func,  # outcome reward (func sel + args + exec)
            format_reward,  # XML format reward
        ],
        args=grpo_args,
        train_dataset=dataset,
        awpo_config=awpo_cfg,
    )

    logger.info("[AWPO] Starting training...")
    trainer.train()

    out_dir = grpo_args.output_dir
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info(f"[AWPO] Model saved → {out_dir}")
