"""
awpo_trainer.py
────────────────
AWPO Trainer  (arxiv 2512.19126)
"Enhancing Tool-Use of LLMs through Adaptive Integration of Reasoning Rewards"

What this file implements (paper Section 3)
────────────────────────────────────────────
Standard GRPOTrainer computes:
    A_i = (R_i - μ_g) / (σ_g + ε)     [group-normalised advantage]

AWPOTrainer overrides compute_advantages to instead compute:

    For each group g:
        1. σ²_out = Var(R_out_i for i in g)
        2. g_gate  = 0 if σ²_out > τ else 1         [Eq. 23 — variance gate]
        3. w        = 4 · μ_out · (1 − μ_out)        [Eq. 22 — difficulty weight]
        4. r_mix_i  = (1-g_gate)·R_out_i + g_gate·R_reason_i
        5. A_i      = w · (r_mix_i − μ_mix) / (σ_mix + ε)

And overrides the PPO clip radius:
    ε_clip → ε_clip · (1 + g_gate · δ)              [adaptive clipping]
    where δ = 0.2 (wider window when reasoning signal is active)

This file also stores per-completion reasoning rewards so that
compute_advantages can access both reward components simultaneously.
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
    awpo_group_advantages,
    compute_group_stats,
    reasoning_quality as _rq,
)
from src.reward.base_reward import format_reward, reasoning_quality
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
    4. compute_loss — widens clip when gate is open (uses self._last_clip_delta)
    """

    def __init__(
        self,
        *args,
        awpo_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        cfg = awpo_config or {}

        # Variance gate threshold τ (Eq. 23)
        self.tau = cfg.get("tau", 0.5)

        # Adaptive clip expansion factor δ (Section 3.3)
        self.adaptive_clip_delta = cfg.get("adaptive_clip_delta", 0.2)

        # Outcome reward weights
        self.outcome_weights = cfg.get(
            "outcome_weights",
            {
                "func_selection": 0.4,
                "args_accuracy": 0.3,
                "execution": 0.3,
            },
        )

        # Cache for reasoning rewards (populated by _generate_and_score_completions)
        # Maps completion text → reasoning score
        self._reasoning_cache: dict[str, float] = {}

        # Last computed adaptive clip delta (set in compute_advantages)
        self._last_clip_delta: float = 0.0

        logger.info(f"[AWPO] τ={self.tau}  δ_clip={self.adaptive_clip_delta}")

    # ── Populate reasoning cache during rollout generation ────────────────────

    def _generate_and_score_completions(self, prompts, **kwargs):
        """
        Override to compute reasoning scores for each completion.
        TRL calls this to generate rollouts and compute rewards.
        We interleave to fill the reasoning cache.
        """
        completions, rewards = super()._generate_and_score_completions(prompts, **kwargs)
        for comp in completions:
            if comp not in self._reasoning_cache:
                self._reasoning_cache[comp] = reasoning_quality(comp)
        return completions, rewards

    # ── AWPO advantage computation  (core override) ───────────────────────────

    def compute_advantages(
        self,
        rewards: torch.Tensor,          # [B]  outcome rewards from reward_funcs
        group_indices: torch.Tensor,    # [B]  which group each sample belongs to
        completions: list[str] | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Replace standard GRPO (R-μ)/σ normalisation with AWPO algorithm.

        Returns
        ───────
        (advantages, adaptive_clip_expansion)
            advantages              : [B] float tensor
            adaptive_clip_expansion : scalar δ (mean over all groups)
                                      used to widen clip in compute_loss
        """
        B = rewards.shape[0]
        advantages = torch.zeros(B, dtype=torch.float32)
        clip_deltas: list[float] = []

        unique_groups = group_indices.unique().tolist()

        for gid in unique_groups:
            mask = (group_indices == gid).nonzero(as_tuple=True)[0]
            g_idx = mask.tolist()

            # Outcome rewards for this group
            out_r = [float(rewards[i]) for i in g_idx]

            # Reasoning rewards for this group
            if completions is not None:
                reason_r = [
                    self._reasoning_cache.get(completions[i], 0.0) for i in g_idx
                ]
            else:
                # Fallback: if completions not provided, compute on the fly (expensive)
                reason_r = [0.0] * len(g_idx)

            # Compute AWPO advantages for this group
            group_advs, clip_delta = awpo_group_advantages(
                outcome_rewards=out_r,
                reasoning_rewards=reason_r,
                tau=self.tau,
            )

            for local_idx, global_idx in enumerate(g_idx):
                advantages[global_idx] = group_advs[local_idx]
            clip_deltas.append(clip_delta)

        # Store the mean clip delta for use in compute_loss
        self._last_clip_delta = sum(clip_deltas) / max(len(clip_deltas), 1)
        return advantages, self._last_clip_delta

    # ── Adaptive clipping in loss ─────────────────────────────────────────────

    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """
        Intercept compute_loss to apply adaptive clip expansion when
        the variance gate is open (reasoning signal active).

        When g=1 (gate open), the reasoning rewards have higher variance
        than outcome rewards; the standard ε clip is too tight and causes
        premature saturation.  We widen it by (1 + δ) per the paper.
        """
        clip_delta = getattr(self, "_last_clip_delta", 0.0)

        if clip_delta > 0.0:
            original_eps = self.args.epsilon
            # Temporarily widen clip
            self.args.epsilon = original_eps * (1.0 + clip_delta)

        loss_output = super().compute_loss(
            model, inputs, return_outputs, num_items_in_batch
        )

        if clip_delta > 0.0:
            self.args.epsilon = original_eps  # restore

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