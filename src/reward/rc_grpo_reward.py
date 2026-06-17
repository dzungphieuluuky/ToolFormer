"""
rc_grpo_reward.py
──────────────────
RC-GRPO Reward  (arxiv 2602.03025)

The paper's core insight
─────────────────────────
Standard GRPO stalls on tool-calling because groups collapse to
all-0 or all-1 reward → normalised advantage A_j = R_j - μ_g ≈ 0
→ vanishing gradient.

RC-GRPO solution (two-phase)
────────────────────────────
Phase 1 — RCTP-FT (done in run_sft.py):
    Fine-tune on mixed-quality trajectories where the reward goal token
    <|high_reward|> or <|low_reward|> is prepended to every prompt.
    Objective: min_θ Σ -log π_θ(a | h, r)
    This teaches the model to produce distinct quality levels on demand.

Phase 2 — RC-GRPO RL (done here):
    Within each GRPO rollout group, SAMPLE diverse reward tokens so that
    roughly half the rollouts are conditioned on <|high_reward|> and half
    on <|low_reward|>.  This deliberately injects within-group diversity,
    restoring a non-zero advantage signal even on hard prompts.

    The reward function itself is still binary (0/1) — the novelty is
    entirely in the rollout sampling strategy, not the reward shape.

Paper equation (advantage after reward-token conditioning):
    A_j = R_j - μ_g     where μ_g = mean over the group
    Because tokens force diversity, R_j varies → A_j ≠ 0.
"""

from __future__ import annotations
from .base_reward import schema_valid, func_selection_ok, args_accuracy, format_reward

# Special tokens — must be added to tokenizer vocabulary in RCTP-FT phase
HIGH_REWARD_TOKEN = "<|high_reward|>"
LOW_REWARD_TOKEN = "<|low_reward|>"


# ── Core reward ───────────────────────────────────────────────────────────────


def rc_grpo_reward(
    response: str,
    ground_truth: dict,
    sandbox=None,
    args_threshold: float = 0.8,
) -> float:
    """
    Binary reward: 1.0 iff schema valid AND correct function AND
    args_accuracy ≥ threshold AND (optionally) execution succeeds.

    This is the verifiable outcome reward R_j used in the paper.
    The variance that drives learning comes from the reward-token
    sampling strategy, not from reward shaping.
    """
    if not schema_valid(response):
        return 0.0
    if not func_selection_ok(response, ground_truth.get("function", "")):
        return 0.0
    if args_accuracy(response, ground_truth.get("arguments", {})) < args_threshold:
        return 0.0
    if sandbox is not None:
        from src.utils.sandbox import Sandbox

        if not sandbox.execute(response):
            return 0.0
    return 1.0


# ── TRL-compatible wrappers ───────────────────────────────────────────────────


def rc_grpo_reward_func(
    completions: list[str],
    ground_truth: list[dict] | None = None,
    **kwargs,
) -> list[float]:
    if ground_truth is None:
        ground_truth = kwargs.get("ground_truths", [{}] * len(completions))
    return [
        rc_grpo_reward(c, gt if isinstance(gt, dict) else {})
        for c, gt in zip(completions, ground_truth)
    ]


def rc_grpo_format_func(completions: list[str], **kwargs) -> list[float]:
    return format_reward(completions, **kwargs)
