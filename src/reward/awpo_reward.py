"""
awpo_reward.py
──────────────
AWPO Reward  (arxiv 2512.19126)
"Enhancing Tool-Use of LLMs through Adaptive Integration of Reasoning Rewards"

Core algorithm (from paper Section 3)
──────────────────────────────────────
AWPO computes a MIXED advantage that blends outcome and reasoning signals.
It has three gating / weighting mechanisms:

1. Variance-aware gating  (Eq. 23 in paper)
   ─────────────────────────────────────────
   Gate g_i controls HOW MUCH reasoning reward enters the advantage.

   σ²_out = variance of outcome rewards within the group
   τ       = threshold hyperparameter (default 0.5, tunable)

   g_i = 0   if  σ²_out  > τ   ← outcome rewards are spread out
                                   (already informative; don't add noise)
   g_i = 1   if  σ²_out ≤ τ   ← outcome variance is low
                                   (group is uniformly easy/hard;
                                    inject reasoning signal for guidance)

   The gating is HARD (binary): reasoning reward is either fully included
   or fully excluded per group. This prevents partial mixing that could
   destabilise training.

2. Difficulty-aware weighting  (Eq. 22 in paper)
   ───────────────────────────────────────────────
   μ_out = mean outcome reward in the group = proxy for difficulty

   w_i = 4 · μ_out · (1 - μ_out)   ← inverted-U: peaks at μ_out = 0.5
                                        (medium-difficulty groups get
                                         highest weight; easy/hard get less)

   This concentrates learning on the groups with the most headroom for
   improvement, following the theoretical upper bound derived in Eq. 20.

3. Mixed advantage with adaptive clipping
   ────────────────────────────────────────
   For each response i in a group:

   r_mix_i = (1 - g_i) · r_out_i  +  g_i · r_reason_i

   A_i = w_i · (r_mix_i - μ_mix) / (σ_mix + ε)

   The clipping radius in the PPO-style loss is scaled by (1 + g_i · δ)
   where δ is an adaptive expansion factor, because reasoning rewards
   are higher-variance and need a wider clip window to avoid excessive
   constraint.

Paper notation mapping to this code:
    r_out     ← binary outcome reward (func selection + args + exec)
    r_reason  ← reasoning quality score (CoT length, structure, etc.)
    g_i       ← variance gate (0 or 1, shared across the group)
    w_i       ← difficulty weight (scalar per group)
    A_i       ← final AWPO advantage fed to the policy loss
"""

from __future__ import annotations

import math
from .base_reward import (
    func_selection_ok, args_accuracy, schema_valid,
    format_reward, reasoning_quality
)


# ── Outcome reward ────────────────────────────────────────────────────────────

def _outcome_reward(
    response:        str,
    ground_truth:    dict,
    sandbox,
    outcome_weights: dict,
) -> float:
    """
    Weighted sum of verifiable outcome components.
    Default weights from paper ablation (Table 4): equal-weight components.
    """
    fn_ok    = func_selection_ok(response, ground_truth.get("function", ""))
    args_ok  = args_accuracy(response, ground_truth.get("arguments", {}))

    if sandbox is not None:
        from src.utils.sandbox import Sandbox
        exec_ok = 1.0 if sandbox.execute(response) else 0.0
    else:
        exec_ok = schema_valid(response)

    return (
        outcome_weights["func_selection"] * fn_ok
        + outcome_weights["args_accuracy"]  * args_ok
        + outcome_weights["execution"]      * exec_ok
    )


# ── Group-level statistics ────────────────────────────────────────────────────

def compute_group_stats(rewards: list[float]) -> dict:
    n    = len(rewards)
    mean = sum(rewards) / n
    var  = sum((r - mean) ** 2 for r in rewards) / n
    std  = math.sqrt(var)
    return {"mean": mean, "var": var, "std": std}


# ── Variance-aware gate  (Eq. 23) ─────────────────────────────────────────────

def variance_gate(sigma2_out: float, tau: float = 0.5) -> float:
    """
    Hard binary gate.
    g = 0 if variance > τ  (outcome is already informative → skip reasoning)
    g = 1 if variance ≤ τ  (outcome is low-variance → admit reasoning signal)
    """
    return 0.0 if sigma2_out > tau else 1.0


# ── Difficulty-aware weight  (Eq. 22) ─────────────────────────────────────────

def difficulty_weight(mu_out: float) -> float:
    """
    w = 4 · μ · (1 − μ)
    Peaks at μ = 0.5 (medium difficulty), approaches 0 at μ = 0 or 1.
    This is the inverted-U weighting derived from the policy-improvement
    upper bound in the paper.
    """
    return 4.0 * mu_out * (1.0 - mu_out)


# ── Full AWPO advantage computation for a group ───────────────────────────────

def awpo_group_advantages(
    outcome_rewards:   list[float],
    reasoning_rewards: list[float],
    tau:               float = 0.5,
    eps:               float = 1e-8,
) -> tuple[list[float], float]:
    """
    Compute AWPO advantages for one rollout group.

    Parameters
    ──────────
    outcome_rewards   : R_out for each response in the group
    reasoning_rewards : R_reason for each response in the group
    tau               : variance gate threshold
    eps               : numerical stability

    Returns
    ───────
    (advantages, adaptive_clip_expansion)
        advantages              : list[float], one per response
        adaptive_clip_expansion : δ factor to expand PPO clip radius
                                  when reasoning signal is active
    """
    n = len(outcome_rewards)
    assert n == len(reasoning_rewards), "Group sizes must match"

    # ── Step 1: group statistics on outcome rewards ───────────────────────────
    out_stats = compute_group_stats(outcome_rewards)
    mu_out    = out_stats["mean"]
    sigma2_out= out_stats["var"]

    # ── Step 2: variance-aware gate (shared for the whole group) ─────────────
    g = variance_gate(sigma2_out, tau)

    # ── Step 3: difficulty weight (shared for the whole group) ───────────────
    w = difficulty_weight(mu_out)

    # ── Step 4: mixed reward per response ────────────────────────────────────
    mixed = [
        (1.0 - g) * r_out + g * r_rea
        for r_out, r_rea in zip(outcome_rewards, reasoning_rewards)
    ]

    # ── Step 5: normalise mixed rewards within the group ─────────────────────
    mix_stats = compute_group_stats(mixed)
    mu_mix    = mix_stats["mean"]
    std_mix   = mix_stats["std"]

    # ── Step 6: weighted, normalised advantages ───────────────────────────────
    advantages = [
        w * (r - mu_mix) / (std_mix + eps)
        for r in mixed
    ]

    # ── Step 7: adaptive clip expansion factor ────────────────────────────────
    # When reasoning gate is open (g=1), clip window widens by δ=0.2
    # to accommodate higher-variance reasoning signal (paper Section 3.3)
    adaptive_clip_delta = g * 0.2

    return advantages, adaptive_clip_delta


# ── Per-response AWPO reward (single call, used in TRL wrapper) ──────────────

def awpo_reward(
    response:        str,
    ground_truth:    dict,
    sandbox=None,
    group_stats:     dict | None = None,
    outcome_weights: dict | None = None,
    tau:             float = 0.5,
) -> float:
    """
    Returns the scalar mixed reward for a single response.
    When group_stats are available (passed from trainer), the variance
    gate is applied; otherwise falls back to pure outcome reward.

    Note: Full AWPO advantage computation (with difficulty weighting and
    normalisation) happens inside AWPOTrainer.compute_advantages(), which
    operates over the whole group at once.  This function only returns
    the raw mixed reward signal that gets stored per response.
    """
    ow = outcome_weights or {
        "func_selection": 0.4,
        "args_accuracy":  0.3,
        "execution":      0.3,
    }

    r_out    = _outcome_reward(response, ground_truth, sandbox, ow)
    r_reason = reasoning_quality(response)

    if group_stats is None:
        return r_out   # no group context yet → return pure outcome

    sigma2_out = group_stats.get("var", 1.0)
    g          = variance_gate(sigma2_out, tau)

    return (1.0 - g) * r_out + g * r_reason


# ── TRL-compatible wrapper ────────────────────────────────────────────────────

def awpo_reward_func(
    completions:  list[str],
    ground_truth: list[dict] | None = None,
    **kwargs,
) -> list[float]:
    """
    TRL GRPOTrainer reward_funcs= compatible.

    Important: TRL calls this BEFORE advantage computation.
    We compute per-response outcome rewards here; the actual
    AWPO advantage (with variance gate + difficulty weight) is
    applied inside AWPOTrainer.compute_advantages().
    """
    if ground_truth is None:
        ground_truth = kwargs.get("ground_truths", [{}] * len(completions))

    outcome_weights = {
        "func_selection": 0.4,
        "args_accuracy":  0.3,
        "execution":      0.3,
    }

    # Compute raw outcome rewards for all completions
    out_rewards = [
        _outcome_reward(c, gt if isinstance(gt, dict) else {},
                        None, outcome_weights)
        for c, gt in zip(completions, ground_truth)
    ]
    return out_rewards