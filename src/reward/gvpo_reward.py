"""
gvpo_reward.py
──────────────
GVPO — Group Verification-based Policy Optimization
"Group Verification-based Policy Optimization for Interactive Coding Agents"
OpenReview: https://openreview.net/pdf?id=RY47Tq0VsV
ICLR 2026

╔══════════════════════════════════════════════════════════════════════════════╗
║                        WHAT THIS PAPER ACTUALLY IS                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  GVPO extends GRPO by replacing its flat, outcome-only advantage with a     ║
║  SHAPED advantage that integrates TWO verifiable signal types:              ║
║                                                                             ║
║  1. OUTCOME-VERIFIABLE reward  R_out(τ)                                     ║
║     Binary signal at the END of the trajectory:                             ║
║     Did the full agent trajectory succeed?                                  ║
║     (e.g., did all function calls execute correctly and solve the query?)   ║
║                                                                             ║
║  2. PROCESS-VERIFIABLE reward  r_proc(s_t, a_t)  per intermediate step     ║
║     Dense signal at EACH reasoning/action step:                             ║
║     Did this specific intermediate step succeed when executed?              ║
║     (e.g., did this individual function call parse and execute without      ║
║      errors? syntax valid? arguments within constraints?)                   ║
║                                                                             ║
║  GRPO's problem: it broadcasts the same advantage A_i to ALL tokens in a   ║
║  response, regardless of which steps were good or bad. This gives the same  ║
║  gradient weight to a correct function call and a malformed one inside the  ║
║  same trajectory — inaccurate credit assignment.                            ║
║                                                                             ║
║  GVPO's fix: shape the advantage at the STEP level using process feedback,  ║
║  so each reasoning step gets credit proportional to its own correctness.    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Core Algorithm (paper Section 3)
──────────────────────────────────
GRPO baseline advantage (flat, outcome-only):

    Â_i = (R_out(τ_i) − μ_g) / (σ_g + ε)       ← same for every token in τ_i

GVPO shaped advantage at step t of trajectory i:

    Ã_i,t = Â_i + b · φ(s_t, a_t)               ← (paper Eq. 4 / Fig. 1)

where:
    Â_i     = group-normalised OUTCOME advantage (identical to GRPO baseline)
    b       = shaping coefficient  (hyperparameter, paper uses b=1.0 default,
                                    ablated in paper Table 3 / RQ4)
    φ(s,a)  = step-level process-verifiable shaping term (defined below)

Process shaping term φ(s_t, a_t)  (paper Section 3.2):

    φ(s_t, a_t) = r_proc(s_t, a_t) − μ_proc_i

where:
    r_proc(s_t, a_t) = per-step process reward ∈ {0, 1}
                       1 if the action at step t passes ALL checks:
                         • schema validity (parseable call block)
                         • correct function selection
                         • argument accuracy ≥ threshold
                         • execution success (sandbox)
                       0 otherwise

    μ_proc_i = mean of r_proc over all steps in trajectory i
               (group-mean analogue, but within the trajectory,
                so φ is zero-mean within each trajectory —
                this prevents the shaping term from shifting
                the overall advantage baseline)

Advantage assignment to tokens (paper Section 3.3):
    Each token t in trajectory i gets Ã_i,t:
      • Tokens inside a <reasoning> block  → Â_i  (outcome advantage only,
                                                    no process shaping, since
                                                    reasoning is not directly
                                                    executable)
      • Tokens inside a <call> block at step t → Ã_i,t = Â_i + b·φ(s_t, a_t)
      • All other tokens                       → Â_i

This means the gradient signal for function-call tokens is STRONGER when the
call executed correctly (φ > 0) and WEAKER (possibly negative) when it failed
(φ < 0), within the same trajectory.

Adaptation for Telecom Function Calling
─────────────────────────────────────────
The paper targets multi-turn interactive coding agents (AppWorld benchmark).
We adapt to our telecom tool-calling setting:

  • "trajectory" = one complete model response (may contain 1 or more <call> blocks)
  • "step t"     = one <call>...</call> block within the response
  • "process reward" = did this specific call pass schema + function + args + exec?
  • "outcome reward" = did the FULL response satisfy the ground truth?

Key properties preserved from the paper:
  ✓ Zero-mean shaping within each trajectory (φ is centred)
  ✓ Outcome advantage Â_i is identical to vanilla GRPO (additive shaping)
  ✓ Shaping coefficient b is the sole new hyperparameter
  ✓ Token-level granularity: call tokens get shaped, non-call tokens don't
  ✓ No importance-sampling ratio changes — only the advantage signal is modified
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from .base_reward import (
    extract_call,
    extract_all_calls,
    args_accuracy,
    schema_valid,
    func_selection_ok,
    format_reward,
    _CALL_RE,
    _REASON_RE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Step-level process reward
# ──────────────────────────────────────────────────────────────────────────────

def process_reward_step(
    call:               dict,
    ground_truth:       dict,
    sandbox=None,
    args_threshold:     float = 0.8,
) -> float:
    """
    Binary process-verifiable reward for ONE step (one <call> block).

    Returns 1.0 if ALL of the following pass, else 0.0:
      1. Schema valid (the call dict is well-formed)
      2. Correct function selected
      3. Argument accuracy ≥ args_threshold
      4. Execution success (if sandbox provided)

    This is r_proc(s_t, a_t) in the paper.
    """
    # Gate 1: schema — call must be a non-None dict
    if not isinstance(call, dict) or not call:
        return 0.0

    # Gate 2: correct function
    if call.get("function") != ground_truth.get("function", ""):
        return 0.0

    # Gate 3: argument accuracy
    if args_accuracy(call, ground_truth.get("arguments", {})) < args_threshold:
        return 0.0

    # Gate 4: execution (optional)
    if sandbox is not None:
        from src.utils.sandbox import Sandbox
        if not sandbox.execute(call):
            return 0.0

    return 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Outcome reward (trajectory-level)
# ──────────────────────────────────────────────────────────────────────────────

def outcome_reward(
    response:     str,
    ground_truth: dict,
    sandbox=None,
    args_threshold: float = 0.8,
) -> float:
    """
    Binary outcome-verifiable reward R_out(τ) for the FULL trajectory.

    Returns 1.0 only if EVERY <call> block in the response passes all checks.
    This is the τ-level reward used for group normalisation (identical to
    vanilla GRPO's reward signal).
    """
    calls = extract_all_calls(response)
    if not calls:
        # No call produced — fails for non-abstention samples
        return 0.0 if ground_truth.get("workflow") != "abstention" else 1.0

    for call in calls:
        if process_reward_step(call, ground_truth, sandbox, args_threshold) < 1.0:
            return 0.0
    return 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Process shaping term φ(s_t, a_t) per trajectory
# ──────────────────────────────────────────────────────────────────────────────

def compute_process_shaping(
    response:       str,
    ground_truth:   dict,
    sandbox=None,
    args_threshold: float = 0.8,
) -> list[float]:
    """
    Compute the zero-mean process shaping term φ for each step in a trajectory.

    Steps
    ─────
    1. Extract all <call> blocks from the response → steps [c_1, c_2, ..., c_T]
    2. Compute r_proc(t) for each step t  ∈ {0, 1}
    3. Centre:  φ(t) = r_proc(t) − mean(r_proc)
       This ensures Σ_t φ(t) = 0, preserving the outcome advantage baseline.

    Returns
    ───────
    list[float] of length T (number of <call> blocks found).
    Empty list if no calls found (shaping term is vacuous).

    Note: The returned values are ALREADY centred (zero-mean within trajectory).
    The trainer maps each φ(t) back to the token positions of the t-th <call>
    block when constructing the per-token advantage tensor.
    """
    calls = extract_all_calls(response)
    if not calls:
        return []

    # Step-level process rewards r_proc(t) ∈ {0, 1}
    step_rewards = [
        process_reward_step(call, ground_truth, sandbox, args_threshold)
        for call in calls
    ]

    # Centre within the trajectory: φ(t) = r_proc(t) − μ_proc
    mu_proc = sum(step_rewards) / len(step_rewards)
    phi     = [r - mu_proc for r in step_rewards]

    return phi


# ──────────────────────────────────────────────────────────────────────────────
# Token-level advantage assignment
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GVPOTokenAdvantages:
    """
    Holds the per-token shaped advantages for one response.

    Fields
    ──────
    outcome_advantage   : Â_i  — scalar, same for all tokens  (from GRPO)
    per_token_shaped    : Ã_i,t — [seq_len] tensor, token-level shaped advantage
    step_phis           : list[float] — one φ value per <call> block
    step_process_rewards: list[float] — raw r_proc before centering
    """
    outcome_advantage:    float
    per_token_shaped:     torch.Tensor           # shape [seq_len]
    step_phis:            list[float]
    step_process_rewards: list[float]


def build_per_token_advantages(
    response:         str,
    tokenizer,
    outcome_adv:      float,
    ground_truth:     dict,
    sandbox=None,
    shaping_coeff:    float = 1.0,
    args_threshold:   float = 0.8,
    max_seq_len:      int   = 512,
) -> GVPOTokenAdvantages:
    """
    Build the full per-token shaped advantage tensor for one response.

    Token assignment rules (paper Section 3.3):
    ──────────────────────────────────────────────
    • Tokens in <call>...</call> block t  →  Â_i + b · φ(t)
    • All other tokens (reasoning, text)  →  Â_i  (outcome only)

    Parameters
    ──────────
    response       : raw model output string
    tokenizer      : HF tokenizer (for span → token-position mapping)
    outcome_adv    : Â_i from group-normalised GRPO advantage
    ground_truth   : {"function": ..., "arguments": ...}
    sandbox        : optional Sandbox instance
    shaping_coeff  : b  (paper hyperparameter, default 1.0)
    args_threshold : threshold for process_reward_step
    max_seq_len    : maximum sequence length to allocate

    Returns
    ───────
    GVPOTokenAdvantages with per_token_shaped tensor of shape [seq_len]
    """
    # Compute process shaping terms φ(t) for each <call> block
    phi_list = compute_process_shaping(response, ground_truth, sandbox, args_threshold)

    # Raw process rewards for logging/debugging
    calls = extract_all_calls(response)
    raw_proc = [
        process_reward_step(c, ground_truth, sandbox, args_threshold)
        for c in calls
    ]

    # Initialise per-token advantages to Â_i (outcome-only, like GRPO)
    per_token = torch.full((max_seq_len,), fill_value=outcome_adv, dtype=torch.float32)

    if not phi_list:
        # No calls found — all tokens get outcome advantage only
        return GVPOTokenAdvantages(
            outcome_advantage    = outcome_adv,
            per_token_shaped     = per_token,
            step_phis            = [],
            step_process_rewards = [],
        )

    # Map <call> character spans → token positions
    # We use a simple approach: tokenise the full response, then find
    # which token indices correspond to each <call>...</call> span.
    try:
        call_token_spans = _find_call_token_spans(response, tokenizer, max_seq_len)
    except Exception:
        # Fallback: if tokenisation mapping fails, use outcome advantage only
        return GVPOTokenAdvantages(
            outcome_advantage    = outcome_adv,
            per_token_shaped     = per_token,
            step_phis            = phi_list,
            step_process_rewards = raw_proc,
        )

    # Apply shaping: tokens in <call> block t → Â_i + b · φ(t)
    for step_idx, (tok_start, tok_end) in enumerate(call_token_spans):
        if step_idx >= len(phi_list):
            break
        shaped_value = outcome_adv + shaping_coeff * phi_list[step_idx]
        per_token[tok_start:tok_end] = shaped_value

    return GVPOTokenAdvantages(
        outcome_advantage    = outcome_adv,
        per_token_shaped     = per_token,
        step_phis            = phi_list,
        step_process_rewards = raw_proc,
    )


def _find_call_token_spans(
    response:    str,
    tokenizer,
    max_seq_len: int,
) -> list[tuple[int, int]]:
    """
    Find the token-index spans [start, end) for each <call>...</call> block.

    Uses the tokenizer's offset mapping to go from character positions
    to token positions precisely.
    """
    # Tokenise with offset mapping
    enc = tokenizer(
        response,
        return_offsets_mapping = True,
        truncation             = True,
        max_length             = max_seq_len,
        add_special_tokens     = False,
    )
    offsets = enc["offset_mapping"]   # list of (char_start, char_end) per token

    spans: list[tuple[int, int]] = []

    for match in _CALL_RE.finditer(response):
        char_start = match.start()   # position of '<' in '<call>'
        char_end   = match.end()     # position after '>' in '</call>'

        tok_start = None
        tok_end   = None
        for tok_idx, (off_s, off_e) in enumerate(offsets):
            if tok_start is None and off_e > char_start:
                tok_start = tok_idx
            if off_s < char_end:
                tok_end = tok_idx + 1

        if tok_start is not None and tok_end is not None:
            spans.append((tok_start, min(tok_end, max_seq_len)))

    return spans


# ──────────────────────────────────────────────────────────────────────────────
# TRL-compatible reward wrapper
# ──────────────────────────────────────────────────────────────────────────────

def gvpo_reward_func(
    completions:  list[str],
    ground_truth: list[dict] | None = None,
    **kwargs,
) -> list[float]:
    """
    TRL reward_funcs= compatible wrapper.

    Returns per-response OUTCOME rewards R_out(τ) ∈ {0, 1}.
    These are used by GRPOTrainer to compute group-normalised advantages Â_i.

    The PROCESS shaping (φ terms and per-token shaped advantages Ã_i,t)
    is computed inside GVPOTrainer.compute_loss(), which has access to
    both the model outputs and the token-level structure needed for mapping.

    The separation is intentional:
      - reward_funcs → outcome reward only (standard TRL interface)
      - compute_loss → applies process shaping to the advantage tensor
    """
    if ground_truth is None:
        ground_truth = kwargs.get("ground_truths", [{}] * len(completions))

    return [
        outcome_reward(c, gt if isinstance(gt, dict) else {})
        for c, gt in zip(completions, ground_truth)
    ]