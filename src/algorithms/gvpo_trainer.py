"""
gvpo_trainer.py
────────────────
GVPO Trainer — Group Verification-based Policy Optimization
"Group Verification-based Policy Optimization for Interactive Coding Agents"
OpenReview: https://openreview.net/pdf?id=RY47Tq0VsV
ICLR 2026

What this file implements
──────────────────────────
The paper modifies GRPO's advantage tensor before the PPO-clip loss is applied.
Standard GRPO:
    All tokens in response i get the SAME advantage: Â_i = (R_i − μ_g)/σ_g

GVPO modification — two-stage advantage computation:

Stage 1 — Group-level outcome advantage  (identical to GRPO):
    Â_i = (R_out_i − μ_g) / (σ_g + ε)
    where μ_g, σ_g are the mean/std of outcome rewards across the rollout group.

Stage 2 — Step-level process shaping  (GVPO innovation):
    For each step t (each <call> block) in trajectory i:
        Ã_i,t = Â_i + b · φ(s_t, a_t)
    where:
        φ(s_t, a_t) = r_proc(t) − mean_t(r_proc)    [zero-mean within trajectory]
        b           = shaping_coeff  (hyperparameter)
    Tokens outside any <call> block retain the flat outcome advantage Â_i.

The PPO-clip loss operates on Ã_i,t (shaped) instead of Â_i (flat).
No other changes to the loss function — importance sampling ratio and
clipping are identical to GRPO.

TRL integration strategy
─────────────────────────
TRL's GRPOTrainer computes advantages inside the training loop and
stores them as a flat tensor (one scalar per response, broadcast to all tokens).
We override compute_loss to:
  1. Retrieve the flat advantage tensor built by the parent
  2. Re-shape it token-by-token using process-verifiable feedback
  3. Feed the shaped advantage into the standard PPO-clip loss

This override is minimal — we touch only the advantage tensor shape,
not the loss formula, clipping, or KL penalty.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from trl import GRPOTrainer

from .base_trainer import load_model, build_grpo_config, load_grpo_dataset
from src.reward.gvpo_reward import (
    gvpo_reward_func,
    build_per_token_advantages,
    compute_process_shaping,
    outcome_reward,
)
from src.reward.base_reward import format_reward
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────


class GVPOTrainer(GRPOTrainer):
    """
    GRPO trainer with GVPO step-level process shaping.

    The sole meaningful change vs. vanilla GRPOTrainer:
        advantage tensor is reshaped token-by-token before the PPO-clip loss.

    All other components — rollout generation, reward calling, KL penalty,
    clipping, logging — are inherited from GRPOTrainer unchanged.

    Parameters (via gvpo_config dict)
    ──────────────────────────────────
    shaping_coeff   : b in the paper  (default 1.0, ablated in Table 3)
                      Controls how strongly process feedback modifies the
                      per-token advantage relative to the outcome advantage.
                      b=0 degrades to vanilla GRPO.
    args_threshold  : minimum argument accuracy for a step to be "process-correct"
                      (default 0.8, matches our telecom domain)
    """

    def __init__(
        self,
        *args,
        gvpo_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        cfg = gvpo_config or {}

        # b — shaping coefficient (paper hyperparameter)
        # Paper default: b=1.0; ablation range: b ∈ {0.5, 1.0, 2.0}
        self.shaping_coeff = cfg.get("shaping_coeff", 1.0)

        # Per-step argument accuracy threshold for process reward
        self.args_threshold = cfg.get("args_threshold", 0.8)

        # Store ground truth refs for access inside compute_loss
        # (populated by _cache_ground_truths called from compute_loss)
        self._current_ground_truths: list[dict] = []
        self._current_completions: list[str] = []

        logger.info(
            f"[GVPO] shaping_coeff b={self.shaping_coeff}  "
            f"args_threshold={self.args_threshold}"
        )

    # ── Ground truth caching hook ─────────────────────────────────────────────

    def _cache_inputs(self, completions: list[str], ground_truths: list[dict]) -> None:
        """Cache completions and ground truths for use in compute_loss."""
        self._current_completions = completions
        self._current_ground_truths = ground_truths

    # ── Core override: reshape advantage tensor before PPO-clip loss ──────────

    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """
        Apply GVPO process shaping to the advantage tensor, then compute
        the standard PPO-clip loss on shaped advantages.

        The parent GRPOTrainer.compute_loss:
          1. Runs a forward pass to get current log-probs
          2. Computes per-token PPO-clip loss using inputs["advantages"]
          3. Applies KL penalty

        We intercept at step 2 by replacing inputs["advantages"] with
        the GVPO-shaped per-token advantage tensor before calling super().

        Fallback: if we cannot reshape (missing completions / ground truths),
        we fall back silently to vanilla GRPO by not modifying advantages.
        """
        # ── Attempt to apply GVPO shaping ────────────────────────────────────
        shaped = self._shape_advantages(inputs)
        if shaped is not None:
            # Replace the flat group-normalised advantages with shaped ones
            inputs["advantages"] = shaped

        # ── Standard PPO-clip loss on shaped advantages ───────────────────────
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def _shape_advantages(
        self,
        inputs: dict[str, Any],
    ) -> torch.Tensor | None:
        """
        Build the GVPO per-token shaped advantage tensor.

        Paper algorithm:
            Ã_i,t = Â_i + b · φ(s_t, a_t)

        where Â_i comes from the group-normalised advantages already
        computed by the parent trainer (stored in inputs["advantages"]).

        Returns
        ───────
        torch.Tensor of shape [B, T]  — shaped per-token advantages
        None                          — if shaping cannot be applied
                                        (falls back to flat GRPO advantages)
        """
        flat_advs = inputs.get("advantages")  # [B] or [B, T]
        attention_mask = inputs.get("attention_mask")  # [B, T]
        completions = self._current_completions
        ground_truths = self._current_ground_truths

        # Validate all required inputs are present
        if flat_advs is None or attention_mask is None:
            return None
        if not completions or not ground_truths:
            return None
        if len(completions) != len(ground_truths):
            return None

        B, T = attention_mask.shape
        if len(completions) != B:
            return None

        device = flat_advs.device

        # If advantages are already [B] (sequence-level), broadcast to [B, T]
        if flat_advs.dim() == 1:
            flat_advs_2d = flat_advs.unsqueeze(1).expand(B, T).clone()
        else:
            flat_advs_2d = flat_advs.clone()  # already [B, T]

        shaped_advs = flat_advs_2d.clone()

        # ── Apply step-level shaping per response ─────────────────────────────
        for i, (completion, gt) in enumerate(zip(completions, ground_truths)):
            if not isinstance(gt, dict):
                continue

            # Outcome advantage Â_i (scalar from group normalisation)
            outcome_adv = float(
                flat_advs[i] if flat_advs.dim() == 1 else flat_advs[i].mean()
            )

            # Compute φ(t) for each <call> block in this response
            phi_list = compute_process_shaping(
                response=completion,
                ground_truth=gt,
                sandbox=None,  # sandbox excluded from training loop
                args_threshold=self.args_threshold,
            )

            if not phi_list:
                # No call blocks found — keep flat outcome advantage
                continue

            # Find token spans for each <call> block
            try:
                from src.reward.gvpo_reward import _find_call_token_spans

                call_spans = _find_call_token_spans(
                    response=completion,
                    tokenizer=self.processing_class,
                    max_seq_len=T,
                )
            except Exception as e:
                logger.debug(f"[GVPO] Token span mapping failed for sample {i}: {e}")
                continue

            # Apply shaping: Ã_i,t = Â_i + b · φ(t) for call-block tokens
            for step_idx, (tok_start, tok_end) in enumerate(call_spans):
                if step_idx >= len(phi_list):
                    break
                shaped_value = outcome_adv + self.shaping_coeff * phi_list[step_idx]
                shaped_advs[i, tok_start:tok_end] = shaped_value

        return shaped_advs.to(device)

    # ── Hook to capture completions + ground truths from reward step ──────────

    def _compute_rewards(self, *args, **kwargs):
        """
        Intercept the reward computation step to cache completions and
        ground truths needed for process shaping in compute_loss.

        TRL calls this method to invoke reward_funcs and collect rewards.
        We capture the inputs here so that compute_loss can access them.
        """
        result = super()._compute_rewards(*args, **kwargs)

        # Try to extract completions and ground_truths from kwargs
        # (TRL passes these to reward_funcs as positional/keyword args)
        try:
            completions = kwargs.get("completions", args[0] if args else [])
            ground_truths = kwargs.get("ground_truth", [])
            if completions and ground_truths:
                self._cache_inputs(completions, ground_truths)
        except Exception:
            pass  # Non-fatal — shaping will fall back to flat advantages

        return result


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────


def train_gvpo(config: dict) -> None:
    """
    Full GVPO training run.

    Differences from vanilla GRPO:
      - Uses GVPOTrainer (process shaping) instead of GRPOTrainer
      - gvpo_reward_func returns outcome rewards (same interface as GRPO)
      - Process shaping happens silently inside compute_loss
    """
    gvpo_cfg = config.get("gvpo", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model, tokenizer = load_model(config)

    # ── 2. Load function library ──────────────────────────────────────────────
    with open(data_cfg["function_library_path"], "r") as fh:
        function_library = json.load(fh)

    # ── 3. Load + format dataset ──────────────────────────────────────────────
    dataset = load_grpo_dataset(data_cfg["train_path"], function_library, tokenizer)

    # ── 4. Build GRPOConfig ───────────────────────────────────────────────────
    grpo_args = build_grpo_config(
        config,
        output_dir=train_cfg.get("output_dir", "outputs/gvpo_model"),
    )

    # ── 5. Instantiate GVPOTrainer ────────────────────────────────────────────
    trainer = GVPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            gvpo_reward_func,  # outcome reward R_out(τ) ∈ {0,1}
            format_reward,  # XML format reward
        ],
        args=grpo_args,
        train_dataset=dataset,
        gvpo_config=gvpo_cfg,
    )

    logger.info("[GVPO] Starting training...")
    logger.info(
        f"[GVPO] shaping_coeff b={trainer.shaping_coeff}  "
        f"(b=0 degrades to vanilla GRPO)"
    )
    trainer.train()

    # ── 6. Save ───────────────────────────────────────────────────────────────
    out_dir = grpo_args.output_dir
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info(f"[GVPO] Model saved → {out_dir}")
