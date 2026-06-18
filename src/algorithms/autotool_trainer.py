"""
autotool_trainer.py
────────────────────
AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints
ICLR 2026  (arxiv 2603.13348)

Paper Implementation (from abstract and discussion)
─────────────────────────────────────────────────────
- SFT warm-up: pre-train on mixed-difficulty tasks to help model distinguish easy vs hard.
- RL with decoupled entropy constraints: separate entropy penalties for reasoning tokens vs tool-call tokens.
- Long-short reasoning fusion: adaptively adjust reasoning length based on query difficulty.
- We implement: (1) entropy bonus in loss with token-type specific coefficients;
  (2) length penalty in reward based on workflow_type (single/parallel/sequential).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import torch
import torch.nn.functional as F
from trl import GRPOTrainer

from .base_trainer import load_model, build_grpo_config, load_grpo_dataset
from src.reward.base_reward import format_reward, extract_reasoning, extract_call
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Reward function with length penalty
# ──────────────────────────────────────────────────────────────────────────────


def autotool_reward_func(
    completions: list[str],
    ground_truth: list[dict] | None = None,
    **kwargs,
) -> list[float]:
    """
    Outcome reward (binary) with a length penalty based on workflow difficulty.
    Penalises long responses for simple (single_call) tasks.
    """
    if ground_truth is None:
        ground_truth = kwargs.get("ground_truths", [{}] * len(completions))

    from src.reward.base_reward import func_selection_ok, args_accuracy, schema_valid

    rewards = []
    for comp, gt in zip(completions, ground_truth):
        if not isinstance(gt, dict):
            gt = {}
        # Outcome reward (binary)
        ok = (
            schema_valid(comp)
            and func_selection_ok(comp, gt.get("function", "")) == 1.0
            and args_accuracy(comp, gt.get("arguments", {})) >= 0.8
        )
        outcome = 1.0 if ok else 0.0

        # Length penalty: penalise overly long responses for simple tasks
        workflow_type = gt.get("workflow", "single_call")
        length = len(comp.split())
        if workflow_type == "single_call":
            # Simpler tasks should have shorter responses
            penalty = max(
                0.0, (length - 100) / 200.0
            )  # penalty for length > 100 tokens
        elif workflow_type == "parallel":
            penalty = max(0.0, (length - 150) / 300.0)
        else:  # sequential or abstention
            penalty = 0.0  # no penalty for complex tasks

        # Combine: outcome minus penalty, clipped to [0,1]
        reward = max(0.0, min(1.0, outcome - 0.1 * penalty))
        rewards.append(reward)

    return rewards


# ──────────────────────────────────────────────────────────────────────────────
# AutoTool Trainer
# ──────────────────────────────────────────────────────────────────────────────


class AutoToolTrainer(GRPOTrainer):
    """
    Extends GRPOTrainer with:
      1. Decoupled entropy constraints: separate entropy coefficients for reasoning vs tool tokens.
      2. Length penalty integrated into reward via autotool_reward_func.
    """

    def __init__(
        self,
        *args,
        autotool_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        cfg = autotool_config or {}

        # Entropy coefficients for different token types
        self.entropy_coeff_reasoning = cfg.get("entropy_coeff_reasoning", 0.01)
        self.entropy_coeff_tool = cfg.get("entropy_coeff_tool", 0.005)
        self.entropy_coeff_other = cfg.get("entropy_coeff_other", 0.001)

        logger.info(
            f"[AutoTool] entropy_coeff_reasoning={self.entropy_coeff_reasoning}  "
            f"tool={self.entropy_coeff_tool}  other={self.entropy_coeff_other}"
        )

    # ── Core override: add entropy bonus to loss ──────────────────────────────

    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """
        Override loss to add token-type weighted entropy bonus.
        We compute entropy per token from logits, mask by token type,
        and add to the PPO loss.
        """
        # First, get the standard PPO loss from parent
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Compute entropy bonus
        # We need logits from the forward pass; outputs should contain logits
        logits = outputs.logits if hasattr(outputs, "logits") else None
        if logits is None:
            return loss if not return_outputs else (loss, outputs)

        # Get attention mask and token type mask
        attention_mask = inputs.get("attention_mask")
        # We need to classify tokens: reasoning, tool_call, or other.
        # We'll use the token IDs to map to token types.
        # This is simplified; we assume the tokenizer has been extended with special tokens.
        # We'll use a heuristic: token IDs for known tags.
        # For a robust implementation, we'd pre-compute token spans during data loading.
        # Here we use a simplified version: we check if the token is part of a reasoning or call block.
        # Since we don't have span info, we'll approximate by checking if the token is a tag token.
        # We'll assume we have special tokens <reasoning>, </reasoning>, <tool_call>, </tool_call>.
        # We'll get token IDs for these and create a mask.

        tokenizer = self.processing_class
        try:
            reasoning_start = tokenizer.convert_tokens_to_ids("<reasoning>")
            reasoning_end = tokenizer.convert_tokens_to_ids("</reasoning>")
            call_start = tokenizer.convert_tokens_to_ids("<tool_call>")
            call_end = tokenizer.convert_tokens_to_ids("</tool_call>")
        except Exception:
            # Fallback: no special tokens → treat all tokens as 'other'
            reasoning_start = reasoning_end = call_start = call_end = None

        # We'll create masks using token IDs (the input_ids)
        input_ids = inputs.get("input_ids")
        if input_ids is None or reasoning_start is None:
            # Fallback: uniform entropy coefficient
            coeff = self.entropy_coeff_other
            if logits is not None and attention_mask is not None:
                # Compute entropy per token
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, T]
                entropy_loss = (
                    -coeff * (entropy * attention_mask).sum() / attention_mask.sum()
                )
                loss = loss + entropy_loss
            return loss if not return_outputs else (loss, outputs)

        # We'll create token-type masks (reasoning, tool, other)
        # This is a simplified version; in production, we'd use token spans from the response.
        # We'll treat tokens between <reasoning> and </reasoning> as reasoning,
        # and between <tool_call> and </tool_call> as tool.
        # This requires tracking span state; we'll do it token by token.

        B, T = input_ids.shape
        reasoning_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        tool_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # For each sequence, track whether we are inside a reasoning or tool block
        for b in range(B):
            in_reasoning = False
            in_tool = False
            for t in range(T):
                token_id = input_ids[b, t].item()
                if token_id == reasoning_start:
                    in_reasoning = True
                elif token_id == reasoning_end:
                    in_reasoning = False
                elif token_id == call_start:
                    in_tool = True
                elif token_id == call_end:
                    in_tool = False
                # Mark tokens inside blocks
                if in_reasoning:
                    reasoning_mask[b, t] = True
                elif in_tool:
                    tool_mask[b, t] = True

        # Compute entropy per token
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, T]

        # Apply attention mask and coefficient masks
        mask = attention_mask.bool()
        entropy_reasoning = (entropy * reasoning_mask * mask).sum()
        entropy_tool = (entropy * tool_mask * mask).sum()
        entropy_other = (entropy * (~(reasoning_mask | tool_mask)) * mask).sum()

        total_tokens = mask.sum().float()
        # Normalise by number of tokens
        entropy_loss = -(
            self.entropy_coeff_reasoning * entropy_reasoning / (mask.sum() + 1e-8)
            + self.entropy_coeff_tool * entropy_tool / (mask.sum() + 1e-8)
            + self.entropy_coeff_other * entropy_other / (mask.sum() + 1e-8)
        )

        loss = loss + entropy_loss

        if not return_outputs:
            return loss
        return loss, outputs


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────


def train_autotool(config: dict) -> None:
    autotool_cfg = config.get("autotool", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    model, tokenizer = load_model(config)

    # Register reasoning/tool tags as special tokens so token-type masking works reliably
    try:
        special = ["<reasoning>", "</reasoning>", "<tool_call>", "</tool_call>"]
        existing = set(getattr(tokenizer, "additional_special_tokens", [])) | set(
            getattr(tokenizer, "all_special_tokens", [])
        )
        to_add = [t for t in special if t not in existing]
        if to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": to_add})
            # If model supports resizing embeddings, do so
            try:
                model.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass
            logger.info(f"[AutoTool] Registered special tokens: {to_add}")
    except Exception:
        logger.debug("[AutoTool] Failed to register special tokens; continuing.")

    with open(data_cfg["function_library_path"], "r") as fh:
        function_library = json.load(fh)

    dataset = load_grpo_dataset(data_cfg["train_path"], function_library, tokenizer)

    grpo_args = build_grpo_config(
        config,
        output_dir=train_cfg.get("output_dir", "outputs/autotool_model"),
    )

    trainer = AutoToolTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            autotool_reward_func,  # outcome + length penalty
            format_reward,  # XML format reward
        ],
        args=grpo_args,
        train_dataset=dataset,
        autotool_config=autotool_cfg,
    )

    logger.info("[AutoTool] Starting training...")
    trainer.train()

    out_dir = grpo_args.output_dir
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info(f"[AutoTool] Model saved → {out_dir}")
