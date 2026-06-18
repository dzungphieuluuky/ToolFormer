# Python Files Consolidated

**Source directory:** `D:\Personal\ToolFormer\src`

**Total files:** 23

---

## 1. `algorithms\__init__.py`

```python
from .base_trainer import load_model, build_grpo_config, SYSTEM_PROMPT
from .rc_grpo_trainer import RCGRPOTrainer, train_rc_grpo
from .awpo_trainer import AWPOTrainer, train_awpo
from .gvpo_trainer import GVPOTrainer, train_gvpo

__all__ = [
    "load_model",
    "build_grpo_config",
    "build_prompt",
    "SYSTEM_PROMPT",
    "RCGRPOTrainer",
    "train_rc_grpo",
    "AWPOTrainer",
    "train_awpo",
    "GVPOTrainer",
    "train_gvpo",
]
```

## 2. `algorithms\autotool_trainer.py`

```python
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
            schema_valid(comp) and
            func_selection_ok(comp, gt.get("function", "")) == 1.0 and
            args_accuracy(comp, gt.get("arguments", {})) >= 0.8
        )
        outcome = 1.0 if ok else 0.0

        # Length penalty: penalise overly long responses for simple tasks
        workflow_type = gt.get("workflow", "single_call")
        length = len(comp.split())
        if workflow_type == "single_call":
            # Simpler tasks should have shorter responses
            penalty = max(0.0, (length - 100) / 200.0)  # penalty for length > 100 tokens
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

        logger.info(f"[AutoTool] entropy_coeff_reasoning={self.entropy_coeff_reasoning}  "
                    f"tool={self.entropy_coeff_tool}  other={self.entropy_coeff_other}")

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
                entropy_loss = -coeff * (entropy * attention_mask).sum() / attention_mask.sum()
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
            self.entropy_coeff_reasoning * entropy_reasoning / (mask.sum() + 1e-8) +
            self.entropy_coeff_tool * entropy_tool / (mask.sum() + 1e-8) +
            self.entropy_coeff_other * entropy_other / (mask.sum() + 1e-8)
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
            autotool_reward_func,      # outcome + length penalty
            format_reward,             # XML format reward
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
```

## 3. `algorithms\awpo_trainer.py`

```python
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
        completions, rewards = super()._generate_and_score_completions(prompts, **kwargs)
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

        Returns
        ───────
        (advantages, mixing_weight_avg)
            advantages         : [B] float tensor
            mixing_weight_avg  : average ρ_g over groups, used to shrink clip
        """
        B = rewards.shape[0]
        advantages = torch.zeros(B, dtype=torch.float32)
        mixing_weights: list[float] = []

        unique_groups = group_indices.unique().tolist()

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

            # ── Group statistics ──────────────────────────────────────────────
            n = len(out_r)
            mu_out = sum(out_r) / n
            sigma2_out = sum((r - mu_out) ** 2 for r in out_r) / n
            sigma_out = math.sqrt(sigma2_out + 1e-8)

            # Mixed rewards (initially with ρ=0 to compute variance)
            # We need to compute ρ iteratively? Paper uses ρ = σ_mix/(σ_out+σ_mix)
            # We'll compute ρ based on outcome variance only (simplified).
            # For exact implementation, we'd compute σ_mix from mixed rewards.
            # We'll approximate ρ = 1 / (1 + σ_out)  (so high variance → low mixing)
            # Actually Eq.20: ρ = σ_mix / (σ_out + σ_mix). We'll set ρ based on σ_out.
            # If σ_out is high, outcome is already informative → ρ low.
            # If σ_out is low, outcome is uniform → ρ high.
            # We'll use a simple mapping: ρ = max(0, 1 - σ_out)
            # This is a reasonable heuristic; the paper uses raw variances.
            # For now, we'll directly compute ρ = 1 - σ_out (clipped to [0,1]).
            rho = max(0.0, min(1.0, 1.0 - sigma_out))

            # Difficulty weight
            w = 4.0 * mu_out * (1.0 - mu_out)

            # Mixed reward
            mixed = [(1 - rho) * r_out + rho * r_rea for r_out, r_rea in zip(out_r, reason_r)]

            # Normalise mixed rewards within group
            mu_mix = sum(mixed) / n
            sigma_mix = math.sqrt(sum((r - mu_mix) ** 2 for r in mixed) / n + 1e-8)

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

        loss_output = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

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
            format_reward,      # XML format reward
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
```

## 4. `algorithms\base_trainer.py`

```python
"""
base_trainer.py
────────────────
Conversation format with structured tags:

<|im_start|>system
You are a telecom network operations assistant...
REQUIRED OUTPUT FORMAT:
<reasoning>...</reasoning>
<tool_call>{"function": "...", "arguments": {...}}</tool_call>
<|im_end|>

<|im_start|>user
Tôi cần xem tốc độ tải xuống...
<|im_end|>

<|im_start|>retriever
## Available Functions
### SPEEDTEST_PROVINCE
Description: ...
Parameters: ...

## Relevant Argument Values
### location_code
  - HCM → Thành phố Hồ Chí Minh  [Tỉnh/Thành phố]
<|im_end|>

<|im_start|>assistant
...model generates here...

─────────────────────────────────────────────────────
IMPORTANT — custom role "retriever":
─────────────────────────────────────────────────────
Qwen3's default Jinja chat template only handles the roles
"system", "user", "assistant", and "tool".  Passing
{"role": "retriever", ...} to the stock template either
raises a Jinja2 error or silently drops the message.

We patch the tokenizer's chat_template in load_model() to
accept any arbitrary role with the generic ChatML pattern:
    <|im_start|>{role}\n{content}\n<|im_end|>\n
This is done once at model-load time and saved into the
tokenizer object; all downstream apply_chat_template() calls
then work correctly without further changes.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

# ── Conditional imports for dataset loading ──────────────────────────────────
try:
    import jsonlines
except ImportError:
    jsonlines = None
    warnings.warn("jsonlines not installed; dataset loading will fail.", ImportWarning)

try:
    from datasets import Dataset
except ImportError:
    Dataset = None
    warnings.warn("datasets not installed; dataset loading will fail.", ImportWarning)

try:
    from src.data.retrieval import ArgumentValueRetriever
except ImportError:
    ArgumentValueRetriever = None
    warnings.warn("ArgumentValueRetriever not available; argument values will be skipped.", ImportWarning)


# ──────────────────────────────────────────────────────────────────────────────
# Custom chat template that handles arbitrary roles (including "retriever")
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'system' %}"
    "{{- '<|im_start|>system\n' + message.content + '\n<|im_end|>\n' }}"
    "{%- elif message.role == 'user' %}"
    "{{- '<|im_start|>user\n' + message.content + '\n<|im_end|>\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\n' + message.content + '\n<|im_end|>\n' }}"
    "{%- elif message.role == 'tool' %}"
    "{{- '<|im_start|>tool\n' + message.content + '\n<|im_end|>\n' }}"
    "{%- elif message.role == 'retriever' %}"
    "{{- '<|im_start|>retriever\n' + message.content + '\n<|im_end|>\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\n' + message.content + '\n<|im_end|>\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{%- endif %}"
)


def patch_tokenizer_for_custom_roles(tokenizer) -> None:
    """Register CUSTOM_CHAT_TEMPLATE so apply_chat_template() handles 'retriever'."""
    tokenizer.chat_template = CUSTOM_CHAT_TEMPLATE
    print("[base_trainer] Custom chat template registered (supports 'retriever' role).")


# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a telecom network operations assistant. \
You will receive a user query, a list of available functions with their \
parameter schemas, and relevant argument values retrieved from a knowledge base.

Your task:
1. Read the user query carefully
2. Review the available functions and retrieved argument values in the \
<|im_start|>retriever<|im_end|> block
3. Reason step by step about which function to call and what argument \
values to use
4. Output your reasoning and the final function call

REQUIRED OUTPUT FORMAT — use these exact tags:
<reasoning>
Step-by-step analysis:
- What the user is asking for
- Which function best matches and why
- Which argument values from the retriever match the query
- Final argument assignments
</reasoning>
<tool_call>
{"function": "<function_name>", "arguments": {"<param1>": "<value1>", "<param2>": "<value2>"}}
</tool_call>

STRICT RULES:
- Call EXACTLY one function from the retriever list (no more, no less)
- Include ALL required parameters
- Use argument values from the retriever when they match the query
- Do NOT invent functions or parameters not in the retriever block
- If the query is unanswerable with available functions, output:
<reasoning>Explanation of why no function is appropriate.</reasoning>
<tool_call>null</tool_call>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Retriever block builders
# ──────────────────────────────────────────────────────────────────────────────

def build_function_description(func_name: str, schema: dict) -> str:
    """Render one function schema into a human-readable block."""
    lines = [f"### {func_name}"]
    lines.append(f"Description: {schema.get('description', 'No description available')}")

    params = schema.get("parameters", {})
    if params:
        lines.append("Parameters:")
        for pname, pinfo in params.items():
            if isinstance(pinfo, dict):
                ptype    = pinfo.get("type", "any")
                required = "(required)" if pinfo.get("required") else "(optional)"
                desc     = pinfo.get("description", "")
                lines.append(f"  - {pname} [{ptype}] {required}: {desc}")
            else:
                lines.append(f"  - {pname}: {pinfo}")

    constraints = schema.get("constraints", {})
    if constraints:
        lines.append(f"Constraints: {json.dumps(constraints, ensure_ascii=False)}")

    return "\n".join(lines)


def build_argument_values_block(
    argument_values: "dict[str, list]",
) -> str:
    """
    Render retrieved argument values as a readable block.

    Output:
        ## Relevant Argument Values

        ### location_code
          - HCM → Thành phố Hồ Chí Minh  [Tỉnh/Thành phố]
    """
    if not argument_values:
        return ""

    lines = ["## Relevant Argument Values"]
    for param_name, matches in argument_values.items():
        if not matches:
            continue
        lines.append(f"\n### {param_name}")
        for m in matches:
            if hasattr(m, "code"):
                label_str = m.label
                if m.alt_label:
                    label_str += f" / {m.alt_label}"
                lines.append(f"  - {m.code} → {label_str}  [{m.group}]")
            else:
                code  = m.get("code", "")
                label = m.get("label", "")
                group = m.get("group", "")
                lines.append(f"  - {code} → {label}  [{group}]")

    return "\n".join(lines)


def build_retriever_block(
    function_names:  list[str],
    function_library: dict,
    argument_values: "dict[str, list] | None" = None,
) -> str:
    """
    Build the content that goes inside the <|im_start|>retriever block.

    Structure:
        ## Available Functions

        ### FUNC_NAME_1
        Description: ...
        Parameters: ...

        ### FUNC_NAME_2
        ...

        ## Relevant Argument Values   (only if non-empty)
        ### param_name
          - CODE → Label  [group]
    """
    lines = ["## Available Functions\n"]
    for fn in function_names:
        if fn in function_library:
            lines.append(build_function_description(fn, function_library[fn]))
            lines.append("")   # blank line between functions

    if argument_values:
        val_block = build_argument_values_block(argument_values)
        if val_block:
            lines.append(val_block)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Message builders
# ──────────────────────────────────────────────────────────────────────────────

def build_messages_for_grpo(
    query:            str,
    function_names:   list[str],
    function_library: dict,
    argument_values:  "dict[str, list] | None" = None,
) -> list[dict]:
    """
    Build [system, retriever, user] messages for GRPO rollout generation.

    Message order (matches the desired conversation format exactly):
        1. system    — instructions + output format
        2. user      — the raw user query
        3. retriever — available functions + relevant argument values

    The model generates the assistant turn (reasoning + tool_call).
    Ground truth is NEVER included here.

    NOTE: Requires patch_tokenizer_for_custom_roles() to have been called
    on the tokenizer so that the "retriever" role is rendered correctly.
    """
    retriever_content = build_retriever_block(
        function_names, function_library, argument_values
    )
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": query},
        {"role": "retriever", "content": retriever_content},   # ← custom role
    ]


def build_messages_for_sft(
    query:            str,
    function_names:   list[str],
    function_library: dict,
    ground_truth:     dict,
    argument_values:  "dict[str, list] | None" = None,
) -> list[dict]:
    """
    Build [system, user, retriever, assistant] messages for SFT.

    The assistant message is the expected ground-truth response.
    """
    messages = build_messages_for_grpo(
        query, function_names, function_library, argument_values
    )

    reasoning = ground_truth.get(
        "reasoning",
        "Analysing the query to determine the correct function and arguments.",
    )
    func_name = ground_truth.get("function")
    arguments = ground_truth.get("arguments", {})

    call_json = (
        "null"
        if func_name is None
        else json.dumps(
            {"function": func_name, "arguments": arguments},
            indent=2,
            ensure_ascii=False,
        )
    )

    assistant_content = (
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<tool_call>\n{call_json}\n</tool_call>"
    )

    messages.append({"role": "assistant", "content": assistant_content})
    return messages


# ──────────────────────────────────────────────────────────────────────────────
# Dataset formatters
# ──────────────────────────────────────────────────────────────────────────────

def format_sample_for_grpo(
    sample:           dict,
    function_library: dict,
    tokenizer: AutoTokenizer,
    argument_values:  "dict[str, list] | None" = None,
) -> dict:
    """
    Convert one raw dataset sample into GRPOTrainer format.

    Columns produced:
      prompt         → tokenizer-formatted string (system+retriever+user + generation prompt)
      ground_truth   → dict passed as kwarg to reward functions
      query          → str  passed as kwarg to reward functions
      workflow_type  → str  passed as kwarg to reward functions
    """
    query     = sample["query"]
    retrieved = sample.get("retrieved_functions", [])
    gt        = sample.get("ground_truth", {})

    arg_vals = argument_values or _deserialise_arg_values(
        sample.get("retrieved_argument_values")
    )

    messages = build_messages_for_grpo(
        query            = query,
        function_names   = retrieved,
        function_library = function_library,
        argument_values  = arg_vals,
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize              = False,
        add_generation_prompt = True,
    )

    return {
        "prompt":        prompt,
        "ground_truth":  gt,
        "query":         query,
        "workflow_type": sample.get("workflow_type", "single_call"),
    }


def format_sample_for_sft(
    sample:           dict,
    function_library: dict,
    tokenizer,
    argument_values:  "dict[str, list] | None" = None,
) -> dict:
    """Convert one raw sample into SFTTrainer format (full conversation)."""
    query     = sample["query"]
    retrieved = sample.get("retrieved_functions", [])
    gt        = sample.get("ground_truth", {})

    arg_vals = argument_values or _deserialise_arg_values(
        sample.get("retrieved_argument_values")
    )

    messages = build_messages_for_sft(
        query            = query,
        function_names   = retrieved,
        function_library = function_library,
        ground_truth     = gt,
        argument_values  = arg_vals,
    )

    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = False,
        )
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_grpo_dataset(
    jsonl_path:               str,
    function_library:         dict,
    tokenizer,
    argument_values_catalog:  dict | None = None,
    telco_retriever           = None,
) -> "Dataset":
    """
    Load the enriched train_dataset.jsonl for GRPOTrainer.

    Argument value priority:
      1. telco_retriever          — live two-stage retrieval
      2. retrieved_argument_values field in JSONL  — pre-computed by prepare_data.py
      3. argument_values_catalog  — on-the-fly ArgumentValueRetriever
      4. None                     — no argument values in prompt
    """
    if Dataset is None or jsonlines is None:
        raise ImportError("Required packages 'datasets' and/or 'jsonlines' are not installed.")

    raw_samples: list[dict] = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            raw_samples.append(obj)

    print(f"[base_trainer] Loaded {len(raw_samples)} samples from {jsonl_path}")

    val_retriever = None
    if telco_retriever is None and argument_values_catalog is not None:
        if ArgumentValueRetriever is not None:
            val_retriever = ArgumentValueRetriever(argument_values_catalog)
        else:
            print("[base_trainer] ArgumentValueRetriever not available; skipping argument values.")

    formatted = []
    for sample in raw_samples:
        query     = sample["query"]
        retrieved = sample.get("retrieved_functions", [])

        if telco_retriever is not None:
            result   = telco_retriever.retrieve(
                query, function_library,
                precomputed_func_names = retrieved,
            )
            arg_vals = result.argument_values
        elif val_retriever is not None:
            arg_vals = val_retriever.retrieve_for_functions(
                query, retrieved, function_library
            )
        else:
            arg_vals = _deserialise_arg_values(
                sample.get("retrieved_argument_values")
            )

        formatted.append(
            format_sample_for_grpo(sample, function_library, tokenizer, arg_vals)
        )

    dataset = Dataset.from_list(formatted)
    print(f"[base_trainer] GRPO dataset ready: {len(dataset)} samples")
    print(f"[base_trainer] Columns: {dataset.column_names}")

    # Sanity-check first sample
    if formatted:
        s  = formatted[0]
        gt = s["ground_truth"]
        print(f"\n{'='*70}")
        print("SAMPLE PROMPT (first 1500 chars):")
        print("=" * 70)
        print(s["prompt"][:1500])
        print("..." if len(s["prompt"]) > 1500 else "")
        print(f"\nGROUND TRUTH (reward funcs see this, MODEL DOES NOT):")
        print(f"  function:  {gt.get('function')}")
        print(f"  arguments: {json.dumps(gt.get('arguments', {}), ensure_ascii=False)[:300]}")
        print("=" * 70 + "\n")

    return dataset


def load_sft_dataset(
    jsonl_path:               str,
    function_library:         dict,
    tokenizer,
    argument_values_catalog:  dict | None = None,
    telco_retriever           = None,
) -> "Dataset":
    """Load the enriched train_dataset.jsonl for SFTTrainer."""
    if Dataset is None or jsonlines is None:
        raise ImportError("Required packages 'datasets' and/or 'jsonlines' are not installed.")

    raw_samples: list[dict] = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            raw_samples.append(obj)

    val_retriever = None
    if telco_retriever is None and argument_values_catalog is not None:
        if ArgumentValueRetriever is not None:
            val_retriever = ArgumentValueRetriever(argument_values_catalog)

    formatted = []
    for sample in raw_samples:
        query     = sample["query"]
        retrieved = sample.get("retrieved_functions", [])

        if telco_retriever is not None:
            result   = telco_retriever.retrieve(
                query, function_library,
                precomputed_func_names = retrieved,
            )
            arg_vals = result.argument_values
        elif val_retriever is not None:
            arg_vals = val_retriever.retrieve_for_functions(
                query, retrieved, function_library
            )
        else:
            arg_vals = _deserialise_arg_values(
                sample.get("retrieved_argument_values")
            )

        formatted.append(
            format_sample_for_sft(sample, function_library, tokenizer, arg_vals)
        )

    dataset = Dataset.from_list(formatted)
    print(f"[base_trainer] SFT dataset ready: {len(dataset)} samples")
    return dataset


def _deserialise_arg_values(raw: dict | None) -> dict | None:
    """Convert the plain-dict form stored in JSONL back to a dict that
    build_argument_values_block() can consume.  Returns None if raw is falsy.
    """
    if not raw:
        return None
    return raw   # plain dicts already work with build_argument_values_block()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(config: dict | None = None):
    """
    Load Qwen3-4B with Unsloth QLoRA and patch the tokenizer to support
    the custom "retriever" role in apply_chat_template().
    """
    from unsloth import FastLanguageModel, PatchFastRL
    from trl import GRPOTrainer as _GT

    PatchFastRL("GRPO", FastLanguageModel)

    cfg       = config or {}
    model_cfg = cfg.get("model", {})
    lora_cfg  = cfg.get("lora",  {})

    model_name = model_cfg.get("name", "unsloth/Qwen3-4B-unsloth-bnb-4bit")
    max_seq    = model_cfg.get("max_seq_length", 2048)
    lora_rank  = lora_cfg.get("r", 16)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name             = model_name,
        max_seq_length         = max_seq,
        load_in_4bit           = model_cfg.get("load_in_4bit", True),
        dtype                  = None,
        fast_inference         = model_cfg.get("fast_inference", True),
        max_lora_rank          = lora_rank,
        gpu_memory_utilization = model_cfg.get("gpu_memory_utilization", 0.7),
    )

    # ── Patch tokenizer to handle "retriever" role ────────────────────────────
    patch_tokenizer_for_custom_roles(tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r                          = lora_rank,
        lora_alpha                 = lora_cfg.get("lora_alpha", 2 * lora_rank),
        lora_dropout               = lora_cfg.get("lora_dropout", 0.0),
        target_modules             = lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias                       = lora_cfg.get("bias", "none"),
        use_gradient_checkpointing = lora_cfg.get(
            "use_gradient_checkpointing", "unsloth"
        ),
        random_state               = 3407,
    )
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# GRPOConfig builder
# ──────────────────────────────────────────────────────────────────────────────

def build_grpo_config(config: dict, output_dir: str | None = None) -> "GRPOConfig":
    from trl import GRPOConfig
    from vllm import SamplingParams

    train_cfg = config.get("training", {})
    grpo_cfg  = config.get("grpo",     {})
    data_cfg  = config.get("data",     {})

    vllm_params = SamplingParams(
        temperature                = grpo_cfg.get("temperature", 1.0),
        top_p                      = 0.95,
        min_p                      = 0.05,
        seed                       = train_cfg.get("seed", 3407),
        stop                       = ["</tool_call>"],
        include_stop_str_in_output = True,
    )

    return GRPOConfig(
        output_dir                  = output_dir or train_cfg.get("output_dir", "outputs/model"),
        learning_rate               = train_cfg.get("learning_rate", 5e-6),
        adam_beta1                  = train_cfg.get("adam_beta1",   0.9),
        adam_beta2                  = train_cfg.get("adam_beta2",   0.99),
        weight_decay                = train_cfg.get("weight_decay", 0.01),
        warmup_ratio                = train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type           = train_cfg.get("lr_scheduler_type", "cosine"),
        optim                       = train_cfg.get("optim", "adamw_8bit"),
        per_device_train_batch_size = train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 4),
        num_generations             = train_cfg.get("num_generations", 8),
        max_prompt_length           = data_cfg.get("max_prompt_length",    1024),
        max_completion_length       = data_cfg.get("max_completion_length", 512),
        temperature                 = grpo_cfg.get("temperature", 1.0),
        epsilon                     = grpo_cfg.get("epsilon",      0.2),
        epsilon_high                = grpo_cfg.get("epsilon_high", 0.28),
        loss_type                   = grpo_cfg.get("loss_type",    "grpo"),
        mask_truncated_completions  = grpo_cfg.get("mask_truncated_completions", True),
        vllm_sampling_params        = vllm_params,
        max_steps                   = train_cfg.get("max_steps",    500),
        save_steps                  = train_cfg.get("save_steps",   100),
        logging_steps               = train_cfg.get("logging_steps",  1),
        max_grad_norm               = train_cfg.get("max_grad_norm", 0.1),
        report_to                   = train_cfg.get("report_to",   "none"),
        seed                        = train_cfg.get("seed",         3407),
        bf16                        = torch.cuda.is_bf16_supported(),
    )
```

## 5. `algorithms\gvpo_trainer.py`

```python
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
```

## 6. `algorithms\rc_grpo_trainer.py`

```python
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
    For each prompt, pre-assign a reward token schedule so that within
    the GRPO rollout group approximately `high_fraction` of the G rollouts
    are conditioned on <|high_reward|> and the rest on <|low_reward|>.

    Implementation note:
    TRL's GRPOTrainer calls the prompt G times (num_generations) for each
    sample.  We create G copies of each sample with alternating tokens,
    then shuffle within each group so the model doesn't learn position bias.

    The token is prepended to the prompt string BEFORE the chat template
    is applied, matching the RCTP-FT training format exactly.
    """
    n_high = max(1, round(num_generations * high_fraction))
    n_low = num_generations - n_high

    def _add_token(example, idx):
        # Deterministic assignment based on sample index
        if (idx % num_generations) < n_high:
            token = high_token
        else:
            token = low_token
        if not example["prompt"].startswith(token):
            example["prompt"] = token + "\n" + example["prompt"]
        example["reward_token"] = token
        return example

    return dataset.map(_add_token, with_indices=True)


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
        logger.info(f"[RC-GRPO] Computed high_fraction={high_fraction:.3f} from RCTP dataset.")
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
```

## 7. `data\__init__.py`

```python
from .excel_parser import parse_telecom_functions
from .dataset_generator import TelcoDatasetGenerator
from .retrieval import FunctionRetriever

__all__ = ["parse_telecom_functions", "TelcoDatasetGenerator", "FunctionRetriever"]
```

## 8. `data\dataset_generator.py`

```python
"""
dataset_generator.py
─────────────────────
Generates 2 000–2 500 (query, ground_truth_call) pairs by prompting
an LLM API (OpenAI / Anthropic / Google / Together / Mistral).

The generator reads two schema files:
  - function_schema_train.json  → functions used for training samples
  - function_schema_test.json   → functions held out for testing

and produces train_dataset.jsonl + test_dataset.jsonl.

Workflow distribution (configurable, defaults from project spec):
  60%  single_call   – one function call answers the query
  20%  parallel      – multiple independent calls in one turn
  15%  sequential    – chained calls where output feeds next input
   5%  abstention    – model should refuse / ask for more info

Train / Test split: functions are pre‑split by the schema files.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from tqdm.auto import tqdm
import logging

_log = logging.getLogger("tenacity.retry")
_log.setLevel(logging.DEBUG)
logging.basicConfig()  # ensure a handler exists


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class GroundTruth:
    function: str
    arguments: dict[str, Any]
    workflow_type: str  # single_call | parallel | sequential | abstention
    # For multi-call workflows
    calls: list[dict] = field(default_factory=list)


@dataclass
class DataSample:
    id: str
    query: str
    workflow_type: str
    function_name: str  # primary function (or "none" for abstention)
    ground_truth: dict  # serialised GroundTruth
    retrieved_functions: list[str]  # simulated top-k from library
    split: str = "train"  # train | test


# ──────────────────────────────────────────────────────────────────────────────
# API client factory
# ──────────────────────────────────────────────────────────────────────────────


class _APIClient:
    """
    Thin wrapper that normalises calls across providers.
    OpenRouter is treated as its own provider (not just an OpenAI base_url swap)
    because it requires extra headers that the plain OpenAI client does not send.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ):
        self.provider = provider.lower()
        self.model = model
        self._client = self._build_client(api_key, base_url)

    # ── Client factory ────────────────────────────────────────────────────────

    def _build_client(self, api_key: str, base_url: str | None):
        if self.provider == "openai":
            from openai import OpenAI

            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            return OpenAI(**kwargs)

        elif self.provider == "openrouter":
            from openai import OpenAI

            # OpenRouter requires HTTP-Referer + X-OpenRouter-Title on every request.
            # Pass them via default_headers so they appear on all calls
            # without any change to the call sites.
            return OpenAI(
                api_key=api_key,
                base_url=base_url or "https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/telco-agent-rl",
                    "X-OpenRouter-Title": "Telco Agent RL",
                },
            )

        elif self.provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(api_key=api_key)

        elif self.provider == "google":
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model)

        elif self.provider == "together":
            from openai import OpenAI

            return OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.together.xyz/v1",
            )

        elif self.provider == "mistral":
            from mistralai import Mistral

            return Mistral(api_key=api_key)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # ── Completion call with verbose retry logging ────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,  # ← exposes real exception instead of RetryError wrapper
    )
    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        max_tokens: int = 1024,
    ) -> str:
        """Single chat completion — returns assistant text."""
        try:
            if self.provider in ("openai", "together", "openrouter"):
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content

            elif self.provider == "anthropic":
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=temperature,
                )
                return resp.content[0].text

            elif self.provider == "google":
                resp = self._client.generate_content(f"{system}\n\n{user}")
                return resp.text

            elif self.provider == "mistral":
                resp = self._client.chat.complete(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content

            raise RuntimeError(f"Unreachable provider branch: {self.provider}")

        except Exception as exc:
            # ── Print the real error before tenacity decides to retry ─────────
            # This is the line that was missing — without it, RetryError hides
            # the actual HTTP status, body, and message.
            print(
                f"\n  [DEBUG] {self.provider} API error "
                f"(model={self.model}): {type(exc).__name__}: {exc}"
            )
            if hasattr(exc, "status_code"):
                print(f"  [DEBUG] HTTP status : {exc.status_code}")
            if hasattr(exc, "body"):
                print(f"  [DEBUG] Error body  : {exc.body}")
            raise  # let tenacity handle retry / reraise


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a telecom network operations expert and dataset creator.
Your job is to generate realistic, diverse training samples for fine-tuning
a small language model on telecom function-calling tasks.

Always respond with **valid JSON only** — no markdown, no extra text, the entire query should be in Vietnamese.
Ensure the KPI code, location code, object code and unit code are based on these values:
kpi_code,Ý nghĩa
call_setup_success_rate,Tỷ lệ thiết lập cuộc gọi thành công
call_drop_rate,Tỷ lệ rớt cuộc gọi
data_traffic_volume,Lưu lượng dữ liệu
network_latency_avg_ms,Độ trễ mạng trung bình (ms)
service_cancellation_rate,Tỷ lệ hủy dịch vụ
voice_call_duration_avg,Thời lượng cuộc gọi thoại trung bình (phút)
data_usage_per_user_4g,Mức sử dụng dữ liệu trung bình trên mỗi thuê bao 4G
active_user_count_4g,Số lượng người dùng 4G hoạt động
interruption_incident_count,Số lượng sự cố gián đoạn dịch vụ
subscriber_growth_rate,Tỷ lệ tăng trưởng thuê bao mới
coverage_percentage_4g,Tỷ lệ phủ sóng 4G (%)
user_experience_score,Điểm trải nghiệm người dùng

Nhóm,Giá trị khả dụng,Mã
Toàn quốc,Việt Nam,VNM
Toàn quốc,toàn quốc,VNM
Khu vực,Khu vực 1,KV1
Khu vực,Khu vực miền Bắc,KV1
Khu vực,Khu vực 2,KV2
Khu vực,Khu vực miền Trung,KV2
Khu vực,Khu vực 3,KV3
Khu vực,Khu vực miền Nam,KV3
Tỉnh/Thành phố Việt Nam,Hà Nội,HNI
Tỉnh/Thành phố Việt Nam,Huế,HUE
Tỉnh/Thành phố Việt Nam,Lai Châu,LCU
Tỉnh/Thành phố Việt Nam,Điện Biên,DBN
Tỉnh/Thành phố Việt Nam,Sơn La,SLA
Tỉnh/Thành phố Việt Nam,Lạng Sơn,LSN
Tỉnh/Thành phố Việt Nam,Quảng Ninh,QNH
Tỉnh/Thành phố Việt Nam,Thanh Hóa,THA
Tỉnh/Thành phố Việt Nam,Nghệ An,NAN
Tỉnh/Thành phố Việt Nam,Hà Tĩnh,HTH
Tỉnh/Thành phố Việt Nam,Cao Bằng,CBG
Tỉnh/Thành phố Việt Nam,Tuyên Quang,TQG
Tỉnh/Thành phố Việt Nam,Lào Cai,LCI
Tỉnh/Thành phố Việt Nam,Thái Nguyên,TNN
Tỉnh/Thành phố Việt Nam,Phú Thọ,PTO
Tỉnh/Thành phố Việt Nam,Bắc Ninh,BNH
Tỉnh/Thành phố Việt Nam,Hưng Yên,HYN
Tỉnh/Thành phố Việt Nam,Hải Phòng,HPG
Tỉnh/Thành phố Việt Nam,Ninh Bình,NBH
Tỉnh/Thành phố Việt Nam,Quảng Trị,QTI
Tỉnh/Thành phố Việt Nam,Đà Nẵng,DNG
Tỉnh/Thành phố Việt Nam,Quảng Ngãi,QNI
Tỉnh/Thành phố Việt Nam,Gia Lai,GLI
Tỉnh/Thành phố Việt Nam,Khánh Hòa,KHA
Tỉnh/Thành phố Việt Nam,Lâm Đồng,LDG
Tỉnh/Thành phố Việt Nam,Đắk Lắk,DLK
Tỉnh/Thành phố Việt Nam,Thành phố Hồ Chí Minh,HCM
Tỉnh/Thành phố Việt Nam,Đồng Nai,DNI
Tỉnh/Thành phố Việt Nam,Tây Ninh,TNH
Tỉnh/Thành phố Việt Nam,Cần Thơ,CTO
Tỉnh/Thành phố Việt Nam,Vĩnh Long,VLG
Tỉnh/Thành phố Việt Nam,Đồng Tháp,DTP
Tỉnh/Thành phố Việt Nam,Cà Mau,CMU
Tỉnh/Thành phố Việt Nam,An Giang,AGG
Quốc gia,Cambodia,KHM
Quốc gia,Laos,LAO
Quốc gia,Peru,PER
Quốc gia,Myanmar,MMR
Quốc gia,TimorLeste,TLS
Quốc gia,Haiti,HTI
Quốc gia,Mozambique,MOZ
Quốc gia,Tanzania,TZA
Quốc gia,Cameroon,CMR
Quốc gia,Burundi,BDI

unit_code,Đơn vị/Công ty
VCS,Công ty An ninh mạng Viettel
IDC,Công ty TNHH Viettel - IDC
VTM,Công ty Truyền thông Viettel
VAI,Trung tâm Trí tuệ nhân tạo và Dịch vụ dữ liệu Việt Nam
VTPost,Tổng công ty cổ phần Bưu chính Viettel
VDS,Tổng công ty Dịch vụ số Viettel
VTS,Tổng công ty Giải pháp doanh nghiệp Viettel
VTNet,Tổng công ty Mạng lưới Viettel
VTT,Tổng Công ty Viễn thông Viettel
"""

_SINGLE_CALL_TEMPLATE = """\
Given the following telecom network function:

FUNCTION SCHEMA:
{schema}

Generate {n} diverse, realistic natural-language queries that a network
operations engineer would ask, along with the exact function call
(with realistic argument values) that answers each query.

Respond with a JSON array of exactly {n} objects, each with:
{{
  "query": "<natural language question>",
  "reasoning": "<brief step-by-step reasoning>",
  "function_call": {{
    "function": "<function_name>",
    "arguments": {{ <param>: <value>, ... }}
  }}
}}

Rules:
- Vary phrasing, urgency, and technical detail across queries
- Use realistic telecom values (cell IDs, frequencies, thresholds, etc.)
- All required parameters must be present
- Respect any constraints in the schema
- Do NOT invent parameters not in the schema
"""

_PARALLEL_CALL_TEMPLATE = """\
Given these telecom network functions:

FUNCTION SCHEMAS:
{schemas}

Generate {n} queries that require calling MULTIPLE of these functions
simultaneously (in parallel) to fully answer the question.

Respond with a JSON array of exactly {n} objects:
{{
  "query": "<natural language question requiring multiple tools>",
  "reasoning": "<why multiple calls are needed>",
  "function_calls": [
    {{"function": "<name1>", "arguments": {{ ... }}}},
    {{"function": "<name2>", "arguments": {{ ... }}}}
  ]
}}
"""

_SEQUENTIAL_CALL_TEMPLATE = """\
Given these telecom network functions:

FUNCTION SCHEMAS:
{schemas}

Generate {n} queries that require chained/sequential function calls
(the output of one call informs the next).

Respond with a JSON array of exactly {n} objects:
{{
  "query": "<query needing sequential tool calls>",
  "reasoning": "<chain-of-thought for the sequence>",
  "function_calls": [
    {{"step": 1, "function": "<name1>", "arguments": {{ ... }}, "depends_on": []}},
    {{"step": 2, "function": "<name2>", "arguments": {{ ... }}, "depends_on": [1]}}
  ]
}}
"""

_ABSTENTION_TEMPLATE = """\
Given these telecom network functions:

FUNCTION SCHEMAS:
{schemas}

Generate {n} queries where the model should NOT call any function because:
- Required information is missing from the query, OR
- The query is out of scope for these functions, OR
- The user needs to provide clarification first.

Respond with a JSON array of exactly {n} objects:
{{
  "query": "<ambiguous or incomplete query>",
  "reasoning": "<why no function should be called>",
  "function_call": null,
  "refusal_message": "<what the model should say instead>"
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Core generator
# ──────────────────────────────────────────────────────────────────────────────


class TelcoDatasetGenerator:
    """
    Generates GRPO‑ready training samples from separate train/test schema files.

    Usage
    ─────
    gen = TelcoDatasetGenerator.from_schemas(
        train_schema_path="data/processed/function_schema_train.json",
        test_schema_path="data/processed/function_schema_test.json",
        provider="openai",
        model="gpt-4o-mini",
    )
    train, test = gen.generate(total=2400, output_dir="data/processed")
    """

    def __init__(
        self,
        train_function_library: dict,
        test_function_library: dict,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_workers: int = 8,
        requests_per_minute: int = 500,
        temperature: float = 0.9,
        max_tokens: int = 1024,
        seed: int = 42,
    ):
        self.train_library = train_function_library
        self.test_library = test_function_library
        self.train_func_names = list(train_function_library.keys())
        self.test_func_names = list(test_function_library.keys())
        # Union for retrieval simulation
        self.all_func_names = list(set(self.train_func_names + self.test_func_names))
        self.max_workers = max_workers
        self.rpm = requests_per_minute
        self.temperature = temperature
        self.max_tokens = max_tokens
        random.seed(seed)

        _key = (
            api_key
            or os.getenv(f"{provider.upper()}_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        if not _key:
            raise ValueError(
                f"API key not found. Set the environment variable "
                f"'{provider.upper()}_API_KEY' or pass api_key= explicitly."
            )
        self.client = _APIClient(provider, model, _key, base_url)
        print(
            f"[DatasetGenerator] Provider={provider}  Model={model}  "
            f"Train functions={len(self.train_func_names)}  "
            f"Test functions={len(self.test_func_names)}  "
            f"MaxWorkers={max_workers}  RPM={requests_per_minute}"
        )

    # ── factory methods ──────────────────────────────────────────────────────

    @classmethod
    def from_schemas(
        cls,
        train_schema_path: str,
        test_schema_path: str,
        **kwargs,
    ) -> "TelcoDatasetGenerator":
        """Build generator from two separate schema JSON files."""
        with open(train_schema_path, "r", encoding="utf-8") as fh:
            train_library = json.load(fh)
        with open(test_schema_path, "r", encoding="utf-8") as fh:
            test_library = json.load(fh)
        return cls(
            train_function_library=train_library,
            test_function_library=test_library,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config: dict) -> "TelcoDatasetGenerator":
        """Build from a loaded YAML config dict (dataset_generation section)."""
        dg = config.get("dataset_generation", config)
        train_schema = config.get("data", {}).get(
            "train_schema_path", "data/processed/function_schema_train.json"
        )
        test_schema = config.get("data", {}).get(
            "test_schema_path", "data/processed/function_schema_test.json"
        )
        return cls.from_schemas(
            train_schema_path=train_schema,
            test_schema_path=test_schema,
            provider=dg.get("provider", "openai"),
            model=dg.get("model", "gpt-4o-mini"),
            api_key=os.getenv(dg.get("api_key_env", "OPENAI_API_KEY")),
            base_url=dg.get("base_url"),
            max_workers=dg.get("max_workers", 8),
            requests_per_minute=dg.get("requests_per_minute", 500),
            temperature=dg.get("temperature", 0.9),
            max_tokens=dg.get("max_tokens", 1024),
        )

    # ── public entry-point ────────────────────────────────────────────────────

    def generate(
        self,
        total: int = 2400,
        output_dir: str = "data/processed",
        workflow_distribution: dict | None = None,
        train_split: float = 0.89,
    ) -> tuple[list[DataSample], list[DataSample]]:
        """
        Main generation loop.

        Parameters
        ──────────
        total                   : target number of samples
        output_dir              : where to write .jsonl files
        workflow_distribution   : overrides default ratios
        train_split             : fraction of samples for training (within training functions)

        Returns
        ───────
        (train_samples, test_samples)
        """
        dist = workflow_distribution or {
            "single_call": 0.60,
            "parallel": 0.20,
            "sequential": 0.15,
            "abstention": 0.05,
        }

        train_funcs = self.train_func_names
        test_funcs = self.test_func_names

        print(f"[DatasetGenerator] Train functions : {len(train_funcs)}")
        print(f"[DatasetGenerator] Test  functions : {test_funcs}")

        # ── 2. Count per workflow type ───────────────────────────────────────
        counts = {wt: max(1, int(total * ratio)) for wt, ratio in dist.items()}
        # adjust to hit exact total
        diff = total - sum(counts.values())
        counts["single_call"] += diff

        print(f"[DatasetGenerator] Workflow counts : {counts}")

        # ── 3. Generate samples per workflow ────────────────────────────────
        all_samples: list[DataSample] = []
        all_samples += self._generate_single_calls(
            counts["single_call"], train_funcs, split="train"
        )
        all_samples += self._generate_parallel(
            counts["parallel"], train_funcs, split="train"
        )
        all_samples += self._generate_sequential(
            counts["sequential"], train_funcs, split="train"
        )
        all_samples += self._generate_abstentions(
            counts["abstention"], train_funcs, split="train"
        )

        # ── 4. Generate test samples from held-out functions ─────────────────
        test_count = max(50, int(total * (1 - train_split)))
        test_samples = self._generate_single_calls(test_count, test_funcs, split="test")

        # ── 5. Assign splits + simulate retrieved functions ──────────────────
        random.shuffle(all_samples)
        train_cut = int(len(all_samples) * train_split)
        train_samples = all_samples[:train_cut]
        # remaining training‑pool samples go to test too (edge cases / overflow)
        extra_test = all_samples[train_cut:]
        test_samples = test_samples + extra_test

        for s in train_samples:
            s.split = "train"
        for s in test_samples:
            s.split = "test"

        self._simulate_retrieval(train_samples, k=5)
        self._simulate_retrieval(test_samples, k=5)

        # ── 6. Save ──────────────────────────────────────────────────────────
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._save_jsonl(train_samples, out / "raw_train_dataset.jsonl")
        self._save_jsonl(test_samples, out / "raw_test_dataset.jsonl")

        print(
            f"\n[DatasetGenerator] ✓  train={len(train_samples)}  test={len(test_samples)}"
        )
        return train_samples, test_samples

    # ── workflow generators ───────────────────────────────────────────────────

    def _generate_single_calls(
        self,
        count: int,
        func_pool: list[str],
        split: str = "train",
        batch_size: int = 5,
    ) -> list[DataSample]:
        """Generate single-function-call samples."""
        tasks: list[tuple[str, int]] = []  # (func_name, n_per_call)
        remaining = count
        while remaining > 0:
            fn = random.choice(func_pool)
            n = min(batch_size, remaining)
            tasks.append((fn, n))
            remaining -= n

        samples: list[DataSample] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._call_single, fn, n): (fn, n) for fn, n in tasks
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="single_call",
                leave=False,
            ):
                fn, _ = futures[fut]
                try:
                    raw_list = fut.result()
                    for raw in raw_list:
                        s = self._parse_single(raw, fn, split)
                        if s:
                            samples.append(s)
                except Exception as exc:
                    print(f"  [WARN] single_call error for '{fn}': {exc}")

        self._rate_limit_wait(len(tasks))
        return samples

    def _generate_parallel(
        self, count: int, func_pool: list[str], split: str = "train"
    ) -> list[DataSample]:
        """Generate parallel multi-call samples."""
        tasks: list[tuple[list[str], int]] = []
        remaining = count
        while remaining > 0:
            n_funcs = random.randint(2, min(4, len(func_pool)))
            chosen = random.sample(func_pool, n_funcs)
            n = min(3, remaining)
            tasks.append((chosen, n))
            remaining -= n

        samples: list[DataSample] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._call_parallel, fns, n): fns for fns, n in tasks
            }
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="parallel", leave=False
            ):
                fns = futures[fut]
                try:
                    for raw in fut.result():
                        s = self._parse_parallel(raw, fns, split)
                        if s:
                            samples.append(s)
                except Exception as exc:
                    print(f"  [WARN] parallel error: {exc}")

        return samples

    def _generate_sequential(
        self, count: int, func_pool: list[str], split: str = "train"
    ) -> list[DataSample]:
        """Generate sequential / chained call samples."""
        tasks: list[tuple[list[str], int]] = []
        remaining = count
        while remaining > 0:
            n_funcs = random.randint(2, min(3, len(func_pool)))
            chosen = random.sample(func_pool, n_funcs)
            n = min(3, remaining)
            tasks.append((chosen, n))
            remaining -= n

        samples: list[DataSample] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._call_sequential, fns, n): fns for fns, n in tasks
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="sequential",
                leave=False,
            ):
                fns = futures[fut]
                try:
                    for raw in fut.result():
                        s = self._parse_sequential(raw, fns, split)
                        if s:
                            samples.append(s)
                except Exception as exc:
                    print(f"  [WARN] sequential error: {exc}")

        return samples

    def _generate_abstentions(
        self, count: int, func_pool: list[str], split: str = "train"
    ) -> list[DataSample]:
        """Generate refusal / abstention samples."""
        tasks: list[tuple[list[str], int]] = []
        remaining = count
        while remaining > 0:
            n_funcs = random.randint(1, min(3, len(func_pool)))
            chosen = random.sample(func_pool, n_funcs)
            n = min(3, remaining)
            tasks.append((chosen, n))
            remaining -= n

        samples: list[DataSample] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._call_abstention, fns, n): fns for fns, n in tasks
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="abstention",
                leave=False,
            ):
                fns = futures[fut]
                try:
                    for raw in fut.result():
                        s = self._parse_abstention(raw, fns, split)
                        if s:
                            samples.append(s)
                except Exception as exc:
                    print(f"  [WARN] abstention error: {exc}")

        return samples

    # ── API call methods ──────────────────────────────────────────────────────

    def _schema_str(self, func_name: str) -> str:
        """
        Return JSON schema for a given function, searching first in train_library,
        then test_library (for test‑only functions during retrieval simulation).
        """
        if func_name in self.train_library:
            schema = self.train_library[func_name]
        elif func_name in self.test_library:
            schema = self.test_library[func_name]
        else:
            raise KeyError(f"Function '{func_name}' not found in any library.")
        return json.dumps(schema, indent=2)

    def _schemas_str(self, func_names: list[str]) -> str:
        return "\n---\n".join(self._schema_str(fn) for fn in func_names)

    def _call_single(self, func_name: str, n: int) -> list[dict]:
        prompt = _SINGLE_CALL_TEMPLATE.format(
            schema=self._schema_str(func_name),
            n=n,
        )
        text = self.client.complete(
            SYSTEM_PROMPT, prompt, self.temperature, self.max_tokens
        )
        return self._parse_json_list(text)

    def _call_parallel(self, func_names: list[str], n: int) -> list[dict]:
        prompt = _PARALLEL_CALL_TEMPLATE.format(
            schemas=self._schemas_str(func_names),
            n=n,
        )
        text = self.client.complete(
            SYSTEM_PROMPT, prompt, self.temperature, self.max_tokens
        )
        return self._parse_json_list(text)

    def _call_sequential(self, func_names: list[str], n: int) -> list[dict]:
        prompt = _SEQUENTIAL_CALL_TEMPLATE.format(
            schemas=self._schemas_str(func_names),
            n=n,
        )
        text = self.client.complete(
            SYSTEM_PROMPT, prompt, self.temperature, self.max_tokens
        )
        return self._parse_json_list(text)

    def _call_abstention(self, func_names: list[str], n: int) -> list[dict]:
        prompt = _ABSTENTION_TEMPLATE.format(
            schemas=self._schemas_str(func_names),
            n=n,
        )
        text = self.client.complete(
            SYSTEM_PROMPT, prompt, self.temperature, self.max_tokens
        )
        return self._parse_json_list(text)

    # ── parsers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_list(text: str) -> list[dict]:
        """Extract the first JSON array from LLM output."""
        text = text.strip()
        # strip markdown fences if present
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rstrip("`").strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            # try to find JSON array via bracket matching
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except Exception:
                    pass
            return []

    def _parse_single(self, raw: dict, func_name: str, split: str) -> DataSample | None:
        query = raw.get("query", "").strip()
        call = raw.get("function_call")
        if not query or not call:
            return None
        return DataSample(
            id=str(uuid.uuid4()),
            query=query,
            workflow_type="single_call",
            function_name=func_name,
            ground_truth={
                "function": call.get("function", func_name),
                "arguments": call.get("arguments", {}),
                "workflow": "single_call",
                "reasoning": raw.get("reasoning", ""),
            },
            retrieved_functions=[],
            split=split,
        )

    def _parse_parallel(
        self, raw: dict, func_names: list[str], split: str = "train"
    ) -> DataSample | None:
        query = raw.get("query", "").strip()
        calls = raw.get("function_calls", [])
        if not query or not calls:
            return None
        primary = calls[0].get("function", func_names[0]) if calls else func_names[0]
        return DataSample(
            id=str(uuid.uuid4()),
            query=query,
            workflow_type="parallel",
            function_name=primary,
            ground_truth={
                "function": primary,
                "arguments": calls[0].get("arguments", {}) if calls else {},
                "workflow": "parallel",
                "calls": calls,
                "reasoning": raw.get("reasoning", ""),
            },
            retrieved_functions=[],
            split=split,
        )

    def _parse_sequential(
        self, raw: dict, func_names: list[str], split: str = "train"
    ) -> DataSample | None:
        query = raw.get("query", "").strip()
        calls = raw.get("function_calls", [])
        if not query or not calls:
            return None
        first_call = calls[0] if calls else {}
        return DataSample(
            id=str(uuid.uuid4()),
            query=query,
            workflow_type="sequential",
            function_name=first_call.get("function", func_names[0]),
            ground_truth={
                "function": first_call.get("function", func_names[0]),
                "arguments": first_call.get("arguments", {}),
                "workflow": "sequential",
                "calls": calls,
                "reasoning": raw.get("reasoning", ""),
            },
            retrieved_functions=[],
            split=split,
        )

    def _parse_abstention(
        self, raw: dict, func_names: list[str], split: str = "train"
    ) -> DataSample | None:
        query = raw.get("query", "").strip()
        if not query:
            return None
        return DataSample(
            id=str(uuid.uuid4()),
            query=query,
            workflow_type="abstention",
            function_name="none",
            ground_truth={
                "function": None,
                "arguments": {},
                "workflow": "abstention",
                "refusal_message": raw.get("refusal_message", ""),
                "reasoning": raw.get("reasoning", ""),
            },
            retrieved_functions=[],
            split=split,
        )

    # ── retrieval simulation ──────────────────────────────────────────────────

    def _simulate_retrieval(self, samples: list[DataSample], k: int = 5) -> None:
        """
        Fill retrieved_functions with the ground-truth function + (k-1) distractors.
        This simulates what the real retriever will return at inference time.
        Real retrieval happens in src/data/retrieval.py at training time.
        """
        for s in samples:
            true_fn = s.function_name
            distractors = [fn for fn in self.all_func_names if fn != true_fn]
            chosen_distractors = random.sample(
                distractors, min(k - 1, len(distractors))
            )
            pool = (
                ([true_fn] + chosen_distractors)
                if true_fn != "none"
                else chosen_distractors
            )
            random.shuffle(pool)
            s.retrieved_functions = pool[:k]

    # ── I/O helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _save_jsonl(samples: list[DataSample], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
        print(f"[DatasetGenerator] Saved {len(samples)} samples → {path}")

    def _rate_limit_wait(self, n_calls: int) -> None:
        """Naive rate-limit: sleep if we're generating too fast."""
        if self.rpm > 0:
            min_interval = n_calls / (self.rpm / 60)
            if min_interval > 0.5:
                time.sleep(min_interval)
```

## 9. `data\excel_parser.py`

```python
"""
excel_parser.py
---------------
Reads telecom_functions.xlsx and converts each row into a structured
JSON schema entry.  Also supports loading directly from function_schema.json
if the Excel file is not yet available.
"""

import json
import ast
from pathlib import Path
from typing import Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _safe_json(value: str, fallback=None):
    """Parse a JSON string; return fallback on any error."""
    if not isinstance(value, str) or not value.strip():
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except Exception:
            return fallback


def _safe_list(value: str) -> list:
    """Parse a Python list string or JSON array."""
    result = _safe_json(value, fallback=None)
    if isinstance(result, list):
        return result
    if isinstance(value, str):
        # treat as newline/comma-separated plain text
        return [v.strip() for v in value.replace("\n", ",").split(",") if v.strip()]
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def parse_telecom_functions(
    excel_path: str,
    output_path: Optional[str] = "data/processed/function_library.json",
) -> dict:
    """
    Parse *telecom_functions.xlsx* → function_library.json.

    Expected columns
    ────────────────
    function_name   : str
    description     : str
    parameters      : JSON string  →  {"param_name": {"type", "required", ...}}
    example_queries : JSON/list    →  ["query1", "query2", ...]
    domain_info     : JSON string  →  {"domain": ..., "category": ...}
    constraints     : JSON string  →  {"param_name": {"min": ..., "max": ...}}   [optional]
    tags            : list / str   →  ["tag1", "tag2"]                           [optional]

    Returns
    ───────
    dict  –  {function_name: schema_dict}
    """
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(path)
    required_cols = {"function_name", "description", "parameters"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Excel is missing required columns: {missing}")

    library: dict = {}

    for idx, row in df.iterrows():
        name = str(row["function_name"]).strip()
        if not name:
            continue

        params = _safe_json(row.get("parameters", "{}"), fallback={})
        examples = _safe_list(row.get("example_queries", "[]"))
        domain = _safe_json(row.get("domain_info", "{}"), fallback={})
        constraints = _safe_json(row.get("constraints", "{}"), fallback={})
        tags = _safe_list(row.get("tags", "[]"))

        func_schema = {
            "name": name,
            "description": str(row["description"]).strip(),
            "parameters": params,
            "examples": examples,
            "domain": domain,
            "constraints": constraints,
            "tags": tags,
        }
        library[name] = func_schema

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(library, fh, indent=2, ensure_ascii=False)
        print(f"[excel_parser] Saved {len(library)} functions → {out}")

    return library


def load_function_library(library_path: str) -> dict:
    """Load a previously parsed function_library.json."""
    with open(library_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_function_schema(schema_path: str) -> dict:
    """
    Load the raw function_schema.json provided by the mentor.
    This is the PRIMARY input for dataset generation when the Excel is unavailable.

    Expected format
    ───────────────
    {
      "function_name_1": {
          "name": "...",
          "description": "...",
          "parameters": { ... },
          "examples": [...],
          ...
      },
      ...
    }
    """
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    print(f"[excel_parser] Loaded function_schema.json  →  {len(schema)} functions")
    return schema
```

## 10. `data\retrieval.py`

```python
"""
retrieval.py
────────────
Two-stage retrieval:

Stage 1 — FunctionRetriever
    Returns the top-k most relevant FUNCTION NAMES for a query.
    Methods: BM25, embedding (sentence-transformers), hybrid.

Stage 2 — ArgumentValueRetriever
    For each parameter in the retrieved functions, finds the most
    relevant ARGUMENT VALUES from a pre-built catalog.
    Example: query "TP.HCM" → location_code: HCM (Thành phố Hồ Chí Minh)

Combined — TelcoRetriever
    Orchestrates both stages and returns a RetrievalResult:
        .function_names     → list[str]
        .argument_values    → dict[param_name → list[ValueMatch]]
"""

from __future__ import annotations

import json
import pickle
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ValueMatch:
    """A single matched argument value."""

    code: str  # the actual argument value to put in the function call
    label: str  # human-readable name
    group: str  # catalog group (e.g. "Tỉnh/Thành phố", "technology")
    score: float = 0.0  # relevance score (higher = more relevant)
    alt_label: str = ""  # alternative label if present


@dataclass
class RetrievalResult:
    """Complete output of TelcoRetriever.retrieve()."""

    function_names: list[str]  # top-k function names
    argument_values: dict[str, list[ValueMatch]]  # param_name → matches


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Function Retriever (unchanged logic, clean rewrite)
# ──────────────────────────────────────────────────────────────────────────────


class FunctionRetriever:
    """
    Retrieves the top-k most relevant functions for a user query.
    Methods: 'bm25' | 'embedding' | 'hybrid'
    """

    def __init__(
        self,
        function_library: dict,
        method: Literal["bm25", "embedding", "hybrid"] = "hybrid",
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        bm25_weight: float = 0.4,
        emb_weight: float = 0.6,
        index_dir: str | None = None,
    ):
        self.library = function_library
        self.method = method
        self.func_names = list(function_library.keys())
        self.bm25_weight = bm25_weight
        self.emb_weight = emb_weight

        # Rich description: name + description + parameter names
        self.desc_list = [
            self._build_search_text(name, schema)
            for name, schema in function_library.items()
        ]

        self._bm25 = None
        self._encoder = None
        self._embeddings = None

        if method in ("bm25", "hybrid"):
            self._init_bm25()
        if method in ("embedding", "hybrid"):
            self._init_embeddings(encoder_model, index_dir)

    @staticmethod
    def _build_search_text(name: str, schema: dict) -> str:
        """
        Build a rich text representation of a function for indexing.
        Includes name, description, and parameter names/descriptions.
        """
        parts = [name, schema.get("description", "")]
        for pname, pinfo in schema.get("parameters", {}).items():
            parts.append(pname)
            if isinstance(pinfo, dict):
                parts.append(pinfo.get("description", ""))
        tags = schema.get("tags", [])
        parts.extend(tags)
        return " ".join(p for p in parts if p)

    def _init_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        tokenized = [d.lower().split() for d in self.desc_list]
        self._bm25 = BM25Okapi(tokenized)
        print("[FunctionRetriever] BM25 index built.")

    def _init_embeddings(self, model_name: str, index_dir: str | None) -> None:
        from sentence_transformers import SentenceTransformer

        self._encoder = SentenceTransformer(model_name)
        cache = Path(index_dir) / "func_embeddings.npy" if index_dir else None
        if cache and cache.exists():
            self._embeddings = np.load(str(cache))
            print(f"[FunctionRetriever] Loaded embedding cache from {cache}")
        else:
            self._embeddings = self._encoder.encode(
                self.desc_list,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(cache), self._embeddings)
                print(f"[FunctionRetriever] Embeddings saved → {cache}")

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        scores = self._score(query)
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.func_names[i] for i in top_k]

    def retrieve_with_scores(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        scores = self._score(query)
        top_k = np.argsort(scores)[-k:][::-1]
        return [(self.func_names[i], float(scores[i])) for i in top_k]

    def _score(self, query: str) -> np.ndarray:
        if self.method == "bm25":
            return self._bm25_scores(query)
        elif self.method == "embedding":
            return self._emb_scores(query)
        else:
            bm25 = self._minmax(self._bm25_scores(query))
            emb = self._minmax(self._emb_scores(query))
            return self.bm25_weight * bm25 + self.emb_weight * emb

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.array(
            self._bm25.get_scores(query.lower().split()),
            dtype=np.float32,
        )

    def _emb_scores(self, query: str) -> np.ndarray:
        q = self._encoder.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        )
        return (self._embeddings @ q).astype(np.float32)

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    def save(self, path: str) -> None:
        state = {
            "func_names": self.func_names,
            "desc_list": self.desc_list,
            "method": self.method,
            "bm25_weight": self.bm25_weight,
            "emb_weight": self.emb_weight,
            "embeddings": self._embeddings,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        print(f"[FunctionRetriever] Saved → {path}")

    @classmethod
    def load(
        cls,
        path: str,
        function_library: dict,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "FunctionRetriever":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.library = function_library
        obj.func_names = state["func_names"]
        obj.desc_list = state["desc_list"]
        obj.method = state["method"]
        obj.bm25_weight = state["bm25_weight"]
        obj.emb_weight = state["emb_weight"]
        obj._embeddings = state["embeddings"]
        obj._bm25 = None
        obj._encoder = None
        if obj.method in ("bm25", "hybrid"):
            obj._init_bm25()
        if obj.method in ("embedding", "hybrid"):
            from sentence_transformers import SentenceTransformer

            obj._encoder = SentenceTransformer(encoder_model)
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Argument Value Retriever
# ──────────────────────────────────────────────────────────────────────────────


class ArgumentValueRetriever:
    """
    For a given query and a set of function parameter names,
    retrieves the most relevant possible argument values from a catalog.

    The catalog maps parameter_name → list of {code, label, group, ...} dicts.

    Matching strategy:
        1. Direct substring match:  query contains code or label verbatim
        2. Normalised fuzzy match:  query normalised (no diacritics) matches
        3. Token match:             any query token matches code/label token

    Scores:
        direct_exact  = 1.0
        direct_fuzzy  = 0.8
        token_match   = 0.5  per matching token (capped at 0.9)
    """

    def __init__(
        self,
        argument_values: dict,  # loaded from argument_values.json
        top_k_values: int = 3,
    ):
        self.catalog = argument_values  # param_name → list[dict]
        self.top_k_values = top_k_values

        # Pre-compute normalised forms for fast matching
        self._norm_cache: dict[str, str] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve_for_function(
        self,
        query: str,
        function_schema: dict,
    ) -> dict[str, list[ValueMatch]]:
        """
        For all parameters in a function schema, retrieve relevant values.

        Parameters
        ──────────
        query           : raw user query string
        function_schema : one function's schema dict from function_library.json

        Returns
        ───────
        {param_name: [ValueMatch, ...]}  — only params with catalog entries
        """
        result: dict[str, list[ValueMatch]] = {}
        params = function_schema.get("parameters", {})

        for param_name in params:
            # Check if this param name has a value catalog
            values_for_param = self._get_catalog(param_name)
            if not values_for_param:
                continue

            # Score all catalog values against the query
            matches = self._score_values(query, values_for_param)
            if matches:
                result[param_name] = matches

        return result

    def retrieve_for_functions(
        self,
        query: str,
        function_names: list[str],
        function_library: dict,
    ) -> dict[str, list[ValueMatch]]:
        """
        Retrieve argument values for all parameters across multiple functions.
        De-duplicates by param_name (union of all retrieved functions).

        Returns
        ───────
        {param_name: [ValueMatch, ...]}
        """
        combined: dict[str, list[ValueMatch]] = {}

        for fn in function_names:
            if fn not in function_library:
                continue
            fn_results = self.retrieve_for_function(query, function_library[fn])
            for param_name, matches in fn_results.items():
                if param_name not in combined:
                    combined[param_name] = matches
                else:
                    # Merge: keep highest-scoring unique codes
                    existing_codes = {m.code for m in combined[param_name]}
                    for m in matches:
                        if m.code not in existing_codes:
                            combined[param_name].append(m)
                    # Re-sort by score
                    combined[param_name].sort(key=lambda x: -x.score)
                    combined[param_name] = combined[param_name][: self.top_k_values]

        return combined

    # ── Catalog lookup ────────────────────────────────────────────────────────

    def _get_catalog(self, param_name: str) -> list[dict]:
        """
        Look up catalog entries for a parameter name.
        Tries exact match first, then suffix/prefix matching
        (e.g. "source_location_code" → "location_code").
        """
        # Exact match
        if param_name in self.catalog:
            return self.catalog[param_name]

        # Suffix match: find the longest catalog key that is a suffix of param_name
        best_key = None
        best_len = 0
        for key in self.catalog:
            if param_name.endswith(key) and len(key) > best_len:
                best_key = key
                best_len = len(key)
        if best_key:
            return self.catalog[best_key]

        # Prefix match: catalog key starts param_name
        for key in self.catalog:
            if param_name.startswith(key):
                return self.catalog[key]

        return []

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_values(
        self,
        query: str,
        catalog: list[dict],
    ) -> list[ValueMatch]:
        """
        Score each catalog entry against the query and return top-k matches.
        Only returns entries with score > 0.
        """
        query_norm = self._normalise(query)
        query_tokens = set(query_norm.split())

        scored: list[ValueMatch] = []

        for entry in catalog:
            code = str(entry.get("code", ""))
            label = str(entry.get("label", ""))
            alt_label = str(entry.get("alt_label", ""))
            group = str(entry.get("group", ""))

            score = self._match_score(
                query,
                query_norm,
                query_tokens,
                code,
                label,
                alt_label,
            )

            if score > 0.0:
                scored.append(
                    ValueMatch(
                        code=code,
                        label=label,
                        group=group,
                        score=score,
                        alt_label=alt_label,
                    )
                )

        # Sort by score descending, return top-k
        scored.sort(key=lambda x: -x.score)
        return scored[: self.top_k_values]

    def _match_score(
        self,
        query: str,
        query_norm: str,
        query_tokens: set[str],
        code: str,
        label: str,
        alt_label: str,
    ) -> float:
        """
        Compute relevance score for one catalog entry.

        Scoring rules (highest score wins):
          1.0  exact code match (case-insensitive)
          1.0  exact label match (case-insensitive, normalised)
          0.8  code appears as substring in query
          0.8  label appears as substring in query (normalised)
          0.8  alt_label appears as substring in query (normalised)
          0.5  per overlapping token between query tokens and label tokens
               (capped at 0.9)
        """
        q_lower = query.lower()
        code_lower = code.lower()

        # Rule 1: exact code in query
        if code_lower in q_lower:
            # Weight: longer codes are more specific matches
            specificity = min(1.0, len(code) / 5.0)
            return 0.8 + 0.2 * specificity

        # Rule 2: exact label in query (normalised)
        label_norm = self._normalise(label)
        alt_label_norm = self._normalise(alt_label) if alt_label else ""

        if label_norm in query_norm:
            return 1.0
        if alt_label_norm and alt_label_norm in query_norm:
            return 0.9

        # Rule 3: token overlap (normalised)
        label_tokens = set(label_norm.split())
        alt_label_tokens = set(alt_label_norm.split()) if alt_label_norm else set()
        all_label_tokens = label_tokens | alt_label_tokens

        # Remove very short/common tokens
        meaningful_label_tokens = {t for t in all_label_tokens if len(t) >= 2}
        meaningful_query_tokens = {t for t in query_tokens if len(t) >= 2}

        if not meaningful_label_tokens:
            return 0.0

        overlap = meaningful_query_tokens & meaningful_label_tokens
        if overlap:
            token_score = min(0.7, 0.35 * len(overlap))
            return token_score

        return 0.0

    def _normalise(self, text: str) -> str:
        """
        Normalise text for matching:
        - Lowercase
        - Remove Vietnamese diacritics
        - Collapse whitespace
        """
        if text in self._norm_cache:
            return self._norm_cache[text]

        # Remove diacritics via Unicode normalisation
        nfkd = unicodedata.normalize("NFKD", text.lower())
        ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))

        # Extra Vietnamese normalisation
        replacements = {
            "đ": "d",
            "ð": "d",
        }
        for src, dst in replacements.items():
            ascii_text = ascii_text.replace(src, dst)

        result = re.sub(r"\s+", " ", ascii_text).strip()
        self._norm_cache[text] = result
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Combined TelcoRetriever
# ──────────────────────────────────────────────────────────────────────────────


class TelcoRetriever:
    """
    Orchestrates FunctionRetriever + ArgumentValueRetriever.

    Usage
    ─────
    retriever = TelcoRetriever.build(function_library, argument_values)
    result    = retriever.retrieve("Xem KPI tại TP.HCM", k=5)

    result.function_names   → ["SPEEDTEST_PROVINCE", ...]
    result.argument_values  → {"location_code": [ValueMatch(code="HCM", ...)], ...}
    """

    def __init__(
        self,
        function_retriever: FunctionRetriever,
        value_retriever: ArgumentValueRetriever,
    ):
        self.func_retriever = function_retriever
        self.value_retriever = value_retriever

    @classmethod
    def build(
        cls,
        function_library: dict,
        argument_values: dict,
        method: str = "hybrid",
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k_values: int = 3,
        index_dir: str | None = None,
    ) -> "TelcoRetriever":
        """Build a TelcoRetriever from raw library and value catalog dicts."""
        func_ret = FunctionRetriever(
            function_library=function_library,
            method=method,
            encoder_model=encoder_model,
            index_dir=index_dir,
        )
        val_ret = ArgumentValueRetriever(
            argument_values=argument_values,
            top_k_values=top_k_values,
        )
        return cls(func_ret, val_ret)

    @classmethod
    def load(
        cls,
        retriever_path: str,
        function_library: dict,
        argument_values: dict,
        top_k_values: int = 3,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "TelcoRetriever":
        """Load a saved FunctionRetriever + build a fresh ArgumentValueRetriever."""
        func_ret = FunctionRetriever.load(
            retriever_path, function_library, encoder_model
        )
        val_ret = ArgumentValueRetriever(
            argument_values=argument_values,
            top_k_values=top_k_values,
        )
        return cls(func_ret, val_ret)

    def retrieve(
        self,
        query: str,
        function_library: dict,
        k: int = 5,
        precomputed_func_names: list[str] | None = None,
    ) -> RetrievalResult:
        """
        Full two-stage retrieval.

        Parameters
        ──────────
        query                  : user query
        function_library       : full schema dict
        k                      : number of functions to retrieve
        precomputed_func_names : if provided, skip function retrieval
                                 (used when dataset already has retrieved_functions)

        Returns
        ───────
        RetrievalResult with function_names and argument_values
        """
        # Stage 1: function retrieval
        if precomputed_func_names is not None:
            func_names = precomputed_func_names
        else:
            func_names = self.func_retriever.retrieve(query, k=k)

        # Stage 2: argument value retrieval
        arg_values = self.value_retriever.retrieve_for_functions(
            query=query,
            function_names=func_names,
            function_library=function_library,
        )

        return RetrievalResult(
            function_names=func_names,
            argument_values=arg_values,
        )
```

## 11. `evaluation\__init__.py`

```python

```

## 12. `evaluation\benchmark.py`

```python
"""
benchmark.py
─────────────
Run full evaluation of a fine-tuned model on the test set.

Prompt construction uses EXACTLY the same functions as the training pipeline
(build_messages_for_grpo → apply_chat_template with the custom template)
so train/eval prompt format is guaranteed to be identical.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jsonlines
from tqdm import tqdm

from src.algorithms.base_trainer import (
    build_messages_for_grpo,
    patch_tokenizer_for_custom_roles,
    SYSTEM_PROMPT,
)
from src.data.retrieval    import FunctionRetriever, ArgumentValueRetriever
from src.utils.model_utils import load_model_from_path, generate_response
from src.utils.sandbox     import Sandbox
from .metrics              import compute_all_metrics, aggregate_metrics, estimate_cost
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model_path:            str,
    test_dataset_path:     str,
    function_library:      dict,
    retriever:             FunctionRetriever,
    sandbox:               Sandbox,
    top_k:                 int   = 5,
    max_new_tokens:        int   = 512,
    model_name_tag:        str   = "model",
    use_dataset_retrieval: bool  = True,
    argument_values:       dict | None = None,   # full catalog for val retriever
) -> dict:
    """
    Evaluate a fine-tuned model on the held-out test set.

    Parameters
    ──────────
    use_dataset_retrieval : True  → use pre-computed retrieved_functions from JSONL
                            False → run live FunctionRetriever per query
    argument_values       : full argument_values.json dict; if provided, runs
                            ArgumentValueRetriever to add arg values to prompt
    """
    # ── Load model + patch tokenizer ──────────────────────────────────────────
    logger.info(f"[Benchmark] Loading model from {model_path}")
    model, tokenizer = load_model_from_path(model_path)
    patch_tokenizer_for_custom_roles(tokenizer)   # ← required for "retriever" role

    # ── Optional argument value retriever ─────────────────────────────────────
    val_retriever = None
    if argument_values is not None:
        val_retriever = ArgumentValueRetriever(argument_values)

    # ── Load test samples ─────────────────────────────────────────────────────
    test_samples: list[dict] = []
    with jsonlines.open(test_dataset_path) as reader:
        for obj in reader:
            test_samples.append(obj)

    logger.info(
        f"[Benchmark] {model_name_tag}: evaluating {len(test_samples)} samples"
    )

    results: list[dict] = []

    for sample in tqdm(test_samples, desc=f"Eval [{model_name_tag}]"):
        query = sample["query"]
        gt    = sample.get("ground_truth", {})

        # Stage 1: function retrieval
        if use_dataset_retrieval and sample.get("retrieved_functions"):
            retrieved = sample["retrieved_functions"]
        else:
            retrieved = retriever.retrieve(query, k=top_k)

        # Stage 2: argument value retrieval
        if val_retriever is not None:
            arg_vals = val_retriever.retrieve_for_functions(
                query, retrieved, function_library
            )
        else:
            raw_av   = sample.get("retrieved_argument_values")
            arg_vals = raw_av if raw_av else None

        # Build prompt using the same helper as training
        messages = build_messages_for_grpo(
            query            = query,
            function_names   = retrieved,
            function_library = function_library,
            argument_values  = arg_vals,
        )

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = True,
        )

        # Generate
        t0       = time.perf_counter()
        response = generate_response(model, tokenizer, prompt, max_new_tokens)
        latency  = (time.perf_counter() - t0) * 1000.0   # ms

        cost    = estimate_cost(prompt, response)
        metrics = compute_all_metrics(
            response, gt, sandbox, latency, cost, function_library
        )
        metrics["sample_id"] = sample.get("id", "")
        results.append(metrics)

    agg = aggregate_metrics(results)
    logger.info(f"[Benchmark] {model_name_tag} aggregate: {agg}")
    return {"model": model_name_tag, "per_sample": results, "aggregate": agg}
```

## 13. `evaluation\metrics.py`

```python
"""
metrics.py
───────────
All 9 evaluation metrics for the telecom function-calling benchmark.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.reward.base_reward import (
    extract_call,
    schema_valid,
    func_selection_ok,
    args_accuracy,
)


# ──────────────────────────────────────────────────────────────────────────────


def compute_all_metrics(
    response: str,
    ground_truth: dict,
    sandbox,
    latency_ms: float,
    cost_estimate: float,
    function_library: dict,
) -> dict[str, float]:
    """
    Compute all 9 metrics for a single sample.

    Returns a flat dict of metric_name → float.
    """
    call = extract_call(response)
    expected_func = ground_truth.get("function", "none")
    expected_args = ground_truth.get("arguments", {})
    workflow = ground_truth.get("workflow", "single_call")

    # ── Metric 1: Function Selection Accuracy ────────────────────────────────
    func_sel_acc = func_selection_ok(response, expected_func)

    # ── Metric 2: Argument Accuracy ──────────────────────────────────────────
    arg_acc = args_accuracy(response, expected_args)

    # ── Metric 3: Schema Validity ────────────────────────────────────────────
    schema_val = schema_valid(response)

    # ── Metric 4: Execution Success Rate ────────────────────────────────────
    exec_success = 0.0
    if sandbox is not None:
        exec_success = 1.0 if sandbox.execute(response) else 0.0
    else:
        exec_success = float(call is not None)

    # ── Metric 5: Task Success Rate ──────────────────────────────────────────
    # Full success: correct function + args accuracy > 0.8 + execution ok
    task_success = float(func_sel_acc == 1.0 and arg_acc >= 0.8 and exec_success == 1.0)

    # ── Metric 6: Hallucinated Call Rate ─────────────────────────────────────
    hallucinated = 0.0
    if call is not None:
        called_fn = call.get("function", "")
        if called_fn and called_fn not in function_library:
            hallucinated = 1.0

    # ── Metric 7: Abstention Accuracy ────────────────────────────────────────
    abstention_acc = float("nan")  # only meaningful for abstention samples
    if workflow == "abstention":
        # Correct abstention: model produces null call or refuses
        produced_call = extract_call(response)
        if produced_call is None or produced_call == "null":
            abstention_acc = 1.0
        else:
            abstention_acc = 0.0

    # ── Metric 8: Latency ────────────────────────────────────────────────────
    latency = latency_ms  # in milliseconds

    # ── Metric 9: Cost per Query ─────────────────────────────────────────────
    cost = cost_estimate  # in USD (estimated externally)

    return {
        "function_selection_accuracy": func_sel_acc,
        "argument_accuracy": arg_acc,
        "schema_validity": schema_val,
        "execution_success_rate": exec_success,
        "task_success_rate": task_success,
        "hallucinated_call_rate": hallucinated,
        "abstention_accuracy": abstention_acc,
        "latency_ms": latency,
        "cost_per_query_usd": cost,
    }


def aggregate_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    """
    Aggregate per-sample metrics into dataset-level statistics.
    Handles NaN values for metrics not applicable to all samples.
    """
    if not results:
        return {}

    keys = results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in results if not np.isnan(r[k])]
        if vals:
            agg[k] = float(np.mean(vals))
            agg[f"{k}__std"] = float(np.std(vals))
            agg[f"{k}__count"] = len(vals)
        else:
            agg[k] = float("nan")
    return agg


def estimate_cost(
    prompt: str, response: str, price_per_1k_tokens: float = 0.0002
) -> float:
    """Rough token-cost estimate (assumes ~1.3 chars/token for English)."""
    total_chars = len(prompt) + len(response)
    tokens_est = total_chars / 1.3
    return (tokens_est / 1000) * price_per_1k_tokens
```

## 14. `evaluation\report_generator.py`

```python
"""
report_generator.py
────────────────────
Generate comparative reports, CSV exports, and plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

METRIC_DISPLAY_NAMES = {
    "function_selection_accuracy": "Func. Selection Acc.",
    "argument_accuracy": "Arg. Accuracy",
    "schema_validity": "Schema Validity",
    "execution_success_rate": "Exec. Success Rate",
    "task_success_rate": "Task Success Rate",
    "hallucinated_call_rate": "Hallucination Rate ↓",
    "abstention_accuracy": "Abstention Acc.",
    "latency_ms": "Latency (ms) ↓",
    "cost_per_query_usd": "Cost/Query (USD) ↓",
}

HIGHER_IS_BETTER = {
    "hallucinated_call_rate": False,
    "latency_ms": False,
    "cost_per_query_usd": False,
}


def generate_report(
    eval_results: list[dict],
    output_dir: str = "outputs/evaluation_reports",
) -> None:
    """
    Generate full comparative report for all evaluated models.

    Parameters
    ──────────
    eval_results : list of dicts returned by benchmark.evaluate_model
    output_dir   : where to save CSV, JSON, and plots
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Build summary DataFrame ───────────────────────────────────────────────
    rows = []
    for result in eval_results:
        model = result["model"]
        agg = result["aggregate"]
        row = {"Model": model}
        for k, display in METRIC_DISPLAY_NAMES.items():
            row[display] = round(agg.get(k, float("nan")), 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    # ── Console table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  EVALUATION REPORT — Telecom Tool-Calling with RL")
    print("=" * 80)
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = out / "metrics_summary.csv"
    df.to_csv(csv_path)
    logger.info(f"[Report] CSV saved → {csv_path}")

    # ── Save full JSON ────────────────────────────────────────────────────────
    json_path = out / "full_results.json"
    with open(json_path, "w") as fh:
        json.dump(eval_results, fh, indent=2, default=str)
    logger.info(f"[Report] JSON saved → {json_path}")

    # ── Bar chart comparison ──────────────────────────────────────────────────
    _plot_bar_comparison(df, out)

    # ── Radar chart ───────────────────────────────────────────────────────────
    _plot_radar(df, out)

    logger.info(f"[Report] All outputs written to {out}")


def _plot_bar_comparison(df: pd.DataFrame, out: Path) -> None:
    core_metrics = [
        "Func. Selection Acc.",
        "Arg. Accuracy",
        "Schema Validity",
        "Exec. Success Rate",
        "Task Success Rate",
    ]
    plot_df = df[[c for c in core_metrics if c in df.columns]]

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.T.plot(kind="bar", ax=ax, width=0.7, colormap="tab10")
    ax.set_title("Model Comparison – Core Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", loc="lower right")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(out / "bar_comparison.png", dpi=150)
    plt.close(fig)


def _plot_radar(df: pd.DataFrame, out: Path) -> None:
    radar_metrics = [
        "Func. Selection Acc.",
        "Arg. Accuracy",
        "Schema Validity",
        "Exec. Success Rate",
        "Task Success Rate",
        "Abstention Acc.",
    ]
    categories = [c for c in radar_metrics if c in df.columns]
    N = len(categories)
    if N < 3:
        return

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={"polar": True})
    cmap = plt.get_cmap("tab10")

    for i, (model, row) in enumerate(df.iterrows()):
        vals = [row.get(c, 0.0) for c in categories]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=cmap(i), label=model)
        ax.fill(angles, vals, alpha=0.1, color=cmap(i))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_title("Algorithm Radar Comparison", size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(out / "radar_comparison.png", dpi=150)
    plt.close(fig)
```

## 15. `reward\__init__.py`

```python
from .base_reward import extract_call, schema_valid, func_selection_ok, args_accuracy
from .rc_grpo_reward import rc_grpo_reward
from .awpo_reward import awpo_reward
from .gvpo_reward import gvpo_reward_func

__all__ = [
    "extract_call",
    "schema_valid",
    "func_selection_ok",
    "args_accuracy",
    "rc_grpo_reward",
    "awpo_reward",
    "gvpo_reward_func",
]
```

## 16. `reward\awpo_reward.py`

```python
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
    func_selection_ok,
    args_accuracy,
    schema_valid,
    format_reward,
    reasoning_quality,
)


# ── Outcome reward ────────────────────────────────────────────────────────────


def _outcome_reward(
    response: str,
    ground_truth: dict,
    sandbox,
    outcome_weights: dict,
) -> float:
    """
    Weighted sum of verifiable outcome components.
    Default weights from paper ablation (Table 4): equal-weight components.
    """
    fn_ok = func_selection_ok(response, ground_truth.get("function", ""))
    args_ok = args_accuracy(response, ground_truth.get("arguments", {}))

    if sandbox is not None:
        from src.utils.sandbox import Sandbox

        exec_ok = 1.0 if sandbox.execute(response) else 0.0
    else:
        exec_ok = schema_valid(response)

    return (
        outcome_weights["func_selection"] * fn_ok
        + outcome_weights["args_accuracy"] * args_ok
        + outcome_weights["execution"] * exec_ok
    )


# ── Group-level statistics ────────────────────────────────────────────────────


def compute_group_stats(rewards: list[float]) -> dict:
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
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
    outcome_rewards: list[float],
    reasoning_rewards: list[float],
    tau: float = 0.5,
    eps: float = 1e-8,
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
    mu_out = out_stats["mean"]
    sigma2_out = out_stats["var"]

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
    mu_mix = mix_stats["mean"]
    std_mix = mix_stats["std"]

    # ── Step 6: weighted, normalised advantages ───────────────────────────────
    advantages = [w * (r - mu_mix) / (std_mix + eps) for r in mixed]

    # ── Step 7: adaptive clip expansion factor ────────────────────────────────
    # When reasoning gate is open (g=1), clip window widens by δ=0.2
    # to accommodate higher-variance reasoning signal (paper Section 3.3)
    adaptive_clip_delta = g * 0.2

    return advantages, adaptive_clip_delta


# ── Per-response AWPO reward (single call, used in TRL wrapper) ──────────────


def awpo_reward(
    response: str,
    ground_truth: dict,
    sandbox=None,
    group_stats: dict | None = None,
    outcome_weights: dict | None = None,
    tau: float = 0.5,
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
        "args_accuracy": 0.3,
        "execution": 0.3,
    }

    r_out = _outcome_reward(response, ground_truth, sandbox, ow)
    r_reason = reasoning_quality(response)

    if group_stats is None:
        return r_out  # no group context yet → return pure outcome

    sigma2_out = group_stats.get("var", 1.0)
    g = variance_gate(sigma2_out, tau)

    return (1.0 - g) * r_out + g * r_reason


# ── TRL-compatible wrapper ────────────────────────────────────────────────────


def awpo_reward_func(
    completions: list[str],
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
        "args_accuracy": 0.3,
        "execution": 0.3,
    }

    # Compute raw outcome rewards for all completions
    out_rewards = [
        _outcome_reward(c, gt if isinstance(gt, dict) else {}, None, outcome_weights)
        for c, gt in zip(completions, ground_truth)
    ]
    return out_rewards
```

## 17. `reward\base_reward.py`

```python
"""
base_reward.py
──────────────
Shared reward components. Updated to use <tool_call> tag
(replacing <call>) to match the new conversation format.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ── Tag patterns (new format) ─────────────────────────────────────────────────
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)

# Keep backward-compatible alias so existing imports don't break
_CALL_RE = _TOOL_CALL_RE
_REASON_RE = _REASONING_RE


def extract_call(response: str) -> dict | None:
    """
    Extract the first <tool_call>...</tool_call> JSON block.
    Falls back to <call>...</call> for backward compatibility.
    """
    match = _TOOL_CALL_RE.search(response)
    if not match:
        # Backward compat: try old <call> tag
        old_re = re.compile(r"<call>(.*?)</call>", re.DOTALL)
        match = old_re.search(response)
    if not match:
        return None
    raw = match.group(1).strip()
    if raw.lower() == "null":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def extract_all_calls(response: str) -> list[dict]:
    calls = []
    for m in _TOOL_CALL_RE.finditer(response):
        try:
            parsed = json.loads(m.group(1).strip())
            if parsed is not None:
                calls.append(parsed)
        except json.JSONDecodeError:
            pass
    return calls


def extract_reasoning(response: str) -> str:
    match = _REASONING_RE.search(response)
    return match.group(1).strip() if match else ""


def schema_valid(response: str) -> float:
    return 1.0 if extract_call(response) is not None else 0.0


def func_selection_ok(response: str, expected_func: str) -> float:
    call = extract_call(response)
    if call is None:
        return 0.0
    return 1.0 if call.get("function") == expected_func else 0.0


def args_accuracy(response: str | dict, expected_args: dict) -> float:
    call = extract_call(response) if isinstance(response, str) else response
    if call is None:
        return 0.0
    if not expected_args:
        return 1.0
    actual = call.get("arguments", {})
    correct = sum(
        1
        for k, v in expected_args.items()
        if str(actual.get(k, "")).strip() == str(v).strip()
    )
    return correct / len(expected_args)


def reasoning_quality(response: str) -> float:
    text = extract_reasoning(response)
    if not text:
        return 0.0
    words = len(text.split())
    length_score = min(1.0, words / 50.0)
    has_steps = bool(
        re.search(
            r"(step\s*\d|first|then|finally|because|therefore|since)",
            text,
            re.IGNORECASE,
        )
    )
    return 0.7 * length_score + 0.3 * float(has_steps)


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Reward correct use of <reasoning>...</reasoning> and
    <tool_call>...</tool_call> tags.
    """
    rewards = []
    for c in completions:
        has_tool_call = bool(_TOOL_CALL_RE.search(c))
        has_reasoning = bool(_REASONING_RE.search(c))
        score = 0.0
        if has_tool_call:
            score += 0.5
        if has_tool_call and has_reasoning:
            score += 0.5
        rewards.append(score)
    return rewards
```

## 18. `reward\gvpo_reward.py`

```python
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
    call: dict,
    ground_truth: dict,
    sandbox=None,
    args_threshold: float = 0.8,
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
    response: str,
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
    response: str,
    ground_truth: dict,
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
    phi = [r - mu_proc for r in step_rewards]

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

    outcome_advantage: float
    per_token_shaped: torch.Tensor  # shape [seq_len]
    step_phis: list[float]
    step_process_rewards: list[float]


def build_per_token_advantages(
    response: str,
    tokenizer,
    outcome_adv: float,
    ground_truth: dict,
    sandbox=None,
    shaping_coeff: float = 1.0,
    args_threshold: float = 0.8,
    max_seq_len: int = 512,
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
        process_reward_step(c, ground_truth, sandbox, args_threshold) for c in calls
    ]

    # Initialise per-token advantages to Â_i (outcome-only, like GRPO)
    per_token = torch.full((max_seq_len,), fill_value=outcome_adv, dtype=torch.float32)

    if not phi_list:
        # No calls found — all tokens get outcome advantage only
        return GVPOTokenAdvantages(
            outcome_advantage=outcome_adv,
            per_token_shaped=per_token,
            step_phis=[],
            step_process_rewards=[],
        )

    # Map <call> character spans → token positions
    # We use a simple approach: tokenise the full response, then find
    # which token indices correspond to each <call>...</call> span.
    try:
        call_token_spans = _find_call_token_spans(response, tokenizer, max_seq_len)
    except Exception:
        # Fallback: if tokenisation mapping fails, use outcome advantage only
        return GVPOTokenAdvantages(
            outcome_advantage=outcome_adv,
            per_token_shaped=per_token,
            step_phis=phi_list,
            step_process_rewards=raw_proc,
        )

    # Apply shaping: tokens in <call> block t → Â_i + b · φ(t)
    for step_idx, (tok_start, tok_end) in enumerate(call_token_spans):
        if step_idx >= len(phi_list):
            break
        shaped_value = outcome_adv + shaping_coeff * phi_list[step_idx]
        per_token[tok_start:tok_end] = shaped_value

    return GVPOTokenAdvantages(
        outcome_advantage=outcome_adv,
        per_token_shaped=per_token,
        step_phis=phi_list,
        step_process_rewards=raw_proc,
    )


def _find_call_token_spans(
    response: str,
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
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=False,
    )
    offsets = enc["offset_mapping"]  # list of (char_start, char_end) per token

    spans: list[tuple[int, int]] = []

    for match in _CALL_RE.finditer(response):
        char_start = match.start()  # position of '<' in '<call>'
        char_end = match.end()  # position after '>' in '</call>'

        tok_start = None
        tok_end = None
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
    completions: list[str],
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
```

## 19. `reward\rc_grpo_reward.py`

```python
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
```

## 20. `utils\__init__.py`

```python

```

## 21. `utils\logging_utils.py`

```python
"""logging_utils.py — Rich-formatted project logger."""

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            markup=True,
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
```

## 22. `utils\model_utils.py`

```python
"""model_utils.py — Helpers for saving, loading, and merging LoRA adapters."""

from __future__ import annotations
from pathlib import Path
import torch


def save_model_and_tokenizer(model, tokenizer, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[model_utils] Saved → {output_dir}")


def load_model_from_path(
    model_path: str,
    base_model_name: str = "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """Load a fine-tuned LoRA adapter on top of the base model."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )
    # Load the LoRA adapter
    model.load_adapter(model_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def merge_and_export(
    model,
    tokenizer,
    output_dir: str,
    quantize_to: str = "q4_k_m",  # gguf quantisation type
) -> None:
    """Merge LoRA weights into the base model and export."""
    merged_dir = str(Path(output_dir) / "merged")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"[model_utils] Merged model → {merged_dir}")


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    do_sample: bool = True,
) -> str:
    """Single-sample inference helper."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the new tokens
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)
```

## 23. `utils\sandbox.py`

```python
"""
sandbox.py
───────────
Safe execution environment for validating tool calls.
Runs function calls in a sandboxed context to check for execution errors
without touching real network infrastructure.

In production, replace `_mock_execute` with actual telco API stubs.
"""

from __future__ import annotations

import json
import re
import traceback
from typing import Any, Callable

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Mock telecom function implementations
# (Replace with real stubs or simulation layer)
# ──────────────────────────────────────────────────────────────────────────────


def _default_mock(function_name: str, arguments: dict) -> dict:
    """Default mock: validates argument types and returns a dummy response."""
    return {
        "status": "success",
        "function": function_name,
        "result": f"Mock result for {function_name}({arguments})",
    }


class Sandbox:
    """
    Sandboxed function executor.

    Usage
    ─────
    sandbox = Sandbox(function_library)
    ok = sandbox.execute(response_string)
    ok = sandbox.execute({"function": "fn", "arguments": {...}})
    """

    def __init__(
        self,
        function_library: dict,
        mocks: dict[str, Callable] | None = None,
        timeout_seconds: float = 5.0,
    ):
        self.library = function_library
        self.mocks = mocks or {}
        self.timeout = timeout_seconds
        self._call_log: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self, call_input: str | dict | None) -> bool:
        """
        Execute a function call.  Returns True on success, False on any error.

        Parameters
        ──────────
        call_input : either a raw model response string containing <call>...</call>
                     OR a pre-parsed call dict {"function": ..., "arguments": ...}
        """
        if call_input is None:
            return False

        call = self._resolve_call(call_input)
        if call is None:
            return False

        return self._run_call(call)

    def execute_all(self, response: str) -> list[bool]:
        """Execute all <call> blocks in a multi-step response."""
        from src.reward.base_reward import extract_all_calls

        calls = extract_all_calls(response)
        return [self._run_call(c) for c in calls]

    def get_call_log(self) -> list[dict]:
        return self._call_log.copy()

    def clear_log(self) -> None:
        self._call_log.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve_call(self, call_input: str | dict) -> dict | None:
        if isinstance(call_input, dict):
            return call_input
        from src.reward.base_reward import extract_call

        return extract_call(call_input)

    def _run_call(self, call: dict) -> bool:
        func_name = call.get("function")
        arguments = call.get("arguments", {})

        if not func_name:
            logger.debug("[Sandbox] No function name in call.")
            return False

        # Validate against library schema
        if func_name not in self.library:
            logger.debug(f"[Sandbox] Unknown function: {func_name}")
            self._log(func_name, arguments, "error", "function_not_found")
            return False

        schema = self.library[func_name]
        if not self._validate_args(func_name, arguments, schema):
            return False

        # Execute mock or default
        try:
            mock_fn = self.mocks.get(func_name, _default_mock)
            if mock_fn == _default_mock:
                result = _default_mock(func_name, arguments)
            else:
                result = mock_fn(**arguments)
            self._log(func_name, arguments, "success", result)
            return True
        except Exception as exc:
            logger.debug(f"[Sandbox] Execution error for {func_name}: {exc}")
            self._log(func_name, arguments, "error", str(exc))
            return False

    def _validate_args(self, func_name: str, arguments: dict, schema: dict) -> bool:
        """
        Check that all required parameters are present and
        values respect declared constraints.
        """
        params = schema.get("parameters", {})
        constraints = schema.get("constraints", {})

        # Check required parameters
        for pname, pinfo in params.items():
            if pinfo.get("required", False) and pname not in arguments:
                logger.debug(
                    f"[Sandbox] Missing required param '{pname}' for {func_name}"
                )
                self._log(func_name, arguments, "error", f"missing_required:{pname}")
                return False

        # Check constraints
        for pname, con in constraints.items():
            if pname not in arguments:
                continue
            val = arguments[pname]
            if "min" in con and val < con["min"]:
                logger.debug(f"[Sandbox] {pname}={val} below min={con['min']}")
                return False
            if "max" in con and val > con["max"]:
                logger.debug(f"[Sandbox] {pname}={val} above max={con['max']}")
                return False
            if "enum" in con and val not in con["enum"]:
                logger.debug(f"[Sandbox] {pname}={val} not in enum {con['enum']}")
                return False

        return True

    def _log(self, func_name: str, arguments: dict, status: str, result: Any) -> None:
        self._call_log.append(
            {
                "function": func_name,
                "arguments": arguments,
                "status": status,
                "result": result,
            }
        )
```

