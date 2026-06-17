"""
base_trainer.py
────────────────
Conversation format with structured tags:

  <system>...</system>
  <user>...</user>
  <retriever>...</retriever>   ← NEW: retrieved functions + argument values
  <reasoning>...</reasoning>   ← model output
  <tool_call>...</tool_call>   ← model output (renamed from <call>)

The <retriever> block is injected as a separate message role so the model
learns to treat it as external context (not as the user's words).
In the Qwen3 chat template this maps to:
  system  → <|im_start|>system
  user    → <|im_start|>user
  (the retriever block is appended to the user message with clear tags)
  assistant → <|im_start|>assistant  (model generates from here)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import datasets
import torch


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
<retriever> block
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
- Call EXACTLY one function from the <retriever> block
- Include ALL required parameters
- Use argument values from the retriever when they match the query
- Do NOT invent functions or parameters not in the retriever block
- If the query is unanswerable with available functions, output:
<reasoning>Explanation of why no function is appropriate.</reasoning>
<tool_call>null</tool_call>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Retriever block builder
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
    argument_values: "dict[str, list]",   # param_name → list[ValueMatch]
) -> str:
    """
    Render the retrieved argument values as a readable block.

    Output example:
        ## Relevant Argument Values

        ### location_code
          - HCM → Thành phố Hồ Chí Minh  [Tỉnh/Thành phố]
          - VNM → Việt Nam  [Toàn quốc]

        ### tech_type
          - 4G → 4G LTE  [technology]
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
                # ValueMatch dataclass
                label_str = m.label
                if m.alt_label:
                    label_str += f" / {m.alt_label}"
                lines.append(f"  - {m.code} → {label_str}  [{m.group}]")
            else:
                # Plain dict fallback
                code  = m.get("code", "")
                label = m.get("label", "")
                group = m.get("group", "")
                lines.append(f"  - {code} → {label}  [{group}]")

    return "\n".join(lines)


def build_retriever_block(
    function_names:   list[str],
    function_library: dict,
    argument_values:  "dict[str, list] | None" = None,
) -> str:
    """
    Build the complete <retriever> block content.

    Structure:
        ## Available Functions
        ### FUNC_NAME_1
        Description: ...
        Parameters: ...

        ### FUNC_NAME_2
        ...

        ## Relevant Argument Values   (only if argument_values non-empty)
        ### param_name
          - CODE → Label [group]
    """
    # ── Available functions ───────────────────────────────────────────────────
    lines = ["## Available Functions\n"]
    for fn in function_names:
        if fn in function_library:
            lines.append(build_function_description(fn, function_library[fn]))
            lines.append("")   # blank line between functions

    # ── Argument values ───────────────────────────────────────────────────────
    if argument_values:
        val_block = build_argument_values_block(argument_values)
        if val_block:
            lines.append(val_block)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Full message builders
# ──────────────────────────────────────────────────────────────────────────────

def build_messages_for_grpo(
    query:            str,
    function_names:   list[str],
    function_library: dict,
    argument_values:  "dict[str, list] | None" = None,
) -> list[dict]:
    """
    Build the [system, user] messages for GRPO rollout generation.

    The user message contains:
      <user> query </user>
      <retriever> functions + argument values </retriever>

    The model then generates from the assistant turn.
    Ground truth is NEVER included here.
    """
    retriever_content = build_retriever_block(
        function_names, function_library, argument_values
    )

    user_content = (
        f"<user>\n{query}\n</user>\n\n"
        f"<retriever>\n{retriever_content}\n</retriever>"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def build_messages_for_sft(
    query:            str,
    function_names:   list[str],
    function_library: dict,
    ground_truth:     dict,
    argument_values:  "dict[str, list] | None" = None,
) -> list[dict]:
    """
    Build the complete [system, user, assistant] messages for SFT.

    The assistant message contains the expected reasoning + tool_call.
    """
    messages = build_messages_for_grpo(
        query, function_names, function_library, argument_values
    )

    reasoning = ground_truth.get(
        "reasoning",
        "Analysing the query to determine the correct function and arguments."
    )
    func_name = ground_truth.get("function")
    arguments = ground_truth.get("arguments", {})

    if func_name is None:
        call_json = "null"
    else:
        call_json = json.dumps(
            {"function": func_name, "arguments": arguments},
            indent=2,
            ensure_ascii=False,
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
    tokenizer,
    argument_values:  "dict[str, list] | None" = None,
) -> dict:
    """
    Convert one raw dataset sample into GRPOTrainer format.

    If argument_values is None, the sample's pre-computed
    retrieved_argument_values field is used (if present).

    Dataset columns produced:
      prompt          → tokenizer-formatted [system, user] with generation prompt
      ground_truth    → dict (passed as kwarg to reward functions)
      query           → str  (passed as kwarg to reward functions)
      workflow_type   → str  (passed as kwarg to reward functions)
    """
    query     = sample["query"]
    retrieved = sample.get("retrieved_functions", [])
    gt        = sample.get("ground_truth", {})

    # Use pre-computed argument values if available in sample
    arg_vals = argument_values or sample.get("retrieved_argument_values")

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
    """
    Convert one raw sample into SFTTrainer format.
    Includes the full conversation (system + user + assistant).
    """
    query     = sample["query"]
    retrieved = sample.get("retrieved_functions", [])
    gt        = sample.get("ground_truth", {})

    arg_vals = argument_values or sample.get("retrieved_argument_values")

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
) -> "datasets.Dataset":
    """
    Load the final enriched .jsonl (train_dataset.jsonl) for GRPOTrainer.

    Argument value source priority:
      1. telco_retriever   — live retrieval (slowest, most up-to-date)
      2. sample field      — pre-computed retrieved_argument_values in the JSONL
      3. argument_values_catalog — re-run ArgumentValueRetriever on the fly
      4. None              — no argument values in the prompt

    The enrichment step in prepare_data.py fills (2), so by default
    the loader just reads the pre-computed values — no extra API or model calls.
    """
    from datasets import Dataset
    import jsonlines

    raw_samples = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            raw_samples.append(obj)

    print(f"[base_trainer] Loaded {len(raw_samples)} samples from {jsonl_path}")

    val_retriever = None
    if telco_retriever is None and argument_values_catalog is not None:
        from src.data.retrieval import ArgumentValueRetriever
        val_retriever = ArgumentValueRetriever(argument_values_catalog)

    formatted = []
    for sample in raw_samples:
        query     = sample["query"]
        retrieved = sample.get("retrieved_functions", [])

        # Argument value priority: live > pre-computed in sample > on-the-fly
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
            # Use whatever was stored by prepare_data.py — may be None
            raw_av   = sample.get("retrieved_argument_values")
            arg_vals = _deserialise_arg_values(raw_av) if raw_av else None

        formatted.append(
            format_sample_for_grpo(sample, function_library, tokenizer, arg_vals)
        )

    dataset = Dataset.from_list(formatted)
    print(f"[base_trainer] GRPO dataset ready: {len(dataset)} samples")
    print(f"[base_trainer] Columns: {dataset.column_names}")

    # ── Sanity-check first sample ─────────────────────────────────────────────
    if formatted:
        s  = formatted[0]
        gt = s["ground_truth"]
        print(f"\n{'='*70}")
        print("SAMPLE PROMPT (first 1200 chars):")
        print("=" * 70)
        print(s["prompt"][:1200])
        print("..." if len(s["prompt"]) > 1200 else "")
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
) -> "datasets.Dataset":
    """
    Load the final enriched .jsonl for SFTTrainer.
    Same argument value priority logic as load_grpo_dataset.
    """
    from datasets import Dataset
    import jsonlines

    raw_samples = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            raw_samples.append(obj)

    val_retriever = None
    if telco_retriever is None and argument_values_catalog is not None:
        from src.data.retrieval import ArgumentValueRetriever
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
            raw_av   = sample.get("retrieved_argument_values")
            arg_vals = _deserialise_arg_values(raw_av) if raw_av else None

        formatted.append(
            format_sample_for_sft(sample, function_library, tokenizer, arg_vals)
        )

    dataset = Dataset.from_list(formatted)
    print(f"[base_trainer] SFT dataset ready: {len(dataset)} samples")
    return dataset


def _deserialise_arg_values(raw: dict) -> dict:
    """
    Convert the plain-dict form stored in JSONL back to a dict that
    build_argument_values_block() can consume.

    The JSONL stores:
        {"location_code": [{"code": "HCM", "label": "...", "group": "...",
                            "score": 0.8, "alt_label": ""}]}

    build_argument_values_block() accepts either ValueMatch dataclasses
    or plain dicts with the same keys — so we return plain dicts directly.
    """
    # Plain dicts work fine with build_argument_values_block because it
    # does isinstance(m, dict) and falls through to m.get("code") etc.
    return raw  # already the right format

# ──────────────────────────────────────────────────────────────────────────────
# Model loading + GRPOConfig builder (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def load_model(config: dict | None = None):
    from unsloth import FastLanguageModel, PatchFastRL
    from trl import GRPOTrainer as _GT
    PatchFastRL("GRPO", FastLanguageModel)

    cfg       = config or {}
    model_cfg = cfg.get("model", {})
    lora_cfg  = cfg.get("lora",  {})

    model_name = model_cfg.get("name", "unsloth/qwen3-4b-unsloth-bnb-4bit")
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

    model = FastLanguageModel.get_peft_model(
        model,
        r                          = lora_rank,
        lora_alpha                 = lora_cfg.get("lora_alpha", lora_rank),
        lora_dropout               = lora_cfg.get("lora_dropout", 0.0),
        target_modules             = lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias                       = lora_cfg.get("bias", "none"),
        use_gradient_checkpointing = lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state               = 3407,
    )
    return model, tokenizer


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
        stop                       = ["</tool_call>"],    # updated stop token
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
        max_steps                   = train_cfg.get("max_steps",   500),
        save_steps                  = train_cfg.get("save_steps",  100),
        logging_steps               = train_cfg.get("logging_steps", 1),
        max_grad_norm               = train_cfg.get("max_grad_norm", 0.1),
        report_to                   = train_cfg.get("report_to",  "none"),
        seed                        = train_cfg.get("seed",        3407),
        bf16                        = torch.cuda.is_bf16_supported(),
    )