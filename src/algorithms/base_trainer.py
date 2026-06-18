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