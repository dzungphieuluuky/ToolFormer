import json
import warnings
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

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

try:
    from src.reward.rc_grpo_reward import REWARD_TOKENS
except ImportError:
    REWARD_TOKENS = []

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
    tokenizer.chat_template = CUSTOM_CHAT_TEMPLATE
    existing_special = tokenizer.additional_special_tokens or []
    tokens_to_add = [t for t in REWARD_TOKENS if t not in existing_special]
    if tokens_to_add:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing_special + tokens_to_add}
        )
        print(f"[base_trainer] Added reward tokens: {tokens_to_add}")
    print("[base_trainer] Custom chat template registered (supports 'retriever' role).")


SYSTEM_PROMPT = """\
You are a telecom network operations assistant. \
You will receive a user query, a list of available functions with their \
parameter schemas, and relevant argument values retrieved from a knowledge base.

Your task:
1. Read the user query carefully
2. Review the available functions and retrieved argument values in the \
<|im_start|>retriever<|im_end|> block
3. Reason step by step about which function(s) to call and what argument \
values to use
4. Output your reasoning and the function call(s)

REQUIRED OUTPUT FORMAT -- use these exact tags:
<reasoning>
Step-by-step analysis:
- What the user is asking for
- Which function(s) best match and why
- Which argument values from the retriever match the query
- Final argument assignments
</reasoning>
<tool_call>
{"function": "<function_name>", "arguments": {"<param1>": "<value1>", ...}}
</tool_call>

If the query requires MULTIPLE sequential calls, output multiple <tool_call> blocks:
<tool_call>
{"function": "<function_name_1>", "arguments": {...}}
</tool_call>
<tool_call>
{"function": "<function_name_2>", "arguments": {...}}
</tool_call>

STRICT RULES:
- Call ONLY functions from the retriever list
- Include ALL required parameters
- Use argument values from the retriever when they match the query
- Do NOT invent functions or parameters not in the retriever block
- If the query is unanswerable with available functions, output:
<reasoning>Explanation of why no function is appropriate.</reasoning>
<tool_call>null</tool_call>
"""


def build_function_description(func_name: str, schema: dict) -> str:
    lines = [f"### {func_name}"]
    lines.append(f"Description: {schema.get('description', 'No description available')}")
    params = schema.get("parameters", {})
    constraints = schema.get("constraints", {})
    if params:
        lines.append("Parameters:")
        for pname, pinfo in params.items():
            if isinstance(pinfo, dict):
                ptype = pinfo.get("type", "any")
                required = "(required)" if pinfo.get("required") else "(optional)"
                desc = pinfo.get("description", "")
                line = f"  - {pname} [{ptype}] {required}: {desc}"
                param_con = constraints.get(pname, {})
                if "enum" in param_con:
                    line += f"  Allowed values: {param_con['enum']}"
                lines.append(line)
            else:
                lines.append(f"  - {pname}: {pinfo}")
    if constraints:
        other_con = {
            k: v for k, v in constraints.items()
            if not (isinstance(v, dict) and list(v.keys()) == ["enum"])
        }
        if other_con:
            lines.append(f"Constraints: {json.dumps(other_con, ensure_ascii=False)}")
    return "\n".join(lines)


def build_argument_values_block(argument_values: dict[str, list]) -> str:
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
                lines.append(f"  - {m.code} -> {label_str}  [{m.group}]")
            else:
                code = m.get("code", "")
                label = m.get("label", "")
                group = m.get("group", "")
                lines.append(f"  - {code} -> {label}  [{group}]")
    return "\n".join(lines)


def build_retriever_block(
    function_names: list[str],
    function_library: dict,
    argument_values: dict[str, list] | None = None,
    include_all_threshold: int = 10,
) -> str:
    lines = ["## Available Functions\n"]
    for fn in function_names:
        if fn in function_library:
            lines.append(build_function_description(fn, function_library[fn]))
            lines.append("")

    if argument_values:
        val_lines = ["## Relevant Argument Values"]
        for param_name, matches in argument_values.items():
            if not matches:
                continue
            val_lines.append(f"\n### {param_name}")
            for m in matches:
                if hasattr(m, "code"):
                    label_str = m.label
                    if m.alt_label:
                        label_str += f" / {m.alt_label}"
                    val_lines.append(f"  - {m.code} -> {label_str}  [{m.group}]")
                else:
                    code = m.get("code", "")
                    label = m.get("label", "")
                    group = m.get("group", "")
                    val_lines.append(f"  - {code} -> {label}  [{group}]")
        if len(val_lines) > 1:
            lines.append("\n".join(val_lines))

    return "\n".join(lines)


def build_messages_for_grpo(
    query: str,
    function_names: list[str],
    function_library: dict,
    argument_values: dict[str, list] | None = None,
    include_all_threshold: int = 10,
) -> list[dict]:
    retriever_content = build_retriever_block(
        function_names, function_library, argument_values, include_all_threshold,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "retriever", "content": retriever_content},
    ]


def build_messages_for_sft(
    query: str,
    function_names: list[str],
    function_library: dict,
    ground_truth: dict,
    argument_values: dict[str, list] | None = None,
) -> list[dict]:
    messages = build_messages_for_grpo(query, function_names, function_library, argument_values)
    reasoning = ground_truth.get(
        "reasoning",
        "Analysing the query to determine the correct function and arguments.",
    )

    calls = ground_truth.get("calls", [])

    if not calls:
        calls_str = "<tool_call>\nnull\n</tool_call>"
    else:
        call_blocks = []
        for call in calls:
            call_json = json.dumps(
                {"function": call["function"], "arguments": call.get("arguments", {})},
                indent=2, ensure_ascii=False,
            )
            call_blocks.append(f"<tool_call>\n{call_json}\n</tool_call>")
        calls_str = "\n".join(call_blocks)

    assistant_content = f"<reasoning>\n{reasoning}\n</reasoning>\n{calls_str}"
    messages.append({"role": "assistant", "content": assistant_content})
    return messages


def format_sample_for_grpo(
    sample: dict,
    function_library: dict,
    tokenizer,
    argument_values: dict[str, list] | None = None,
    include_all_threshold: int = 10,
) -> dict:
    query = sample["query"]
    retrieved = sample.get("retrieved_functions", [])
    gt = sample.get("ground_truth", {})
    if not isinstance(gt, dict):
        gt = {}
    arg_vals = argument_values or sample.get("retrieved_argument_values")
    messages = build_messages_for_grpo(query, retrieved, function_library, arg_vals, include_all_threshold)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # FIXED: serialise ground_truth to JSON string to avoid pyarrow struct-
    # schema inference on heterogeneous argument value types (ArrowInvalid).
    # Reward functions must json.loads() this back out (they do, via _parse_gt).
    return {
        "prompt": prompt,
        "ground_truth": json.dumps(gt, ensure_ascii=False),
        "query": query,
        "workflow_type": sample.get("workflow_type", "single_call"),
    }


def format_sample_for_sft(
    sample: dict,
    function_library: dict,
    tokenizer,
    argument_values: dict[str, list] | None = None,
) -> dict:
    query = sample["query"]
    retrieved = sample.get("retrieved_functions", [])
    gt = sample.get("ground_truth", {})
    arg_vals = argument_values or sample.get("retrieved_argument_values")
    messages = build_messages_for_sft(query, retrieved, function_library, gt, arg_vals)
    return {
        "text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    }


def load_model(config: dict | None = None):
    from unsloth import FastLanguageModel, PatchFastRL

    PatchFastRL("GRPO", FastLanguageModel)
    cfg = config or {}
    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    model_name = model_cfg.get("name", "unsloth/Qwen3-4B-Base")
    max_seq = model_cfg.get("max_seq_length", 2048)
    lora_rank = lora_cfg.get("r", 16)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        load_in_4bit=model_cfg.get("load_in_4bit", False),
        fast_inference=model_cfg.get("fast_inference", True),
        max_lora_rank=lora_rank,
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.5),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_alpha=lora_rank * 2,
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=3407,
    )

    patch_tokenizer_for_custom_roles(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    print(
        f"[base_trainer] Tokenizer vocab size: {len(tokenizer)} "
        f"(includes reward tokens: {REWARD_TOKENS})"
    )

    return model, tokenizer


def build_grpo_config(config: dict, output_dir: str | None = None) -> "GRPOConfig":
    from trl import GRPOConfig
    from vllm import SamplingParams

    train_cfg = config.get("training", {})
    grpo_cfg = config.get("grpo", {})
    data_cfg = config.get("data", {})

    vllm_params = SamplingParams(
        temperature=grpo_cfg.get("temperature", 1.0),
        top_p=0.95,
        min_p=0.05,
        seed=train_cfg.get("seed", 3407),
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )

    return GRPOConfig(
        output_dir=output_dir or train_cfg.get("output_dir", "outputs/model"),
        learning_rate=train_cfg.get("learning_rate", 1e-6),
        adam_beta1=train_cfg.get("adam_beta1", 0.9),
        adam_beta2=train_cfg.get("adam_beta2", 0.99),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        optim=train_cfg.get("optim", "adamw_8bit"),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        num_generations=train_cfg.get("num_generations", 5),
        max_prompt_length=data_cfg.get("max_prompt_length", 1024),
        max_completion_length=data_cfg.get("max_completion_length", 512),
        temperature=grpo_cfg.get("temperature", 1.0),
        epsilon=grpo_cfg.get("epsilon", 0.2),
        beta=grpo_cfg.get("kl_coefficient", 0.1),
        loss_type=grpo_cfg.get("loss_type", "grpo"),
        mask_truncated_completions=grpo_cfg.get("mask_truncated_completions", True),
        vllm_sampling_params=vllm_params,
        max_steps=train_cfg.get("max_steps", 500),
        save_steps=train_cfg.get("save_steps", 100),
        logging_steps=train_cfg.get("logging_steps", 1),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        report_to=train_cfg.get("report_to", "none"),
        seed=train_cfg.get("seed", 3407),
        bf16=torch.cuda.is_bf16_supported(),
    )


def load_grpo_dataset(
    jsonl_path: str,
    function_library: dict,
    tokenizer,
    argument_values_catalog: dict | None = None,
    telco_retriever=None,
    include_all_threshold: int = 10,
) -> "Dataset":
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

    formatted = []
    for sample in raw_samples:
        query = sample["query"]
        retrieved = sample.get("retrieved_functions", [])
        gt = sample.get("ground_truth", {})
        if not isinstance(gt, dict):
            gt = {}

        if telco_retriever is not None:
            result = telco_retriever.retrieve(query, function_library, precomputed_func_names=retrieved)
            arg_vals = result.argument_values
        elif val_retriever is not None:
            arg_vals = val_retriever.retrieve_for_functions(query, retrieved, function_library)
        else:
            arg_vals = sample.get("retrieved_argument_values")

        formatted.append(
            format_sample_for_grpo(sample, function_library, tokenizer, arg_vals, include_all_threshold)
        )

    dataset = Dataset.from_list(formatted)
    print(f"[base_trainer] GRPO dataset ready: {len(dataset)} samples")
    print(f"[base_trainer] Columns: {dataset.column_names}")

    # FIXED: ground_truth is now a JSON string (see format_sample_for_grpo).
    # Parse it back for display in the sanity check.
    if len(dataset) > 0:
        s = dataset[0]
        gt_raw = s.get("ground_truth", "{}")
        try:
            gt = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        except json.JSONDecodeError:
            gt = {}
        wt = s.get("workflow_type", "single_call")
        print(f"\n{'=' * 70}")
        print("SAMPLE PROMPT (first 1500 chars):")
        print("=" * 70)
        print(s["prompt"][:1500])
        print("..." if len(s["prompt"]) > 1500 else "")
        calls = gt.get("calls", [])
        print(f"\nGROUND TRUTH (reward funcs see this, MODEL DOES NOT):")
        print(f"  workflow:  {wt}")
        print(f"  calls:     {len(calls)} steps")
        if calls:
            print(f"  first:     {json.dumps(calls[0], ensure_ascii=False)[:300]}")
        print("=" * 70 + "\n")

    return dataset


def load_sft_dataset(
    jsonl_path: str,
    function_library: dict,
    tokenizer,
    argument_values_catalog: dict | None = None,
    telco_retriever=None,
) -> "Dataset":
    if Dataset is None or jsonlines is None:
        raise ImportError("datasets and/or jsonlines not installed.")

    raw_samples = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            raw_samples.append(obj)

    val_retriever = None
    if telco_retriever is None and argument_values_catalog is not None:
        if ArgumentValueRetriever is not None:
            val_retriever = ArgumentValueRetriever(argument_values_catalog)

    formatted = []
    for sample in raw_samples:
        query = sample["query"]
        retrieved = sample.get("retrieved_functions", [])

        if telco_retriever is not None:
            result = telco_retriever.retrieve(query, function_library, precomputed_func_names=retrieved)
            arg_vals = result.argument_values
        elif val_retriever is not None:
            arg_vals = val_retriever.retrieve_for_functions(query, retrieved, function_library)
        else:
            arg_vals = sample.get("retrieved_argument_values")

        formatted.append(format_sample_for_sft(sample, function_library, tokenizer, arg_vals))

    dataset = Dataset.from_list(formatted)
    print(f"[base_trainer] SFT dataset ready: {len(dataset)} samples")
    return dataset


def build_trl_reward_functions(algorithm: str = "rc_grpo"):
    if algorithm == "rc_grpo":
        from src.reward.rc_grpo_reward import rc_grpo_reward_func, rc_grpo_format_func
        return [rc_grpo_reward_func, rc_grpo_format_func]
    from src.reward.base_reward import function_reward, argument_reward, format_reward
    return [function_reward, argument_reward, format_reward]


def _deserialise_arg_values(raw: dict | None) -> dict | None:
    if not raw:
        return None
    return raw


def inject_reward_token_into_prompt(
    prompt: str,
    reward_token: str,
) -> str:
    marker = "<|im_start|>assistant\n"
    idx = prompt.rfind(marker)
    if idx == -1:
        return prompt + f"\n[Reward Goal: {reward_token}]\n"
    inject_str = f"[Reward Goal: {reward_token}]\n"
    return prompt[:idx] + inject_str + prompt[idx:]
