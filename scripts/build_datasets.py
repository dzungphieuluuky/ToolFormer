#!/usr/bin/env python3
"""Build four pre-built dataset files from gold data + failures.

Outputs (all to data/generated/v2.0/):
  - sft_dataset.jsonl     (gold only)
  - rctp_dataset.jsonl    (gold + failures)
  - grpo_dataset.jsonl    (gold only)
  - rcgrpo_dataset.jsonl  (gold only, same format as GRPO)

Usage:
    python scripts/build_datasets.py \
        --input-base data/generated/v2.0/train_dataset_cleaned.jsonl \
        --input-failures data/generated/v2.0/failures_dataset.jsonl \
        --output-dir data/generated/v2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Reward token constants (from cell 21) ─────────────────────────────

HIGH_REWARD_TOKEN = "<|high_reward|>"
LOW_REWARD_TOKEN = "<|low_reward|>"
REWARD_TOKENS = [HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN]


def binary_reward_to_token(reward: int) -> str:
    """Convert binary reward to its corresponding token string.

    Args:
        reward: Binary reward value (0 or 1).

    Returns:
        High-reward token for 1, low-reward token for 0.
    """
    assert reward in (0, 1), "Reward must be binary (0 or 1)."
    return HIGH_REWARD_TOKEN if reward == 1 else LOW_REWARD_TOKEN


# ── Chat template (replicates Jinja WITHOUT tokenizer, from cell 23) ──


def apply_chat_template(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """Format a list of message dicts into a chat template string.

    Args:
        messages: List of dicts with 'role' and 'content' keys.
        add_generation_prompt: If True, appends the assistant start token.

    Returns:
        Concatenated chat-formatted string with role markers.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("system", "user", "assistant", "tool", "retriever"):
            parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# ── SYSTEM PROMPT (from cell 23) ──────────────────────────────────────

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

REQUIRED OUTPUT FORMAT — use these exact tags:
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


# ── Ground truth normalization (from cell 20/23) ──────────────────────


def normalize_ground_truth(gt: Any) -> dict:
    """Normalise ground truth to a canonical dict with 'calls' and 'reasoning'.

    Args:
        gt: Raw ground truth value, expected to be a dict.

    Returns:
        Normalised dict with 'calls' (list) and 'reasoning' (str) keys.
    """
    if not isinstance(gt, dict):
        return {}
    calls = gt.get("calls", [])
    if not isinstance(calls, list):
        calls = []
    normalized_calls = []
    for c in calls:
        if isinstance(c, dict):
            func = c.get("function")
            args = c.get("arguments", {})
            if func:
                normalized_calls.append({"function": func, "arguments": args})
    return {
        "calls": normalized_calls,
        "reasoning": gt.get("reasoning", ""),
    }


# ── Prompt building helpers (from cell 23) ────────────────────────────


def build_function_description(func_name: str, schema: dict) -> str:
    """Build a human-readable description of a function from its schema.

    Args:
        func_name: Name of the function.
        schema: Function schema dict with 'description', 'parameters', 'constraints'.

    Returns:
        Formatted string with name, description, parameters, and constraints.
    """
    lines = [f"### {func_name}"]
    lines.append(
        f"Description: {schema.get('description', 'No description available')}"
    )
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
            k: v
            for k, v in constraints.items()
            if not (isinstance(v, dict) and list(v.keys()) == ["enum"])
        }
        if other_con:
            lines.append(f"Constraints: {json.dumps(other_con, ensure_ascii=False)}")
    return "\n".join(lines)


def build_argument_values_block(argument_values: dict[str, list]) -> str:
    """Build a formatted block of relevant argument values for the prompt.

    Args:
        argument_values: Dict mapping parameter names to lists of value dicts.

    Returns:
        Formatted string, or empty string if no argument values provided.
    """
    if not argument_values:
        return ""
    lines = ["## Relevant Argument Values"]
    for param_name, matches in argument_values.items():
        if not matches:
            continue
        lines.append(f"\n### {param_name}")
        for m in matches:
            if isinstance(m, dict):
                code = m.get("code", "")
                label = m.get("label", "")
                group = m.get("group", "")
                alt_label = m.get("alt_label", "")
                label_str = label
                if alt_label:
                    label_str += f" / {alt_label}"
                lines.append(f"  - {code} \u2192 {label_str}  [{group}]")
            else:
                lines.append(f"  - {m}")
    return "\n".join(lines)


def build_retriever_block(
    function_names: list[str],
    function_library: dict,
    argument_values: dict[str, list] | None = None,
    include_all_threshold: int = 10,
) -> str:
    """Build the retriever block listing available functions and argument values.

    Args:
        function_names: Names of functions to include.
        function_library: Dict of all function schemas.
        argument_values: Optional dict of relevant argument values.
        include_all_threshold: Max values to include before truncation.

    Returns:
        Formatted retriever block string.
    """
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
                if isinstance(m, dict):
                    code = m.get("code", "")
                    label = m.get("label", "")
                    group = m.get("group", "")
                    alt_label = m.get("alt_label", "")
                    label_str = label
                    if alt_label:
                        label_str += f" / {alt_label}"
                    val_lines.append(f"  - {code} \u2192 {label_str}  [{group}]")
                else:
                    val_lines.append(f"  - {m}")
        if len(val_lines) > 1:
            lines.append("\n".join(val_lines))

    return "\n".join(lines)


def build_messages_for_grpo(
    query: str,
    function_names: list[str],
    function_library: dict,
    argument_values: dict[str, list] | None = None,
    include_all_threshold: int = 10,
) -> list[dict[str, str]]:
    """Build the message list for GRPO/RC-GRPO training (system + user + retriever).

    Args:
        query: The user's natural-language query.
        function_names: Retrieved function names to include.
        function_library: Dict of function schemas.
        argument_values: Optional argument values for the retriever block.
        include_all_threshold: Threshold for including all values.

    Returns:
        List of message dicts with roles and content.
    """
    retriever_content = build_retriever_block(
        function_names,
        function_library,
        argument_values,
        include_all_threshold,
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
) -> list[dict[str, str]]:
    """Build the full message list for SFT training, including the assistant response.

    Args:
        query: The user's natural-language query.
        function_names: Retrieved function names to include.
        function_library: Dict of function schemas.
        ground_truth: Dict with 'calls' and 'reasoning' keys.
        argument_values: Optional argument values for the retriever block.

    Returns:
        List of message dicts including the gold assistant response.
    """
    messages = build_messages_for_grpo(
        query, function_names, function_library, argument_values
    )
    reasoning = ground_truth.get(
        "reasoning",
        "Analysing the query to determine the correct function and arguments.",
    )
    gt = normalize_ground_truth(ground_truth)
    calls = gt.get("calls", [])

    if not calls:
        calls_str = "<tool_call>\nnull\n</tool_call>"
    else:
        call_blocks = []
        for call in calls:
            call_json = json.dumps(
                {"function": call["function"], "arguments": call["arguments"]},
                indent=2,
                ensure_ascii=False,
            )
            call_blocks.append(f"<tool_call>\n{call_json}\n</tool_call>")
        calls_str = "\n".join(call_blocks)

    assistant_content = f"<reasoning>\n{reasoning}\n</reasoning>\n{calls_str}"
    messages.append({"role": "assistant", "content": assistant_content})
    return messages


# ── Build gold assistant response text ────────────────────────────────


def build_gold_response(ground_truth: dict) -> str:
    """Build the gold assistant response text from ground truth.

    Args:
        ground_truth: Dict with 'calls' and 'reasoning' keys.

    Returns:
        Formatted response string with reasoning and tool call blocks.
    """
    gt = normalize_ground_truth(ground_truth)
    calls = gt.get("calls", [])
    if not calls:
        return "<reasoning>\nAnalysing the query to determine the correct function and arguments.\n</reasoning>\n<tool_call>\nnull\n</tool_call>"
    call_blocks = []
    for c in calls:
        c_json = json.dumps(
            {"function": c["function"], "arguments": c.get("arguments", {})},
            indent=2,
            ensure_ascii=False,
        )
        call_blocks.append(f"<tool_call>\n{c_json}\n</tool_call>")
    calls_str = "\n".join(call_blocks)
    reasoning = gt.get(
        "reasoning",
        "Analysing the query to determine the correct function and arguments.",
    )
    return f"<reasoning>\n{reasoning}\n</reasoning>\n{calls_str}"


# ── Trajectory dataclass (from cell 21) ───────────────────────────────


@dataclass
class Trajectory:
    """A training trajectory with prompt, response, and reward signal.

    Attributes:
        prompt_messages: List of message dicts forming the prompt.
        response_text: The model response text.
        reward: Binary reward (0 or 1).
        reward_token: Auto-computed reward token string.
    """
    prompt_messages: list[dict[str, str]]
    response_text: str
    reward: int
    _source_id: str | None = None
    reward_token: str = field(init=False)

    def __post_init__(self):
        """Set reward_token based on the binary reward value."""
        self.reward_token = binary_reward_to_token(self.reward)


# ── Build failure response text ───────────────────────────────────────


def build_failure_response(failure_gt: dict) -> str:
    """Build a failure response text from failure ground truth.

    Args:
        failure_gt: Dict with 'calls' and 'reasoning' keys (corrupted).

    Returns:
        Formatted response string with reasoning and tool call blocks.
    """
    gt = normalize_ground_truth(failure_gt)
    calls = gt.get("calls", [])
    if not calls:
        return "<reasoning>\nAnalyzing the query and determining no matching function is available.\n</reasoning>\n<tool_call>\nnull\n</tool_call>"
    call_blocks = []
    for c in calls:
        c_json = json.dumps(
            {"function": c["function"], "arguments": c.get("arguments", {})},
            indent=2,
            ensure_ascii=False,
        )
        call_blocks.append(f"<tool_call>\n{c_json}\n</tool_call>")
    calls_str = "\n".join(call_blocks)
    reasoning = gt.get(
        "reasoning",
        "Analyzing the query to determine the correct function and arguments.",
    )
    return f"<reasoning>\n{reasoning}\n</reasoning>\n{calls_str}"


# ── Function extraction helper ────────────────────────────────────────


def _extract_functions_from_text(text: str) -> list[str]:
    """Extract unique function names from <tool_call> blocks in text.

    Parses each <tool_call>...</tool_call> block as JSON and extracts the
    ``function`` key. Handles ``null`` blocks (abstention/failure responses)
    gracefully by skipping them.

    Args:
        text: The response or chat text containing tool call blocks.

    Returns:
        Deduplicated list of function name strings found in tool calls.
    """
    functions: list[str] = []
    pattern = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)
    for match in pattern.finditer(text):
        content = match.group(1).strip()
        if not content or content == "null":
            continue
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                func = parsed.get("function")
                if func and isinstance(func, str):
                    functions.append(func)
        except json.JSONDecodeError:
            continue
    # Deduplicate while preserving order
    seen: set[str] = set()
    return [f for f in functions if not (f in seen or seen.add(f))]


# ── I/O helpers ───────────────────────────────────────────────────────


def load_jsonl(path: str) -> list[dict]:
    """Load a JSON Lines file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON dicts.
    """
    samples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_json(path: str) -> dict:
    """Load a JSON file into a dict.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, records: list[dict]) -> None:
    """Write a list of dicts to a JSON Lines file.

    Args:
        path: Output path for the JSONL file.
        records: List of dicts to write.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Written %d records to %s", len(records), path)


# ── Dataset builders ──────────────────────────────────────────────────


def build_sft_dataset(
    samples: list[dict],
    function_library: dict,
    argument_values: dict,
) -> list[dict]:
    """Build the SFT dataset from gold samples.

    Each sample produces a complete chat-formatted text string including
    the gold assistant response.

    Args:
        samples: List of preprocessed dataset samples.
        function_library: Dict mapping function names to schemas.
        argument_values: Dict of argument value catalogs.

    Returns:
        List of dicts with 'text' key for SFT training.
    """
    records = []
    for sample in samples:
        query = sample["query"]
        retrieved = sample.get("retrieved_functions", [])
        gt = sample.get("ground_truth", {})
        arg_vals = sample.get("retrieved_argument_values") or argument_values
        messages = build_messages_for_sft(
            query, retrieved, function_library, gt, arg_vals
        )
        text = apply_chat_template(messages, add_generation_prompt=False)
        records.append({"text": text, "_source_id": sample.get("id")})
    return records


def build_grpo_dataset(
    samples: list[dict],
    function_library: dict,
    argument_values: dict,
) -> list[dict]:
    """Build the GRPO/RC-GRPO dataset from gold samples.

    Each sample produces a prompt, serialized ground truth, query, and
    workflow type for GRPO training.

    Args:
        samples: List of preprocessed dataset samples.
        function_library: Dict mapping function names to schemas.
        argument_values: Dict of argument value catalogs.

    Returns:
        List of dicts with 'prompt', 'ground_truth', 'query', 'workflow_type'.
    """
    records = []
    for sample in samples:
        query = sample["query"]
        retrieved = sample.get("retrieved_functions", [])
        gt = sample.get("ground_truth", {})
        if not isinstance(gt, dict):
            gt = {}
        gt = normalize_ground_truth(gt)
        arg_vals = sample.get("retrieved_argument_values") or argument_values
        messages = build_messages_for_grpo(query, retrieved, function_library, arg_vals)
        prompt = apply_chat_template(messages, add_generation_prompt=True)
        records.append(
            {
                "prompt": prompt,
                "ground_truth": json.dumps(gt, ensure_ascii=False),
                "query": query,
                "workflow_type": sample.get("workflow_type", "single_call"),
                "_source_id": sample.get("id"),
            }
        )
    return records


def build_rctp_dataset(
    samples: list[dict],
    failures: list[dict],
    function_library: dict,
    argument_values: dict,
    failures_per_expert: int = 1,
) -> list[Trajectory]:
    """Build the RCTP dataset from gold samples and failure trajectories.

    Each gold sample produces one expert trajectory (reward=1) and up to
    `failures_per_expert` failure trajectories (reward=0) matched by ID.

    Args:
        samples: List of gold dataset samples.
        failures: List of failure records with matching IDs.
        function_library: Dict mapping function names to schemas.
        argument_values: Dict of argument value catalogs.
        failures_per_expert: Number of failure trajectories per expert (default: 1).

    Returns:
        List of Trajectory objects with balanced expert/failure samples.
    """
    failure_by_id: dict[str, list[dict]] = {}
    for fs in failures:
        fs_id = fs.get("id")
        if fs_id:
            failure_by_id.setdefault(fs_id, []).append(fs)

    trajectories: list[Trajectory] = []

    for sample in samples:
        query = sample["query"]
        retrieved = sample.get("retrieved_functions", [])
        gt = normalize_ground_truth(sample.get("ground_truth", {}))
        gold_calls = gt.get("calls", [])

        arg_vals = sample.get("retrieved_argument_values") or argument_values
        prompt_messages = build_messages_for_grpo(
            query, retrieved, function_library, arg_vals
        )

        # Expert (success) trajectory
        expert_response = build_gold_response(gt)
        trajectories.append(
            Trajectory(
                prompt_messages=prompt_messages,
                response_text=expert_response,
                reward=1,
                _source_id=sample.get("id"),
            )
        )

        # If abstention — only expert trajectory
        if not gold_calls:
            continue

        # Failure trajectories
        sample_id = sample.get("id")
        if sample_id and sample_id in failure_by_id:
            matching = failure_by_id[sample_id]
            for i in range(failures_per_expert):
                if i < len(matching):
                    fail_gt = matching[i].get("ground_truth", {})
                    failure_response = build_failure_response(fail_gt)
                    trajectories.append(
                        Trajectory(
                            prompt_messages=prompt_messages,
                            response_text=failure_response,
                            reward=0,
                            _source_id=matching[i].get("id"),
                        )
                    )

    return trajectories


# ── Argument value check helper ───────────────────────────────────────


def _check_retrieved_argument_values(
    sample: dict,
    argument_values: dict,
    label: str,
    idx: int,
) -> list[str]:
    """Check that retrieved argument value codes exist in the catalog.

    Args:
        sample: A dataset sample dict with an optional ``retrieved_argument_values`` key.
        argument_values: Catalog dict mapping parameter names to lists of value dicts.
        label: Dataset label for error messages (e.g. ``"sft"``).
        idx: Sample index for error messages.

    Returns:
        List of error messages (empty if all codes are valid).
    """
    errs: list[str] = []
    rav = sample.get("retrieved_argument_values")
    if not rav or not isinstance(rav, dict):
        return errs
    for param_name, entries in rav.items():
        if param_name not in argument_values:
            continue  # Skip parameters not in catalog (e.g., dynamic date values)
        if not isinstance(entries, list):
            continue
        catalog_entries = argument_values[param_name]
        codes_in_catalog = {
            e.get("code") for e in catalog_entries if isinstance(e, dict)
        }
        for j, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            code = entry.get("code")
            if not code:
                continue
            if code not in codes_in_catalog:
                errs.append(
                    f"{label}[{idx}]: retrieved_argument_values.{param_name}[{j}].code={code!r} not in argument_values catalog"
                )
    return errs


# ── Validation ────────────────────────────────────────────────────────


def validate_sft(
    records: list[dict],
    expected_count: int,
    function_library: dict | None = None,
) -> list[str]:
    """Validate SFT dataset records.

    Args:
        records: List of SFT record dicts.
        expected_count: Expected number of records.
        function_library: Optional dict to validate function names against.

    Returns:
        List of error messages (empty if valid).
    """
    errs = []
    if len(records) != expected_count:
        errs.append(f"sft: expected {expected_count} records, got {len(records)}")
    for i, rec in enumerate(records):
        if "text" not in rec:
            errs.append(f"sft[{i}]: missing 'text' field")
        elif not isinstance(rec["text"], str) or not rec["text"]:
            errs.append(f"sft[{i}]: 'text' is empty or not a string")
        elif rec.get("text"):
            text = rec["text"]
            if "<|im_start|>system\n" not in text:
                errs.append(f"sft[{i}]: text missing <|im_start|>system marker")
            if "<|im_start|>user\n" not in text:
                errs.append(f"sft[{i}]: text missing <|im_start|>user marker")
            if "<|im_start|>assistant\n" not in text:
                errs.append(f"sft[{i}]: text missing <|im_start|>assistant marker")
            if not text.strip().endswith("<|im_end|>"):
                errs.append(f"sft[{i}]: text does not end with <|im_end|>")
    if function_library:
        for i, rec in enumerate(records):
            funcs = _extract_functions_from_text(rec.get("text", ""))
            for func in funcs:
                if func not in function_library:
                    errs.append(
                        f"sft[{i}]: function {func!r} in text not found in function_library"
                    )
    return errs


def validate_grpo(
    records: list[dict],
    expected_count: int,
    function_library: dict | None = None,
) -> list[str]:
    """Validate GRPO/RC-GRPO dataset records.

    Args:
        records: List of GRPO record dicts.
        expected_count: Expected number of records.
        function_library: Optional dict to validate function names against.

    Returns:
        List of error messages (empty if valid).
    """
    errs = []
    if len(records) != expected_count:
        errs.append(f"grpo: expected {expected_count} records, got {len(records)}")
    required = {"prompt", "ground_truth", "query", "workflow_type"}
    for i, rec in enumerate(records):
        missing = required - set(rec.keys())
        if missing:
            errs.append(f"grpo[{i}]: missing fields {missing}")
        prompt = rec.get("prompt", "")
        if not prompt.endswith("<|im_start|>assistant\n"):
            errs.append(f"grpo[{i}]: prompt does not end with generation prompt")
        gt = rec.get("ground_truth", "")
        if gt:
            try:
                parsed = json.loads(gt)
                if not isinstance(parsed, dict):
                    errs.append(f"grpo[{i}]: ground_truth is not a dict after parse")
            except json.JSONDecodeError:
                errs.append(f"grpo[{i}]: ground_truth is not valid JSON")
    if function_library:
        for i, rec in enumerate(records):
            gt = rec.get("ground_truth", "")
            if gt:
                try:
                    parsed = json.loads(gt)
                    calls = parsed.get("calls", [])
                    for func in (c.get("function") for c in calls if isinstance(c, dict)):
                        if func and func not in function_library:
                            errs.append(
                                f"grpo[{i}]: function {func!r} in ground_truth not found in function_library"
                            )
                except (json.JSONDecodeError, TypeError):
                    pass
    return errs


def validate_rctp(
    records: list[Trajectory],
    min_count: int,
    function_library: dict | None = None,
) -> list[str]:
    """Validate RCTP dataset trajectories.

    Args:
        records: List of Trajectory objects.
        min_count: Minimum expected number of records.
        function_library: Optional dict to validate function names against.

    Returns:
        List of error messages (empty if valid).
    """
    errs = []
    if len(records) < min_count:
        errs.append(f"rctp: expected at least {min_count} records, got {len(records)}")
    rewards = set(t.reward for t in records)
    if not rewards:
        errs.append("rctp: no records")
    elif rewards != {0, 1}:
        wt = (
            f"only reward={list(rewards)[0]}"
            if len(rewards) == 1
            else f"rewards={rewards}"
        )
        logger.warning(
            "rctp: records contain %s (no failure samples matched to gold by id)", wt
        )
    for i, t in enumerate(records):
        if not t.prompt_messages:
            errs.append(f"rctp[{i}]: empty prompt_messages")
        if not t.response_text:
            errs.append(f"rctp[{i}]: empty response_text")
        if t.reward not in (0, 1):
            errs.append(f"rctp[{i}]: reward must be 0 or 1, got {t.reward}")
        if not t.response_text.startswith("<reasoning>"):
            errs.append(f"rctp[{i}]: response_text does not start with <reasoning>")
        if "</reasoning>" not in t.response_text:
            errs.append(f"rctp[{i}]: response_text missing closing </reasoning>")
        if "<tool_call>" not in t.response_text or "</tool_call>" not in t.response_text:
            errs.append(
                f"rctp[{i}]: response_text missing <tool_call> block"
            )
    if function_library:
        for i, t in enumerate(records):
            funcs = _extract_functions_from_text(t.response_text)
            for func in funcs:
                if func not in function_library:
                    errs.append(
                        f"rctp[{i}]: function {func!r} in response_text not found in function_library"
                    )
    return errs


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: parse args, load data, build all datasets, validate, and save.

    Builds SFT, GRPO, RC-GRPO, and RCTP dataset files from gold data and
    failure trajectories. Validates each output and logs a summary.
    """
    parser = argparse.ArgumentParser(
        description="Build SFT, RCTP, GRPO, and RC-GRPO dataset files from gold data + failures."
    )
    parser.add_argument(
        "--input-base",
        default="data/generated/v2.0/train_dataset_cleaned.jsonl",
        help="Path to gold training dataset (JSONL)",
    )
    parser.add_argument(
        "--input-failures",
        default="data/generated/v2.0/failures_dataset.jsonl",
        help="Path to failures dataset (JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/generated/v2.0",
        help="Output directory for built dataset files",
    )
    parser.add_argument(
        "--function-library",
        default=None,
        help="Function library JSON (default: derived from --input-base dir)",
    )
    parser.add_argument(
        "--argument-values",
        default=None,
        help="Argument values catalog JSON (default: derived from --input-base dir)",
    )
    parser.add_argument(
        "--failures-per-expert",
        type=int,
        default=1,
        help="Number of failure trajectories per gold (non-abstention) sample for RCTP",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    input_dir = Path(args.input_base).parent.resolve()
    function_library_path = args.function_library or str(
        input_dir / "function_library.json"
    )
    argument_values_path = args.argument_values or str(
        input_dir / "argument_values.json"
    )

    logger.info("Loading gold samples from %s", args.input_base)
    gold_samples = load_jsonl(args.input_base)
    logger.info("Loaded %d gold samples", len(gold_samples))

    logger.info("Loading failures from %s", args.input_failures)
    failure_samples = load_jsonl(args.input_failures)
    logger.info("Loaded %d failure samples", len(failure_samples))

    logger.info("Loading function library from %s", function_library_path)
    function_library = load_json(function_library_path)
    logger.info("Loaded %d functions", len(function_library))

    logger.info("Loading argument values from %s", argument_values_path)
    argument_values = load_json(argument_values_path)
    logger.info("Loaded argument catalog with %d param keys", len(argument_values))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errs: list[str] = []

    # ── 0. Validate gold input argument values ──────────────────────
    logger.info("Validating argument value codes against catalog...")
    for i, sample in enumerate(gold_samples):
        errs.extend(
            _check_retrieved_argument_values(sample, argument_values, "gold", i)
        )
    if errs:
        logger.error("Found %d argument value validation errors", len(errs))

    # ── 1. SFT dataset ─────────────────────────────────────────────
    logger.info("Building SFT dataset...")
    sft_records = build_sft_dataset(gold_samples, function_library, argument_values)
    sft_errs = validate_sft(sft_records, len(gold_samples), function_library)
    if sft_errs:
        errs.extend(f"sft: {e}" for e in sft_errs)
    else:
        write_jsonl(str(output_dir / "sft_dataset.jsonl"), sft_records)
        logger.info("SFT dataset: %d records", len(sft_records))

    # ── 2. GRPO dataset ────────────────────────────────────────────
    logger.info("Building GRPO dataset...")
    grpo_records = build_grpo_dataset(gold_samples, function_library, argument_values)
    grpo_errs = validate_grpo(grpo_records, len(gold_samples), function_library)
    if grpo_errs:
        errs.extend(f"grpo: {e}" for e in grpo_errs)
    else:
        write_jsonl(str(output_dir / "grpo_dataset.jsonl"), grpo_records)
        logger.info("GRPO dataset: %d records", len(grpo_records))

    # ── 3. RC-GRPO dataset (same format as GRPO) ───────────────────
    logger.info("Building RC-GRPO dataset...")
    rcgrpo_records = build_grpo_dataset(gold_samples, function_library, argument_values)
    rcgrpo_errs = validate_grpo(rcgrpo_records, len(gold_samples), function_library)
    if rcgrpo_errs:
        errs.extend(f"rcgrpo: {e}" for e in rcgrpo_errs)
    else:
        write_jsonl(str(output_dir / "rcgrpo_dataset.jsonl"), rcgrpo_records)
        logger.info("RC-GRPO dataset: %d records", len(rcgrpo_records))

    # ── 4. RCTP dataset ───────────────────────────────────────────
    logger.info("Building RCTP dataset...")
    rctp_trajectories = build_rctp_dataset(
        gold_samples,
        failure_samples,
        function_library,
        argument_values,
        failures_per_expert=args.failures_per_expert,
    )
    rctp_errs = validate_rctp(rctp_trajectories, len(gold_samples), function_library)
    if rctp_errs:
        errs.extend(f"rctp: {e}" for e in rctp_errs)
    else:
        rctp_records = [
            {
                "prompt_messages": t.prompt_messages,
                "response_text": t.response_text,
                "reward": t.reward,
                "_source_id": t._source_id,
            }
            for t in rctp_trajectories
        ]
        write_jsonl(str(output_dir / "rctp_dataset.jsonl"), rctp_records)
        n_success = sum(1 for t in rctp_trajectories if t.reward == 1)
        n_failure = len(rctp_trajectories) - n_success
        logger.info(
            "RCTP dataset: %d records (reward=1: %d, reward=0: %d)",
            len(rctp_trajectories),
            n_success,
            n_failure,
        )

    if errs:
        for e in errs:
            logger.error("VALIDATION ERROR: %s", e)
        sys.exit(1)
    else:
        logger.info("All datasets built and validated successfully.")


if __name__ == "__main__":
    main()
