import re
import json
from typing import Any

# ─────────────────────────────────────────────────────────────────────
# _parse_gt: safely coerce a ground_truth column value (JSON string
# OR already-a-dict, for backward compatibility) into a dict.
#
# BUG FIXED: format_sample_for_grpo now serialises ground_truth to a
# JSON string to avoid pyarrow struct-schema inference on heterogeneous
# argument value types (ArrowInvalid). Every consumer that receives
# ground_truth from the HF Dataset must route it through this helper.
# ─────────────────────────────────────────────────────────────────────

def _parse_gt(gt) -> dict:
    """Coerce ground_truth (JSON string or dict) into a dict."""
    if isinstance(gt, dict):
        return gt
    if isinstance(gt, str):
        try:
            parsed = json.loads(gt)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)

_CALL_RE = _TOOL_CALL_RE
_REASON_RE = _REASONING_RE


def extract_call(response: str) -> dict | None:
    match = _TOOL_CALL_RE.search(response)
    if not match:
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
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        return None


def extract_all_calls(response: str) -> list[dict]:
    calls = []
    for m in _TOOL_CALL_RE.finditer(response):
        raw = m.group(1).strip()
        if raw.lower() == "null":
            continue
        try:
            parsed = json.loads(raw)
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


def compute_action_coverage_reward(
    agent_calls: list[dict],
    gold_calls: list[dict],
) -> int:
    def _call_matches(agent_call: dict, gold_call: dict) -> bool:
        if agent_call.get("function") != gold_call.get("function"):
            return False
        gold_args = gold_call.get("arguments", {})
        agent_args = agent_call.get("arguments", {})
        for param_key, param_val in gold_args.items():
            if str(agent_args.get(param_key, "")).strip() != str(param_val).strip():
                return False
        return True

    for gold_call in gold_calls:
        if not any(_call_matches(a, gold_call) for a in agent_calls):
            return 0
    return 1


def compute_state_consistency_reward(
    final_state: dict,
    gold_state: dict,
) -> int:
    return int(final_state == gold_state)


def compute_trajectory_reward(
    final_state: dict,
    gold_state: dict,
    agent_calls: list[dict],
    gold_calls: list[dict],
) -> int:
    r_state = compute_state_consistency_reward(final_state, gold_state)
    r_action = compute_action_coverage_reward(agent_calls, gold_calls)
    return r_state * r_action


def make_reward_function(
    ground_truth: dict,
    function_library: dict,
    sandbox_cls=None,
    weights: dict | None = None,
):
    w = weights or {
        "format": 0.10,
        "function": 0.30,
        "arguments": 0.40,
        "reasoning": 0.10,
        "execution": 0.10,
    }

    gold_calls = [
        {"function": c["function"], "arguments": c["arguments"]}
        for c in ground_truth.get("calls", [])
    ]

    def _reward(response_text: str) -> float:
        score = 0.0

        has_tool_call = bool(_TOOL_CALL_RE.search(response_text))
        has_reasoning = bool(_REASONING_RE.search(response_text))
        fmt = 0.0
        if has_tool_call:
            fmt += 0.5
        if has_tool_call and has_reasoning:
            fmt += 0.5
        score += w["format"] * fmt

        agent_calls = extract_all_calls(response_text)

        if not agent_calls:
            return score

        gold_func_names = [c["function"] for c in gold_calls]
        agent_func_names = [c.get("function") for c in agent_calls]
        func_hits = sum(1 for gf in gold_func_names if gf in agent_func_names)
        score += w["function"] * (func_hits / len(gold_func_names))

        arg_scores = []
        for gc in gold_calls:
            best = 0.0
            for ac in agent_calls:
                if ac.get("function") == gc["function"]:
                    best = max(best, args_accuracy(ac, gc.get("arguments", {})))
            arg_scores.append(best)
        score += w["arguments"] * (sum(arg_scores) / len(arg_scores))

        score += w["reasoning"] * reasoning_quality(response_text)

        if sandbox_cls is not None:
            try:
                sb = sandbox_cls(function_library)
                results = sb.execute_all(response_text)
                exec_score = sum(results) / max(len(results), 1)
                score += w["execution"] * exec_score
            except Exception:
                pass

        return score

    return _reward


def make_binary_reward_function(ground_truth: dict, function_library: dict):
    gold_calls = [
        {"function": c["function"], "arguments": c["arguments"]}
        for c in ground_truth.get("calls", [])
    ]

    def _reward(response_text: str) -> int:
        agent_calls = extract_all_calls(response_text)
        if not agent_calls:
            return 0
        return compute_action_coverage_reward(agent_calls, gold_calls)

    return _reward


def format_reward(completions: list[str], **kwargs) -> list[float]:
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


def function_reward(completions: list[str], ground_truth: list, **kwargs) -> list[float]:
    rewards = []
    for c, gt_raw in zip(completions, ground_truth):
        gt = _parse_gt(gt_raw)
        calls = gt.get("calls", [])
        expected = calls[0]["function"] if calls else ""
        rewards.append(func_selection_ok(c, expected))
    return rewards


def argument_reward(completions: list[str], ground_truth: list, **kwargs) -> list[float]:
    rewards = []
    for c, gt_raw in zip(completions, ground_truth):
        gt = _parse_gt(gt_raw)
        calls = gt.get("calls", [])
        expected_args = calls[0].get("arguments", {}) if calls else {}
        rewards.append(args_accuracy(c, expected_args))
    return rewards


def composite_reward(completions: list[str], ground_truth: list, **kwargs) -> list[float]:
    rewards = []
    for c, gt_raw in zip(completions, ground_truth):
        gt = _parse_gt(gt_raw)
        calls = gt.get("calls", [])
        first_call = calls[0] if calls else {}
        expected_func = first_call.get("function", "")
        expected_args = first_call.get("arguments", {})
        r = (
            0.10 * (1.0 if bool(_TOOL_CALL_RE.search(c)) and bool(_REASONING_RE.search(c)) else
                    0.5 if bool(_TOOL_CALL_RE.search(c)) else 0.0)
            + 0.30 * func_selection_ok(c, expected_func)
            + 0.40 * args_accuracy(c, expected_args)
            + 0.20 * reasoning_quality(c)
        )
        rewards.append(r)
    return rewards
