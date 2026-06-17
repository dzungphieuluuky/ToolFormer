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
