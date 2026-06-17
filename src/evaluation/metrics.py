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
