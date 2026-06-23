import numpy as np

from src.reward.base_reward import (
    extract_call,
    schema_valid,
    func_selection_ok,
    args_accuracy,
)


def compute_all_metrics(response: str, ground_truth: dict, sandbox, latency_ms: float,
                        cost_estimate: float, function_library: dict) -> dict[str, float]:
    call = extract_call(response)
    expected_func = ground_truth.get("function", "none")
    expected_args = ground_truth.get("arguments", {})
    workflow = ground_truth.get("workflow", "single_call")
    func_sel_acc = func_selection_ok(response, expected_func)
    arg_acc = args_accuracy(response, expected_args)
    schema_val = schema_valid(response)
    exec_success = 0.0
    if sandbox is not None:
        exec_success = 1.0 if sandbox.execute(response) else 0.0
    else:
        exec_success = float(call is not None)
    task_success = float(func_sel_acc == 1.0 and arg_acc >= 0.8 and exec_success == 1.0)
    hallucinated = 0.0
    if call is not None:
        called_fn = call.get("function", "")
        if called_fn and called_fn not in function_library:
            hallucinated = 1.0
    abstention_acc = float("nan")
    if workflow == "abstention":
        produced_call = extract_call(response)
        abstention_acc = 1.0 if (produced_call is None or produced_call == "null") else 0.0
    latency = latency_ms
    cost = cost_estimate
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


def estimate_cost(prompt: str, response: str, price_per_1k_tokens: float = 0.0002) -> float:
    total_chars = len(prompt) + len(response)
    tokens_est = total_chars / 1.3
    return (tokens_est / 1000) * price_per_1k_tokens
