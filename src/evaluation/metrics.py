import numpy as np

from src.reward.base_reward import (
    extract_call,
    extract_all_calls,
    schema_valid,
    func_selection_ok,
    args_accuracy,
    compute_action_coverage_reward,
)


def compute_all_metrics(response: str, ground_truth: dict, sandbox, latency_ms: float,
                        cost_estimate: float, function_library: dict) -> dict[str, float]:
    workflow = ground_truth.get("workflow", "single_call")
    gold_calls = [
        {"function": c["function"], "arguments": c["arguments"]}
        for c in ground_truth.get("calls", [])
    ]
    agent_calls = extract_all_calls(response)
    schema_val = 1.0 if agent_calls else 0.0

    if gold_calls:
        # Per-call metrics: best match for each gold call
        func_hits = 0
        arg_scores = []
        for gc in gold_calls:
            best_func = 0.0
            best_arg = 0.0
            for ac in agent_calls:
                if ac.get("function") == gc["function"]:
                    best_func = 1.0
                    best_arg = max(best_arg, args_accuracy(ac, gc.get("arguments", {})))
            func_hits += best_func
            arg_scores.append(best_arg)
        func_sel_acc = func_hits / len(gold_calls)
        arg_acc = sum(arg_scores) / len(arg_scores)

        task_success = float(compute_action_coverage_reward(agent_calls, gold_calls))

        exec_success = 0.0
        if sandbox is not None:
            exec_success = 1.0 if sandbox.execute(response) else 0.0
        else:
            exec_success = float(len(agent_calls) > 0)
    else:
        # Abstention — expect no tool call
        func_sel_acc = 0.0
        arg_acc = 0.0
        task_success = 0.0
        exec_success = 0.0

    hallucinated = 0.0
    for ac in agent_calls:
        called_fn = ac.get("function", "")
        if called_fn and called_fn not in function_library:
            hallucinated = 1.0
            break

    abstention_acc = float("nan")
    if workflow == "abstention":
        produced_call = extract_call(response)
        abstention_acc = 1.0 if (produced_call is None or produced_call == "null") else 0.0

    return {
        "function_selection_accuracy": func_sel_acc,
        "argument_accuracy": arg_acc,
        "schema_validity": schema_val,
        "execution_success_rate": exec_success,
        "task_success_rate": task_success,
        "hallucinated_call_rate": hallucinated,
        "abstention_accuracy": abstention_acc,
        "latency_ms": latency_ms,
        "cost_per_query_usd": cost_estimate,
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
