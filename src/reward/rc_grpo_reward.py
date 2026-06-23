from src.reward.base_reward import (
    _parse_gt,
    format_reward,
    extract_all_calls,
    compute_action_coverage_reward,
)

HIGH_REWARD_TOKEN = "<|high_reward|>"
LOW_REWARD_TOKEN = "<|low_reward|>"
REWARD_TOKENS = [HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN]


def rc_grpo_reward(
    response: str,
    ground_truth: dict,
    sandbox=None,
    args_threshold: float = 0.8,
) -> float:
    """
    R(tau) = R_state * R_action (Eq. 5), binary in {0, 1}.

    ground_truth.calls is always present (normalised format). Builds gold_calls
    from the full calls list and uses extract_all_calls +
    compute_action_coverage_reward to handle single and multi-call uniformly.
    """
    gold_calls = [
        {"function": c["function"], "arguments": c["arguments"]}
        for c in ground_truth.get("calls", [])
    ]

    if not gold_calls:
        return 0.0

    agent_calls = extract_all_calls(response)

    if not agent_calls:
        return 0.0

    r_action = compute_action_coverage_reward(agent_calls, gold_calls)

    if sandbox is not None:
        from src.utils.sandbox import Sandbox
        if not sandbox.execute(response):
            return 0.0

    return float(r_action)


def rc_grpo_reward_func(
    completions: list[str],
    ground_truth: list | None = None,
    **kwargs,
) -> list[float]:
    """
    R(tau) = R_state * R_action (Eq. 5), binary in {0, 1}.

    FIXED: ground_truth now arrives as a JSON string column (see
    format_sample_for_grpo) — must json.loads() it first via _parse_gt.
    FIXED: delegates to rc_grpo_reward which handles multi-call workflows.
    """
    if ground_truth is None:
        ground_truth = kwargs.get("ground_truths", [{}] * len(completions))
    return [
        rc_grpo_reward(c, _parse_gt(gt_raw))
        for c, gt_raw in zip(completions, ground_truth)
    ]


def rc_grpo_format_func(completions: list[str], **kwargs) -> list[float]:
    return format_reward(completions, **kwargs)
