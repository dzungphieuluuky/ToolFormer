from src.reward.base_reward import schema_valid, func_selection_ok, args_accuracy, format_reward

HIGH_REWARD_TOKEN = "<|high_reward|>"
LOW_REWARD_TOKEN = "<|low_reward|>"
REWARD_TOKENS = [HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN]


def rc_grpo_reward(
    response: str,
    ground_truth: dict,
    sandbox=None,
    args_threshold: float = 0.8,
) -> float:
    if not schema_valid(response):
        return 0.0
    if not func_selection_ok(response, ground_truth.get("function", "")):
        return 0.0
    if args_accuracy(response, ground_truth.get("arguments", {})) < args_threshold:
        return 0.0
    if sandbox is not None:
        from src.utils.sandbox import Sandbox
        if not sandbox.execute(response):
            return 0.0
    return 1.0


def rc_grpo_reward_func(
    completions: list[str],
    ground_truth: list[dict] | None = None,
    **kwargs,
) -> list[float]:
    if ground_truth is None:
        ground_truth = kwargs.get("ground_truths", [{}] * len(completions))
    return [
        rc_grpo_reward(c, gt if isinstance(gt, dict) else {})
        for c, gt in zip(completions, ground_truth)
    ]


def rc_grpo_format_func(completions: list[str], **kwargs) -> list[float]:
    return format_reward(completions, **kwargs)
