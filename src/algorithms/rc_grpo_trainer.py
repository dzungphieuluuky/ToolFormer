import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

from src.reward.base_reward import (
    compute_state_consistency_reward,
    compute_action_coverage_reward,
    compute_trajectory_reward,
    extract_call,
    extract_all_calls,
)
from src.reward.rc_grpo_reward import (
    HIGH_REWARD_TOKEN,
    LOW_REWARD_TOKEN,
    REWARD_TOKENS,
    rc_grpo_reward_func,
    rc_grpo_format_func,
)
from src.algorithms.base_trainer import (
    load_model,
    build_grpo_config,
    load_grpo_dataset,
    build_trl_reward_functions,
    inject_reward_token_into_prompt,
    format_sample_for_grpo,
    build_messages_for_grpo,
)


# ── 1. Reward token helpers (Eq. 1) ──────────────────────────────────────────

def binary_reward_to_token(reward: int) -> str:
    assert reward in (0, 1), "Reward must be binary (0 or 1)."
    return HIGH_REWARD_TOKEN if reward == 1 else LOW_REWARD_TOKEN


# ── 2. Group-normalized advantage (Eq. 6, ddof=0) ────────────────────────────

def compute_group_normalized_advantages(
    group_rewards: List[float],
    epsilon_stable: float = 1e-8,
) -> List[float]:
    rewards = np.array(group_rewards, dtype=np.float64)
    mu_g = rewards.mean()
    sigma_g = rewards.std(ddof=0)
    return ((rewards - mu_g) / (sigma_g + epsilon_stable)).tolist()


# ── 3. Reward-conditioned rollout sampling (Eq. 3) ───────────────────────────

def sample_reward_token_for_group(
    group_size: int,
    high_reward_probability: float = 0.5,
) -> List[str]:
    return [
        HIGH_REWARD_TOKEN if random.random() < high_reward_probability else LOW_REWARD_TOKEN
        for _ in range(group_size)
    ]


# ── 4. RCTP dataset types & builders (Section 3.2 / Appendix B) ──────────────

@dataclass
class Trajectory:
    prompt_messages: List[Dict]
    response_text: str
    reward: int
    reward_token: str = field(init=False)

    def __post_init__(self):
        self.reward_token = binary_reward_to_token(self.reward)


def _make_failure_response(gold_function: str, gold_arguments: dict, function_library: dict) -> str:
    import json as _json
    failure_kind = random.choice(["wrong_function", "missing_arg", "wrong_arg_value"])
    reasoning = "Analyzing the query and selecting the matching function and arguments."

    if failure_kind == "wrong_function" and len(function_library) > 1:
        candidates = [f for f in function_library if f != gold_function]
        wrong_func = random.choice(candidates) if candidates else gold_function
        call = {"function": wrong_func, "arguments": gold_arguments}
    elif failure_kind == "missing_arg" and gold_arguments:
        corrupted = dict(gold_arguments)
        drop_key = random.choice(list(corrupted.keys()))
        corrupted.pop(drop_key)
        call = {"function": gold_function, "arguments": corrupted}
    elif failure_kind == "wrong_arg_value" and gold_arguments:
        corrupted = dict(gold_arguments)
        change_key = random.choice(list(corrupted.keys()))
        corrupted[change_key] = f"__WRONG__{corrupted[change_key]}"
        call = {"function": gold_function, "arguments": corrupted}
    else:
        call = {"function": gold_function, "arguments": {}}

    call_json = _json.dumps(call, indent=2, ensure_ascii=False)
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<tool_call>\n{call_json}\n</tool_call>"


def build_rctp_dataset_from_jsonl(
    jsonl_path: str,
    function_library: dict,
    argument_values_catalog: dict | None = None,
    telco_retriever=None,
    failures_per_expert: int = 1,
    seed: int = 3407,
) -> List[Trajectory]:
    import json as _json
    import jsonlines as _jsonlines
    from src.algorithms.base_trainer import build_messages_for_grpo
    from src.data.retrieval import ArgumentValueRetriever

    random.seed(seed)

    raw_samples: list[dict] = []
    with _jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            raw_samples.append(obj)

    val_retriever = None
    if telco_retriever is None and argument_values_catalog is not None:
        val_retriever = ArgumentValueRetriever(argument_values_catalog)

    trajectories: List[Trajectory] = []

    for sample in raw_samples:
        query = sample["query"]
        retrieved = sample.get("retrieved_functions", [])
        gt = sample.get("ground_truth", {})
        if not isinstance(gt, dict):
            continue
        gold_function = gt.get("function")
        gold_arguments = gt.get("arguments", {})
        if gold_function is None:
            continue

        if telco_retriever is not None:
            result = telco_retriever.retrieve(query, function_library, precomputed_func_names=retrieved)
            arg_vals = result.argument_values
        elif val_retriever is not None:
            arg_vals = val_retriever.retrieve_for_functions(query, retrieved, function_library)
        else:
            arg_vals = sample.get("retrieved_argument_values")

        prompt_messages = build_messages_for_grpo(query, retrieved, function_library, arg_vals)

        reasoning = gt.get(
            "reasoning",
            "Analysing the query to determine the correct function and arguments.",
        )
        gold_call_json = _json.dumps(
            {"function": gold_function, "arguments": gold_arguments},
            indent=2, ensure_ascii=False,
        )
        expert_response = f"<reasoning>\n{reasoning}\n</reasoning>\n<tool_call>\n{gold_call_json}\n</tool_call>"
        trajectories.append(Trajectory(prompt_messages=prompt_messages, response_text=expert_response, reward=1))

        for _ in range(failures_per_expert):
            failure_response = _make_failure_response(gold_function, gold_arguments, function_library)
            trajectories.append(Trajectory(prompt_messages=prompt_messages, response_text=failure_response, reward=0))

    n_success = sum(1 for t in trajectories if t.reward == 1)
    n_failure = sum(1 for t in trajectories if t.reward == 0)
    print(f"[RCTP-FT data] {n_success} success / {n_failure} failure "
          f"(ratio {n_success}:{n_failure}, p_high={n_success / max(1, n_success + n_failure):.3f})")

    random.shuffle(trajectories)
    return trajectories


# ── 5. RCTP Dataset class ────────────────────────────────────────────────────

class RCTPDataset(Dataset):
    def __init__(
        self,
        trajectories: List[Trajectory],
        tokenizer,
        max_length: int = 2048,
    ):
        self.trajectories = trajectories
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trajectory = self.trajectories[idx]
        text = self._build_reward_conditioned_prompt(trajectory)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _build_reward_conditioned_prompt(self, trajectory: Trajectory) -> str:
        reward_prefix = f"[Reward Goal: {trajectory.reward_token}]"
        messages = []
        injected = False
        for msg in trajectory.prompt_messages:
            if msg["role"] == "user" and not injected:
                messages.append({"role": "user", "content": f"{msg['content']}\n{reward_prefix}"})
                injected = True
            else:
                messages.append(msg)
        messages.append({"role": "assistant", "content": trajectory.response_text})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


# ── 6. RCTP Fine-tuning config & runner ──────────────────────────────────────

@dataclass
class RCTPFinetuningConfig:
    learning_rate: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 5
    save_path: str = "checkpoints/rctp"


def run_rctp_finetuning(
    base_model_name: str,
    trajectories: List[Trajectory],
    config: RCTPFinetuningConfig = RCTPFinetuningConfig(),
    device: str = "cuda",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": REWARD_TOKENS})

    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16).to(device)

    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    dataset = RCTPDataset(trajectories, tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(dataloader),
        num_training_steps=len(dataloader) * config.num_epochs,
    )

    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[RCTP-FT] Epoch {epoch + 1}/{config.num_epochs}  Loss: {avg_loss:.4f}")

    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[RCTP-FT] Saved RCTP checkpoint -> {save_dir}")

    return model, tokenizer


# ── 7. Rollout helpers ───────────────────────────────────────────────────────

def _build_conditioned_prompt(
    tokenizer,
    prompt_messages: List[Dict],
    reward_token: str,
) -> str:
    conditioned_messages = []
    injected = False
    for msg in prompt_messages:
        if msg["role"] == "user" and not injected:
            conditioned_messages.append({
                "role": "user",
                "content": f"{msg['content']}\n[Reward Goal: {reward_token}]",
            })
            injected = True
        else:
            conditioned_messages.append(msg)

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            conditioned_messages, tokenize=False, add_generation_prompt=True,
        )
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conditioned_messages)


@dataclass
class RolloutResult:
    text: str
    full_ids: torch.Tensor
    prompt_length: int
    generated_ids: torch.Tensor
    old_log_probs: torch.Tensor
    attention_mask: torch.Tensor


def generate_rollout(
    model,
    tokenizer,
    prompt_messages: List[Dict],
    reward_token: str,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    device: str = "cuda",
) -> RolloutResult:
    prompt_text = _build_conditioned_prompt(tokenizer, prompt_messages, reward_token)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-7),
            pad_token_id=tokenizer.eos_token_id,
        )

    full_ids = output_ids
    generated_ids = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    with torch.no_grad():
        logits = model(full_ids).logits
        gen_logits = logits[0, prompt_len - 1: -1, :]
        log_probs_all = F.log_softmax(gen_logits, dim=-1)
        old_log_probs = log_probs_all.gather(dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)

    attention_mask = torch.ones_like(full_ids)

    return RolloutResult(
        text=generated_text,
        full_ids=full_ids.detach(),
        prompt_length=prompt_len,
        generated_ids=generated_ids.detach(),
        old_log_probs=old_log_probs.detach(),
        attention_mask=attention_mask.detach(),
    )


def compute_current_log_probs(
    model,
    full_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    prompt_length: int,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=full_ids, attention_mask=attention_mask).logits
    gen_logits = logits[0, prompt_length - 1: -1, :]
    log_probs_all = F.log_softmax(gen_logits, dim=-1)
    current_log_probs = log_probs_all.gather(dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)
    return current_log_probs


def compute_reference_log_probs(
    ref_model,
    full_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    prompt_length: int,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        logits = ref_model(input_ids=full_ids, attention_mask=attention_mask).logits
        gen_logits = logits[0, prompt_length - 1: -1, :]
        log_probs_all = F.log_softmax(gen_logits, dim=-1)
        ref_log_probs = log_probs_all.gather(dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)
    return ref_log_probs


# ── 8. Token-level importance ratio (Eq. 7) ──────────────────────────────────

def compute_token_importance_ratios(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(current_log_probs - old_log_probs)


# ── 9. Clipped surrogate (Eq. 9, token-level) ────────────────────────────────

def compute_clipped_surrogate_loss(
    token_ratios: torch.Tensor,
    advantage: float,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    adv = torch.tensor(advantage, dtype=token_ratios.dtype, device=token_ratios.device)
    unclipped = token_ratios * adv
    clipped = torch.clamp(token_ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    return torch.min(unclipped, clipped).mean()


# ── 10. KL penalty (Schulman k3 estimator) ──────────────────────────────────

def compute_kl_penalty(
    current_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_coefficient: float = 0.1,
) -> torch.Tensor:
    log_ratio = ref_log_probs - current_log_probs
    kl_per_token = torch.exp(log_ratio) - 1.0 - log_ratio
    kl_per_token = torch.clamp(kl_per_token, min=0.0)
    return kl_coefficient * kl_per_token.mean()


# ── 11. RC-GRPO full loss (Eq. 8) ───────────────────────────────────────────

def compute_rc_grpo_loss(
    surrogate_terms: List[torch.Tensor],
    kl_terms: List[torch.Tensor],
) -> torch.Tensor:
    G = len(surrogate_terms)
    surrogate_mean = sum(surrogate_terms) / G
    kl_mean = sum(kl_terms) / G
    loss = -(surrogate_mean - kl_mean)
    return loss


# ── 12. Config ───────────────────────────────────────────────────────────────

@dataclass
class RCGRPOConfig:
    learning_rate: float = 1e-6
    kl_coefficient: float = 0.1
    group_size: int = 5
    batch_size: int = 256
    clip_epsilon: float = 0.2
    num_epochs: int = 400
    high_reward_probability: float = 0.5
    epsilon_stable: float = 1e-8
    max_new_tokens: int = 512
    temperature: float = 1.0


# ── 13. Single update step (Algorithm 1 lines 13-19) ─────────────────────────

def rc_grpo_update_step_for_one_prompt(
    policy_model,
    reference_model,
    tokenizer,
    prompt_messages: List[Dict],
    reward_function: Callable,
    optimizer,
    config: RCGRPOConfig,
    device: str = "cuda",
) -> Dict[str, float]:
    G = config.group_size

    sampled_reward_tokens = sample_reward_token_for_group(G, config.high_reward_probability)

    policy_model.eval()
    rollouts: List[RolloutResult] = []
    for r_j in sampled_reward_tokens:
        rollout = generate_rollout(
            model=policy_model, tokenizer=tokenizer, prompt_messages=prompt_messages,
            reward_token=r_j, max_new_tokens=config.max_new_tokens,
            temperature=config.temperature, device=device,
        )
        rollouts.append(rollout)

    group_rewards = [reward_function(r.text) for r in rollouts]
    advantages_list = compute_group_normalized_advantages(group_rewards, config.epsilon_stable)

    policy_model.train()
    surrogate_terms = []
    kl_terms = []

    for j, (rollout, adv_j) in enumerate(zip(rollouts, advantages_list)):
        current_lp = compute_current_log_probs(
            model=policy_model, full_ids=rollout.full_ids,
            generated_ids=rollout.generated_ids, prompt_length=rollout.prompt_length,
            attention_mask=rollout.attention_mask,
        )

        ref_lp = compute_reference_log_probs(
            ref_model=reference_model, full_ids=rollout.full_ids,
            generated_ids=rollout.generated_ids, prompt_length=rollout.prompt_length,
            attention_mask=rollout.attention_mask,
        )

        token_ratios = compute_token_importance_ratios(
            current_log_probs=current_lp, old_log_probs=rollout.old_log_probs,
        )

        kl_j = compute_kl_penalty(
            current_log_probs=current_lp, ref_log_probs=ref_lp,
            kl_coefficient=config.kl_coefficient,
        )

        surrogate_j = compute_clipped_surrogate_loss(
            token_ratios=token_ratios, advantage=adv_j, clip_epsilon=config.clip_epsilon,
        )

        surrogate_terms.append(surrogate_j)
        kl_terms.append(kl_j)

    loss = compute_rc_grpo_loss(surrogate_terms, kl_terms)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()

    adv_tensor = torch.tensor(advantages_list)
    return {
        "loss": loss.item(),
        "mean_reward": float(np.mean(group_rewards)),
        "advantage_spread": float(adv_tensor.max() - adv_tensor.min()),
        "reward_tokens": sampled_reward_tokens,
        "group_rewards": group_rewards,
        "mean_kl": float(sum(k.item() for k in kl_terms) / G),
    }


# ── 14. Reference model loading ──────────────────────────────────────────────

def load_reference_model(
    rctp_checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(rctp_checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        rctp_checkpoint_path, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[RC-GRPO] Loaded frozen pi_ref from {rctp_checkpoint_path}")
    return model, tokenizer


# ── 15. Full RC-GRPO training loop (Algorithm 1) ─────────────────────────────

def run_rc_grpo_training(
    rctp_model,
    tokenizer,
    training_prompts: List[List[Dict]],
    reward_function: Callable,
    rctp_checkpoint_path: str,
    config: RCGRPOConfig = RCGRPOConfig(),
    device: str = "cuda",
) -> AutoModelForCausalLM:
    reference_model, _ = load_reference_model(rctp_checkpoint_path, device)

    policy_model = rctp_model.to(device)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate)

    print(f"[RC-GRPO] Starting training: {config.num_epochs} epochs, "
          f"G={config.group_size}, p(high)={config.high_reward_probability}")

    for epoch in range(config.num_epochs):
        epoch_stats = {"loss": [], "mean_reward": [], "advantage_spread": [], "mean_kl": []}
        random.shuffle(training_prompts)

        for prompt_messages in training_prompts:
            step_stats = rc_grpo_update_step_for_one_prompt(
                policy_model=policy_model, reference_model=reference_model,
                tokenizer=tokenizer, prompt_messages=prompt_messages,
                reward_function=reward_function, optimizer=optimizer,
                config=config, device=device,
            )
            for key in epoch_stats:
                epoch_stats[key].append(step_stats[key])

        print(
            f"[RC-GRPO] Epoch {epoch + 1:>4}/{config.num_epochs}"
            f"  Loss: {np.mean(epoch_stats['loss']):.4f}"
            f"  Reward: {np.mean(epoch_stats['mean_reward']):.3f}"
            f"  Adv.Spread: {np.mean(epoch_stats['advantage_spread']):.3f}"
            f"  KL: {np.mean(epoch_stats['mean_kl']):.6f}"
        )

    return policy_model


# ── 16. Inference — condition on <|high_reward|> (Section 4.1) ───────────────

def run_inference_with_high_reward_conditioning(
    trained_policy_model,
    tokenizer,
    prompt_messages: List[Dict],
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> str:
    prompt_text = _build_conditioned_prompt(
        tokenizer, prompt_messages, HIGH_REWARD_TOKEN,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    trained_policy_model.eval()
    with torch.no_grad():
        output_ids = trained_policy_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


# ── 17. Variance lower bound (Proposition 4.3) ───────────────────────────────

def compute_variance_lower_bound(
    group_size: int,
    high_reward_probability: float,
    conditional_mean_gap: float,
) -> float:
    p = high_reward_probability
    kappa = ((group_size - 1) / group_size) * p * (1 - p)
    return kappa * (conditional_mean_gap ** 2)


# ── 18. TRL integration ──────────────────────────────────────────────────────

from trl import GRPOTrainer as _GRPOTrainer


def sample_reward_tokens_for_group(
    group_size: int,
    high_reward_probability: float = 0.5,
) -> List[str]:
    return [
        HIGH_REWARD_TOKEN if random.random() < high_reward_probability else LOW_REWARD_TOKEN
        for _ in range(group_size)
    ]


def inject_reward_token_into_messages(
    prompt_messages: List[Dict],
    reward_token: str,
) -> List[Dict]:
    out = []
    injected = False
    for msg in prompt_messages:
        if msg.get("role") == "user" and not injected:
            out.append({**msg, "content": f"{msg['content']}\n[Reward Goal: {reward_token}]"})
            injected = True
        else:
            out.append(msg)
    return out


class RCGRPOTrainer(_GRPOTrainer):
    def __init__(self, *args, high_reward_probability: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.high_reward_probability = high_reward_probability
        self._last_reward_tokens: List[str] = []

    @staticmethod
    def register_reward_tokens(tokenizer) -> int:
        existing = set(tokenizer.all_special_tokens)
        to_add = [t for t in REWARD_TOKENS if t not in existing]
        if not to_add:
            return 0
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        return num_added

    def _generate_and_score_completions(self, inputs):
        G = self.args.num_generations

        conditioned_inputs = []
        for group_start in range(0, len(inputs), G):
            group = inputs[group_start: group_start + G]
            reward_tokens = sample_reward_tokens_for_group(len(group), self.high_reward_probability)
            self._last_reward_tokens.extend(reward_tokens)
            for example, r_j in zip(group, reward_tokens):
                conditioned_example = dict(example)
                prompt_messages = conditioned_example["prompt"]
                conditioned_example["prompt"] = inject_reward_token_into_messages(prompt_messages, r_j)
                conditioned_example["_reward_token"] = r_j
                conditioned_inputs.append(conditioned_example)

        return super()._generate_and_score_completions(conditioned_inputs)


def rc_grpo_reward_func_trl(completions: list[str], ground_truth: list[dict], **kwargs) -> list[float]:
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        if not isinstance(gt, dict) or gt.get("function") is None:
            rewards.append(0.0)
            continue
        gold_calls = [{"function": gt.get("function"), "arguments": gt.get("arguments", {})}]
        call = extract_call(completion)
        agent_calls = [call] if call else []
        r_action = compute_action_coverage_reward(agent_calls, gold_calls)
        r_state = r_action
        rewards.append(float(r_state * r_action))
    return rewards


def rc_grpo_format_func_trl(completions: list[str], **kwargs) -> list[float]:
    from src.reward.base_reward import format_reward
    return format_reward(completions, **kwargs)
