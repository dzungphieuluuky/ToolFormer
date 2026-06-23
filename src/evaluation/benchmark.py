import time
from pathlib import Path

import jsonlines
from tqdm import tqdm

from src.algorithms.base_trainer import build_messages_for_grpo, patch_tokenizer_for_custom_roles
from src.data.retrieval import FunctionRetriever, ArgumentValueRetriever
from src.utils.model_utils import load_model_for_inference, generate_response
from src.utils.sandbox import Sandbox
from src.evaluation.metrics import compute_all_metrics, aggregate_metrics, estimate_cost
from src.utils.logging_utils import get_logger
from src.reward.rc_grpo_reward import HIGH_REWARD_TOKEN

logger = get_logger(__name__)


def evaluate_model(
    model_path: str,
    test_dataset_path: str,
    function_library: dict,
    retriever: FunctionRetriever,
    sandbox: Sandbox,
    top_k: int = 5,
    max_new_tokens: int = 512,
    model_name_tag: str = "model",
    use_dataset_retrieval: bool = True,
    argument_values: dict | None = None,
    condition_on_high_reward: bool = True,
) -> dict:
    logger.info(f"[Benchmark] Loading model from {model_path}")
    model, tokenizer = load_model_for_inference(model_path)
    patch_tokenizer_for_custom_roles(tokenizer)

    val_retriever = None
    if argument_values is not None:
        val_retriever = ArgumentValueRetriever(argument_values)

    test_samples = []
    with jsonlines.open(test_dataset_path) as reader:
        for obj in reader:
            test_samples.append(obj)

    logger.info(f"[Benchmark] {model_name_tag}: evaluating {len(test_samples)} samples")
    results = []

    for sample in tqdm(test_samples, desc=f"Eval [{model_name_tag}]"):
        query = sample["query"]
        gt = sample.get("ground_truth", {})
        if use_dataset_retrieval and sample.get("retrieved_functions"):
            retrieved = sample["retrieved_functions"]
        else:
            retrieved = retriever.retrieve(query, k=top_k)

        if val_retriever is not None:
            arg_vals = val_retriever.retrieve_for_functions(query, retrieved, function_library)
        else:
            arg_vals = sample.get("retrieved_argument_values")

        messages = build_messages_for_grpo(query, retrieved, function_library, arg_vals)

        if condition_on_high_reward and HIGH_REWARD_TOKEN in tokenizer.all_special_tokens:
            from src.algorithms.base_trainer import inject_reward_token_into_prompt

            def _inject_into_messages(messages, token):
                out = []
                injected = False
                for msg in messages:
                    if msg.get("role") == "user" and not injected:
                        out.append({**msg, "content": f"{msg['content']}\n[Reward Goal: {token}]"})
                        injected = True
                    else:
                        out.append(msg)
                return out

            messages = _inject_into_messages(messages, HIGH_REWARD_TOKEN)

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        t0 = time.perf_counter()
        response = generate_response(model, tokenizer, prompt, max_new_tokens, temperature=0.0, do_sample=False)
        latency = (time.perf_counter() - t0) * 1000.0
        cost = estimate_cost(prompt, response)
        metrics = compute_all_metrics(response, gt, sandbox, latency, cost, function_library)
        metrics["sample_id"] = sample.get("id", "")
        results.append(metrics)

    agg = aggregate_metrics(results)
    logger.info(f"[Benchmark] {model_name_tag} aggregate: {agg}")
    return {"model": model_name_tag, "per_sample": results, "aggregate": agg}
