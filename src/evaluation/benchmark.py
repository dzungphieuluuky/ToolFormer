"""
benchmark.py
─────────────
Run full evaluation of a fine-tuned model on the test set.

Prompt construction uses EXACTLY the same functions as the training pipeline
(build_messages_for_grpo → apply_chat_template with the custom template)
so train/eval prompt format is guaranteed to be identical.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jsonlines
from tqdm import tqdm

from src.algorithms.base_trainer import (
    build_messages_for_grpo,
    patch_tokenizer_for_custom_roles,
    SYSTEM_PROMPT,
)
from src.data.retrieval import FunctionRetriever, ArgumentValueRetriever
from src.utils.model_utils import load_model_from_path, generate_response
from src.utils.sandbox import Sandbox
from .metrics import compute_all_metrics, aggregate_metrics, estimate_cost
from src.utils.logging_utils import get_logger

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
    argument_values: dict | None = None,  # full catalog for val retriever
) -> dict:
    """
    Evaluate a fine-tuned model on the held-out test set.

    Parameters
    ──────────
    use_dataset_retrieval : True  → use pre-computed retrieved_functions from JSONL
                            False → run live FunctionRetriever per query
    argument_values       : full argument_values.json dict; if provided, runs
                            ArgumentValueRetriever to add arg values to prompt
    """
    # ── Load model + patch tokenizer ──────────────────────────────────────────
    logger.info(f"[Benchmark] Loading model from {model_path}")
    model, tokenizer = load_model_from_path(model_path)
    patch_tokenizer_for_custom_roles(tokenizer)  # ← required for "retriever" role

    # ── Optional argument value retriever ─────────────────────────────────────
    val_retriever = None
    if argument_values is not None:
        val_retriever = ArgumentValueRetriever(argument_values)

    # ── Load test samples ─────────────────────────────────────────────────────
    test_samples: list[dict] = []
    with jsonlines.open(test_dataset_path) as reader:
        for obj in reader:
            test_samples.append(obj)

    logger.info(f"[Benchmark] {model_name_tag}: evaluating {len(test_samples)} samples")

    results: list[dict] = []

    for sample in tqdm(test_samples, desc=f"Eval [{model_name_tag}]"):
        query = sample["query"]
        gt = sample.get("ground_truth", {})

        # Stage 1: function retrieval
        if use_dataset_retrieval and sample.get("retrieved_functions"):
            retrieved = sample["retrieved_functions"]
        else:
            retrieved = retriever.retrieve(query, k=top_k)

        # Stage 2: argument value retrieval
        if val_retriever is not None:
            arg_vals = val_retriever.retrieve_for_functions(
                query, retrieved, function_library
            )
        else:
            raw_av = sample.get("retrieved_argument_values")
            arg_vals = raw_av if raw_av else None

        # Build prompt using the same helper as training
        messages = build_messages_for_grpo(
            query=query,
            function_names=retrieved,
            function_library=function_library,
            argument_values=arg_vals,
        )

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate
        t0 = time.perf_counter()
        response = generate_response(model, tokenizer, prompt, max_new_tokens)
        latency = (time.perf_counter() - t0) * 1000.0  # ms

        cost = estimate_cost(prompt, response)
        metrics = compute_all_metrics(
            response, gt, sandbox, latency, cost, function_library
        )
        metrics["sample_id"] = sample.get("id", "")
        results.append(metrics)

    agg = aggregate_metrics(results)
    logger.info(f"[Benchmark] {model_name_tag} aggregate: {agg}")
    return {"model": model_name_tag, "per_sample": results, "aggregate": agg}
