"""
benchmark.py
─────────────
Run full evaluation of a fine-tuned model on the test set.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jsonlines
from tqdm.auto import tqdm

from src.utils.model_utils   import load_model_from_path, generate_response
from src.utils.sandbox        import Sandbox
from src.data.retrieval       import FunctionRetriever
from src.algorithms.base_trainer import build_prompt, SYSTEM_PROMPT
from .metrics import compute_all_metrics, aggregate_metrics, estimate_cost
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_path:       str,
    test_dataset_path: str,
    function_library:  dict,
    retriever:         FunctionRetriever,
    sandbox:           Sandbox,
    top_k:             int   = 5,
    max_new_tokens:    int   = 512,
    model_name_tag:    str   = "model",
) -> dict:
    """
    Evaluate a fine-tuned model on the held-out test set.

    Returns aggregated metrics dict.
    """
    # Load model
    logger.info(f"[Benchmark] Loading model from {model_path}")
    model, tokenizer = load_model_from_path(model_path)

    # Load test samples
    test_samples = []
    with jsonlines.open(test_dataset_path) as reader:
        for obj in reader:
            test_samples.append(obj)

    logger.info(f"[Benchmark] Evaluating on {len(test_samples)} test samples...")
    results = []

    for sample in tqdm(test_samples, desc=f"Eval [{model_name_tag}]"):
        query     = sample["query"]
        gt        = sample.get("ground_truth", {})

        # Retrieve top-k functions
        retrieved = retriever.retrieve(query, k=top_k)

        # Build prompt
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(query, retrieved, function_library)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        # Time the inference
        t0       = time.perf_counter()
        response = generate_response(model, tokenizer, prompt, max_new_tokens)
        latency  = (time.perf_counter() - t0) * 1000.0  # ms

        cost = estimate_cost(prompt, response)
        metrics = compute_all_metrics(
            response, gt, sandbox, latency, cost, function_library
        )
        metrics["sample_id"] = sample.get("id", "")
        results.append(metrics)

    agg = aggregate_metrics(results)
    logger.info(f"[Benchmark] {model_name_tag} results: {agg}")
    return {"model": model_name_tag, "per_sample": results, "aggregate": agg}