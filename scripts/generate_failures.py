#!/usr/bin/env python3
"""Generate failure samples for RCTP training using three-tier hybrid approach.

Three tiers (tried in order):
  Tier 1 (LLM): TelcoDatasetGenerator._call_failure() via OpenRouter
  Tier 2 (heuristic): catalog-aware inline heuristic (retrieval-aware function
    selection, real alternative values from catalog)
  Tier 3 (legacy): original __WRONG__ prefix behaviour

Usage:
    python scripts/generate_failures.py
    python scripts/generate_failures.py \\
        --input data/generated/v1.0/train_dataset_cleaned.jsonl \\
        --output data/generated/v1.0/failures_dataset.jsonl \\
        --failures-per-sample 1 \\
    --provider openrouter \\
    --model openai/gpt-oss-120b:free \\
    --workers 8 \\
    --dry-run

    python scripts/generate_failures.py \\
        --provider cerebras \\
        --model llama4-scout-17b-16e-instruct

    python scripts/generate_failures.py \\
        --provider groq \\
        --model llama-4-scout-17b-16e-instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

__all__ = ["generate_failures", "FAILURE_TYPES"]


# ── Constants ──────────────────────────────────────────────────────────────────

FAILURE_TYPES = ["wrong_function", "wrong_value", "missing_arg", "hallucinated_arg"]

REASONING_POOL: dict[str, list[str]] = {
    "wrong_function": [
        "Mô hình đã chọn sai hàm do nhầm lẫn ngữ nghĩa câu hỏi.",
        "Mô hình đã hiểu sai ngữ cảnh và chọn hàm không phù hợp.",
        "Mô hình đã nhầm lẫn giữa các hàm có chức năng tương tự.",
        "Mô hình đã chọn hàm không liên quan đến yêu cầu của người dùng.",
    ],
    "wrong_value": [
        "Mô hình đã chọn sai tham số do không khớp giá trị catalog.",
        "Mô hình đã sử dụng giá trị tham số không đúng với yêu cầu.",
        "Mô hình đã nhầm lẫn mã địa điểm/công nghệ trong tham số.",
        "Mô hình đã dùng giá trị mặc định thay vì giá trị người dùng yêu cầu.",
    ],
    "missing_arg": [
        "Mô hình đã bỏ qua tham số không bắt buộc dẫn đến thiếu thông tin.",
        "Mô hình đã không truyền đủ tham số cho hàm.",
        "Mô hình đã thiếu tham số quan trọng mặc dù không bắt buộc.",
    ],
    "hallucinated_arg": [
        "Mô hình đã thêm tham số không tồn tại trong schema.",
        "Mô hình đã phát minh ra tham số không có trong định nghĩa hàm.",
        "Mô hình đã sử dụng tham số không được hỗ trợ bởi hàm.",
    ],
}

HALLUCINATED_PARAMS = [
    "priority_level",
    "service_type",
    "bandwidth",
    "signal_strength",
    "frequency_band",
    "channel_number",
    "encryption_type",
    "protocol_version",
]

# ── Token bucket rate limiter ──────────────────────────────────────────────────


class TokenBucket:
    """Thread-safe token bucket rate limiter for pacing Tier 1 LLM calls.

    Maintains a budget of tokens that refills continuously at a fixed rate.
    Workers call ``acquire()`` before making an API call — if the bucket is
    empty the caller blocks until a token becomes available.  This paces
    requests in **real time** across all threads, unlike a post-hoc sleep.

    The bucket can accumulate up to ``rate_per_minute`` tokens (one minute of
    burst capacity), so short bursts within the limit are allowed while the
    long-term average is enforced.

    Args:
        rate_per_minute: Maximum sustained requests per minute.
    """

    def __init__(self, rate_per_minute: float) -> None:
        self._rate_per_sec = rate_per_minute / 60.0
        self._tokens = 0.0
        self._max_tokens = float(rate_per_minute)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available.

        If the rate limit is 0 or negative, returns immediately (no limit).
        """
        if self._rate_per_sec <= 0:
            return  # No rate limit
        if self._rate_per_sec <= 0:
            return  # No rate limit
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Time until one token is available
                wait = (1.0 - self._tokens) / self._rate_per_sec
            # Sleep outside the lock so other threads can refill concurrently
            time.sleep(max(wait, 0.001))

    def _refill(self) -> None:
        """Refill the token bucket based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._max_tokens, self._tokens + elapsed * self._rate_per_sec
        )
        self._last_refill = now


HALLUCINATED_VALUES = [
    "high",
    "medium",
    "low",
    "auto",
    "standard",
    "enabled",
    "disabled",
    "v1",
    "v2",
]


# ── Thread-safe random helper ─────────────────────────────────────────────────

_local = threading.local()


def _rand() -> random.Random:
    """Return a thread-local Random instance for thread-safe operations.

    Returns:
        Thread-local random.Random instance.
    """
    if not hasattr(_local, "rng"):
        _local.rng = random.Random()
    return _local.rng


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts.

    Args:
        path: Path to JSONL file.

    Returns:
        List of parsed JSON objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path: str) -> dict:
    """Load a JSON file into a dict.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_failure_record(failure: Any) -> bool:
    """Check that a failure dict has the required keys for _build_record.

    Args:
        failure: Raw failure dict to validate.

    Returns:
        True if the dict has failure_type, tool_call, and reasoning keys.
    """
    return (
        isinstance(failure, dict)
        and "failure_type" in failure
        and "tool_call" in failure
        and "reasoning" in failure
    )


# ── Tier 2: smarter heuristic (inline, no API) ────────────────────────────────


def _h_wrong_function(
    sample: dict,
    gold_call: dict,
    function_library: dict,
    catalog: dict,
) -> dict | None:
    """Generate a wrong-function failure: pick a wrong function from candidates.

    Tier 2 heuristic. Selects a different function from the retrieved list
    and fills required args from the catalog.

    Args:
        sample: Gold dataset sample.
        gold_call: The correct tool_call dict.
        function_library: Dict of function_name → schema.
        catalog: Argument value catalog dict.

    Returns:
        Failure dict with wrong function, or None if impossible.
    """
    rng = _rand()
    gold_fn = gold_call["function"]
    retrieved = sample.get("retrieved_functions", [])
    candidates = [fn for fn in retrieved if fn != gold_fn]
    if not candidates:
        candidates = [fn for fn in function_library if fn != gold_fn]
    if not candidates:
        return None

    wrong_fn = rng.choice(candidates)
    schema = function_library.get(wrong_fn, {})
    params = schema.get("parameters", {})
    wrong_args: dict[str, Any] = {}
    for k, v in params.items():
        if v.get("required", False):
            # Try catalog first for a realistic value, then fall back to default
            if k in catalog and catalog[k]:
                wrong_args[k] = catalog[k][0].get("code", "")
            else:
                wrong_args[k] = v.get("default", "")

    return {
        "failure_type": "wrong_function",
        "tool_call": {"function": wrong_fn, "arguments": wrong_args},
        "reasoning": rng.choice(REASONING_POOL["wrong_function"]),
    }


def _h_wrong_value(
    sample: dict,
    gold_call: dict,
    function_library: dict,
    catalog: dict,
) -> dict | None:
    """Generate a wrong-value failure: keep correct fn, use wrong arg value.

    Tier 2 heuristic. Picks a catalog value different from the gold value.

    Args:
        sample: Gold dataset sample.
        gold_call: The correct tool_call dict.
        function_library: Dict of function_name → schema.
        catalog: Argument value catalog dict.

    Returns:
        Failure dict with wrong argument value, or None if impossible.
    """
    rng = _rand()
    gold_fn = gold_call["function"]
    gold_args = gold_call.get("arguments", {})
    catalog_keys = set(catalog.keys())
    eligible = [k for k in gold_args if k in catalog_keys]
    if not eligible:
        return None

    target = rng.choice(eligible)
    current = gold_args[target]
    alternatives = [e for e in catalog[target] if e.get("code", "") != current]
    if not alternatives:
        return None

    chosen = rng.choice(alternatives)
    wrong_args = dict(gold_args)
    wrong_args[target] = chosen["code"]

    return {
        "failure_type": "wrong_value",
        "tool_call": {"function": gold_fn, "arguments": wrong_args},
        "reasoning": rng.choice(REASONING_POOL["wrong_value"]),
    }


def _h_missing_arg(
    sample: dict,
    gold_call: dict,
    function_library: dict,
    catalog: dict,
) -> dict | None:
    """Generate a missing-argument failure: drop a non-required parameter.

    Tier 2 heuristic. Selects a non-required parameter from the schema
    and removes it from the arguments.

    Args:
        sample: Gold dataset sample.
        gold_call: The correct tool_call dict.
        function_library: Dict of function_name → schema.
        catalog: Argument value catalog dict.

    Returns:
        Failure dict with missing argument, or None if impossible.
    """
    rng = _rand()
    gold_fn = gold_call["function"]
    gold_args = gold_call.get("arguments", {})
    schema = function_library.get(gold_fn, {})
    params = schema.get("parameters", {})

    non_required = [
        k for k, v in params.items() if not v.get("required", True) and k in gold_args
    ]
    if not non_required:
        return None

    drop = rng.choice(non_required)
    wrong_args = {k: v for k, v in gold_args.items() if k != drop}

    return {
        "failure_type": "missing_arg",
        "tool_call": {"function": gold_fn, "arguments": wrong_args},
        "reasoning": rng.choice(REASONING_POOL["missing_arg"]),
    }


def _h_hallucinated_arg(
    sample: dict,
    gold_call: dict,
    function_library: dict,
    catalog: dict,
) -> dict | None:
    """Generate a hallucinated-argument failure: add an invented parameter.

    Tier 2 heuristic. Picks a parameter name from the hallucination pool
    that doesn't exist in the gold args.

    Args:
        sample: Gold dataset sample.
        gold_call: The correct tool_call dict.
        function_library: Dict of function_name → schema.
        catalog: Argument value catalog dict.

    Returns:
        Failure dict with hallucinated argument, or None if impossible.
    """
    rng = _rand()
    gold_fn = gold_call["function"]
    gold_args = gold_call.get("arguments", {})

    # Only pick a param that doesn't already exist in gold_args
    existing = set(gold_args.keys())
    candidates = [p for p in HALLUCINATED_PARAMS if p not in existing]
    if not candidates:
        return None

    fake = rng.choice(candidates)
    wrong_args = dict(gold_args)
    wrong_args[fake] = rng.choice(HALLUCINATED_VALUES)

    return {
        "failure_type": "hallucinated_arg",
        "tool_call": {"function": gold_fn, "arguments": wrong_args},
        "reasoning": rng.choice(REASONING_POOL["hallucinated_arg"]),
    }


_TIER2_DISPATCH = {
    "wrong_function": _h_wrong_function,
    "wrong_value": _h_wrong_value,
    "missing_arg": _h_missing_arg,
    "hallucinated_arg": _h_hallucinated_arg,
}


def _tier2_failure(
    sample: dict,
    gold_call: dict,
    function_library: dict,
    catalog: dict,
    failure_type: str | None = None,
) -> dict | None:
    """Generate a Tier 2 heuristic failure for a specific failure type.

    Dispatches to the appropriate _h_* function based on failure_type.

    Args:
        sample: Gold dataset sample.
        gold_call: The correct tool_call dict.
        function_library: Dict of function_name → schema.
        catalog: Argument value catalog dict.
        failure_type: Specific failure type, or None to pick randomly.

    Returns:
        Failure dict, or None if the chosen type cannot be generated.
    """
    if failure_type is None:
        failure_type = _rand().choice(FAILURE_TYPES)
    func = _TIER2_DISPATCH.get(failure_type)
    if func is None:
        return None
    return func(sample, gold_call, function_library, catalog)


# ── Tier 3: legacy heuristic (original __WRONG__ prefix behaviour) ────────────


def _tier3_failure(
    sample: dict,
    gold_call: dict,
    function_library: dict,
    catalog: dict,
    failure_type: str | None = None,
) -> dict | None:
    """Generate a Tier 3 legacy failure (__WRONG__ prefix behaviour).

    Fallback heuristic that prefixes argument values with "__WRONG__"
    or uses simple random selection.

    Args:
        sample: Gold dataset sample.
        gold_call: The correct tool_call dict.
        function_library: Dict of function_name → schema.
        catalog: Argument value catalog dict.
        failure_type: Specific failure type, or None to pick randomly.

    Returns:
        Failure dict, or None if the chosen type cannot be generated.
    """
    rng = _rand()
    if failure_type is None:
        failure_type = rng.choice(FAILURE_TYPES)

    gold_fn = gold_call["function"]
    gold_args = gold_call.get("arguments", {})
    all_funcs = list(function_library.keys())

    if failure_type == "wrong_function":
        candidates = [fn for fn in all_funcs if fn != gold_fn]
        wrong_fn = rng.choice(candidates) if candidates else gold_fn
        wrong_args = {k: f"__WRONG__{v}" for k, v in gold_args.items()}
    elif failure_type == "wrong_value":
        wrong_fn = gold_fn
        wrong_args = {k: f"__WRONG__{v}" for k, v in gold_args.items()}
    elif failure_type == "missing_arg":
        wrong_fn = gold_fn
        keys = list(gold_args.keys())
        if not keys:
            return None
        drop = rng.choice(keys)
        wrong_args = {k: v for k, v in gold_args.items() if k != drop}
    elif failure_type == "hallucinated_arg":
        wrong_fn = gold_fn
        wrong_args = dict(gold_args)
        existing = set(gold_args.keys())
        candidates = [p for p in HALLUCINATED_PARAMS if p not in existing]
        if not candidates:
            return None
        wrong_args[rng.choice(candidates)] = rng.choice(HALLUCINATED_VALUES)
    else:
        return None

    return {
        "failure_type": failure_type,
        "tool_call": {"function": wrong_fn, "arguments": wrong_args},
        "reasoning": rng.choice(
            REASONING_POOL.get(
                failure_type, ["Mô hình đã tạo ra lỗi do nhầm lẫn ngẫu nhiên."]
            )
        ),
    }


# ── Output record builder ─────────────────────────────────────────────────────


def _build_record(sample: dict, failure: dict) -> dict:
    """Build a complete failure record from a sample and a failure dict.

    Args:
        sample: Original gold dataset sample.
        failure: Failure dict with tool_call, failure_type, and reasoning.

    Returns:
        Complete failure record dict matching the output schema.
    """
    tc = failure["tool_call"]
    ft = failure["failure_type"]
    return {
        "id": sample["id"],
        "query": sample["query"],
        "workflow_type": sample.get("workflow_type", ""),
        "ground_truth": {
            "calls": [tc],
            "reasoning": failure["reasoning"],
            "refusal_message": None,
        },
        "retrieved_functions": sample.get("retrieved_functions", []),
        "retrieved_argument_values": sample.get("retrieved_argument_values", {}),
        "split": sample.get("split", "train"),
        "failure_type": ft,
    }


# ── Single-sample processing (called by workers) ──────────────────────────────


def _process_sample(
    sample: dict,
    function_library: dict,
    catalog: dict,
    failures_per_sample: int,
    tier1_generator: Any | None,
    rate_limiter: TokenBucket | None = None,
) -> list[dict]:
    """Generate failure records for a single gold sample using three tiers.

    Tries Tier 1 (LLM) first, then falls back to Tier 2 (heuristic),
    then Tier 3 (legacy) for any remaining slots.

    Args:
        sample: Gold dataset sample.
        function_library: Dict of function_name → schema.
        catalog: Argument value catalog dict.
        failures_per_sample: Number of failures to generate.
        tier1_generator: Optional TelcoDatasetGenerator for LLM tier.
        rate_limiter: Optional TokenBucket for rate limiting.

    Returns:
        List of failure record dicts.
    """
    gold_calls = sample.get("ground_truth", {}).get("calls", [])
    if not gold_calls:
        return []
    gold_call = gold_calls[0]

    results: list[dict] = []

    # Tier 1: LLM (one call, returns up to 4 failure candidates)
    tier1_pool: list[dict] = []
    if tier1_generator is not None:
        try:
            if rate_limiter is not None:
                rate_limiter.acquire()
            raw = tier1_generator._call_failure(sample)
            if isinstance(raw, list):
                tier1_pool = [f for f in raw if _validate_failure_record(f)]
        except Exception:
            pass  # Tier 1 LLM call failed — will rely on Tier 2/3 fallback

    for failure in tier1_pool[:failures_per_sample]:
        results.append(_build_record(sample, failure))

    remaining = failures_per_sample - len(results)
    if remaining <= 0:
        return results

    # Fill remaining slots with Tier 2 / Tier 3
    available_types = list(FAILURE_TYPES)
    rng = _rand()
    for _ in range(remaining):
        if available_types:
            ft = available_types.pop(rng.randrange(len(available_types)))
        else:
            ft = rng.choice(FAILURE_TYPES)

        # Tier 2: heuristic
        failure = _tier2_failure(sample, gold_call, function_library, catalog, ft)

        # Tier 3: legacy fallback
        if failure is None:
            failure = _tier3_failure(sample, gold_call, function_library, catalog, ft)

        if failure is not None:
            results.append(_build_record(sample, failure))
        else:
            logger.warning(
                "All tiers failed for sample %s (type=%s)",
                sample.get("id", "?"),
                ft,
            )

    return results


# ── Main generation ───────────────────────────────────────────────────────────


def generate_failures(
    input_path: str,
    output_path: str,
    train_schema_path: str,
    test_schema_path: str,
    catalog_path: str,
    failures_per_sample: int = 1,
    provider: str = "opencode",
    model: str = "deepseek-v4-flash-free",
    max_workers: int = 8,
    requests_per_minute: int = 500,
    dry_run: bool = False,
) -> None:
    """Generate failure samples using three-tier hybrid approach.

    Tier 1: LLM via TelcoDatasetGenerator._call_failure()
    Tier 2: Catalog-aware heuristic (inline)
    Tier 3: Legacy __WRONG__ prefix behaviour

    Args:
        input_path: Path to gold dataset JSONL.
        output_path: Path for output failure dataset JSONL.
        train_schema_path: Path to train function schemas.
        test_schema_path: Path to test function schemas.
        catalog_path: Path to argument values catalog JSON.
        failures_per_sample: Number of failures per gold sample.
        provider: LLM provider for Tier 1.
        model: LLM model name for Tier 1.
        max_workers: Number of parallel worker threads.
        requests_per_minute: Rate limit for Tier 1 LLM calls.
        dry_run: If True, only log configuration without generating.
    """
    samples = _load_jsonl(input_path)
    logger.info("Loaded %d samples from %s", len(samples), input_path)

    train_function_library = _load_json(train_schema_path)
    test_function_library = _load_json(test_schema_path)
    logger.info(
        "Function schemas: %d train (%s), %d test (%s)",
        len(train_function_library),
        train_schema_path,
        len(test_function_library),
        test_schema_path,
    )

    catalog = _load_json(catalog_path)
    logger.info("Argument catalog: %d param keys from %s", len(catalog), catalog_path)

    # ── Filter out abstention ──────────────────────────────────────────
    non_abstention: list[dict] = []
    skipped = 0
    for s in samples:
        calls = s.get("ground_truth", {}).get("calls", [])
        wf = s.get("workflow_type", "")
        if wf == "abstention" or not calls:
            skipped += 1
            continue
        non_abstention.append(s)

    logger.info("Non-abstention samples: %d (skipped %d)", len(non_abstention), skipped)

    if not non_abstention:
        logger.warning("No non-abstention samples to process. Exiting.")
        return

    # ── Tier 1 generator setup ─────────────────────────────────────────
    tier1_generator = None
    if not dry_run:
        try:
            input_dir = Path(input_path).parent.resolve()
            av_path = str(input_dir / "argument_values.json")

            from scripts.data_generator import TelcoDatasetGenerator

            api_key = (
                os.getenv(f"{provider.upper()}_API_KEY")
                or os.getenv("OPENROUTER_API_KEY")
                or os.getenv("LLM_API_KEY")
            )

            if api_key:
                tier1_generator = TelcoDatasetGenerator.from_schemas(
                    train_schema_path=train_schema_path,
                    test_schema_path=test_schema_path,
                    argument_values_path=av_path,
                    provider=provider,
                    model=model,
                    max_workers=max_workers,
                    requests_per_minute=requests_per_minute,
                )
                logger.info("Tier 1 (LLM) ready: %s (%s)", model, provider)
            else:
                logger.warning(
                    "No API key for %s. Set %s_API_KEY or LLM_API_KEY. "
                    "Tier 1 will be skipped.",
                    provider,
                    provider.upper(),
                )
        except Exception as exc:
            logger.warning(
                "Tier 1 LLM init failed — falling back to Tier 2/3 heuristics only."
            )

    # ── Dry run ────────────────────────────────────────────────────────
    if dry_run:
        logger.info(
            "[DRY RUN] Would process %d samples, %d failure(s) each",
            len(non_abstention),
            failures_per_sample,
        )
        logger.info("[DRY RUN] Would write to %s", output_path)
        return

    # ── Generate ───────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_expected = len(non_abstention) * failures_per_sample
    progress = (
        tqdm(total=total_expected, desc="Generating failures", unit="failure")
        if tqdm
        else None
    )

    all_records: list[dict] = []
    records_lock = threading.Lock()

    # ── Create rate limiter ──────────────────────────────────────────────
    rate_limiter: TokenBucket | None = (
        TokenBucket(requests_per_minute) if requests_per_minute > 0 else None
    )
    if rate_limiter is not None:
        logger.info(
            "Rate limiter: %d req/min (burst up to %d tokens)",
            requests_per_minute,
            requests_per_minute,
        )

    def _worker(sample: dict) -> list[dict]:
        fn_library = (
            test_function_library
            if sample.get("split") == "test"
            else train_function_library
        )
        return _process_sample(
            sample,
            fn_library,
            catalog,
            failures_per_sample,
            tier1_generator,
            rate_limiter,
        )

    if max_workers <= 1:
        for sample in non_abstention:
            records = _worker(sample)
            all_records.extend(records)
            if progress:
                progress.update(len(records) if records else failures_per_sample)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_worker, sample) for sample in non_abstention]
            for fut in as_completed(futures):
                try:
                    records = fut.result()
                    with records_lock:
                        all_records.extend(records)
                    if progress:
                        progress.update(
                            len(records) if records else failures_per_sample
                        )
                except Exception as exc:
                    logger.warning(
                        "A worker thread encountered an error — check sample data for compatibility."
                    )
                    if progress:
                        progress.update(failures_per_sample)

    if progress:
        progress.close()

    # ── Write output ───────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Written %d failure samples to %s", len(all_records), out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point: generate failure samples for RCTP training."""
    parser = argparse.ArgumentParser(
        description="Generate failure samples for RCTP training (three-tier hybrid)",
    )

    base = "data/generated/v1.0"
    parser.add_argument(
        "--input",
        default=f"{base}/train_dataset_cleaned.jsonl",
        help="Input dataset (JSONL)",
    )
    parser.add_argument(
        "--output",
        default="data/generated/v2.0/failures_dataset.jsonl",
        help="Output failure dataset (JSONL)",
    )
    parser.add_argument(
        "--failures-per-sample",
        type=int,
        default=1,
        dest="failures_per_sample",
        help="Number of failures to generate per gold sample (>= 1)",
    )
    parser.add_argument(
        "--provider",
        default="openrouter",
        help="LLM provider for Tier 1 (openrouter, openai, cerebras, groq, together, anthropic, google, mistral)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-120b:free",
        help="LLM model for Tier 1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        dest="max_workers",
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=50,
        dest="requests_per_minute",
        help="Requests per minute for Tier 1 LLM calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Preview configuration without generating",
    )
    parser.add_argument(
        "--train-schema",
        default=None,
        help="Train function schema JSON (default: derived from --input dir)",
    )
    parser.add_argument(
        "--test-schema",
        default=None,
        help="Test function schema JSON (default: derived from --input dir)",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="Argument values catalog JSON (default: derived from --input dir)",
    )

    args = parser.parse_args()

    if args.failures_per_sample < 1:
        parser.error("--failures-per-sample must be >= 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    input_dir = Path(args.input).parent.resolve()
    catalog = args.catalog or str(input_dir / "argument_values.json")
    train_schema = args.train_schema or str(input_dir / "function_schema_train.json")
    test_schema = args.test_schema or str(input_dir / "function_schema_test.json")

    generate_failures(
        input_path=args.input,
        output_path=args.output,
        train_schema_path=train_schema,
        test_schema_path=test_schema,
        catalog_path=catalog,
        failures_per_sample=args.failures_per_sample,
        provider=args.provider,
        model=args.model,
        max_workers=args.max_workers,
        requests_per_minute=args.requests_per_minute,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
