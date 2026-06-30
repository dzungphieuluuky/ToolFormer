#!/usr/bin/env python3
"""
evaluate_retriever.py — Evaluate retriever performance on the test dataset.

Computes precision, recall, F1, hit-rate, and MRR at various k values
for BM25-based and/or hybrid function retrieval.

Usage:
    python scripts/evaluate_retriever.py
    python scripts/evaluate_retriever.py --methods bm25 hybrid
    python scripts/evaluate_retriever.py --test nonexistent.json

Output:
    - Formatted metric tables to stdout
    - JSON results to outputs/evaluation_reports/retriever_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Ensure scripts/ is on the path for importing project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position
from scripts.retrieval import FunctionRetriever


# ── I/O helpers ───────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skip malformed lines with a warning."""
    samples: list[dict] = []
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"WARNING: {path}:{lineno} malformed JSON — skipping: {e}",
                    file=sys.stderr,
                )
    return samples


def _load_json(path: Path) -> dict:
    """Load a JSON file."""
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _download_encoder_local(model_name: str, cache_dir: Path) -> Path:
    """Download an encoder model to a local directory and return its path.

    Uses ``huggingface_hub.snapshot_download`` to download the model
    into ``cache_dir / model_slug`` (where slug is ``model_name`` with
    ``/`` replaced by ``_``), then returns that local path so
    ``SentenceTransformer`` loads from disk instead of the HF cache.

    If the model is already a local path (no ``/`` and the path exists),
    returns it as-is.
    """
    model_path = Path(model_name)
    if model_path.exists():
        return model_path.resolve()

    local_dir = cache_dir / model_name.replace("/", "_")
    if local_dir.exists():
        print(f"    Encoder already cached at {local_dir}")
        return local_dir

    from huggingface_hub import snapshot_download

    print(f"    Downloading encoder model {model_name} -> {local_dir} ...")
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"    Download complete.")
    return local_dir


def _cleanup_encoder_dir(cache_dir: Path, model_name: str) -> None:
    """Remove the locally downloaded encoder model directory."""
    local_dir = cache_dir / model_name.replace("/", "_")
    if local_dir.exists():
        shutil.rmtree(local_dir)
        print(f"    Cleaned up encoder weights at {local_dir}")


def _extract_relevant(sample: dict) -> set[str]:
    """Extract relevant function names from a sample's ground_truth.

    Handles ground_truth as either a dict or a JSON string.
    Returns an empty set for abstention samples.
    """
    gt = sample.get("ground_truth")
    if gt is None:
        return set()
    # Handle JSON-encoded string
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except (json.JSONDecodeError, TypeError):
            return set()
    if not isinstance(gt, dict):
        return set()
    calls = gt.get("calls", [])
    if not isinstance(calls, list):
        return set()
    return {c.get("function", "") for c in calls if isinstance(c, dict) and c.get("function")}


# ── Metric functions ──────────────────────────────────────────────


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute precision@k: fraction of top-k retrieved items that are relevant.

    When *relevant* is empty, the intersection is empty, so precision is 0.0.
    When *k* exceeds len(retrieved), missing tail positions are treated as
    non-relevant (penalty-free, they simply contribute zero to the count).
    """
    if k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """Compute recall@k: fraction of relevant items found in top-k.

    Returns *None* when the relevant set is empty (division by zero).
    """
    if not relevant:
        return None
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def f1_at_k(prec: float, rec: float | None) -> float | None:
    """Compute the harmonic mean of precision and recall at k.

    Returns *None* when recall is *None* or when *prec + rec == 0*.
    """
    if rec is None:
        return None
    total = prec + rec
    if total == 0.0:
        return None
    return 2.0 * prec * rec / total


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute hit-rate@k: 1.0 if any relevant item is in top-k,
    *float('nan')* if relevant set is empty.
    """
    if not relevant:
        return float("nan")
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Compute the reciprocal rank of the first relevant item.

    Returns *float('nan')* when no relevant item is found (or when
    relevant set is empty).
    """
    if not relevant:
        return float("nan")
    for i, func in enumerate(retrieved):
        if func in relevant:
            return 1.0 / (i + 1)
    return float("nan")


# ── Aggregation helpers ───────────────────────────────────────────


def _nan_safe_mean(values: list[float]) -> float:
    """Arithmetic mean, ignoring NaN values."""
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def _nan_safe_std(values: list[float]) -> float:
    """Population standard deviation, ignoring NaN values."""
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) < 2:
        return 0.0
    mean = sum(clean) / len(clean)
    variance = sum((v - mean) ** 2 for v in clean) / len(clean)
    return math.sqrt(variance)


# ── Evaluation functions ──────────────────────────────────────────


def evaluate_retriever(
    retriever: FunctionRetriever,
    test_samples: list[dict],
    k_values: list[int],
    *,
    label: str = "Retriever",
) -> dict[str, Any]:
    """Evaluate a *FunctionRetriever* on a list of test samples.

    For each sample the retriever fetches *max(k_values)* candidates,
    then all metrics are computed at every k. Abstention samples
    (empty *ground_truth.calls*) contribute zero to precision (since
    nothing is relevant) and are excluded from recall / hit-rate / MRR
    aggregates.

    Args:
        retriever: An initialised *FunctionRetriever* instance.
        test_samples: List of sample dicts (must contain *query* and
            *ground_truth*).
        k_values: Sorted list of k thresholds to report.
        label: Human-readable label for this result set.

    Returns:
        A dict with keys *label*, *total_samples*, *num_with_gt*,
        *num_abstention*, *k* (nested dict keyed by str(k)), *mrr*,
        and *time_seconds*.
    """
    max_k = max(k_values)
    num_with_gt = 0
    num_abstention = 0

    # Per-k accumulators
    prec_vals: dict[int, list[float]] = {k: [] for k in k_values}
    rec_vals: dict[int, list[float]] = {k: [] for k in k_values}
    hit_vals: dict[int, list[float]] = {k: [] for k in k_values}
    rr_vals: list[float] = []

    t0 = time.time()

    for sample in test_samples:
        relevant = _extract_relevant(sample)
        query: str = sample.get("query", "")

        if not relevant:
            num_abstention += 1
        else:
            num_with_gt += 1

        retrieved = retriever.retrieve(query, k=max_k)

        for k in k_values:
            prec_vals[k].append(precision_at_k(retrieved, relevant, k))

            rec = recall_at_k(retrieved, relevant, k)
            if rec is not None:
                rec_vals[k].append(rec)
                hit_vals[k].append(hit_rate_at_k(retrieved, relevant, k))

        rr = reciprocal_rank(retrieved, relevant)
        if not math.isnan(rr):
            rr_vals.append(rr)

    elapsed = time.time() - t0

    result: dict[str, Any] = {
        "label": label,
        "total_samples": len(test_samples),
        "num_with_gt": num_with_gt,
        "num_abstention": num_abstention,
        "k": {},
        "mrr": _nan_safe_mean(rr_vals),
        "time_seconds": round(elapsed, 2),
    }

    for k in k_values:
        prec = _nan_safe_mean(prec_vals[k])
        prec_std = _nan_safe_std(prec_vals[k])
        rec = _nan_safe_mean(rec_vals[k])
        rec_std = _nan_safe_std(rec_vals[k])
        hit = _nan_safe_mean(hit_vals[k])

        f1_val = f1_at_k(prec, rec)
        result["k"][str(k)] = {
            "precision": round(prec, 4),
            "precision_std": round(prec_std, 4),
            "recall": round(rec, 4) if not math.isnan(rec) else None,
            "recall_std": round(rec_std, 4),
            "f1": round(f1_val, 4) if f1_val is not None else None,
            "hit_rate": round(hit, 4) if not math.isnan(hit) else None,
        }

    return result


def evaluate_pre_retrieved(
    test_samples: list[dict],
    k_values: list[int],
) -> dict[str, Any]:
    """Evaluate the pre-computed *retrieved_functions* field in the dataset.

    Works exactly like *evaluate_retriever* but reads the already-retrieved
    list from each sample instead of calling a retriever object — providing
    a baseline to compare live retrieval against.
    """
    num_with_gt = 0
    num_abstention = 0

    prec_vals: dict[int, list[float]] = {k: [] for k in k_values}
    rec_vals: dict[int, list[float]] = {k: [] for k in k_values}
    hit_vals: dict[int, list[float]] = {k: [] for k in k_values}
    rr_vals: list[float] = []

    for sample in test_samples:
        relevant = _extract_relevant(sample)
        retrieved: list[str] = sample.get("retrieved_functions", [])

        if not relevant:
            num_abstention += 1
        else:
            num_with_gt += 1

        for k in k_values:
            prec_vals[k].append(precision_at_k(retrieved, relevant, k))

            rec = recall_at_k(retrieved, relevant, k)
            if rec is not None:
                rec_vals[k].append(rec)
                hit_vals[k].append(hit_rate_at_k(retrieved, relevant, k))

        rr = reciprocal_rank(retrieved, relevant)
        if not math.isnan(rr):
            rr_vals.append(rr)

    result: dict[str, Any] = {
        "label": "Pre-Retrieved",
        "total_samples": len(test_samples),
        "num_with_gt": num_with_gt,
        "num_abstention": num_abstention,
        "k": {},
        "mrr": _nan_safe_mean(rr_vals),
        "time_seconds": 0.0,
    }

    for k in k_values:
        prec = _nan_safe_mean(prec_vals[k])
        prec_std = _nan_safe_std(prec_vals[k])
        rec = _nan_safe_mean(rec_vals[k])
        rec_std = _nan_safe_std(rec_vals[k])
        hit = _nan_safe_mean(hit_vals[k])

        f1_val = f1_at_k(prec, rec)
        result["k"][str(k)] = {
            "precision": round(prec, 4),
            "precision_std": round(prec_std, 4),
            "recall": round(rec, 4) if not math.isnan(rec) else None,
            "recall_std": round(rec_std, 4),
            "f1": round(f1_val, 4) if f1_val is not None else None,
            "hit_rate": round(hit, 4) if not math.isnan(hit) else None,
        }

    return result


# ── Reporting ─────────────────────────────────────────────────────


def _fmt(value: Any, width: int = 8) -> str:
    """Format a single metric value for table display.

    *None* renders as ``--``, NaN as ``N/A``, floats to 4 decimal
    places, everything else via ``str()`` — all right-justified to
    *width*.
    """
    if value is None:
        return "--".rjust(width)
    if isinstance(value, float) and math.isnan(value):
        return "N/A".rjust(width)
    if isinstance(value, float):
        return f"{value:.4f}".rjust(width)
    return str(value).rjust(width)


def format_metric_table(
    results: list[dict],
    metric: str,
    title: str,
    k_values: list[int],
) -> str:
    """Build a formatted table for *metric* across several evaluation runs.

    Rows = methods (e.g. "Pre-Retrieved", "BM25"), columns = k values
    plus MRR. The helper ``_fmt`` handles None/NaN gracefully.

    Args:
        results: List of result dicts from *evaluate_retriever* /
            *evaluate_pre_retrieved*.
        metric: Key into ``result['k'][str(k)]``, e.g. ``"precision"``.
        title: Human-readable table title printed above the table.
        k_values: Sorted list of k thresholds to include as columns.

    Returns:
        A multi-line string ready to print.
    """
    # Each data column: 7 chars for value + 2 spaces
    lines: list[str] = []
    separator = "─" * (14 + len(k_values) * 9 + 9)

    lines.append(f"  {title}")
    lines.append(f"  {separator}")

    # Header row
    hdr_parts = ["  Method".ljust(14)]
    for k in k_values:
        hdr_parts.append(f"  @{k:<6}")
    hdr_parts.append("  MRR".ljust(9))
    lines.append("".join(hdr_parts))
    lines.append(f"  {separator}")

    # Data rows
    for res in results:
        row_parts = [f"  {res['label'][:14]:<14}"]
        for k in k_values:
            val = res["k"].get(str(k), {}).get(metric)
            row_parts.append(f"  {_fmt(val, 7)}")
        row_parts.append(f"  {_fmt(res.get('mrr'), 7)}")
        lines.append("".join(row_parts))

    lines.append(f"  {separator}")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate retriever performance on the test dataset.",
    )
    parser.add_argument(
        "--test",
        default="data/generated/v1.0_k5/test_dataset_cleaned.jsonl",
        help="Path to test dataset JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--func-lib",
        default="data/generated/v1.0_k5/function_library.json",
        help="Path to function library JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["bm25", "hybrid"],
        default=["bm25"],
        help="Retrieval methods to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--encoder",
        default="BAAI/bge-m3",
        help="Encoder model for hybrid retrieval (default: %(default)s)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20, 31],
        help="K values for evaluation (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    test_path = Path(args.test)
    func_lib_path = Path(args.func_lib)
    k_values = sorted(set(args.k_values))

    # ── Validate paths ────────────────────────────────────────────
    if not test_path.exists():
        print(f"ERROR: test dataset not found: {test_path}", file=sys.stderr)
        sys.exit(1)
    if not func_lib_path.exists():
        print(f"ERROR: function library not found: {func_lib_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────
    print(f"Loading test samples from {test_path} ...")
    test_samples = _load_jsonl(test_path)
    print(f"  Loaded {len(test_samples)} samples")

    print(f"Loading function library from {func_lib_path} ...")
    function_library = _load_json(func_lib_path)
    print(f"  Loaded {len(function_library)} functions")
    print(f"  K values: {k_values}")

    # ── Evaluate pre-retrieved baseline ───────────────────────────
    print("\nEvaluating pre-retrieved baseline ...")
    pre_result = evaluate_pre_retrieved(test_samples, k_values)
    print(
        f"  Done.  {pre_result['num_with_gt']} samples with ground truth, "
        f"{pre_result['num_abstention']} abstentions."
    )

    # ── Evaluate each requested method ────────────────────────────
    method_results: list[dict] = [pre_result]
    cache_dir = Path(".cache_hf_local")
    for method in args.methods:
        label = method.upper()
        print(f"\nEvaluating {label} retriever ...")
        encoder = args.encoder if method == "hybrid" else None

        local_encoder: str | None = None
        cleanup_needed = False
        if encoder is not None:
            local_encoder = str(_download_encoder_local(encoder, cache_dir))
            cleanup_needed = True

        try:
            retriever = FunctionRetriever(
                function_library=function_library,
                method=method,
                encoder_model=local_encoder,
            )
            result = evaluate_retriever(retriever, test_samples, k_values, label=label)
        finally:
            if cleanup_needed:
                assert encoder is not None
                _cleanup_encoder_dir(cache_dir, encoder)

        print(
            f"  Done in {result['time_seconds']:.2f}s.  "
            f"{result['num_with_gt']} samples with ground truth, "
            f"{result['num_abstention']} abstentions."
        )
        method_results.append(result)

    # ── Print tables ──────────────────────────────────────────────
    print()

    metrics = [
        ("precision", "Precision@K"),
        ("recall", "Recall@K"),
        ("f1", "F1@K"),
        ("hit_rate", "Hit-Rate@K"),
    ]
    for metric_key, title in metrics:
        print(format_metric_table(method_results, metric_key, title, k_values))
        print()

    # ── Save results ──────────────────────────────────────────────
    out_dir = Path("outputs/evaluation_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retriever_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(method_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
