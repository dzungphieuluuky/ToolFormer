"""
rerank_functions.py
───────────────────
Re-rank retrieved_functions in cleaned datasets to keep only the
top 5 most relevant per sample (by BM25 score against the query).

Usage:
    python scripts/rerank_functions.py \\
        --input-base data/generated/v1.0/train_dataset_cleaned.jsonl \\
        --input-test data/generated/v1.0/test_dataset_cleaned.jsonl \\
        --output-base data/generated/v1.0/train_dataset_cleaned_k5.jsonl \\
        --output-test data/generated/v1.0/test_dataset_cleaned_k5.jsonl
"""

import json
import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.retrieval import FunctionRetriever


def pick_top5(
    query: str,
    candidate_fns: list[str],
    ground_truth: dict,
    retriever: FunctionRetriever,
    name_to_idx: dict[str, int],
) -> list[str]:
    """From candidate_fns, keep top 5 most relevant to query by BM25 score.

    Always retains the ground-truth function if it was in the candidates.
    """
    if not candidate_fns:
        return []

    scores = retriever._score(query)

    scored = []
    for fn in candidate_fns:
        idx = name_to_idx.get(fn)
        if idx is not None:
            scored.append((fn, float(scores[idx])))

    if not scored:
        return []

    scored.sort(key=lambda x: -x[1])
    top5 = [fn for fn, _ in scored[:5]]

    calls = (
        ground_truth.get("calls", []) if isinstance(ground_truth, dict) else []
    )
    true_fn = calls[0]["function"] if calls else None
    if true_fn and true_fn in candidate_fns and true_fn not in top5:
        top5[-1] = true_fn  # slot the ground truth back in

    return top5


def process_file(
    input_path: Path,
    output_path: Path,
    retriever: FunctionRetriever,
    name_to_idx: dict[str, int],
    label: str,
):
    """Load JSONL, re-rank retrieved_functions, save to output_path."""
    samples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    before_counts: dict[int, int] = {}
    after_counts: dict[int, int] = {}
    gt_retained = 0
    gt_total = 0
    unchanged = 0

    for s in samples:
        query = s.get("query", "")
        orig_fns = s.get("retrieved_functions", [])
        gt = s.get("ground_truth", {})

        n_before = len(orig_fns)
        before_counts[n_before] = before_counts.get(n_before, 0) + 1

        if n_before <= 5:
            new_fns = orig_fns[:5]
            unchanged += 1
        else:
            new_fns = pick_top5(query, orig_fns, gt, retriever, name_to_idx)

        # Track ground-truth retention
        calls = gt.get("calls", []) if isinstance(gt, dict) else []
        if calls and len(new_fns) > 0:
            gt_total += 1
            true_fn = calls[0]["function"]
            if true_fn in new_fns:
                gt_retained += 1

        after_counts[len(new_fns)] = after_counts.get(len(new_fns), 0) + 1
        s["retrieved_functions"] = new_fns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n── {label} ──────────────────────────────────")
    print(f"  Samples:         {len(samples)}")
    print(f"  Before dist:     {dict(sorted(before_counts.items()))}")
    print(f"  After dist:      {dict(sorted(after_counts.items()))}")
    print(f"  Unchanged (≤5):  {unchanged}")
    print(f"  GT retained:     {gt_retained}/{gt_total}  ({gt_retained/max(gt_total,1)*100:.1f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-rank retrieved_functions to top 5 most relevant."
    )
    parser.add_argument(
        "--input-base",
        default="data/generated/v1.0/train_dataset_cleaned.jsonl",
    )
    parser.add_argument(
        "--input-test",
        default="data/generated/v1.0/test_dataset_cleaned.jsonl",
    )
    parser.add_argument(
        "--output-base",
        default="data/generated/v1.0/train_dataset_cleaned_k5.jsonl",
    )
    parser.add_argument(
        "--output-test",
        default="data/generated/v1.0/test_dataset_cleaned_k5.jsonl",
    )
    parser.add_argument(
        "--retriever-index",
        default="data/generated/v1.0/retrieval_index/retriever.pkl",
    )
    parser.add_argument(
        "--function-library",
        default="data/generated/v1.0/function_library.json",
    )
    args = parser.parse_args()

    retriever_path = Path(args.retriever_index)
    func_lib_path = Path(args.function_library)

    # Load function library
    print("Loading function library ...")
    with open(func_lib_path) as f:
        func_lib = json.load(f)
    print(f"  {len(func_lib)} functions")

    # Load retriever
    print("Loading FunctionRetriever ...")
    retriever = FunctionRetriever.load(str(retriever_path), func_lib)
    print("  OK")

    name_to_idx = {name: i for i, name in enumerate(retriever.func_names)}

    process_file(
        Path(args.input_base),
        Path(args.output_base),
        retriever,
        name_to_idx,
        "train",
    )
    process_file(
        Path(args.input_test),
        Path(args.output_test),
        retriever,
        name_to_idx,
        "test",
    )

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
