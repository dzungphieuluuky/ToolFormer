#!/usr/bin/env python3
"""
inspect_dataset.py — Print basic statistics about the toolformer dataset.

Usage:
    python inspect_dataset.py [--data-dir data/processed]

Prints sectioned statistics to stdout. Exit 0 always (informational only).
Exit 1 when required input files are missing.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_ground_truth(gt: Any) -> dict | None:
    """Normalise ground_truth from dict or JSON string. Mirrors clean_dataset.py."""
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except (json.JSONDecodeError, TypeError):
            return None
    return gt if isinstance(gt, dict) else None


def extract_calls(gt: Any) -> list[dict]:
    """Return the calls list from ground_truth, or empty list if unparseable."""
    parsed = _parse_ground_truth(gt)
    if parsed is None:
        return []
    calls = parsed.get("calls")
    return calls if isinstance(calls, list) else []


def extract_function_names(sample: dict) -> list[str]:
    """Return function names from ground_truth calls, or ['<missing>']."""
    calls = extract_calls(sample.get("ground_truth"))
    if not calls:
        return ["<missing>"]
    names = [c.get("function", "<unknown>") for c in calls if isinstance(c, dict)]
    return names if names else ["<missing_function_name>"]


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping malformed lines with a warning."""
    samples = []
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return None
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


def load_json(path: Path) -> dict | None:
    """Load a JSON file, return None if missing."""
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print("─" * 60)
    print(f"  {title}")
    print("─" * 60)


def inspect_all(data_dir: str) -> int:
    """Run all inspection checks and print results. Returns exit code."""
    data_path = Path(data_dir)

    # ── Load data ──────────────────────────────────────────────────────
    train_samples = load_jsonl(data_path / "train_dataset_cleaned.jsonl")
    test_samples = load_jsonl(data_path / "test_dataset_cleaned.jsonl")
    function_library = load_json(data_path / "function_library.json")
    argument_values = load_json(data_path / "argument_values.json")

    if train_samples is None or test_samples is None:
        return 1
    if function_library is None:
        print(
            "WARNING: function_library.json not found — skipping function-related stats",
            file=sys.stderr,
        )
    if argument_values is None:
        print(
            "WARNING: argument_values.json not found — skipping arg-value coverage",
            file=sys.stderr,
        )
    print(
        f"Loaded {len(train_samples)} train samples, {len(test_samples)} test samples"
    )
    print(f"Loaded train samples from {data_path / 'train_dataset_cleaned.jsonl'}")
    print(f"Loaded test samples from {data_path / 'test_dataset_cleaned.jsonl'}")

    test_only_funcs: set[str] = set()
    all_funcs: set[str] = set()
    if function_library:
        # Load train/test schemas to determine which are test-only
        train_schema = load_json(data_path / "function_schema_train.json")
        test_schema = load_json(data_path / "function_schema_test.json")
        train_funcs = set(train_schema.keys()) if train_schema else set()
        test_funcs = set(test_schema.keys()) if test_schema else set()
        test_only_funcs = test_funcs - train_funcs
        all_funcs = set(function_library.keys())

    # ── 1. Sample counts ───────────────────────────────────────────────
    print_section("1. Sample counts")
    print(f"  Total:  {len(train_samples) + len(test_samples)}")
    print(f"  Train:  {len(train_samples)}")
    print(f"  Test:   {len(test_samples)}")

    # ── 2. Workflow distribution ───────────────────────────────────────
    print_section("2. Workflow distribution")
    for split_name, samples in [("TRAIN", train_samples), ("TEST", test_samples)]:
        print(f"\n  [{split_name}]")
        wf_counts = Counter(s.get("workflow_type", "<missing>") for s in samples)
        for wf in ["single_call", "parallel", "sequential", "abstention"]:
            count = wf_counts.get(wf, 0)
            pct = 100.0 * count / len(samples) if samples else 0.0
            print(f"    {wf:20s}: {count:5d}  ({pct:5.1f}%)")
        for wf in wf_counts:
            if wf not in {"single_call", "parallel", "sequential", "abstention"}:
                print(f"    {wf:20s}: {wf_counts[wf]:5d}  (UNKNOWN)")

    # ── 3. Function frequency ──────────────────────────────────────────
    # Extracted from ground_truth.calls (not the top-level function_name field,
    # which is dropped by clean_dataset.py). Parallel samples may contribute
    # multiple function names.
    print_section("3. Function frequency")
    for split_name, samples in [("TRAIN", train_samples), ("TEST", test_samples)]:
        print(f"\n  [{split_name}]")
        fn_counts: Counter = Counter()
        for s in samples:
            for fn in extract_function_names(s):
                fn_counts[fn] += 1
        # Add functions from schemas with 0 usage
        if all_funcs:
            for fn in sorted(all_funcs):
                if fn not in fn_counts:
                    fn_counts[fn] = 0
        for fn, count in fn_counts.most_common():
            marker = ""
            if function_library and fn in test_only_funcs:
                marker = "  [TEST-ONLY]"
            pct = 100.0 * count / len(samples) if samples else 0.0
            print(f"    {fn:35s}: {count:5d}  ({pct:5.1f}%){marker}")
        # Functions used but not in library
        if function_library:
            for fn in fn_counts:
                if fn not in all_funcs and fn not in (
                    "<missing>",
                    "<missing_function_name>",
                    "<unknown>",
                ):
                    print(f"    {fn:35s}: {fn_counts[fn]:5d}  (NOT IN LIBRARY)")

    # ── 4. Ground truth call stats ─────────────────────────────────────
    print_section("4. Ground truth call stats")
    for split_name, samples in [("TRAIN", train_samples), ("TEST", test_samples)]:
        print(f"\n  [{split_name}]")
        empty_by_wf: Counter = Counter()
        non_empty = 0
        call_counts: Counter = Counter()
        for s in samples:
            calls = extract_calls(s.get("ground_truth"))
            n = len(calls)
            call_counts[n] += 1
            if n == 0:
                wf = s.get("workflow_type", "<missing>")
                empty_by_wf[wf] += 1
            else:
                non_empty += 1

        print(f"    Non-empty calls:     {non_empty:5d}")
        print(f"    Empty calls total:   {sum(empty_by_wf.values()):5d}")
        for wf in ["single_call", "parallel", "sequential", "abstention", "<missing>"]:
            c = empty_by_wf.get(wf, 0)
            if c > 0:
                print(f"      {wf:20s}: {c:5d}")
        print(f"    Call count distribution:")
        for n in sorted(call_counts):
            print(f"      {n} call(s): {call_counts[n]:5d}")

    # ── 5. Argument value coverage ─────────────────────────────────────
    print_section("5. Argument value coverage")
    for split_name, samples in [("TRAIN", train_samples), ("TEST", test_samples)]:
        if argument_values is None:
            print(f"  [{split_name}] SKIPPED (argument_values.json not loaded)")
            continue
        with_vals = sum(1 for s in samples if s.get("retrieved_argument_values"))
        print(f"  [{split_name}]")
        print(
            f"    With arg values:     {with_vals:5d}  ({100.0 * with_vals / len(samples):5.1f}%)"
        )
        print(
            f"    Without arg values:  {len(samples) - with_vals:5d}  ({100.0 * (len(samples) - with_vals) / len(samples):5.1f}%)"
        )

    # ── 6. Data quality warnings ───────────────────────────────────────
    print_section("6. Data quality warnings")
    has_warnings = False

    # 6a. Non-abstention empty calls (ERROR)
    for split_name, samples in [("TRAIN", train_samples), ("TEST", test_samples)]:
        error_count = 0
        for s in samples:
            wf = s.get("workflow_type")
            if wf == "abstention":
                continue
            calls = extract_calls(s.get("ground_truth"))
            if len(calls) == 0:
                error_count += 1
        if error_count > 0:
            has_warnings = True
            print(
                f"  [ERROR] {split_name}: {error_count} non-abstention samples with empty ground_truth.calls"
            )

    # 6b. Cross-split: train samples using test-only functions
    if test_only_funcs:
        for split_name, samples in [("TRAIN", train_samples)]:
            leak_count = 0
            seen: set[str] = set()
            for s in samples:
                calls = extract_calls(s.get("ground_truth"))
                for c in calls:
                    fn = c.get("function", "")
                    if fn in test_only_funcs:
                        seen.add(fn)
                        leak_count += 1
            if leak_count > 0:
                has_warnings = True
                print(
                    f"  [WARNING] {split_name}: {leak_count} test-only functions existed in train samples ({', '.join(sorted(seen))})"
                )
            else:
                print(f"  [OK] {split_name}: no cross-split function leakage")

    if not has_warnings:
        print("  No data quality warnings found.")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Inspect toolformer dataset statistics"
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Path to the data/processed directory (default: data/processed)",
    )
    args = parser.parse_args()
    exit_code = inspect_all(args.data_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
