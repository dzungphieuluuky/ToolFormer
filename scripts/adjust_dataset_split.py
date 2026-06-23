#!/usr/bin/env python
"""
adjust_dataset_split.py
────────────────────────
Reorganise train/test split so that:

  - Test set  contains samples for ALL functions (both train-schema and
    test-schema functions) for comprehensive evaluation.
  - Train set contains only train-schema functions (never test-schema
    functions).

Strategy:
  1. Segregate all samples by their primary function_name.
  2. Test-schema-function samples → always go to test.
  3. Train-schema-function samples are split by --train-ratio
     (default 0.7 train / 0.3 test) so the test set has broad coverage.
  4. Abstention ("none") samples are split by the same ratio.
  5. Every train-schema function is guaranteed at least one sample in test.

Usage:
    python scripts/adjust_dataset_split.py
    python scripts/adjust_dataset_split.py --train-ratio 0.8
    python scripts/adjust_dataset_split.py --train-ratio 0.7 --dry-run
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


SEED = 3407


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_report(
    label: str,
    records: list[dict],
    train_funcs: set,
    test_funcs: set,
) -> None:
    total = len(records)
    fn_counter: Counter = Counter()
    wf_counter: Counter = Counter()
    for r in records:
        fn = r.get("function_name", "unknown")
        fn_counter[fn] += 1
        wf_counter[r.get("workflow_type", "unknown")] += 1

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  Total samples: {total}")
    print(f"\n  ── Workflow distribution ──")
    for wt, n in sorted(wf_counter.items()):
        print(f"    {wt:20s}: {n:5d}  ({100 * n / total:.1f}%)")

    print(f"\n  ── Function distribution ──")
    print(f"    {'Function':35s} {'Count':>8} {'Source':>10}")
    print(f"    {'─' * 35} {'─' * 8} {'─' * 10}")
    for fn, n in sorted(fn_counter.items()):
        if fn in test_funcs:
            source = "test-only"
        elif fn in train_funcs:
            source = "train-schema"
        else:
            source = "abstention"
        print(f"    {fn:35s} {n:>8} {source:>10}")

    train_fn_in_set = fn_counter.keys() & train_funcs
    test_fn_in_set  = fn_counter.keys() & test_funcs
    print(f"\n  ── Summary ──")
    print(f"    Train-schema functions present : {len(train_fn_in_set)}")
    print(f"    Test-schema  functions present : {len(test_fn_in_set)}")
    if train_funcs - fn_counter.keys():
        print(f"    Train-schema MISSING           : {sorted(train_funcs - fn_counter.keys())}")
    if test_funcs - fn_counter.keys():
        print(f"    Test-schema  MISSING           : {sorted(test_funcs - fn_counter.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganise train/test split by function schema"
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/train_dataset_uncleaned.jsonl"),
        help="Current train JSONL (input)",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/test_dataset_uncleaned.jsonl"),
        help="Current test JSONL (input)",
    )
    parser.add_argument(
        "--train-schema",
        type=Path,
        default=Path("data/processed/function_schema_train.json"),
        help="Train function schema JSON",
    )
    parser.add_argument(
        "--test-schema",
        type=Path,
        default=Path("data/processed/function_schema_test.json"),
        help="Test function schema JSON",
    )
    parser.add_argument(
        "--out-train",
        type=Path,
        default=Path("data/processed/train_dataset_adjusted.jsonl"),
        help="Output train JSONL",
    )
    parser.add_argument(
        "--out-test",
        type=Path,
        default=Path("data/processed/test_dataset_adjusted.jsonl"),
        help="Output test JSONL",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of train-schema samples to keep in train (default 0.7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print report without writing output",
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────
    train_records = load_jsonl(args.train)
    test_records = load_jsonl(args.test)
    train_schema = json.loads(args.train_schema.read_text(encoding="utf-8"))
    test_schema = json.loads(args.test_schema.read_text(encoding="utf-8"))

    train_funcs: set = set(train_schema.keys())
    test_funcs: set = set(test_schema.keys())
    all_records = train_records + test_records

    print(f"Loaded {len(train_records)} train + {len(test_records)} test = {len(all_records)} total")
    print(f"Train-schema functions: {len(train_funcs)}")
    print(f"Test-schema  functions: {len(test_funcs)}")

    # ── Segregate ─────────────────────────────────────────────────────────
    test_only_records: list[dict] = []
    train_func_records: list[dict] = []
    abstention_records: list[dict] = []

    for rec in all_records:
        fn = rec.get("function_name", "")
        if fn == "none":
            abstention_records.append(rec)
        elif fn in test_funcs:
            test_only_records.append(rec)
        elif fn in train_funcs:
            train_func_records.append(rec)
        else:
            # Unknown function — skip (shouldn't happen after cleaning)
            pass

    print(f"\nSegregation:")
    print(f"  Test-schema function samples : {len(test_only_records)}")
    print(f"  Train-schema function samples: {len(train_func_records)}")
    print(f"  Abstention samples           : {len(abstention_records)}")

    # ── Split train-schema records ────────────────────────────────────────
    random.seed(SEED)
    random.shuffle(train_func_records)

    total_train_func = len(train_func_records)
    n_train = int(total_train_func * args.train_ratio)
    # Ensure at least 1 sample per function stays in train
    func_to_records: dict[str, list[dict]] = {}
    for rec in train_func_records:
        func_to_records.setdefault(rec["function_name"], []).append(rec)

    # Reserve 1 sample per function for train if possible
    guaranteed_train: list[dict] = []
    remaining: list[dict] = []
    for fn, recs in func_to_records.items():
        if len(recs) == 1:
            guaranteed_train.append(recs[0])
        else:
            guaranteed_train.append(recs[0])
            remaining.extend(recs[1:])

    # Adjust ratio
    random.shuffle(remaining)
    needed_from_remaining = max(0, n_train - len(guaranteed_train))
    train_from_funcs = guaranteed_train + remaining[:needed_from_remaining]
    test_from_funcs = remaining[needed_from_remaining:]

    # ── Split abstention records ──────────────────────────────────────────
    total_abstention = len(abstention_records)
    n_abstention_train = int(total_abstention * args.train_ratio)
    random.shuffle(abstention_records)
    abstention_train = abstention_records[:n_abstention_train]
    abstention_test = abstention_records[n_abstention_train:]

    # ── Assemble final sets ───────────────────────────────────────────────
    new_train = train_from_funcs + abstention_train
    new_test = test_only_records + test_from_funcs + abstention_test

    # ── Ensure every train-schema function appears in test at least once ──
    train_fn_in_train = {r["function_name"] for r in train_from_funcs}
    test_fn_in_test = {r.get("function_name") for r in new_test}
    for fn in sorted(train_fn_in_train):
        if fn != "none" and fn not in test_fn_in_test:
            for rec in train_func_records:
                if rec["function_name"] == fn:
                    new_test.append(rec)
                    test_fn_in_test.add(fn)
                    # Remove from train if it was the only copy
                    if rec in new_train:
                        new_train.remove(rec)
                    break

    # ── Write ─────────────────────────────────────────────────────────────
    if not args.dry_run:
        args.out_train.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.out_train, new_train)
        write_jsonl(args.out_test, new_test)
        print(f"\n  -> Wrote {len(new_train)} samples to {args.out_train}")
        print(f"  -> Wrote {len(new_test)}  samples to {args.out_test}")

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SPLIT SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Train ratio: {args.train_ratio:.0%}")
    print(f"  Train samples: {len(new_train)} ({100 * len(new_train) / (len(new_train) + len(new_test)):.1f}%)")
    print(f"  Test samples : {len(new_test)} ({100 * len(new_test) / (len(new_train) + len(new_test)):.1f}%)")

    print_report("TRAIN SET", new_train, train_funcs, test_funcs)
    print_report("TEST SET", new_test, train_funcs, test_funcs)

    # ── Validation check ─────────────────────────────────────────────────
    train_fn_set = {r["function_name"] for r in new_train}
    test_fn_set = {r["function_name"] for r in new_test}

    violations = train_fn_set & test_funcs
    if violations:
        print(f"\n  ⚠  WARNING: Train set contains test-only functions: {sorted(violations)}")
    else:
        print(f"\n  ✓  Train set has NO test-only functions")

    missing_from_test = (train_funcs - test_fn_set)
    if missing_from_test:
        print(f"  ⚠  Test set is missing train-schema functions: {sorted(missing_from_test)}")
    else:
        print(f"  ✓  Test set covers ALL functions (train + test schemas)")


if __name__ == "__main__":
    main()
