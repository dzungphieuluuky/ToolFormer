#!/usr/bin/env python
"""
clean_dataset.py
─────────────────
Remove samples with hallucinated functions or invalid argument values from
a JSONL dataset.  Uses the same validation logic as validate_dataset.py.

Samples are removed when:
  - primary function_name / ground_truth["function"] does not exist in schema
  - a function referenced in ground_truth["calls"] does not exist in schema
  - a required parameter is missing from ground_truth["arguments"]
  - an argument value violates type, enum, min, or max constraints

Usage:
    # Clean train_dataset only
    python scripts/clean_dataset.py \\
        --dataset data/processed/train_dataset.jsonl \\
        --schema data/processed/function_schema_train.json \\
        --output data/processed/train_dataset_clean.jsonl

    # Clean both splits at once
    python scripts/clean_dataset.py \\
        --train data/processed/train_dataset.jsonl \\
        --train-schema data/processed/function_schema_train.json \\
        --train-out data/processed/train_dataset_clean.jsonl \\
        --test data/processed/test_dataset.jsonl \\
        --test-schema data/processed/function_schema_test.json \\
        --test-out data/processed/test_dataset_clean.jsonl

    # Dry-run (report only, no output written)
    python scripts/clean_dataset.py \\
        --dataset data/processed/train_dataset.jsonl \\
        --schema data/processed/function_schema_train.json \\
        --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import jsonlines


def load_schema(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict]:
    samples: list[dict] = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            samples.append(obj)
    return samples


def write_jsonl(samples: list[dict], path: str) -> None:
    with jsonlines.open(path, mode="w") as writer:
        for s in samples:
            writer.write(s)


def _validate_type(value, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    elif expected_type == "number":
        return isinstance(value, (int, float))
    elif expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    elif expected_type == "boolean":
        return isinstance(value, bool)
    return True


def _check_args(arguments: dict, func_name: str, schema: dict) -> str | None:
    params = schema.get("parameters", {})
    constraints = schema.get("constraints", {})

    for pname, pinfo in params.items():
        if pinfo.get("required", False) and pname not in arguments:
            return f"missing required param '{pname}' for '{func_name}'"
        if pname in arguments:
            val = arguments[pname]
            expected_type = pinfo.get("type", "string")
            if not _validate_type(val, expected_type):
                return (
                    f"param '{pname}' for '{func_name}' expected type "
                    f"'{expected_type}', got {type(val).__name__}"
                )

    for pname, con in constraints.items():
        if pname not in arguments:
            continue
        val = arguments[pname]
        if isinstance(val, (int, float)):
            if "min" in con and val < con["min"]:
                return f"param '{pname}'={val} below min={con['min']} for '{func_name}'"
            if "max" in con and val > con["max"]:
                return f"param '{pname}'={val} above max={con['max']} for '{func_name}'"
        if "enum" in con and val not in con["enum"]:
            return (
                f"param '{pname}'={val!r} not in allowed values "
                f"{con['enum']} for '{func_name}'"
            )

    return None


def is_sample_valid(
    sample: dict, schema: dict, full_library: dict | None = None
) -> tuple[bool, str]:
    sid = sample.get("id", "unknown")
    gt = sample.get("ground_truth")

    if not isinstance(gt, dict):
        return False, f"sample {sid}: missing or non-dict ground_truth"

    workflow_type = sample.get("workflow_type")

    # ── Workflow mismatch ────────────────────────────────────────────────
    if workflow_type != gt.get("workflow"):
        return (
            False,
            f"sample {sid}: workflow_type '{workflow_type}' != "
            f"ground_truth['workflow'] '{gt.get('workflow')}'",
        )

    # ── Abstention ───────────────────────────────────────────────────────
    if workflow_type == "abstention":
        if gt.get("function") is not None:
            return (
                False,
                f"sample {sid}: abstention but ground_truth['function'] is not None",
            )
        if not isinstance(gt.get("arguments"), dict):
            return (
                False,
                f"sample {sid}: abstention ground_truth['arguments'] is not a dict",
            )
        return True, ""

    # ── Non-abstention: check primary function ───────────────────────────
    func_name = gt.get("function")
    if not func_name or not isinstance(func_name, str):
        return (
            False,
            f"sample {sid}: ground_truth['function'] is missing or not a string",
        )

    func_schema = schema.get(func_name)
    if func_schema is None:
        if full_library and func_name in full_library:
            func_schema = full_library[func_name]
        else:
            return (
                False,
                f"sample {sid}: function '{func_name}' not found in schema or library",
            )

    # ── Check primary function arguments ─────────────────────────────────
    arguments = gt.get("arguments", {})
    if not isinstance(arguments, dict):
        return (
            False,
            f"sample {sid}: ground_truth['arguments'] is not a dict",
        )

    err = _check_args(arguments, func_name, func_schema)
    if err is not None:
        return False, f"sample {sid}: {err}"

    # ── Parallel / sequential: check calls ──────────────────────────────
    if workflow_type in ("parallel", "sequential"):
        calls = gt.get("calls")
        if not isinstance(calls, list) or len(calls) == 0:
            return (
                False,
                f"sample {sid}: {workflow_type} missing or empty 'calls'",
            )

        for i, call in enumerate(calls):
            if not isinstance(call, dict):
                return False, f"sample {sid}: calls[{i}] is not a dict"

            cfunc = call.get("function")
            if not cfunc:
                return False, f"sample {sid}: calls[{i}] missing 'function'"

            cfunc_schema = schema.get(cfunc)
            if cfunc_schema is None:
                if full_library and cfunc in full_library:
                    cfunc_schema = full_library[cfunc]
                else:
                    return (
                        False,
                        f"sample {sid}: calls[{i}] function '{cfunc}' "
                        f"not found in schema or library",
                    )

            cargs = call.get("arguments")
            if not isinstance(cargs, dict):
                return (
                    False,
                    f"sample {sid}: calls[{i}] missing or malformed 'arguments'",
                )

            err = _check_args(cargs, cfunc, cfunc_schema)
            if err is not None:
                return False, f"sample {sid}: calls[{i}] {err}"

    return True, ""


def main():
    parser = argparse.ArgumentParser(
        description="Remove samples with hallucinated functions or invalid args"
    )
    parser.add_argument("--dataset", help="Path to single JSONL dataset")
    parser.add_argument("--schema", help="Path to function schema for --dataset")
    parser.add_argument("--output", help="Output path for cleaned dataset")
    parser.add_argument("--train", help="Path to train JSONL dataset")
    parser.add_argument("--train-schema", help="Path to train function schema")
    parser.add_argument("--train-out", help="Output path for cleaned train dataset")
    parser.add_argument("--test", help="Path to test JSONL dataset")
    parser.add_argument("--test-schema", help="Path to test function schema")
    parser.add_argument("--test-out", help="Output path for cleaned test dataset")
    parser.add_argument(
        "--library",
        default="data/processed/function_library.json",
        help="Path to full function library JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report stats, do not write output",
    )
    args = parser.parse_args()

    full_library = None
    if args.library and Path(args.library).exists():
        with open(args.library, encoding="utf-8") as f:
            full_library = json.load(f)

    def _process(
        samples: list[dict], schema: dict, label: str, out_path: str | None
    ) -> int:
        valid: list[dict] = []
        removed: list[tuple[str, str]] = []

        for s in samples:
            ok, reason = is_sample_valid(s, schema, full_library)
            if ok:
                valid.append(s)
            else:
                removed.append((s.get("id", "unknown"), reason))

        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        print(f"  Total  : {len(samples)}")
        print(f"  Kept   : {len(valid)}")
        print(f"  Removed: {len(removed)}")
        if removed:
            print(f"\n  Removed samples:")
            for rid, reason in removed:
                print(f"    {rid}: {reason}")

        if not args.dry_run and out_path:
            write_jsonl(valid, out_path)
            print(f"\n  → Wrote {len(valid)} samples to {out_path}")

        return len(removed)

    total_removed = 0

    # Single dataset mode
    if args.dataset and args.schema:
        schema = load_schema(args.schema)
        samples = load_jsonl(args.dataset)
        total_removed += _process(
            samples, schema, f"Cleaning: {args.dataset}", args.output
        )
    # Dual dataset mode
    elif args.train and args.train_schema and args.test and args.test_schema:
        train_schema = load_schema(args.train_schema)
        test_schema = load_schema(args.test_schema)
        train_samples = load_jsonl(args.train)
        test_samples = load_jsonl(args.test)

        if args.dry_run:
            total_removed += _process(
                train_samples, train_schema, "TRAIN DATASET", None
            )
            total_removed += _process(
                test_samples, test_schema, "TEST DATASET", None
            )
        else:
            if args.train_out:
                total_removed += _process(
                    train_samples, train_schema, "TRAIN DATASET", args.train_out
                )
            else:
                print("ERROR: --train-out required when --train is specified")
                sys.exit(1)
            if args.test_out:
                total_removed += _process(
                    test_samples, test_schema, "TEST DATASET", args.test_out
                )
            else:
                print("ERROR: --test-out required when --test is specified")
                sys.exit(1)
    else:
        print(
            "Error: provide either --dataset/--schema/--output or all of "
            "--train/--train-schema/--train-out/--test/--test-schema/--test-out"
        )
        sys.exit(1)

    if total_removed == 0:
        print("\n✓  No invalid samples found — dataset is clean.")
    else:
        print(f"\n  Total removed across all datasets: {total_removed}")

    sys.exit(0)


if __name__ == "__main__":
    main()
