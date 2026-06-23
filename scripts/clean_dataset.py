#!/usr/bin/env python
"""
clean_dataset.py
─────────────────
Remove invalid data samples from a JSONL dataset so the output passes
validate_dataset.py with zero errors AND zero warnings.

Samples are removed when ANY of the following is true:

  Top-level integrity
    - required field (id, query, workflow_type, function_name, ground_truth) missing
    - query is empty or whitespace-only
    - duplicate id
    - split is not "train" or "test"

  ground_truth integrity
    - ground_truth is not a dict
    - workflow_type mismatches ground_truth["workflow"]
    - top-level function_name != ground_truth["function"] (non-abstention)

  Function existence
    - primary function or any call function not found in schema or library

  Arguments
    - arguments is not a dict
    - missing required parameter
    - value has wrong type
    - value violates enum, min, or max constraints
    - unknown parameter (not in schema) present

  Abstention
    - ground_truth["function"] is not None
    - ground_truth["arguments"] is not a dict
    - missing refusal_message or reasoning

  Parallel / Sequential
    - calls is missing, not a list, or empty
    - any call is not a dict or missing function/arguments
    - sequential call missing "step"

  Single-call
    - ground_truth has "calls" field

  retrieved_functions
    - not a list
    - primary function missing from list (non-abstention)

  retrieved_argument_values (if present)
    - not a dict
    - any value is not a list
    - any score outside [0, 1]

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
import re
import sys
from pathlib import Path


UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)

REQUIRED_TOP_FIELDS = ("id", "query", "workflow_type", "function_name", "ground_truth")
VALID_SPLITS = ("train", "test")
VALID_WORKFLOWS = ("single_call", "parallel", "sequential", "abstention")


def load_schema(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict]:
    samples: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_jsonl(samples: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _validate_type(value, expected_type: str) -> bool:
    if expected_type in ("string",):
        return isinstance(value, str)
    elif expected_type in ("number",):
        return isinstance(value, (int, float))
    elif expected_type in ("integer", "int"):
        return isinstance(value, int) and not isinstance(value, bool)
    elif expected_type in ("boolean", "bool"):
        return isinstance(value, bool)
    return True


def _check_args(
    arguments: dict, func_name: str, schema: dict
) -> str | None:
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

    for pname in arguments:
        if pname not in params:
            return f"unknown param '{pname}' for '{func_name}'"

    return None


def is_sample_valid(
    sample: dict, schema: dict, full_library: dict | None = None,
    seen_ids: set | None = None,
) -> tuple[bool, str]:
    if seen_ids is None:
        seen_ids = set()
    sid = sample.get("id", "unknown")

    # ── Required top-level fields ──────────────────────────────────────────
    for field in REQUIRED_TOP_FIELDS:
        if field not in sample:
            return False, f"sample {sid}: missing top-level field '{field}'"

    # ── ID uniqueness ──────────────────────────────────────────────────────
    if sid in seen_ids:
        return False, f"sample {sid}: duplicate id"
    seen_ids.add(sid)

    # ── UUID format ────────────────────────────────────────────────────────
    if sid and not UUID_RE.match(sid):
        return False, f"sample {sid}: id is not a valid UUID"

    # ── Query ──────────────────────────────────────────────────────────────
    query = sample.get("query", "")
    if not query or not isinstance(query, str) or not query.strip():
        return False, f"sample {sid}: query is missing or empty"
    if not query.strip():
        return False, f"sample {sid}: query is whitespace-only"

    # ── Split ──────────────────────────────────────────────────────────────
    split = sample.get("split")
    if split is not None and split not in VALID_SPLITS:
        return False, f"sample {sid}: unexpected split value '{split}'"

    # ── Workflow type ──────────────────────────────────────────────────────
    workflow_type = sample.get("workflow_type")
    if workflow_type not in VALID_WORKFLOWS:
        return False, f"sample {sid}: unknown workflow_type '{workflow_type}'"

    # ── retrieved_functions ────────────────────────────────────────────────
    retrieved = sample.get("retrieved_functions", [])
    if not isinstance(retrieved, list):
        return False, f"sample {sid}: 'retrieved_functions' is not a list"

    primary_func = sample.get("function_name")
    if (
        primary_func
        and workflow_type != "abstention"
        and primary_func not in retrieved
    ):
        return (
            False,
            f"sample {sid}: primary function '{primary_func}' not in "
            f"retrieved_functions {retrieved}",
        )

    # ── ground_truth ───────────────────────────────────────────────────────
    gt = sample.get("ground_truth")
    if not isinstance(gt, dict):
        return False, f"sample {sid}: missing or non-dict ground_truth"

    # Workflow mismatch
    if workflow_type != gt.get("workflow"):
        return (
            False,
            f"sample {sid}: workflow_type '{workflow_type}' != "
            f"ground_truth['workflow'] '{gt.get('workflow')}'",
        )

    # ── Function name match ────────────────────────────────────────────────
    top_func_name = sample.get("function_name")
    gt_func = gt.get("function")
    if workflow_type != "abstention":
        if top_func_name and gt_func and top_func_name != gt_func:
            return (
                False,
                f"sample {sid}: top-level 'function_name'='{top_func_name}' != "
                f"ground_truth['function']='{gt_func}'",
            )

    # ── Abstention ─────────────────────────────────────────────────────────
    if workflow_type == "abstention":
        if gt_func is not None:
            return (
                False,
                f"sample {sid}: abstention but ground_truth['function'] is not None",
            )
        if not isinstance(gt.get("arguments"), dict):
            return (
                False,
                f"sample {sid}: abstention ground_truth['arguments'] is not a dict",
            )
        if not gt.get("refusal_message"):
            return (
                False,
                f"sample {sid}: abstention missing 'refusal_message'",
            )
        if not gt.get("reasoning"):
            return (
                False,
                f"sample {sid}: abstention missing 'reasoning'",
            )
        return True, ""

    # ── Non-abstention: check primary function ─────────────────────────────
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

    # ── Check primary function arguments ───────────────────────────────────
    arguments = gt.get("arguments", {})
    if not isinstance(arguments, dict):
        return (
            False,
            f"sample {sid}: ground_truth['arguments'] is not a dict",
        )

    err = _check_args(arguments, func_name, func_schema)
    if err is not None:
        return False, f"sample {sid}: {err}"

    # ── Single_call: must NOT have "calls" ─────────────────────────────────
    if workflow_type == "single_call":
        if "calls" in gt:
            return (
                False,
                f"sample {sid}: single_call ground_truth should not have 'calls'",
            )

    # ── Parallel / sequential: check calls ─────────────────────────────────
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

            # Sequential must have "step"
            if workflow_type == "sequential" and "step" not in call:
                return (
                    False,
                    f"sample {sid}: calls[{i}] missing 'step' for sequential",
                )

    # ── retrieved_argument_values structure (if present) ───────────────────
    arg_vals = sample.get("retrieved_argument_values")
    if arg_vals is not None:
        if not isinstance(arg_vals, dict):
            return (
                False,
                f"sample {sid}: 'retrieved_argument_values' is not a dict",
            )
        for param, matches in arg_vals.items():
            if not isinstance(matches, list):
                return (
                    False,
                    f"sample {sid}: retrieved_argument_values['{param}'] "
                    f"is not a list",
                )
            for mi, m in enumerate(matches):
                if isinstance(m, dict):
                    score = m.get("score")
                    if score is not None and isinstance(score, (int, float)):
                        if score < 0 or score > 1:
                            return (
                                False,
                                f"sample {sid}: "
                                f"retrieved_argument_values['{param}'][{mi}] "
                                f"score={score} outside [0, 1]",
                            )

    return True, ""


def main():
    parser = argparse.ArgumentParser(
        description="Remove invalid samples from JSONL dataset"
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
        seen_ids: set[str] = set()

        for s in samples:
            ok, reason = is_sample_valid(s, schema, full_library, seen_ids)
            if ok:
                valid.append(s)
            else:
                removed.append((s.get("id", "unknown"), reason))

        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")
        print(f"  Total  : {len(samples)}")
        print(f"  Kept   : {len(valid)}")
        print(f"  Removed: {len(removed)}")
        if removed:
            print(f"\n  Removed samples:")
            for rid, reason in removed:
                print(f"    {rid}: {reason}")

        if not args.dry_run and out_path:
            write_jsonl(valid, out_path)
            print(f"\n  -> Wrote {len(valid)} samples to {out_path}")

        return len(removed)

    total_removed = 0

    if args.dataset and args.schema:
        schema = load_schema(args.schema)
        samples = load_jsonl(args.dataset)
        total_removed += _process(
            samples, schema, f"Cleaning: {args.dataset}", args.output
        )
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
        print("\n✓  No invalid samples found - dataset is clean.")
    else:
        print(f"\n  Total removed across all datasets: {total_removed}")

    sys.exit(0)


if __name__ == "__main__":
    main()
