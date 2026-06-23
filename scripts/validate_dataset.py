#!/usr/bin/env python
"""
validate_dataset.py
────────────────────
Validate ground truth correctness and data integrity of train/test datasets.

Checks performed:
  1. Required top-level fields present
  2. ground_truth structural integrity per workflow type
  3. workflow_type matches ground_truth["workflow"]
  4. function_name matches ground_truth["function"] (non-abstention)
  5. All function names exist in the function library
  6. Required parameters present and values respect constraints (types, enums, min/max)
  7. No duplicate IDs
  8. query is non-empty
  9. retrieved_functions contain the primary function

Usage:
    python scripts/validate_dataset.py \
        --dataset data/processed/train_dataset.jsonl \
        --schema data/processed/function_schema_train.json

    python scripts/validate_dataset.py \
        --dataset data/processed/test_dataset.jsonl \
        --schema data/processed/function_schema_test.json

    python scripts/validate_dataset.py \
        --train data/processed/train_dataset.jsonl \
        --train-schema data/processed/function_schema_train.json \
        --test data/processed/test_dataset.jsonl \
        --test-schema data/processed/function_schema_test.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import jsonlines

# ── Severity levels ──────────────────────────────────────────────────────────

class Result:
    def __init__(self):
        self.errors: list[str]   = []
        self.warnings: list[str] = []

    def error(self, sample_id: str, msg: str) -> None:
        self.errors.append(f"[ERROR] {sample_id}: {msg}")

    def warning(self, sample_id: str, msg: str) -> None:
        self.warnings.append(f"[WARN]  {sample_id}: {msg}")

    def ok(self) -> bool:
        return len(self.errors) == 0

    def print_summary(self, label: str) -> None:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        if not self.errors and not self.warnings:
            print("  All checks passed.")
            return
        for item in self.errors:
            print(f"  {item}")
        for item in self.warnings:
            print(f"  {item}")
        print(f"  ── {len(self.errors)} error(s), {len(self.warnings)} warning(s)")


UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def load_jsonl(path: str) -> list[dict]:
    samples: list[dict] = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            samples.append(obj)
    return samples


def load_schema(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_type(value, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    elif expected_type == "number":
        return isinstance(value, (int, float))
    elif expected_type == "integer":
        return isinstance(value, int)
    elif expected_type == "boolean":
        return isinstance(value, bool)
    return True


def validate_value(
    value, param_name: str, param_info: dict, constraints: dict
) -> list[str]:
    errs: list[str] = []
    expected_type   = param_info.get("type", "string")
    if value is not None and not validate_type(value, expected_type):
        errs.append(
            f"param '{param_name}' expected type '{expected_type}', got {type(value).__name__}"
        )

    con = constraints.get(param_name, {})
    if isinstance(value, (int, float)):
        if "min" in con and value < con["min"]:
            errs.append(f"param '{param_name}'={value} below min={con['min']}")
        if "max" in con and value > con["max"]:
            errs.append(f"param '{param_name}'={value} above max={con['max']}")
    if "enum" in con and value not in con["enum"]:
        errs.append(
            f"param '{param_name}'={value!r} not in allowed values {con['enum']}"
        )
    return errs


def validate_ground_truth(
    sample: dict,
    schema: dict,
    result: Result,
    full_library: dict | None = None,
) -> None:
    sid = sample.get("id", "unknown")
    gt  = sample.get("ground_truth")
    if not isinstance(gt, dict):
        result.error(sid, "missing or non-dict 'ground_truth'")
        return

    workflow_type = sample.get("workflow_type")
    gt_workflow   = gt.get("workflow")

    if workflow_type != gt_workflow:
        result.error(
            sid,
            f"workflow_type mismatch: top-level='{workflow_type}', ground_truth['workflow']='{gt_workflow}'",
        )

    # ── Workflow-specific checks ──────────────────────────────────────────
    if workflow_type == "abstention":
        if gt.get("function") is not None:
            result.error(sid, "abstention ground_truth['function'] should be None")
        if not isinstance(gt.get("arguments"), dict):
            result.error(sid, "abstention ground_truth['arguments'] should be a dict")
        if not gt.get("refusal_message"):
            result.warning(sid, "abstention ground_truth missing 'refusal_message'")
        if not gt.get("reasoning"):
            result.warning(sid, "abstention ground_truth missing 'reasoning'")
        return

    # ── Non-abstention checks ─────────────────────────────────────────────
    func_name = gt.get("function")
    if not func_name or not isinstance(func_name, str):
        result.error(sid, "ground_truth['function'] is missing or not a string")
        return

    top_func_name = sample.get("function_name")
    if top_func_name and top_func_name != func_name:
        result.warning(
            sid,
            f"top-level 'function_name'='{top_func_name}' != ground_truth['function']='{func_name}'",
        )

    # Check function exists in schema
    if func_name not in schema:
        # If full_library provided, check there too
        if full_library and func_name in full_library:
            pass
        else:
            result.error(
                sid, f"function '{func_name}' not found in schema or library"
            )
        return

    func_schema = schema[func_name]
    params      = func_schema.get("parameters", {})
    constraints = func_schema.get("constraints", {})
    arguments   = gt.get("arguments", {})
    if not isinstance(arguments, dict):
        result.error(sid, "ground_truth['arguments'] is not a dict")
        return

    # Check required params present
    for pname, pinfo in params.items():
        if pinfo.get("required", False) and pname not in arguments:
            result.error(
                sid,
                f"missing required param '{pname}' for function '{func_name}'",
            )
        elif pname in arguments:
            sub_errs = validate_value(
                arguments[pname], pname, pinfo, constraints
            )
            for e in sub_errs:
                result.error(sid, f"{e}")

    # Warn about unknown params
    for pname in arguments:
        if pname not in params:
            result.warning(
                sid,
                f"unknown param '{pname}' for function '{func_name}'",
            )

    # ── Workflow-specific sub-checks ──────────────────────────────────────
    if workflow_type in ("parallel", "sequential"):
        calls = gt.get("calls")
        if not isinstance(calls, list) or len(calls) == 0:
            result.error(
                sid, f"{workflow_type} ground_truth must have non-empty 'calls' list"
            )
        else:
            for i, call in enumerate(calls):
                if not isinstance(call, dict):
                    result.error(sid, f"calls[{i}] is not a dict")
                    continue
                cfunc = call.get("function")
                cargs = call.get("arguments")
                if not cfunc:
                    result.error(sid, f"calls[{i}] missing 'function'")
                if not isinstance(cargs, dict):
                    result.error(sid, f"calls[{i}] missing/malformed 'arguments'")
                if cfunc and cfunc not in schema:
                    if not (full_library and cfunc in full_library):
                        result.error(
                            sid,
                            f"calls[{i}] function '{cfunc}' not in schema",
                        )
                if cfunc and cfunc in schema and isinstance(cargs, dict):
                    c_params      = schema[cfunc].get("parameters", {})
                    c_constraints = schema[cfunc].get("constraints", {})
                    for pname, pinfo in c_params.items():
                        if pinfo.get("required", False) and pname not in cargs:
                            result.error(
                                sid,
                                f"calls[{i}] missing required param '{pname}' for '{cfunc}'",
                            )
                        elif pname in cargs:
                            sub_errs = validate_value(
                                cargs[pname], pname, pinfo, c_constraints
                            )
                            for e in sub_errs:
                                result.error(sid, f"calls[{i}] {e}")
                    for pname in cargs:
                        if pname not in c_params:
                            result.warning(
                                sid,
                                f"calls[{i}] unknown param '{pname}' for '{cfunc}'",
                            )

        if workflow_type == "sequential":
            for i, call in enumerate(calls):
                if "step" not in call:
                    result.warning(sid, f"calls[{i}] missing 'step' for sequential")

    if workflow_type == "single_call":
        if "calls" in gt:
            result.warning(sid, "single_call ground_truth should not have 'calls'")


def validate_sample(
    sample: dict,
    schema: dict,
    result: Result,
    seen_ids: set,
    full_library: dict | None = None,
) -> None:
    sid = sample.get("id", "unknown")

    # ── Required top-level fields ─────────────────────────────────────────
    for field in ("id", "query", "workflow_type", "function_name", "ground_truth"):
        if field not in sample:
            result.error(sid, f"missing top-level field '{field}'")

    # ── ID uniqueness and format ──────────────────────────────────────────
    sid_val = sample.get("id", "")
    if sid_val in seen_ids:
        result.error(sid_val, f"duplicate id '{sid_val}'")
    seen_ids.add(sid_val)
    if sid_val and not UUID_RE.match(sid_val):
        result.warning(sid_val, f"id '{sid_val}' is not a valid UUID")

    # ── Query ─────────────────────────────────────────────────────────────
    query = sample.get("query", "")
    if not query or not isinstance(query, str) or not query.strip():
        result.error(sid, "query is missing or empty")
    elif not query.strip():
        result.error(sid, "query is whitespace-only")

    # ── Split ─────────────────────────────────────────────────────────────
    split = sample.get("split")
    if split and split not in ("train", "test"):
        result.warning(sid, f"unexpected split value '{split}'")

    # ── retrieved_functions ───────────────────────────────────────────────
    retrieved = sample.get("retrieved_functions", [])
    if not isinstance(retrieved, list):
        result.error(sid, "'retrieved_functions' is not a list")
    else:
        primary = sample.get("function_name")
        if primary and primary not in retrieved and sample.get("workflow_type") != "abstention":
            result.warning(
                sid,
                f"primary function '{primary}' not in retrieved_functions {retrieved}",
            )

    # ── ground_truth ──────────────────────────────────────────────────────
    validate_ground_truth(sample, schema, result, full_library)

    # ── retrieved_argument_values (if present) ────────────────────────────
    arg_vals = sample.get("retrieved_argument_values")
    if arg_vals is not None:
        if not isinstance(arg_vals, dict):
            result.error(sid, "'retrieved_argument_values' is not a dict")
        else:
            for param, matches in arg_vals.items():
                if not isinstance(matches, list):
                    result.warning(
                        sid,
                        f"retrieved_argument_values['{param}'] is not a list",
                    )
                    continue
                for mi, m in enumerate(matches):
                    if not isinstance(m, dict):
                        continue
                    score = m.get("score")
                    if score is not None and isinstance(score, (int, float)):
                        if score < 0 or score > 1:
                            result.warning(
                                sid,
                                f"retrieved_argument_values['{param}'][{mi}] score={score} outside [0, 1]",
                            )


def validate_cross_split(
    train_samples: list[dict],
    test_samples: list[dict],
    train_schema: dict,
    test_schema: dict,
    full_lib: dict | None = None,
) -> dict | None:
    print(f"\n{'='*70}")
    print("  CROSS-SPLIT CONSISTENCY CHECKS")
    print(f"{'='*70}")

    # Check primary function overlap between train/test splits.
    # The train/test split is defined by the primary function_name, not
    # by functions referenced inside ground_truth["calls"] (which can
    # reference any function from the full library).
    train_primaries = set()
    for s in train_samples:
        fn = s.get("function_name")
        if fn and fn != "none":
            train_primaries.add(fn)

    test_primaries = set()
    for s in test_samples:
        fn = s.get("function_name")
        if fn and fn != "none":
            test_primaries.add(fn)

    overlap = train_primaries & test_primaries
    if overlap:
        print(f"  [WARN]  Primary functions appearing in BOTH train and test: {sorted(overlap)}")
    else:
        print("  ✓       No primary function overlap between train and test splits")

    # Collect all functions referenced in calls across both splits
    all_train_funcs = set(train_primaries)
    all_test_funcs  = set(test_primaries)
    for s in train_samples:
        gt = s.get("ground_truth", {})
        if isinstance(gt, dict):
            calls = gt.get("calls", [])
            if isinstance(calls, list):
                for c in calls:
                    if isinstance(c, dict) and c.get("function"):
                        all_train_funcs.add(c["function"])
    for s in test_samples:
        gt = s.get("ground_truth", {})
        if isinstance(gt, dict):
            calls = gt.get("calls", [])
            if isinstance(calls, list):
                for c in calls:
                    if isinstance(c, dict) and c.get("function"):
                        all_test_funcs.add(c["function"])

    # Check test schema functions are subset of full library (if available)
    if full_lib:
        missing_from_lib = all_test_funcs - set(full_lib.keys())
        if missing_from_lib:
            print(f"  [WARN]  Test functions referenced but not in full library: {sorted(missing_from_lib)}")

        # Which test functions are NOT in the test schema (expected if calls reference train functions)?
        test_not_in_schema = all_test_funcs - set(test_schema.keys())
        if test_not_in_schema and full_lib:
            in_full = test_not_in_schema & set(full_lib.keys())
            if in_full:
                print(f"  [INFO]  Test functions not in test schema (referenced via calls from full lib): {sorted(in_full)}")

    return full_lib


def main():
    parser = argparse.ArgumentParser(description="Validate dataset ground truth")
    parser.add_argument("--dataset", help="Path to a single JSONL dataset file")
    parser.add_argument("--schema", help="Path to the function schema JSON file")
    parser.add_argument("--train", help="Path to train JSONL dataset")
    parser.add_argument(
        "--train-schema", help="Path to train function schema JSON"
    )
    parser.add_argument("--test", help="Path to test JSONL dataset")
    parser.add_argument("--test-schema", help="Path to test function schema JSON")
    parser.add_argument(
        "--library",
        default="data/processed/function_library.json",
        help="Path to full function library JSON",
    )
    args = parser.parse_args()

    full_library = None
    if args.library and Path(args.library).exists():
        with open(args.library, encoding="utf-8") as f:
            full_library = json.load(f)

    # ── Single dataset mode ───────────────────────────────────────────────
    if args.dataset or args.schema:
        if not args.dataset or not args.schema:
            print("Error: --dataset and --schema must be used together")
            sys.exit(1)
        schema             = load_schema(args.schema)
        samples            = load_jsonl(args.dataset)
        result             = Result()
        seen_ids: set[str] = set()
        print(f"\nLoaded {len(samples)} samples, {len(schema)} functions in schema")
        for s in samples:
            validate_sample(s, schema, result, seen_ids, full_library)
        label = f"Validation: {args.dataset}"
        result.print_summary(label)
        sys.exit(0 if result.ok() else 1)

    # ── Dual dataset mode ─────────────────────────────────────────────────
    if args.train and args.train_schema and args.test and args.test_schema:
        train_schema  = load_schema(args.train_schema)
        test_schema   = load_schema(args.test_schema)
        train_samples = load_jsonl(args.train)
        test_samples  = load_jsonl(args.test)

        print(f"Train dataset: {len(train_samples)} samples, {len(train_schema)} functions")
        print(f"Test dataset:  {len(test_samples)} samples, {len(test_schema)} functions")

        # Cross-split checks
        fl = validate_cross_split(
            train_samples, test_samples, train_schema, test_schema, full_library
        )
        full_library = full_library or fl

        # Validate each split
        train_result       = Result()
        test_result        = Result()
        seen_ids: set[str] = set()

        for s in train_samples:
            validate_sample(s, train_schema, train_result, seen_ids, full_library)

        for s in test_samples:
            validate_sample(s, test_schema, test_result, seen_ids, full_library)

        train_result.print_summary("TRAIN DATASET VALIDATION")
        test_result.print_summary("TEST DATASET VALIDATION")

        if train_result.ok() and test_result.ok():
            print("\n✓  All datasets passed validation.")
            sys.exit(0)
        else:
            print("\n✗  Validation failed.")
            sys.exit(1)

    print("Error: provide either --dataset/--schema or all four of --train/--train-schema/--test/--test-schema")
    sys.exit(1)


if __name__ == "__main__":
    main()
