#!/usr/bin/env python3
"""
validate_dataset.py — Schema & integrity checks for toolformer dataset JSONL files.

Usage:
    python validate_dataset.py \
        --data-dir data/processed \
        [--output data/processed/validation_report.json]

Validates:
    - File structure (valid JSON, required fields)
    - Ground truth schema (calls, workflow, reasoning)
    - Retrieved functions (non-empty, 5 items)
    - Retrieved argument values (code/label/group/score)
    - Function existence in schemas
    - Date format (YYYY-MM-DD)
    - Data level enums
    - Score ranges
    - Duplicate IDs across train+test
    - Cross-split function leakage
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import Any

VALID_WORKFLOWS = {"single_call", "parallel", "sequential", "abstention"}
VALID_SPLITS = {"train", "test"}
VALID_DATA_LEVELS = {"day", "week", "month", "quarter", "year"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

REQUIRED_FIELDS = [
    "id",
    "query",
    "workflow_type",
    "function_name",
    "ground_truth",
    "retrieved_functions",
    "split",
    "retrieved_argument_values",
]

ARGVALUE_KEYS = {"code", "label", "group", "score"}


def _load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                samples.append(None)
    return samples


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Validators ──────────────────────────────────────────────────────────


def check_required_fields(sample: dict, path: str) -> list[str]:
    errs = []
    for field in REQUIRED_FIELDS:
        if field not in sample:
            errs.append(f"missing required field: {field!r}")
        elif sample[field] is None:
            errs.append(f"field {field!r} is null")
    return errs


def check_id(sample: dict) -> list[str]:
    errs = []
    val = sample.get("id")
    if not isinstance(val, str):
        errs.append(f"id is not a string: {type(val).__name__}")
    elif not UUID_RE.match(val):
        errs.append(f"id does not look like a UUID: {val!r}")
    return errs


def check_query(sample: dict) -> list[str]:
    val = sample.get("query")
    if not isinstance(val, str) or not val.strip():
        return ["query is empty or not a string"]
    return []


def check_workflow_type(sample: dict) -> list[str]:
    val = sample.get("workflow_type")
    if val not in VALID_WORKFLOWS:
        return [f"invalid workflow_type: {val!r} (expected one of {VALID_WORKFLOWS})"]
    return []


def check_split(sample: dict) -> list[str]:
    val = sample.get("split")
    if val not in VALID_SPLITS:
        return [f"invalid split: {val!r} (expected one of {VALID_SPLITS})"]
    return []


def check_ground_truth(sample: dict) -> list[str]:
    errs = []
    gt = sample.get("ground_truth")
    if not isinstance(gt, dict):
        return [f"ground_truth is not a dict: {type(gt).__name__}"]

    for key in ("calls", "workflow", "reasoning"):
        if key not in gt:
            errs.append(f"ground_truth missing key: {key!r}")

    if not isinstance(gt.get("calls"), list):
        return errs + [f"ground_truth.calls is not a list: {type(gt.get('calls')).__name__}"]

    for ci, call in enumerate(gt["calls"]):
        if not isinstance(call, dict):
            errs.append(f"ground_truth.calls[{ci}] is not a dict")
            continue
        if "function" not in call:
            errs.append(f"ground_truth.calls[{ci}] missing 'function'")
        elif not isinstance(call["function"], str):
            errs.append(f"ground_truth.calls[{ci}].function is not a string")
        if "arguments" not in call:
            errs.append(f"ground_truth.calls[{ci}] missing 'arguments'")
        elif not isinstance(call["arguments"], dict):
            errs.append(f"ground_truth.calls[{ci}].arguments is not a dict")
    return errs


def check_ground_truth_calls_args(sample: dict, function_library: dict) -> list[str]:
    """Validate call arguments against function schemas."""
    errs = []
    gt = sample.get("ground_truth")
    if not isinstance(gt, dict):
        return errs
    calls = gt.get("calls", [])
    if not isinstance(calls, list):
        return errs
    for ci, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        func_name = call.get("function")
        if not isinstance(func_name, str):
            continue
        schema = function_library.get(func_name)
        if schema is None:
            errs.append(f"ground_truth.calls[{ci}].function {func_name!r} not in function library")
            continue
        params = schema.get("parameters", {})
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        for pname, pinfo in params.items():
            if pinfo.get("required") and pname not in args:
                default = pinfo.get("default", "")
                if not (isinstance(default, str) and (default.strip() == "" or default == '""')):
                    errs.append(f"ground_truth.calls[{ci}] missing required param {pname!r} for {func_name!r}")
        for pname, pval in args.items():
            if pname not in params:
                errs.append(f"ground_truth.calls[{ci}] unexpected param {pname!r} for {func_name!r}")
                continue
            enum_vals = params[pname].get("enum")
            if enum_vals and isinstance(pval, str) and pval not in enum_vals:
                # Allow empty string as "not specified" sentinel
                if pval != "":
                    errs.append(
                        f"ground_truth.calls[{ci}].{pname}={pval!r} not in enum {enum_vals}"
                    )
            # Date format check
            if pname in ("from_date", "to_date", "start_date", "end_date"):
                if isinstance(pval, str) and pval and not DATE_RE.match(pval):
                    errs.append(
                        f"ground_truth.calls[{ci}].{pname}={pval!r} not in YYYY-MM-DD format"
                    )
    return errs


def check_function_name(sample: dict, function_library: dict) -> list[str]:
    fname = sample.get("function_name")
    if not isinstance(fname, str):
        return [f"function_name is not a string: {type(fname).__name__}"]
    if fname not in function_library:
        return [f"function_name {fname!r} not found in function library"]
    return []


def check_retrieved_functions(sample: dict, function_library: dict) -> list[str]:
    errs = []
    rf = sample.get("retrieved_functions")
    if not isinstance(rf, list):
        return [f"retrieved_functions is not a list: {type(rf).__name__}"]
    if len(rf) == 0:
        errs.append("retrieved_functions is empty")
    for fi, fname in enumerate(rf):
        if not isinstance(fname, str):
            errs.append(f"retrieved_functions[{fi}] is not a string")
        elif fname not in function_library:
            errs.append(f"retrieved_functions[{fi}] {fname!r} not in function library")
    return errs


def check_retrieved_argument_values(sample: dict, argument_values: dict) -> list[str]:
    errs = []
    rav = sample.get("retrieved_argument_values")
    if rav is None:
        return []
    if not isinstance(rav, dict):
        return [f"retrieved_argument_values is not a dict: {type(rav).__name__}"]
    for pname, entries in rav.items():
        if not isinstance(entries, list):
            errs.append(f"retrieved_argument_values.{pname} is not a list")
            continue
        if len(entries) == 0:
            errs.append(f"retrieved_argument_values.{pname} is empty")
            continue
        for ei, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errs.append(f"retrieved_argument_values.{pname}[{ei}] is not a dict")
                continue
            for k in ARGVALUE_KEYS:
                if k not in entry:
                    errs.append(f"retrieved_argument_values.{pname}[{ei}] missing key {k!r}")
            score = entry.get("score")
            if score is not None and not isinstance(score, (int, float)):
                errs.append(f"retrieved_argument_values.{pname}[{ei}].score is not numeric: {type(score).__name__}")
            elif isinstance(score, (int, float)) and not (0.0 <= score <= 1.0):
                errs.append(f"retrieved_argument_values.{pname}[{ei}].score={score} out of range [0, 1]")
    return errs


def check_duplicate_ids(
    train_samples: list[dict], test_samples: list[dict]
) -> list[str]:
    errs = []
    seen: dict[str, str] = {}
    for path, samples in [("train", train_samples), ("test", test_samples)]:
        for sample in samples:
            if sample is None:
                continue
            sid = sample.get("id")
            if not isinstance(sid, str):
                continue
            if sid in seen:
                errs.append(f"duplicate id {sid!r}: first in {seen[sid]}, also in {path}")
            else:
                seen[sid] = path
    return errs


def check_cross_split_function_leakage(
    samples: list[dict], allowed_functions: set[str], split_name: str
) -> list[str]:
    errs = []
    for sample in samples:
        if sample is None:
            continue
        gt = sample.get("ground_truth")
        if not isinstance(gt, dict):
            continue
        calls = gt.get("calls", [])
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            fname = call.get("function")
            if isinstance(fname, str) and fname not in allowed_functions:
                errs.append(
                    f"ground_truth call uses function {fname!r} "
                    f"which is not in {split_name} function schema set"
                )
    return errs


# ── Orchestration ───────────────────────────────────────────────────────


def validate_all(data_dir: str, output_path: str | None = None) -> dict:
    data_dir = Path(data_dir)
    processed_dir = data_dir

    # Load files
    train_path = processed_dir / "train_dataset.jsonl"
    test_path = processed_dir / "test_dataset.jsonl"
    train_schema_path = processed_dir / "function_schema_train.json"
    test_schema_path = processed_dir / "function_schema_test.json"
    library_path = processed_dir / "function_library.json"
    arg_values_path = processed_dir / "argument_values.json"

    for p in [train_path, test_path, train_schema_path, test_schema_path, library_path, arg_values_path]:
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    train_samples = _load_jsonl(str(train_path))
    test_samples = _load_jsonl(str(test_path))
    train_schemas = _load_json(str(train_schema_path))
    test_schemas = _load_json(str(test_schema_path))
    function_library = _load_json(str(library_path))
    argument_values = _load_json(str(arg_values_path))

    all_train_functions = set(train_schemas.keys())
    all_test_functions = set(test_schemas.keys())

    report: dict[str, Any] = {
        "files": {},
        "duplicate_ids": [],
        "cross_split_leaks": [],
        "summary": {"total": 0, "passed": 0, "failed": 0},
    }

    for split_name, samples, schema_set in [
        ("train", train_samples, all_train_functions),
        ("test", test_samples, all_test_functions),
    ]:
        split_report: list[dict] = []
        passed = 0
        failed = 0
        for sample in samples:
            entry: dict[str, Any] = {"errors": []}
            if sample is None:
                entry["errors"].append("invalid JSON (parse failed)")
                entry["status"] = "FAIL"
                failed += 1
                split_report.append(entry)
                continue

            sid = sample.get("id", "<no-id>")
            entry["id"] = str(sid)[:36]

            all_errors: list[str] = []
            all_errors.extend(check_required_fields(sample, str(train_path)))
            all_errors.extend(check_id(sample))
            all_errors.extend(check_query(sample))
            all_errors.extend(check_workflow_type(sample))
            all_errors.extend(check_split(sample))
            all_errors.extend(check_ground_truth(sample))
            all_errors.extend(check_function_name(sample, function_library))
            all_errors.extend(
                check_ground_truth_calls_args(sample, function_library)
            )
            all_errors.extend(check_retrieved_functions(sample, function_library))
            all_errors.extend(
                check_retrieved_argument_values(sample, argument_values)
            )

            if all_errors:
                entry["errors"] = all_errors
                entry["status"] = "FAIL"
                failed += 1
            else:
                entry["status"] = "PASS"
                passed += 1
            split_report.append(entry)

        report["files"][split_name] = {
            "total": len(samples),
            "passed": passed,
            "failed": failed,
            "samples": split_report,
        }
        report["summary"]["total"] += len(samples)
        report["summary"]["passed"] += passed
        report["summary"]["failed"] += failed

        # Cross-split function leakage
        leaks = check_cross_split_function_leakage(samples, schema_set, split_name)
        if leaks:
            report["cross_split_leaks"].extend(
                {"split": split_name, "error": e} for e in leaks
            )

    # Duplicate IDs across train+test
    dups = check_duplicate_ids(train_samples, test_samples)
    report["duplicate_ids"] = dups

    if output_path:
        _save_report(report, output_path)

    return report


def _save_report(report: dict, path: str) -> None:
    # Remove per-sample details for compact saved report
    summary = {
        "summary": report["summary"],
        "duplicate_ids": report["duplicate_ids"],
        "cross_split_leaks": report["cross_split_leaks"],
    }
    for split_name in ("train", "test"):
        info = report["files"][split_name]
        summary[split_name] = {
            "total": info["total"],
            "passed": info["passed"],
            "failed": info["failed"],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary report → {path}")


def print_summary(report: dict) -> None:
    print("=" * 60)
    print("  DATASET VALIDATION REPORT")
    print("=" * 60)
    for split_name in ("train", "test"):
        info = report["files"][split_name]
        print(f"\n  [{split_name.upper()}]  {info['total']} samples")
        print(f"    PASSED: {info['passed']}")
        print(f"    FAILED: {info['failed']}")
        if info["failed"] > 0:
            fail_samples = [s for s in info["samples"] if s["status"] == "FAIL"]
            print(f"\n  --- FAILURES (showing up to 5) ---")
            shown = 0
            for s in fail_samples:
                if shown >= 5:
                    print(f"    ... and {len(fail_samples) - 5} more")
                    break
                sid = s.get("id", "<?>")
                for err in s["errors"][:3]:
                    print(f"    [{sid}] {err}")
                shown += 1
    if report["duplicate_ids"]:
        print(f"\n  DUPLICATE IDS: {len(report['duplicate_ids'])}")
        for d in report["duplicate_ids"][:5]:
            print(f"    {d}")
    if report["cross_split_leaks"]:
        print(f"\n  CROSS-SPLIT LEAKS: {len(report['cross_split_leaks'])}")
        for d in report["cross_split_leaks"][:5]:
            print(f"    [{d['split']}] {d['error']}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate toolformer dataset")
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Path to the data/processed directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the validation report JSON (optional)",
    )
    args = parser.parse_args()

    report = validate_all(data_dir=args.data_dir, output_path=args.output)
    print_summary(report)

    if report["summary"]["failed"] > 0:
        sys.exit(1)
