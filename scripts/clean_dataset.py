#!/usr/bin/env python3
"""
clean_dataset.py — Validate, standardise, and deduplicate toolformer JSONL datasets.

Usage:
    python scripts/clean_dataset.py \\
        --input-train data/generated/v1.0/train_dataset_cleaned.jsonl \\
        --input-test data/generated/v1.0/test_dataset_cleaned.jsonl \\
        --function-library data/generated/v1.0/function_library.json \\
        --output-dir data/processed/clean

Validates every sample against:
  - workflow_type in {single_call, parallel, abstention}
  - call count matches workflow type (1 / >=2 / 0)
  - every call has a known function name
  - every call's arguments match the function schema (keys, required params, enums, types)

Standardises every valid sample to the canonical schema:
  - strips top-level function_name (legacy)
  - strips workflow inside ground_truth (legacy)

Drops invalid samples and writes a JSON report with detailed stats.

stdlib only — no external dependencies.
"""

from email.policy import default
import json
import sys
import argparse
from pathlib import Path
from collections import Counter
from typing import Any


# ── Ground truth parser ────────────────────────────────────────────────


def _parse_ground_truth(gt: Any) -> dict | None:
    """Normalise ground_truth from dict or JSON string to a dict with 'calls' list.

    Args:
        gt: Raw ground_truth value (dict or JSON string).

    Returns:
        Parsed dict, or None if unusable.
    """
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(gt, dict):
        return None
    return gt


# ── Schema standardisation ────────────────────────────────────────────


def _standardize_sample(sample: dict) -> dict:
    """Strip legacy fields (function_name, ground_truth.workflow) in-place.

    Args:
        sample: Dataset sample dict to standardize in-place.

    Returns:
        The same sample dict for convenience.
    """
    sample.pop("function_name", None)
    gt = sample.get("ground_truth")
    if isinstance(gt, dict):
        gt.pop("workflow", None)
    return sample


# ── Core validation ────────────────────────────────────────────────────


def validate_sample(sample: dict, function_library: dict) -> tuple[bool, str]:
    """Validate a dataset sample against the function library.

    Checks workflow_type, call count, function existence, argument schema
    conformance (required params, enums, types). Returns machine-readable
    rejection reasons for report aggregation.

    Args:
        sample: Dataset sample dict.
        function_library: Dict of function_name → schema.

    Returns:
        Tuple of (is_valid: bool, rejection_reason: str).
        Empty rejection_reason string means valid.
    """
    # 1. workflow_type must be one of the three allowed values
    wt = sample.get("workflow_type")
    if wt not in ("single_call", "parallel", "abstention"):
        return False, "invalid_workflow_type"

    # 2. ground_truth must be a dict (or a JSON string that parses to one)
    gt = sample.get("ground_truth")
    gt = _parse_ground_truth(gt)
    if gt is None:
        return False, "missing_ground_truth"

    # 3. calls must be a list
    calls = gt.get("calls")
    if not isinstance(calls, list):
        return False, "calls_not_list"

    # 4-6. Call count must match workflow type
    if wt == "single_call" and len(calls) != 1:
        return False, "single_call_wrong_call_count"
    if wt == "parallel" and len(calls) < 2:
        return False, "parallel_wrong_call_count"
    if wt == "abstention" and len(calls) != 0:
        return False, "abstention_has_calls"

    # If abstention with 0 calls, nothing more to check
    if wt == "abstention":
        return True, ""

    # 7-13. Validate every call
    for call in calls:
        # 7. function name is a non-empty string
        func_name = call.get("function")
        if not isinstance(func_name, str) or not func_name.strip():
            return False, "missing_function_name"

        # 8. function exists in the library
        schema = function_library.get(func_name)
        if schema is None:
            return False, "unknown_function"

        # 9. arguments is a dict
        args = call.get("arguments")
        if not isinstance(args, dict):
            return False, "arguments_not_dict"

        params = schema.get("parameters", {})

        # 10. Every arg key must be a known parameter
        for key in args:
            if key not in params:
                return False, "unknown_argument_key"

        # 11-13. Validate against schema for each parameter
        for pname, pschema in params.items():
            # 11. Required params must have a non-null, non-empty value
            if pschema.get("required"):
                value = args.get(pname)
                if value is None or (isinstance(value, str) and not value.strip()):
                    return False, "missing_required_argument"

            # Only validate if the arg is present and non-null
            value = args.get(pname)
            if value is None:
                continue

            # 12. Enum values must match exactly
            enum_vals = pschema.get("enum")
            if enum_vals is not None and value not in enum_vals:
                return False, "enum_violation"

            # 13. Type check for integer types
            ptype = pschema.get("type", "string")
            if ptype in ("integer", "int"):
                if not isinstance(value, int):
                    if isinstance(value, str) and value.isdigit():
                        pass  # valid: string that looks like an int
                    else:
                        return False, "type_mismatch"

    return True, ""


# ── Pipeline ────────────────────────────────────────────────────────────


def clean_split(
    input_path: Path,
    function_library: dict,
) -> tuple[list[dict], dict[str, Any]]:
    """Process one JSONL file: parse, deduplicate, validate, standardize.

    Args:
        input_path: Path to JSONL file.
        function_library: Dict of function_name → schema.

    Returns:
        Tuple of (list of valid samples, stats dict with detailed counts).
    """
    stats: dict[str, Any] = {
        "total_lines": 0,
        "malformed_json_lines": [],
        "duplicate_ids": [],
        "dropped": Counter(),  # reason -> count
        "dropped_samples": {},  # reason -> list of sample IDs (first 20)
        "workflow_before": Counter(),
        "workflow_after": Counter(),
    }

    # 1. Read + parse JSONL
    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    stats["total_lines"] = len(raw_lines)

    samples: list[dict] = []
    for lineno, line in enumerate(raw_lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            stats["malformed_json_lines"].append(lineno)

    # 2. Deduplicate by id (keep first occurrence)
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for s in samples:
        sid = s.get("id")
        if isinstance(sid, str) and sid in seen_ids:
            stats["duplicate_ids"].append(sid)
        else:
            if isinstance(sid, str):
                seen_ids.add(sid)
            deduped.append(s)

    # 3. Workflow distribution before filtering
    for s in deduped:
        stats["workflow_before"][s.get("workflow_type", "unknown")] += 1

    # 4. Validate every sample
    valid: list[dict] = []
    for s in deduped:
        ok, reason = validate_sample(s, function_library)
        if ok:
            valid.append(s)
        else:
            stats["dropped"][reason] += 1
            sid = s.get("id", "<?>")
            if reason not in stats["dropped_samples"]:
                stats["dropped_samples"][reason] = []
            if len(stats["dropped_samples"][reason]) < 20:
                stats["dropped_samples"][reason].append(sid)

    # 5. Standardise schema (strip legacy fields)
    for s in valid:
        _standardize_sample(s)

    # 6. Workflow distribution after filtering
    for s in valid:
        stats["workflow_after"][s.get("workflow_type", "unknown")] += 1

    stats["output_count"] = len(valid)
    return valid, stats


def write_report(stats: dict[str, Any], output_dir: Path) -> None:
    """Write clean_report.json with human-readable and aggregated stats.

    Args:
        stats: Dict of split_name → split_stats from clean_split.
        output_dir: Directory to write the report file into.
    """
    report: dict[str, Any] = {}

    for split_name, split_stats in stats.items():
        report[split_name] = {
            "input": {
                "total_lines": split_stats["total_lines"],
                "malformed_json_lines_count": len(split_stats["malformed_json_lines"]),
                "malformed_json_lines": split_stats["malformed_json_lines"][:50],
                "duplicate_ids_count": len(split_stats["duplicate_ids"]),
                "duplicate_ids": split_stats["duplicate_ids"][:50],
            },
            "output": {
                "valid_samples": split_stats["output_count"],
            },
            "dropped": {
                "total": int(sum(split_stats["dropped"].values())),
                "by_reason": dict(
                    sorted(split_stats["dropped"].items(), key=lambda x: -x[1])
                ),
                "sample_ids_by_reason": {
                    k: v for k, v in split_stats["dropped_samples"].items()
                },
            },
            "workflow_distribution": {
                "before": dict(
                    sorted(split_stats["workflow_before"].items(), key=lambda x: -x[1])
                ),
                "after": dict(
                    sorted(split_stats["workflow_after"].items(), key=lambda x: -x[1])
                ),
            },
        }

    # Summary across splits
    total_in = sum(s["total_lines"] for s in stats.values())
    total_out = sum(s["output_count"] for s in stats.values())
    total_dropped = total_in - total_out
    report["summary"] = {
        "total_input_lines": total_in,
        "total_valid_output": total_out,
        "total_dropped": total_dropped,
        "dropped_rate": f"{total_dropped / max(total_in, 1) * 100:.2f}%",
    }

    report_path = output_dir / "clean_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  Report → {report_path}")


def print_summary(stats: dict[str, Any]) -> None:
    """Print a concise per-split and total summary to stdout.

    Args:
        stats: Dict of split_name → split_stats from clean_split.
    """
    grand_in = 0
    grand_out = 0
    for split_name, s in stats.items():
        in_ = s["total_lines"]
        out_ = s["output_count"]
        dropped = int(sum(s["dropped"].values()))
        grand_in += in_
        grand_out += out_

        print(f"\n  [{split_name}]")
        print(f"    Lines read:      {in_}")
        print(f"    Malformed JSON:  {len(s['malformed_json_lines'])}")
        print(f"    Duplicates:      {len(s['duplicate_ids'])}")
        print(f"    Dropped:         {dropped}")
        print(f"    Valid output:    {out_}")
        if s["dropped"]:
            print(f"    By reason:")
            for reason, count in s["dropped"].most_common():
                print(f"      {reason}: {count}")
        print(f"    Workflow before: {dict(s['workflow_before'])}")
        print(f"    Workflow after:  {dict(s['workflow_after'])}")

    print(f"\n  TOTAL: {grand_in} → {grand_out} (dropped {grand_in - grand_out})")
    print()


def main() -> None:
    """CLI entry point: validate, clean, and report on dataset JSONL files."""
    parser = argparse.ArgumentParser(
        description="Strict validation & cleanup of toolformer dataset JSONL files."
    )
    parser.add_argument(
        "--input-train",
        default=None,
        required=True,
        help="Path to training JSONL need to clean",
    )
    parser.add_argument(
        "--input-test",
        default=None,
        required=True,
        help="Path to test JSONL need to clean",
    )
    parser.add_argument(
        "--function-library",
        default=None,
        required=True,
        help="Path to function library JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        required=True,
        help="Output directory for cleaned files (default: data/cleaned)",
    )
    args = parser.parse_args()

    input_train = Path(args.input_train)
    input_test = Path(args.input_test)
    library_path = Path(args.function_library)
    output_dir = Path(args.output_dir)

    # Validate inputs exist
    for p in [input_train, input_test, library_path]:
        if not p.exists():
            print(f"Error: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Load function library
    with open(library_path, "r", encoding="utf-8") as f:
        function_library: dict = json.load(f)
    print(f"Loaded {len(function_library)} functions from {library_path}")

    # Process each split
    stats: dict[str, Any] = {}
    for split_name, input_path in [("train", input_train), ("test", input_test)]:
        print(f"\nProcessing {split_name}...")
        valid_samples, split_stats = clean_split(input_path, function_library)
        stats[split_name] = split_stats

        # Write output
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{split_name}_dataset_cleaned.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for s in valid_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(valid_samples)} samples → {output_path}")

    # Write report
    write_report(stats, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("  DATASET CLEANING REPORT")
    print("=" * 60)
    print_summary(stats)


if __name__ == "__main__":
    main()
