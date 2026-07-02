#!/usr/bin/env python3
"""
verify_token_counts.py — Verify dataset token counts against model limits.

Verifies that every sample in every training dataset file fits within the
model's configured token limits. This is a critical gate before training:
samples exceeding the limit will be silently truncated, potentially
corrupting the completion/response.

Usage:
    # Check all datasets in the default directory
    python scripts/verify_token_counts.py

    # Check a specific data directory
    python scripts/verify_token_counts.py \\
        --data-dir data/generated/v1.0_k5

    # Override model or limits (e.g. if using a different base model)
    python scripts/verify_token_counts.py \\
        --model-name unsloth/Qwen3-4B-Instruct-2507 \\
        --max-seq-length 8192 \\
        --max-prompt-length 7680

    # Verbose: show every sample that exceeds limits
    python scripts/verify_token_counts.py --verbose

Integrated into the data readiness pipeline:
    python scripts/validate_dataset.py --data-dir data/generated/v1.0_k5 && \\
    python scripts/verify_token_counts.py --data-dir data/generated/v1.0_k5 && \\
    echo "Dataset certified training-ready"

Dataset formats detected automatically:
  - SFT (.jsonl with 'text' field):              full text ≤ max_seq_length
  - GRPO/RC-GRPO (.jsonl with 'prompt' field):    prompt ≤ max_prompt_length
  - RCTP (.jsonl with 'prompt_messages' field):   formatted messages ≤ max_seq_length
  - Raw (.jsonl with 'query' field):              estimated prompt ≤ max_prompt_length (informational)
"""

import json
import sys
import argparse
from pathlib import Path
from collections import Counter
from statistics import median
from typing import Any


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSON Lines file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects. Malformed lines are skipped with a warning.
    """
    samples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [WARN] {path.name}:{lineno} — malformed JSON, skipping")
    return samples


def detect_format(samples: list[dict]) -> str:
    """Detect the dataset format from field presence in the first sample.

    Priority order: text > prompt_messages > prompt > query.

    Args:
        samples: List of dataset samples (must be non-empty).

    Returns:
        One of 'sft', 'stage2', 'rctp', 'raw'.
    """
    if not samples:
        return "unknown"
    keys = set(samples[0].keys())
    if "text" in keys:
        return "sft"
    if "prompt_messages" in keys:
        return "rctp"
    if "prompt" in keys:
        return "stage2"
    if "query" in keys:
        return "raw"
    return "unknown"


def _compute_stats(token_counts: list[int], limit: int) -> dict[str, Any]:
    """Compute token-count statistics for a list of samples.

    Args:
        token_counts: List of token counts per sample.
        limit: The maximum allowed token count.

    Returns:
        Dict with total, min, max, mean, median, over_limit, over_limit_pct,
        and worst sample info.
    """
    if not token_counts:
        return {
            "total": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0,
            "over_limit": 0,
            "over_limit_pct": 0.0,
        }
    over = sum(1 for t in token_counts if t > limit)
    return {
        "total": len(token_counts),
        "min": min(token_counts),
        "max": max(token_counts),
        "mean": round(sum(token_counts) / len(token_counts), 1),
        "median": sorted(token_counts)[len(token_counts) // 2],
        "over_limit": over,
        "over_limit_pct": round(100.0 * over / len(token_counts), 1),
    }


def check_sft_format(
    samples: list[dict],
    tokenizer: Any,
    max_seq_length: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Verify SFT-format samples (``text`` field) against max_seq_length.

    Tokenizes the full ``text`` field of each sample and checks it does not
    exceed ``max_seq_length``. Samples that exceed the limit will have their
    completion silently truncated during SFT training.

    Args:
        samples: List of samples with a ``text`` field.
        tokenizer: HuggingFace AutoTokenizer instance.
        max_seq_length: Maximum allowed total tokens (prompt + completion).
        verbose: If True, print details of every sample over the limit.

    Returns:
        Dict with token-count statistics and list of over-limit sample indices.
    """
    token_counts: list[int] = []
    over_limit_indices: list[int] = []

    for i, sample in enumerate(samples):
        text = sample.get("text", "")
        if not isinstance(text, str) or not text:
            token_counts.append(0)
            continue
        tokens = tokenizer.encode(text)
        token_counts.append(len(tokens))
        if len(tokens) > max_seq_length:
            over_limit_indices.append(i)
            if verbose:
                print(
                    f"  SFT sample {i}: {len(tokens)} tokens "
                    f"(limit {max_seq_length}) — exceeds by {len(tokens) - max_seq_length}"
                )

    stats = _compute_stats(token_counts, max_seq_length)
    stats["over_limit_indices"] = over_limit_indices[:20]  # cap reporting
    return stats


def check_stage2_format(
    samples: list[dict],
    tokenizer: Any,
    max_prompt_length: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Verify stage2-format samples (``prompt`` field) against max_prompt_length.

    Tokenizes the ``prompt`` field of each sample. During GRPO/RC-GRPO training,
    prompts exceeding max_prompt_length are truncated from the left, which can
    drop critical system instructions.

    Args:
        samples: List of samples with a ``prompt`` field.
        tokenizer: HuggingFace AutoTokenizer instance.
        max_prompt_length: Maximum allowed prompt tokens.
        verbose: If True, print details of every sample over the limit.

    Returns:
        Dict with token-count statistics and list of over-limit sample indices.
    """
    token_counts: list[int] = []
    over_limit_indices: list[int] = []

    for i, sample in enumerate(samples):
        prompt = sample.get("prompt", "")
        if not isinstance(prompt, str) or not prompt:
            token_counts.append(0)
            continue
        tokens = tokenizer.encode(prompt)
        token_counts.append(len(tokens))
        if len(tokens) > max_prompt_length:
            over_limit_indices.append(i)
            if verbose:
                print(
                    f"  Stage2 sample {i}: {len(tokens)} prompt tokens "
                    f"(limit {max_prompt_length}) — exceeds by {len(tokens) - max_prompt_length}"
                )

    stats = _compute_stats(token_counts, max_prompt_length)
    stats["over_limit_indices"] = over_limit_indices[:20]
    return stats


def check_rctp_format(
    samples: list[dict],
    tokenizer: Any,
    max_seq_length: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Verify RCTP-format samples (``prompt_messages`` + ``response_text``).

    Reconstructs the full chat-format text by applying the tokenizer's chat
    template to ``prompt_messages`` and appending ``response_text``, then
    checks total tokens against max_seq_length.

    Args:
        samples: List of samples with ``prompt_messages`` and ``response_text``.
        tokenizer: HuggingFace AutoTokenizer instance.
        max_seq_length: Maximum allowed total tokens.
        verbose: If True, print details of every sample over the limit.

    Returns:
        Dict with token-count statistics and list of over-limit sample indices.
    """
    token_counts: list[int] = []
    over_limit_indices: list[int] = []
    template_errors: int = 0

    for i, sample in enumerate(samples):
        messages = sample.get("prompt_messages")
        response = sample.get("response_text", "")

        if not isinstance(messages, list):
            token_counts.append(0)
            continue

        # Ensure every message has a role and content string
        safe_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", "")
            safe_messages.append({"role": role, "content": str(content) if content else ""})

        try:
            prompt_text = tokenizer.apply_chat_template(
                safe_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            template_errors += 1
            token_counts.append(0)
            continue

        full_text = prompt_text + (response if isinstance(response, str) else "")
        tokens = tokenizer.encode(full_text)
        token_counts.append(len(tokens))
        if len(tokens) > max_seq_length:
            over_limit_indices.append(i)
            if verbose:
                print(
                    f"  RCTP sample {i}: {len(tokens)} tokens "
                    f"(limit {max_seq_length}) — exceeds by {len(tokens) - max_seq_length}"
                )

    stats = _compute_stats(token_counts, max_seq_length)
    stats["over_limit_indices"] = over_limit_indices[:20]
    stats["template_errors"] = template_errors
    return stats


def check_raw_format(
    samples: list[dict],
    tokenizer: Any,
    max_prompt_length: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Estimate token counts for raw-format samples (``query`` field).

    This is informational only: raw datasets are not directly tokenized for
    training — they are first processed by ``build_datasets.py`` into SFT/GRPO
    formats. The estimate uses the ``query`` string as a rough proxy.

    Args:
        samples: List of samples with at least a ``query`` field.
        tokenizer: HuggingFace AutoTokenizer instance.
        max_prompt_length: Reference limit for informational reporting.
        verbose: If True, print details of every sample over the limit.

    Returns:
        Dict with estimated token-count statistics.
    """
    token_counts: list[int] = []
    over_limit_indices: list[int] = []

    for i, sample in enumerate(samples):
        query = sample.get("query", "")
        if not isinstance(query, str) or not query:
            token_counts.append(0)
            continue
        tokens = tokenizer.encode(query)
        token_counts.append(len(tokens))
        if len(tokens) > max_prompt_length:
            over_limit_indices.append(i)
            if verbose:
                print(
                    f"  Raw sample {i}: ~{len(tokens)} query tokens "
                    f"(reference limit {max_prompt_length})"
                )

    stats = _compute_stats(token_counts, max_prompt_length)
    stats["over_limit_indices"] = over_limit_indices[:20]
    return stats


FORMAT_DISPATCH = {
    "sft": ("max_seq_length", check_sft_format),
    "stage2": ("max_prompt_length", check_stage2_format),
    "rctp": ("max_seq_length", check_rctp_format),
    "raw": ("max_prompt_length", check_raw_format),
}

FORMAT_LABELS = {
    "sft": "SFT (text field)",
    "stage2": "GRPO/RC-GRPO (prompt field)",
    "rctp": "RCTP (prompt_messages + response_text)",
    "raw": "Raw dataset (query field, informational)",
}


def check_file(
    file_path: Path,
    tokenizer: Any,
    config: dict[str, Any],
    verbose: bool = False,
) -> dict[str, Any]:
    """Run token-count verification on a single dataset file.

    Detects the format automatically, dispatches to the appropriate checker,
    and returns a summary dict with per-format statistics.

    Args:
        file_path: Path to a JSONL dataset file.
        tokenizer: HuggingFace AutoTokenizer instance.
        config: Dict with ``max_seq_length`` and ``max_prompt_length`` keys.
        verbose: If True, print per-sample details for over-limit samples.

    Returns:
        Dict with file name, format, token-count stats, and verdict.
    """
    samples = load_jsonl(file_path)
    fmt = detect_format(samples)

    result: dict[str, Any] = {
        "file": file_path.name,
        "format": fmt,
        "format_label": FORMAT_LABELS.get(fmt, "Unknown"),
        "total_samples": len(samples),
    }

    if fmt == "unknown" or not samples:
        result["status"] = "SKIPPED (unknown format or empty)"
        result["stats"] = {}
        return result

    limit_key, checker = FORMAT_DISPATCH[fmt]
    limit = config[limit_key]

    stats = checker(samples, tokenizer, limit, verbose=verbose)
    limit_label = limit_key.replace("_", " ")
    result["stats"] = stats
    result["limit_type"] = limit_key
    result["limit_value"] = limit

    if stats["over_limit"] > 0:
        result["status"] = "FAIL"
    else:
        result["status"] = "PASS"

    return result


def print_report(results: list[dict[str, Any]]) -> None:
    """Print a formatted summary of all verification results to stdout.

    Args:
        results: List of per-file result dicts from ``check_file``.
    """
    grand_total = 0
    grand_over = 0
    grand_passed = 0
    grand_failed = 0

    print("=" * 72)
    print("  TOKEN COUNT VERIFICATION REPORT")
    print("=" * 72)

    for r in results:
        print(f"\n  [{r['file']}]")
        print(f"    Format:   {r['format_label']}")
        print(f"    Samples:  {r['total_samples']}")

        if r["status"].startswith("SKIPPED"):
            print(f"    Status:   {r['status']}")
            continue

        s = r["stats"]
        print(f"    Tokens:   min={s['min']}, max={s['max']}, "
              f"mean={s['mean']}, median={s['median']}")
        print(f"    Limit:    {r['limit_value']} ({r['limit_type'].replace('_', ' ')})")
        print(f"    Over:     {s['over_limit']} / {s['total']} "
              f"({s['over_limit_pct']}%)")
        print(f"    Status:   {r['status']}")

        grand_total += r["total_samples"]
        if r["status"] == "FAIL":
            grand_failed += 1
            grand_over += s["over_limit"]
        else:
            grand_passed += 1

    print("\n" + "=" * 72)
    print(f"  SUMMARY: {grand_total} total samples across {len(results)} files")
    if grand_over > 0:
        print(f"  ⚠️  {grand_over} samples exceed token limits ({grand_failed} files affected)")
        print(f"  These samples will be silently truncated during training.")
        print(f"  Run with --verbose to see which sample indices are over limit.")
    else:
        print(f"  ✅ All samples fit within token limits — dataset is training-ready")
    print("=" * 72)


def save_evidence(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save a compact verification evidence JSON file.

    Excludes per-sample index lists (which can be large) and keeps only
    aggregate statistics for reproducibility.

    Args:
        results: List of per-file result dicts from ``check_file``.
        output_path: Path to write the evidence JSON file.
    """
    evidence = {
        "tool": "verify_token_counts.py",
        "summary": {
            "total_files": len(results),
            "passed": sum(1 for r in results if r["status"] == "PASS"),
            "failed": sum(1 for r in results if r["status"] == "FAIL"),
            "skipped": sum(1 for r in results if r["status"].startswith("SKIPPED")),
            "total_samples": sum(r["total_samples"] for r in results),
            "total_over_limit": sum(r.get("stats", {}).get("over_limit", 0) for r in results),
        },
        "files": [],
    }
    for r in results:
        entry = {
            "file": r["file"],
            "format": r.get("format_label", r["format"]),
            "total_samples": r["total_samples"],
            "status": r["status"],
        }
        if "stats" in r and r["stats"]:
            s = r["stats"]
            entry["stats"] = {
                "min_tokens": s["min"],
                "max_tokens": s["max"],
                "mean_tokens": s["mean"],
                "median_tokens": s["median"],
                "over_limit": s["over_limit"],
                "over_limit_pct": s["over_limit_pct"],
            }
            if "template_errors" in s:
                entry["stats"]["template_errors"] = s["template_errors"]
        if "limit_value" in r:
            entry["limit"] = {"type": r["limit_type"], "value": r["limit_value"]}
        evidence["files"].append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nEvidence saved → {output_path}")


def main() -> None:
    """CLI entry point: verify token counts for all datasets in a directory.

    Loads the tokenizer, scans for JSONL files, detects each file's format,
    runs the appropriate token-count check, and reports results.
    """
    parser = argparse.ArgumentParser(
        description="Verify dataset token counts against model limits."
    )
    parser.add_argument(
        "--data-dir",
        default="data/generated/v1.0_k5",
        help="Directory containing dataset JSONL files (default: data/generated/v1.0_k5)",
    )
    parser.add_argument(
        "--model-name",
        default="unsloth/Qwen3-4B-Instruct-2507",
        help="HuggingFace model name for tokenizer (default: unsloth/Qwen3-4B-Instruct-2507)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum total sequence length (prompt + completion) (default: 8192)",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=7680,
        help="Maximum prompt length for GRPO/RC-GRPO (default: 7680)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print details of every sample that exceeds limits",
    )
    parser.add_argument(
        "--evidence",
        default=".omo/evidence/verify_token_counts.json",
        help="Path for evidence output (default: .omo/evidence/verify_token_counts.json)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name} ...")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print()

    config = {
        "max_seq_length": args.max_seq_length,
        "max_prompt_length": args.max_prompt_length,
    }

    # Discover JSONL files
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(jsonl_files)} JSONL files in {data_dir}")
    print(f"  Limits: max_seq_length={args.max_seq_length}, "
          f"max_prompt_length={args.max_prompt_length}")
    print()

    # Run checks
    results: list[dict[str, Any]] = []
    for fpath in jsonl_files:
        print(f"Checking {fpath.name} ...")
        result = check_file(fpath, tokenizer, config, verbose=args.verbose)
        results.append(result)
        status_icon = "✅" if result["status"] == "PASS" else "⚠️" if result["status"] == "FAIL" else "⏭️"
        print(f"  {status_icon} {result['status']}")
        print()

    # Print report
    print_report(results)

    # Save evidence
    evidence_path = Path(args.evidence)
    save_evidence(results, evidence_path)

    # Exit code
    total_over = sum(r.get("stats", {}).get("over_limit", 0) for r in results)
    if total_over > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
