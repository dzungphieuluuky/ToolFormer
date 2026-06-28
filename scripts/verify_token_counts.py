#!/usr/bin/env python3
"""Verify that all SFT samples fit within max_seq_length with assistant response intact."""

import json
import sys
from pathlib import Path

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

def main() -> None:
    """Verify that all SFT samples fit within max_seq_length.

    Loads the SFT dataset, tokenizes each sample, and checks whether
    the assistant response is intact within the model's max sequence
    length. Saves evidence to .omo/evidence/.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Instruct-2507")
    max_seq_length = 4096
    assistant_marker = "<|im_start|>assistant\n"

    sft_path = Path("data/generated/v1.0/sft_dataset.jsonl")
    if not sft_path.exists():
        print(f"ERROR: {sft_path} not found")
        sys.exit(1)

    records = load_jsonl(sft_path)
    print(f"Loaded {len(records)} SFT records")

    at_risk = 0
    truncated_responses = 0
    total_tokens = []

    for i, rec in enumerate(records):
        text = rec["text"]
        tokens = tokenizer.encode(text)
        total_tokens.append(len(tokens))

        # Check if assistant marker is present
        idx = text.find(assistant_marker)
        if idx == -1:
            print(f"  WARNING: Sample {i} has no assistant marker")
            at_risk += 1
            continue

        # Check if the response is intact (marker is within first max_seq_length tokens)
        before = text[:idx]
        after = text[idx:]
        before_tokens = tokenizer.encode(before, add_special_tokens=False)
        after_tokens = tokenizer.encode(after, add_special_tokens=False)

        if len(before_tokens) + len(after_tokens) > max_seq_length:
            at_risk += 1
            print(f"  SAMPLE {i}: {len(tokens)} tokens (before={len(before_tokens)}, after={len(after_tokens)})")
            if len(after_tokens) >= max_seq_length - 1:
                truncated_responses += 1
        elif len(tokens) > max_seq_length:
            at_risk += 1
            print(f"  SAMPLE {i}: {len(tokens)} tokens exceeds limit, but assistant marker position may still be intact")

    print(f"\n=== RESULTS ===")
    print(f"Total samples: {len(records)}")
    print(f"At risk (over limit): {at_risk}")
    print(f"Truncated responses: {truncated_responses}")
    if total_tokens:
        print(f"Token range: {min(total_tokens)} - {max(total_tokens)}")
        print(f"Token mean: {sum(total_tokens) / len(total_tokens):.1f}")
        print(f"Token median: {sorted(total_tokens)[len(total_tokens)//2]}")
        over_limit = sum(1 for t in total_tokens if t > max_seq_length)
        print(f"Over {max_seq_length} limit: {over_limit} ({100*over_limit/len(total_tokens):.1f}%)")

    if at_risk == 0:
        print("✅ ALL SAMPLES FIT WITHIN LIMIT")
    else:
        print(f"⚠️  {at_risk} samples need attention, {truncated_responses} with truncated responses")

    # Save evidence
    evidence_path = Path(".omo/evidence/task-5-per-sample-argument-values.json")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "total_samples": len(records),
        "at_risk": at_risk,
        "truncated_responses": truncated_responses,
        "token_min": min(total_tokens) if total_tokens else 0,
        "token_max": max(total_tokens) if total_tokens else 0,
        "token_mean": sum(total_tokens) / len(total_tokens) if total_tokens else 0,
        "over_limit_count": sum(1 for t in total_tokens if t > max_seq_length) if total_tokens else 0
    }
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"\nEvidence saved to {evidence_path}")

if __name__ == "__main__":
    main()
