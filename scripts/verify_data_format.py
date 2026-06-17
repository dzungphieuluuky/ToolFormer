#!/usr/bin/env python
"""
verify_data_format.py
──────────────────────
Print formatted samples to verify the conversation flow is correct.

Usage:
    python scripts/verify_data_format.py
    python scripts/verify_data_format.py --mode sft --n 1
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
import jsonlines

from src.algorithms.base_trainer import (
    SYSTEM_PROMPT,
    build_retriever_block,
    build_messages_for_grpo,
    build_messages_for_sft,
    format_sample_for_grpo,
    format_sample_for_sft,
)

SEP = "=" * 80


class MockTokenizer:
    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kw
    ):
        parts = []
        for m in messages:
            parts.append(f"<{m['role']}>\n{m['content']}\n</{m['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--mode", default="grpo", choices=["grpo", "sft"])
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    data_cfg = cfg["data"]

    with open(data_cfg["function_library_path"], encoding="utf-8") as fh:
        function_library = json.load(fh)

    jsonl_path = (
        data_cfg["train_path"] if args.split == "train" else data_cfg["test_path"]
    )
    raw = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            raw.append(obj)
            if len(raw) >= args.n:
                break

    tokenizer = MockTokenizer()

    for i, sample in enumerate(raw):
        print(f"\n{SEP}")
        print(f"  SAMPLE {i + 1}  |  mode={args.mode}  |  split={args.split}")
        print(f"  id:       {sample['id'][:36]}")
        print(f"  query:    {sample['query']}")
        print(f"  function: {sample['function_name']}")
        print(f"  workflow: {sample['workflow_type']}")
        print(f"{SEP}")

        arg_vals = sample.get("retrieved_argument_values", {})
        retrieved = sample.get("retrieved_functions", [])

        if args.mode == "grpo":
            formatted = format_sample_for_grpo(
                sample, function_library, tokenizer, arg_vals
            )
            print("\n── PROMPT (model input) ──────────────────────────────────────\n")
            print(formatted["prompt"])
            print("\n── GROUND TRUTH (reward funcs see this, MODEL DOES NOT) ────\n")
            gt = formatted["ground_truth"]
            print(json.dumps(gt, indent=2, ensure_ascii=False))

        else:  # sft
            formatted = format_sample_for_sft(
                sample, function_library, tokenizer, arg_vals
            )
            print("\n── FULL SFT CONVERSATION ─────────────────────────────────────\n")
            print(formatted["text"])

        print(f"\n── RETRIEVED ARGUMENT VALUES ────────────────────────────────\n")
        if arg_vals:
            for param, matches in arg_vals.items():
                print(f"  {param}:")
                for m in matches:
                    code = (
                        m.get("code", m.get("code", "?"))
                        if isinstance(m, dict)
                        else m.code
                    )
                    label = (
                        m.get("label", m.get("label", "?"))
                        if isinstance(m, dict)
                        else m.label
                    )
                    score = m.get("score", "?") if isinstance(m, dict) else m.score
                    print(f"    {code} → {label}  (score={score})")
        else:
            print("  (none — run prepare_data.py to enrich)")


if __name__ == "__main__":
    main()
