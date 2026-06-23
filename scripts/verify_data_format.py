#!/usr/bin/env python
"""
verify_data_format.py
──────────────────────
Print formatted samples to visually verify the conversation flow.

Conversation format verified:
    <|im_start|>system
    ...SYSTEM_PROMPT...
    <|im_end|>
    <|im_start|>user
    <raw query>
    <|im_end|>
    <|im_start|>retriever
    ## Available Functions
    ...
    ## Relevant Argument Values
    ...
    <|im_end|>
    <|im_start|>assistant          ← GRPO: model generates here
    <reasoning>...</reasoning>     ← SFT: ground-truth included
    <tool_call>...</tool_call>

Usage:
    python scripts/verify_data_format.py
    python scripts/verify_data_format.py --mode sft --n 2
    python scripts/verify_data_format.py --split test --n 1
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
    CUSTOM_CHAT_TEMPLATE,
    build_retriever_block,
    build_messages_for_grpo,
    build_messages_for_sft,
    format_sample_for_grpo,
    format_sample_for_sft,
    _deserialise_arg_values,
)

SEP = "=" * 80


# ──────────────────────────────────────────────────────────────────────────────
# Mock tokenizer that uses the same custom template as the real one
# ──────────────────────────────────────────────────────────────────────────────


class MockTokenizer:
    """
    Minimal tokenizer mock that renders messages using CUSTOM_CHAT_TEMPLATE.
    Uses the same Jinja2 template string registered in load_model() so the
    output is byte-for-byte identical to what the real tokenizer produces.
    """

    def __init__(self):
        self.chat_template = CUSTOM_CHAT_TEMPLATE

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool              = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> str:
        """Render messages using the custom ChatML template."""
        parts: list[str] = []
        for msg in messages:
            role    = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
        result = "\n".join(parts)
        if add_generation_prompt:
            result += "\n<|im_start|>assistant\n"
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--mode", default="grpo", choices=["grpo", "sft"])
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--n", type=int, default=1, help="Number of samples to print")
    args = parser.parse_args()

    cfg      = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    data_cfg = cfg["data"]

    # Load function library
    with open(data_cfg["function_library_path"], encoding="utf-8") as fh:
        function_library = json.load(fh)
    print(f"Function library: {len(function_library)} functions")

    # Load raw samples
    jsonl_path = (
        data_cfg["train_path"] if args.split == "train" else data_cfg["test_path"]
    )
    raw_samples: list[dict] = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            raw_samples.append(obj)
            if len(raw_samples) >= args.n:
                break

    print(f"Loaded {len(raw_samples)} sample(s) from {jsonl_path}\n")
    tokenizer = MockTokenizer()

    for i, sample in enumerate(raw_samples):
        arg_vals  = _deserialise_arg_values(sample.get("retrieved_argument_values"))
        retrieved = sample.get("retrieved_functions", [])
        gt        = sample.get("ground_truth", {})

        print(f"\n{SEP}")
        print(
            f"  SAMPLE {i + 1}/{len(raw_samples)}  |  mode={args.mode}  |  split={args.split}"
        )
        print(f"  id:           {sample.get('id', 'n/a')[:36]}")
        print(f"  query:        {sample['query'][:100]}")
        print(f"  function:     {sample.get('function_name', 'n/a')}")
        print(f"  workflow:     {sample.get('workflow_type', 'n/a')}")
        print(f"  retrieved fns:{retrieved}")
        print(f"  arg_val keys: {list(arg_vals.keys()) if arg_vals else []}")
        print(SEP)

        if args.mode == "grpo":
            formatted = format_sample_for_grpo(
                sample, function_library, tokenizer, arg_vals
            )
            print("\n── GRPO PROMPT (full, what the model sees) ──────────────────\n")
            print(formatted["prompt"])
            print("\n── GROUND TRUTH (reward funcs see this — model does NOT) ───\n")
            print(json.dumps(formatted["ground_truth"], indent=2, ensure_ascii=False))

        else:  # sft
            formatted = format_sample_for_sft(
                sample, function_library, tokenizer, arg_vals
            )
            print("\n── SFT FULL CONVERSATION ─────────────────────────────────────\n")
            print(formatted["text"])

        # Always show expected model output shape
        print("\n── EXPECTED MODEL OUTPUT FORMAT ─────────────────────────────\n")
        call_json = json.dumps(
            {"function": gt.get("function"), "arguments": gt.get("arguments", {})},
            indent      =2,
            ensure_ascii=False,
        )
        print(
            f"<reasoning>\n{gt.get('reasoning', '...')}\n</reasoning>\n"
            f"<tool_call>\n{call_json}\n</tool_call>"
        )

        # Show retrieved argument values
        if arg_vals:
            print("\n── RETRIEVED ARGUMENT VALUES ────────────────────────────────\n")
            for param, matches in arg_vals.items():
                print(f"  {param}:")
                for m in matches if isinstance(matches, list) else []:
                    if isinstance(m, dict):
                        score = m.get("score", "?")
                        print(
                            f"    {m.get('code')} → {m.get('label')}  "
                            f"[{m.get('group')}]  score={score}"
                        )

    # ── Summary stats ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("DATASET SUMMARY")
    print(SEP)

    all_samples: list[dict] = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            all_samples.append(obj)

    wf_counts: dict[str, int] = {}
    fn_counts: dict[str, int] = {}
    enriched_n                = 0

    for s in all_samples:
        wt = s.get("workflow_type", "unknown")
        fn = s.get("function_name", "unknown")
        wf_counts[wt] = wf_counts.get(wt, 0) + 1
        fn_counts[fn] = fn_counts.get(fn, 0) + 1
        if s.get("retrieved_argument_values"):
            enriched_n += 1

    print(f"Total samples    : {len(all_samples)}")
    print(f"Arg-value-enriched: {enriched_n}/{len(all_samples)}")

    print("\nWorkflow distribution:")
    for wt, n in sorted(wf_counts.items()):
        print(f"  {wt:20s}: {n:5d}  ({100 * n / len(all_samples):.1f}%)")

    print("\nTop-10 functions:")
    for fn, n in sorted(fn_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {fn:35s}: {n:5d}")

    # Check all retrieved functions exist in library
    missing: set[str] = set()
    for s in all_samples:
        for fn in s.get("retrieved_functions", []):
            if fn not in function_library:
                missing.add(fn)
    if missing:
        print(f"\n⚠️  {len(missing)} retrieved fn(s) NOT in library:")
        for fn in sorted(missing):
            print(f"    {fn}")
    else:
        print("\n✓  All retrieved function names exist in function_library.")


if __name__ == "__main__":
    main()
