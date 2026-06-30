#!/usr/bin/env python
"""
generate_stage1_dataset.py
──────────────────────────
Generate ~3500 SFT/RCTP stage1 samples via LLM API.
Uses a provider fallback chain to maximise reliability.

Produces:
  data/generated/v1.0_k5/raw_stage1.jsonl
  data/generated/v1.0_k5/raw_stage1_cleaned.jsonl
  data/generated/v1.0_k5/failures_stage1.jsonl
  data/generated/v1.0_k5/sft_dataset_stage1.jsonl
  data/generated/v1.0_k5/rctp_dataset_stage1.jsonl
  data/generated/v1.0_k5/stage1_query_hashes.json

Usage:
  python scripts/generate_stage1_dataset.py

Environment:
  OPENCODE_API_KEY    (preferred)
  OPENROUTER_API_KEY  (fallback)
"""

import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from scripts.data_generator import TelcoDatasetGenerator
from scripts.build_datasets import (
    build_rctp_dataset,
    build_sft_dataset,
    write_jsonl,
)
from scripts.clean_dataset import clean_split
from scripts.retrieval import FunctionRetriever

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _hash_query(query: str) -> str:
    """SHA256 of lowercased query, truncated to 16 hex chars."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def _generate_failures(
    samples: list[dict],
    generator: TelcoDatasetGenerator,
    failures_per_sample: int = 1,
) -> list[dict]:
    """Generate failure trajectories for non-abstention samples.

    Args:
        samples: List of cleaned sample dicts (must have 'id', 'ground_truth').
        generator: Initialised TelcoDatasetGenerator instance.
        failures_per_sample: Max failures per sample (default: 1).

    Returns:
        List of failure records with 'id' and 'ground_truth' keys.
    """
    failures: list[dict] = []
    for sample in samples:
        gt = sample.get("ground_truth", {})
        calls = gt.get("calls", []) if isinstance(gt, dict) else []
        if not calls:
            continue  # skip abstention

        sample_id = sample.get("id")
        if not sample_id:
            continue

        try:
            failure_calls = generator._call_failure(sample)
        except Exception:
            continue

        for fc in failure_calls[:failures_per_sample]:
            if isinstance(fc, dict):
                call_entry = fc
            else:
                continue
            failures.append(
                {
                    "id": sample_id,
                    "ground_truth": {
                        "calls": [call_entry],
                        "reasoning": (
                            "Incorrect function selection due to "
                            "similar function names or parameter confusion."
                        ),
                    },
                }
            )
    return failures


# --------------------------------------------------------------------------- #
#  Main script
# --------------------------------------------------------------------------- #

# Silence noisy HTTP/API loggers
for _lib in ("httpx", "httpcore", "openai", "urllib3", "requests", "sentence_transformers"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_stage1_dataset")
logger.setLevel(logging.INFO)

BASE_DIR = Path("data/generated/v1.0_k5")

PATHS = {
    "train_schema": BASE_DIR / "function_schema_train.json",
    "test_schema": BASE_DIR / "function_schema_test.json",
    "argument_values": BASE_DIR / "argument_values.json",
    "raw_output": BASE_DIR / "raw_stage1.jsonl",
    "raw_cleaned_output": BASE_DIR / "raw_stage1_cleaned.jsonl",
    "failures_output": BASE_DIR / "failures_stage1.jsonl",
    "sft_output": BASE_DIR / "sft_dataset_stage1.jsonl",
    "rctp_output": BASE_DIR / "rctp_dataset_stage1.jsonl",
    "query_hashes": BASE_DIR / "stage1_query_hashes.json",
}

PROVIDER_CHAIN = [
    {
        "provider": "opencode",
        "model": "deepseek-v4-flash-free",
        "base_url": "https://opencode.ai/zen/v1",
        "api_key_env": "OPENCODE_API_KEY",
        "max_workers": 16,
        "requests_per_minute": 120,
        "temperature": 0.9,
        "max_tokens": 8192,
        "seed": 3407,
    },
    {
        "provider": "openrouter",
        "model": "openrouter/free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "max_workers": 2,
        "requests_per_minute": 15,
        "temperature": 0.9,
        "max_tokens": 8192,
        "seed": 3407,
    },
    {
        "provider": "openrouter",
        "model": "openai/gpt-oss-120b:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "max_workers": 2,
        "requests_per_minute": 15,
        "temperature": 0.9,
        "max_tokens": 8192,
        "seed": 3407,
    },
]


def main() -> None:
    # ── 1. Verify input schemas exist ──────────────────────────────────────
    for key in ("train_schema", "test_schema", "argument_values"):
        path = PATHS[key]
        if not path.exists():
            logger.error("Required file not found: %s", path)
            sys.exit(1)

    logger.info("Loading schemas from %s and %s", PATHS["train_schema"], PATHS["test_schema"])

    # ── 2. Build FunctionRetriever (real retrieval) ────────────────────────
    merged_lib = {}
    for p in [PATHS["train_schema"], PATHS["test_schema"]]:
        merged_lib.update(json.load(open(p)))
    func_retriever = FunctionRetriever(
        merged_lib,
        method="hybrid",
        encoder_model="AITeamVN/Vietnamese_Embedding_v2",
    )

    # ── 3. Try each provider in chain ─────────────────────────────────────
    train_samples = []
    generator: TelcoDatasetGenerator | None = None
    for provider_cfg in PROVIDER_CHAIN:
        api_key = os.getenv(provider_cfg["api_key_env"])
        if not api_key:
            logger.warning(
                "Provider %s: %s not set — skipping",
                provider_cfg["provider"],
                provider_cfg["api_key_env"],
            )
            continue

        logger.info("Trying provider: %s (%s)", provider_cfg["provider"], provider_cfg["model"])
        try:
            generator = TelcoDatasetGenerator.from_schemas(
                train_schema_path=str(PATHS["train_schema"]),
                test_schema_path=str(PATHS["test_schema"]),
                argument_values_path=str(PATHS["argument_values"]),
                provider=provider_cfg["provider"],
                model=provider_cfg["model"],
                base_url=provider_cfg["base_url"],
                api_key=api_key,
                max_workers=provider_cfg["max_workers"],
                requests_per_minute=provider_cfg["requests_per_minute"],
                temperature=provider_cfg["temperature"],
                max_tokens=provider_cfg["max_tokens"],
                seed=provider_cfg.get("seed", 3407),
                retriever=func_retriever,
            )

            logger.info("Generating 3500 samples...")
            generated_train, generated_test = generator.generate(
                total=3500,
                train_split=1.0,
            )
            logger.info(
                "Generated %d train samples (test: %d)",
                len(generated_train),
                len(generated_test),
            )

            if generated_train:
                train_samples = generated_train
                chosen_provider = provider_cfg
                break
            else:
                logger.warning(
                    "Provider %s returned 0 samples — trying next",
                    provider_cfg["provider"],
                )
        except Exception as e:
            logger.error(
                "Provider %s failed: %s — trying next",
                provider_cfg["provider"],
                e,
            )
            time.sleep(5)

    if not train_samples:
        logger.error("All providers exhausted — no samples generated. Aborting.")
        sys.exit(1)

    assert generator is not None  # guaranteed after successful provider chain

    # ── 4. Enrich with argument values ────────────────────────────────────
    logger.info("Enriching samples with argument values...")
    for s in tqdm(train_samples, desc="Enriching", unit="sample"):
        generator._enrich_argument_values(s)

    # ── 5. Serialize raw enriched samples ─────────────────────────────────
    raw_dicts = [asdict(s) for s in train_samples]
    with open(PATHS["raw_output"], "w", encoding="utf-8") as fh:
        for d in raw_dicts:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info("Saved %d raw enriched samples → %s", len(raw_dicts), PATHS["raw_output"])

    # ── 6. Build function library ──────────────────────────────────────────
    with open(PATHS["train_schema"], "r", encoding="utf-8") as fh:
        train_lib = json.load(fh)
    with open(PATHS["test_schema"], "r", encoding="utf-8") as fh:
        test_lib = json.load(fh)
    function_library = {**train_lib, **test_lib}

    with open(PATHS["argument_values"], "r", encoding="utf-8") as fh:
        argument_values = json.load(fh)

    # ── 7. Clean via clean_split ──────────────────────────────────────────
    logger.info("Cleaning raw samples via clean_split...")
    cleaned_samples, clean_stats = clean_split(PATHS["raw_output"], function_library)

    with open(PATHS["raw_cleaned_output"], "w", encoding="utf-8") as fh:
        for s in cleaned_samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(
        "Saved %d cleaned samples → %s", len(cleaned_samples), PATHS["raw_cleaned_output"]
    )

    dropped_total = int(sum(clean_stats["dropped"].values()))
    logger.info(
        "Cleaning: %d total → %d valid, %d dropped",
        clean_stats["total_lines"],
        len(cleaned_samples),
        dropped_total,
    )
    for reason, count in clean_stats["dropped"].most_common():
        logger.info("  dropped[%s]: %d", reason, count)

    # ── 8. Save query hashes for stage2 dedup ─────────────────────────────
    query_hashes = sorted({_hash_query(s["query"]) for s in cleaned_samples})
    with open(PATHS["query_hashes"], "w", encoding="utf-8") as fh:
        json.dump(query_hashes, fh)
    logger.info("Saved %d unique query hashes → %s", len(query_hashes), PATHS["query_hashes"])

    # ── 9. Generate failures for non-abstention samples ───────────────────
    logger.info("Generating failure trajectories...")
    failures = _generate_failures(
        cleaned_samples,
        generator,
        failures_per_sample=1,
    )
    with open(PATHS["failures_output"], "w", encoding="utf-8") as fh:
        for f_rec in failures:
            fh.write(json.dumps(f_rec, ensure_ascii=False) + "\n")
    logger.info("Generated %d failure records → %s", len(failures), PATHS["failures_output"])

    # ── 10. Build SFT dataset ─────────────────────────────────────────────
    logger.info("Building SFT dataset...")
    sft_records = build_sft_dataset(cleaned_samples, function_library, argument_values)
    write_jsonl(str(PATHS["sft_output"]), sft_records)
    logger.info("Built %d SFT records → %s", len(sft_records), PATHS["sft_output"])

    # ── 11. Build RCTP dataset ────────────────────────────────────────────
    logger.info("Building RCTP dataset...")
    rctp_trajectories = build_rctp_dataset(
        cleaned_samples,
        failures,
        function_library,
        argument_values,
        failures_per_expert=1,
    )
    rctp_records = [
        {
            "prompt_messages": t.prompt_messages,
            "response_text": t.response_text,
            "reward": t.reward,
        }
        for t in rctp_trajectories
    ]
    write_jsonl(str(PATHS["rctp_output"]), rctp_records)

    n_success = sum(1 for t in rctp_trajectories if t.reward == 1)
    n_failure = len(rctp_trajectories) - n_success
    logger.info(
        "Built %d RCTP records (reward=1: %d, reward=0: %d) → %s",
        len(rctp_trajectories),
        n_success,
        n_failure,
        PATHS["rctp_output"],
    )

    # ── 12. Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Stage1 dataset generation complete!")
    print(f"  Raw enriched samples : {PATHS['raw_output']} ({len(raw_dicts)} records)")
    print(
        f"  Cleaned samples      : "
        f"{PATHS['raw_cleaned_output']} ({len(cleaned_samples)} records, dropped {dropped_total})"
    )
    print(f"  Query hashes         : {PATHS['query_hashes']} ({len(query_hashes)} hashes)")
    print(f"  Failures             : {PATHS['failures_output']} ({len(failures)} records)")
    print(f"  SFT dataset          : {PATHS['sft_output']} ({len(sft_records)} records)")
    print(
        f"  RCTP dataset         : "
        f"{PATHS['rctp_output']} ({len(rctp_trajectories)} records)"
    )
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user (Ctrl+C). Exiting cleanly.")
        sys.exit(130)
