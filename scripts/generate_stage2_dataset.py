#!/usr/bin/env python
"""
generate_stage2_dataset.py
──────────────────────────
Generate 3000 diverse GRPO/RC‑GRPO stage2 samples via LLM API.
Uses a provider fallback chain to maximise reliability.

Produces:
  data/generated/v1.0_k5/raw_stage2.jsonl
  data/generated/v1.0_k5/raw_stage2_cleaned.jsonl
  data/generated/v1.0_k5/grpo_dataset_stage2.jsonl
  data/generated/v1.0_k5/rcgrpo_dataset_stage2.jsonl

Usage:
  python scripts/generate_stage2_dataset.py

Environment:
  OPENROUTER_API_KEY  (optional — first fallback)
  OPENCODE_API_KEY    (optional — second fallback)
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
from scripts.build_datasets import build_grpo_dataset, write_jsonl
from scripts.clean_dataset import clean_split
from scripts.retrieval import FunctionRetriever

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
logger = logging.getLogger("generate_stage2_dataset")
logger.setLevel(logging.INFO)

BASE_DIR = Path("data/generated/v1.0_k5")

PATHS = {
    "train_schema": BASE_DIR / "function_schema_train.json",
    "test_schema": BASE_DIR / "function_schema_test.json",
    "argument_values": BASE_DIR / "argument_values.json",
    "raw_output": BASE_DIR / "raw_stage2.jsonl",
    "grpo_output": BASE_DIR / "grpo_dataset_stage2.jsonl",
    "rcgrpo_output": BASE_DIR / "rcgrpo_dataset_stage2.jsonl",
    "raw_cleaned_output": BASE_DIR / "raw_stage2_cleaned.jsonl",
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
        "seed": 7043,
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
    },
]


def main() -> None:
    # ── 1. Verify inputs ───────────────────────────────────────────────────
    for key, path in PATHS.items():
        if key.endswith("_schema") or key == "argument_values":
            if not path.exists():
                logger.error("Required file not found: %s", path)
                sys.exit(1)

    logger.info("Loading schemas from %s and %s", PATHS["train_schema"], PATHS["test_schema"])

    # ── 2. Build real retriever (for realistic retrieval during generation) ──
    merged_lib = {}
    for p in [PATHS["train_schema"], PATHS["test_schema"]]:
        merged_lib.update(json.load(open(p)))
    func_retriever = FunctionRetriever(
        merged_lib, method="hybrid",
        encoder_model="AITeamVN/Vietnamese_Embedding_v2",
    )

    # ── 3. Try each provider in chain ─────────────────────────────────────
    train_samples = []
    generator: TelcoDatasetGenerator | None = None
    for provider_cfg in PROVIDER_CHAIN:
        api_key = os.getenv(provider_cfg["api_key_env"])
        if not api_key:
            logger.warning("Provider %s: %s not set — skipping",
                           provider_cfg["provider"], provider_cfg["api_key_env"])
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
                seed=provider_cfg.get("seed", 7043),
                retriever=func_retriever,
            )

            logger.info("Generating 3500 samples...")
            train_samples, test_samples = generator.generate(
                total=3500,
                train_split=1.0,
            )
            logger.info("Generated %d train samples (test: %d)",
                        len(train_samples), len(test_samples))

            if train_samples:
                break  # success
            else:
                logger.warning("Provider %s returned 0 samples — trying next",
                               provider_cfg["provider"])
        except Exception as e:
            logger.error("Provider %s failed: %s — trying next",
                         provider_cfg["provider"], e)
            # Short cooldown before trying next provider
            time.sleep(5)

    if not train_samples:
        logger.error("All providers exhausted — no samples generated. Aborting.")
        sys.exit(1)

    assert generator is not None  # guaranteed after successful provider chain

    # ── 3. Enrich with argument values ─────────────────────────────────────
    logger.info("Enriching samples with argument values...")
    for s in tqdm(train_samples, desc="Enriching", unit="sample"):
        generator._enrich_argument_values(s)

    # ── 4. Serialize raw enriched samples ──────────────────────────────────
    raw_dicts = [asdict(s) for s in train_samples]
    with open(PATHS["raw_output"], "w", encoding="utf-8") as fh:
        for d in raw_dicts:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info("Saved %d raw enriched samples → %s", len(raw_dicts), PATHS["raw_output"])

    # ── 5. Build function library ───────────────────────────────────────────
    with open(PATHS["train_schema"], "r", encoding="utf-8") as fh:
        train_lib = json.load(fh)
    with open(PATHS["test_schema"], "r", encoding="utf-8") as fh:
        test_lib = json.load(fh)
    function_library = {**train_lib, **test_lib}

    with open(PATHS["argument_values"], "r", encoding="utf-8") as fh:
        argument_values = json.load(fh)

    # ── 6. Clean via clean_split ──────────────────────────────────────────
    logger.info("Cleaning raw samples via clean_split...")
    cleaned_samples, clean_stats = clean_split(PATHS["raw_output"], function_library)

    # ── 6b. Dedup against stage1 queries ─────────────────────────────────
    stage1_hashes_path = BASE_DIR / "stage1_query_hashes.json"
    if stage1_hashes_path.exists():
        with open(stage1_hashes_path, "r", encoding="utf-8") as _fh:
            stage1_hashes = set(json.load(_fh))
        before = len(cleaned_samples)
        cleaned_samples = [
            s for s in cleaned_samples
            if hashlib.sha256(s["query"].strip().lower().encode()).hexdigest()[:16]
            not in stage1_hashes
        ]
        deduped = before - len(cleaned_samples)
        if deduped:
            logger.info("Dedup: removed %d samples overlapping with stage1", deduped)

    with open(PATHS["raw_cleaned_output"], "w", encoding="utf-8") as fh:
        for s in cleaned_samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("Saved %d cleaned samples → %s", len(cleaned_samples), PATHS["raw_cleaned_output"])

    dropped_total = int(sum(clean_stats["dropped"].values()))
    logger.info("Cleaning: %d total → %d valid, %d dropped",
                clean_stats["total_lines"], len(cleaned_samples), dropped_total)
    for reason, count in clean_stats["dropped"].most_common():
        logger.info("  dropped[%s]: %d", reason, count)

    # ── 7. Build GRPO & RC‑GRPO ───────────────────────────────────────────
    logger.info("Building GRPO dataset...")
    grpo_records = build_grpo_dataset(cleaned_samples, function_library, argument_values)
    write_jsonl(str(PATHS["grpo_output"]), grpo_records)
    logger.info("Built %d GRPO records → %s", len(grpo_records), PATHS["grpo_output"])

    shutil.copy2(str(PATHS["grpo_output"]), str(PATHS["rcgrpo_output"]))
    logger.info("Copied to %s", PATHS["rcgrpo_output"])

    # ── 8. Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Stage2 dataset generation complete!")
    print(f"  Raw enriched samples : {PATHS['raw_output']} ({len(raw_dicts)} records)")
    print(f"  Cleaned samples      : {PATHS['raw_cleaned_output']} ({len(cleaned_samples)} records, dropped {dropped_total})")
    print(f"  GRPO dataset         : {PATHS['grpo_output']} ({len(grpo_records)} records)")
    print(f"  RC‑GRPO dataset      : {PATHS['rcgrpo_output']} ({len(grpo_records)} records)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user (Ctrl+C). Exiting cleanly.")
        sys.exit(130)