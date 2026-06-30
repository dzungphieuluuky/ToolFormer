#!/usr/bin/env python
"""
generate_stage2_dataset.py
──────────────────────────
Generate 3000 diverse GRPO/RC-GRPO stage2 samples via DeepSeek V4 Flash Free (OpenRouter).

Produces:
  data/generated/v1.0_k5/raw_stage2.jsonl          — enriched raw samples
  data/generated/v1.0_k5/grpo_dataset_stage2.jsonl  — GRPO-formatted records
  data/generated/v1.0_k5/rcgrpo_dataset_stage2.jsonl — identical copy for RC-GRPO

Usage:
  python scripts/generate_stage2_dataset.py

Requires:
  OPENROUTER_API_KEY env var (or LLM_API_KEY)
"""

import json
import logging
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm
from scripts.data_generator import TelcoDatasetGenerator
from scripts.build_datasets import build_grpo_dataset, write_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_stage2_dataset")

BASE_DIR = Path("data/generated/v1.0_k5")

PATHS = {
    "train_schema": BASE_DIR / "function_schema_train.json",
    "test_schema": BASE_DIR / "function_schema_test.json",
    "argument_values": BASE_DIR / "argument_values.json",
    "raw_output": BASE_DIR / "raw_stage2.jsonl",
    "grpo_output": BASE_DIR / "grpo_dataset_stage2.jsonl",
    "rcgrpo_output": BASE_DIR / "rcgrpo_dataset_stage2.jsonl",
}


def main() -> None:
    # ── 1. Verify inputs exist ───────────────────────────────────────────
    for key, path in PATHS.items():
        if key.endswith("_schema") or key == "argument_values":
            if not path.exists():
                logger.error("Required file not found: %s", path)
                sys.exit(1)

    logger.info("Loading schemas from %s and %s", PATHS["train_schema"], PATHS["test_schema"])

    # ── 2. Initialize generator ──────────────────────────────────────────
    generator = TelcoDatasetGenerator.from_schemas(
        train_schema_path=str(PATHS["train_schema"]),
        test_schema_path=str(PATHS["test_schema"]),
        argument_values_path=str(PATHS["argument_values"]),
        provider="openrouter",
        model="deepseek/deepseek-v4-flash:free",
        max_workers=12,
        requests_per_minute=500,
        temperature=0.9,
        max_tokens=1024,
    )

    # ── 3. Generate 3000 samples (train only) ────────────────────────────
    logger.info("Generating 3000 samples...")
    train_samples, test_samples = generator.generate(
        total=3000,
        train_split=1.0,
    )
    logger.info("Generated %d train samples (test: %d)", len(train_samples), len(test_samples))

    if not train_samples:
        logger.error("No samples generated — aborting.")
        sys.exit(1)

    # ── 4. Enrich with argument values ───────────────────────────────────
    logger.info("Enriching samples with argument values...")
    for s in tqdm(train_samples, desc="Enriching", unit="sample"):
        generator._enrich_argument_values(s)

    # ── 5. Serialize raw enriched samples ────────────────────────────────
    raw_dicts = []
    for s in train_samples:
        d = asdict(s)
        raw_dicts.append(d)

    with open(PATHS["raw_output"], "w", encoding="utf-8") as fh:
        for d in raw_dicts:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info("Saved %d raw enriched samples → %s", len(raw_dicts), PATHS["raw_output"])

    # ── 6. Build function library (train + test) ─────────────────────────
    with open(PATHS["train_schema"], "r", encoding="utf-8") as fh:
        train_lib = json.load(fh)
    with open(PATHS["test_schema"], "r", encoding="utf-8") as fh:
        test_lib = json.load(fh)
    function_library = {**train_lib, **test_lib}

    # Load argument values catalog dict (same format as _catalog in generator)
    with open(PATHS["argument_values"], "r", encoding="utf-8") as fh:
        argument_values = json.load(fh)

    # ── 7. Build GRPO dataset ────────────────────────────────────────────
    logger.info("Building GRPO dataset...")
    grpo_records = build_grpo_dataset(raw_dicts, function_library, argument_values)
    write_jsonl(str(PATHS["grpo_output"]), grpo_records)
    logger.info("Built %d GRPO records → %s", len(grpo_records), PATHS["grpo_output"])

    # ── 8. Copy to RC-GRPO ──────────────────────────────────────────────
    shutil.copy2(str(PATHS["grpo_output"]), str(PATHS["rcgrpo_output"]))
    logger.info("Copied to %s", PATHS["rcgrpo_output"])

    # ── 9. Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Stage2 dataset generation complete!")
    print(f"  Raw enriched samples : {PATHS['raw_output']} ({len(raw_dicts)} records)")
    print(f"  GRPO dataset         : {PATHS['grpo_output']} ({len(grpo_records)} records)")
    print(f"  RC-GRPO dataset      : {PATHS['rcgrpo_output']} ({len(grpo_records)} records)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
