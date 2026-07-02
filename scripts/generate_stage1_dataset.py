#!/usr/bin/env python
"""
generate_stage1_dataset.py
──────────────────────────
Generate SFT/RCTP stage1 samples via LLM API.
Uses a provider fallback chain to maximise reliability and a hybrid 
three-tier failure generator with rate-limiting to prevent 429s.

Produces:
  data/generated/v1.0_k5/raw_stage1.jsonl
  data/generated/v1.0_k5/raw_stage1_cleaned.jsonl
  data/generated/v1.0_k5/failures_stage1.jsonl
  data/generated/v1.0_k5/sft_dataset_stage1.jsonl
  data/generated/v1.0_k5/rctp_dataset_stage1.jsonl
  data/generated/v1.0_k5/stage1_query_hashes.json
"""

import hashlib
import json
import logging
import os
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tqdm import tqdm

from scripts.data_generator import TelcoDatasetGenerator
from scripts.build_datasets import (
    build_rctp_dataset,
    build_sft_dataset,
    write_jsonl,
)
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
    function_library: dict,
    catalog: dict,
    failures_per_sample: int = 1,
    max_workers: int = 8,
    llm_ratio: float = 0.15,         # Only query LLM for 15% of samples to avoid 429s
    requests_per_minute: int = 60,   # Soft limit for LLM pacing
) -> list[dict]:
    from scripts.generate_failures import (
        _tier2_failure,
        _tier3_failure,
        TokenBucket,
        FAILURE_TYPES,
    )

    eligible = [
        s for s in samples
        if s.get("ground_truth", {}).get("calls", []) and s.get("id")
    ]
    if not eligible:
        return []

    logger.info(
        "Generating failures for %d eligible samples (LLM Ratio: %.1f%%, workers: %d)...",
        len(eligible), llm_ratio * 100, max_workers,
    )

    # Real-time token bucket rate-limiter for Tier 1 LLM calls
    rate_limiter = TokenBucket(requests_per_minute) if requests_per_minute > 0 else None
    failures: list[dict] = []
    lock = threading.Lock()

    # Re-seeded local Random to guarantee output reproducibility
    rng = random.Random(3407)

    def _process(sample: dict) -> list[dict]:
        sample_id = sample["id"]
        gold_calls = sample.get("ground_truth", {}).get("calls", [])
        gold_call = gold_calls[0] if gold_calls else None
        if not gold_call:
            return []

        results = []
        # Decide whether to attempt Tier 1 LLM failure generation
        use_llm = rng.random() < llm_ratio
        tier1_success = False

        if use_llm:
            try:
                if rate_limiter:
                    rate_limiter.acquire()
                failure_calls = generator._call_failure(sample)
                if isinstance(failure_calls, list) and failure_calls:
                    for fc in failure_calls[:failures_per_sample]:
                        if isinstance(fc, dict):
                            tool_call = fc.get("tool_call") or fc
                            reasoning = fc.get("reasoning") or "Mô hình chọn sai chức năng do nhầm lẫn ngữ cảnh câu hỏi."
                            results.append(
                                {
                                    "id": sample_id,
                                    "ground_truth": {
                                        "calls": [tool_call],
                                        "reasoning": reasoning,
                                    },
                                }
                            )
                    tier1_success = len(results) > 0
            except Exception as e:
                logger.debug("Tier 1 LLM generation bypassed/failed for %s: %s", sample_id, e)

        # Fallback to Tier 2 (Heuristics) or Tier 3 (Legacy) if Tier 1 skipped or failed
        if not tier1_success:
            available_types = list(FAILURE_TYPES)
            for _ in range(failures_per_sample):
                if available_types:
                    ft = available_types.pop(rng.randrange(len(available_types)))
                else:
                    ft = rng.choice(FAILURE_TYPES)

                # Tier 2: Catalog-aware inline heuristic (0 API cost, instantaneous)
                failure_dict = _tier2_failure(sample, gold_call, function_library, catalog, ft)
                
                # Tier 3: Legacy prefix fallback
                if not failure_dict:
                    failure_dict = _tier3_failure(sample, gold_call, function_library, catalog, ft)

                if failure_dict:
                    results.append(
                        {
                            "id": sample_id,
                            "ground_truth": {
                                "calls": [failure_dict["tool_call"]],
                                "reasoning": failure_dict["reasoning"],
                            },
                        }
                    )
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_process, s) for s in eligible]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating failures",
            unit="sample",
        ):
            try:
                results = fut.result()
                with lock:
                    failures.extend(results)
            except Exception:
                pass

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
        "max_workers": 12,
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

    # ── Load cleaned samples ─────────────────────────────────────────────
    cleaned_path = PATHS["raw_cleaned_output"]
    if not cleaned_path.exists():
        logger.error("Pre-built cleaned samples not found at %s", cleaned_path)
        logger.error("Run the full pipeline first to generate intermediates.")
        sys.exit(1)

    with open(cleaned_path, "r", encoding="utf-8") as fh:
        cleaned_samples: list[dict] = [json.loads(line) for line in fh if line.strip()]
    logger.info("Loaded %d cleaned samples from %s", len(cleaned_samples), cleaned_path)

    # ── Load function library ────────────────────────────────────────────
    with open(PATHS["train_schema"], "r", encoding="utf-8") as fh:
        train_lib = json.load(fh)
    with open(PATHS["test_schema"], "r", encoding="utf-8") as fh:
        test_lib = json.load(fh)
    function_library = {**train_lib, **test_lib}
    logger.info("Function library: %d functions total", len(function_library))

    # ── Load argument values ─────────────────────────────────────────────
    with open(PATHS["argument_values"], "r", encoding="utf-8") as fh:
        argument_values = json.load(fh)
    logger.info("Argument values: %d param keys", len(argument_values))

    # ── Load or recompute query hashes ───────────────────────────────────
    if PATHS["query_hashes"].exists():
        with open(PATHS["query_hashes"], "r", encoding="utf-8") as fh:
            query_hashes = json.load(fh)
        logger.info("Loaded %d query hashes from %s", len(query_hashes), PATHS["query_hashes"])
    else:
        query_hashes = sorted({_hash_query(s["query"]) for s in cleaned_samples})
        with open(PATHS["query_hashes"], "w", encoding="utf-8") as fh:
            json.dump(query_hashes, fh)
        logger.info("Computed and saved %d query hashes → %s", len(query_hashes), PATHS["query_hashes"])

    # ── Initialise API client for failure generation ─────────────────────
    logger.info("Initialising API client for failure generation...")
    merged_lib = {}
    for p in [PATHS["train_schema"], PATHS["test_schema"]]:
        merged_lib.update(json.load(open(p)))
    func_retriever = FunctionRetriever(
        merged_lib,
        method="hybrid",
        encoder_model="AITeamVN/Vietnamese_Embedding_v2",
    )

    generator: TelcoDatasetGenerator | None = None
    selected_provider = None
    for provider_cfg in PROVIDER_CHAIN:
        api_key = os.getenv(provider_cfg["api_key_env"])
        if not api_key:
            continue
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
            selected_provider = provider_cfg
            logger.info(
                "Generator ready: %s (%s) with %d workers",
                provider_cfg["provider"],
                provider_cfg["model"],
                provider_cfg["max_workers"],
            )
            break
        except Exception as e:
            logger.warning(
                "Failed to init generator with %s: %s — trying next",
                provider_cfg["provider"],
                e,
            )
            continue

    if generator is None:
        logger.error("All providers exhausted — cannot initialise API client. Aborting.")
        sys.exit(1)

    # ── 9. Generate failures using hybrid three-tier design ───────────────
    logger.info("Generating failure trajectories...")
    rpm_limit = selected_provider.get("requests_per_minute", 60) if selected_provider else 60
    failures = _generate_failures(
        cleaned_samples,
        generator,
        function_library=function_library,
        catalog=argument_values,
        failures_per_sample=1,
        max_workers=8,
        llm_ratio=0.15,            # Limit LLM workload to 15% to safeguard standard tier limits
        requests_per_minute=rpm_limit,
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
    n_raw = 0
    if PATHS["raw_output"].exists():
        with open(PATHS["raw_output"], "r") as fh:
            n_raw = sum(1 for _ in fh)

    print(f"\n{'=' * 60}")
    print(f"  Stage1 dataset generation complete!")
    print(f"  Raw enriched samples : {PATHS['raw_output']} ({n_raw} records)")
    print(
        f"  Cleaned samples      : "
        f"{PATHS['raw_cleaned_output']} ({len(cleaned_samples)} records)"
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