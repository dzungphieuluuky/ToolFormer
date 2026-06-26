#!/usr/bin/env python
"""
prepare_data.py
───────────────
Orchestrates full data preparation:
  1. Parse Excel OR load function_schema.json
  2. Split into train/test schemas (if not provided separately)
  3. Build retrieval index
  4. Generate synthetic dataset via LLM API
     → saves raw_train_dataset.jsonl + raw_test_dataset.jsonl
  5. Enrich dataset with retrieved argument values
     → saves train_dataset.jsonl + test_dataset.jsonl  (final output)

Usage
─────
  python scripts/prepare_data.py [--config config/base_config.yaml] \
                                  [--train-schema data/generated/v1.0/function_schema_train.json] \
                                  [--test-schema  data/processed/function_schema_test.json] \
                                  [--excel  data/raw/telecom_functions.xlsx] \
                                  [--skip-generation] \
                                  [--skip-enrichment]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from tqdm import tqdm
from scripts.excel_parser import parse_telecom_functions, load_function_schema
from scripts.data_generator import TelcoDatasetGenerator
from scripts.retrieval import (
    FunctionRetriever,
    ArgumentValueRetriever,
    TelcoRetriever,
)
from src.utils.logging_utils import get_logger

logger = get_logger("prepare_data")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged from your version)
# ──────────────────────────────────────────────────────────────────────────────


def split_library(
    library: dict,
    test_funcs: list[str] | None = None,
    reserved_test_functions: int = 5,
) -> tuple[dict, dict]:
    """
    Split function library into train and test dictionaries.
    If test_funcs is provided, use that list; otherwise randomly reserve
    `reserved_test_functions` names.
    """
    func_names = list(library.keys())
    if test_funcs is None:
        shuffled = func_names.copy()
        random.shuffle(shuffled)
        test_names = set(shuffled[-reserved_test_functions:])
    else:
        test_names = set(test_funcs)
        missing = test_names - set(func_names)
        if missing:
            raise ValueError(f"Test function(s) not in library: {missing}")

    train_library = {k: v for k, v in library.items() if k not in test_names}
    test_library = {k: v for k, v in library.items() if k in test_names}
    return train_library, test_library


def enrich_dataset_with_arg_values(
    raw_path: str,
    output_path: str,
    val_retriever: ArgumentValueRetriever,
    function_library: dict,
) -> int:
    """
    Read raw_*.jsonl, add retrieved_argument_values to each sample,
    write to output_path.

    Returns the number of samples that received at least one argument value.
    """
    import jsonlines

    samples = []
    with jsonlines.open(raw_path) as reader:
        for obj in reader:
            samples.append(obj)

    enriched_count = 0
    out_samples = []

    for sample in tqdm(samples, desc="Enriching", unit="sample", leave=False):
        query = sample["query"]
        retrieved = sample.get("retrieved_functions", [])

        arg_vals = val_retriever.retrieve_for_functions(
            query=query,
            function_names=retrieved,
            function_library=function_library,
        )

        # Convert ValueMatch dataclasses → plain dicts for JSON serialisation
        serialisable: dict = {}
        for param, matches in arg_vals.items():
            serialisable[param] = [
                {
                    "code": m.code,
                    "label": m.label,
                    "group": m.group,
                    "score": round(m.score, 4),
                    "alt_label": m.alt_label,
                }
                for m in matches
            ]

        sample["retrieved_argument_values"] = serialisable
        if serialisable:
            enriched_count += 1
        out_samples.append(sample)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode="w") as writer:
        for s in out_samples:
            writer.write(s)

    logger.info(
        f"[enrich] {enriched_count}/{len(out_samples)} samples got argument values"
        f"  →  {output_path}"
    )
    return enriched_count


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Prepare telecom tool-calling dataset")
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument(
        "--train-schema",
        default=None,
        help="Path to train function_schema.json (if already split)",
    )
    parser.add_argument(
        "--test-schema", default=None, help="Path to test function_schema.json"
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Path to a single function_schema.json (will be split)",
    )
    parser.add_argument(
        "--excel",
        default=None,
        help="Path to telecom_functions.xlsx (will be converted and split)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip LLM generation (use existing raw JSONL files)",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip argument value enrichment step",
    )
    parser.add_argument(
        "--total", type=int, default=None, help="Override total sample count"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for final enriched datasets (train/test_dataset.jsonl). "
        "Schemas, index, and argument values remain in the original processed dir.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the LLM model name from config (e.g. openai/gpt-4o). "
        "Defaults to the value in dataset_generation.model in the config file.",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    data_cfg = cfg.get("data", {})
    dg_cfg = cfg.get("dataset_generation", {})
    ret_cfg = cfg.get("retrieval", {})
    split_cfg = dg_cfg.get("split", {})

    # ── Step 1: Build / load function library ─────────────────────────────────
    train_schema_path = args.train_schema
    test_schema_path = args.test_schema
    single_schema_path = args.schema or data_cfg.get("function_schema_path")
    excel_path = args.excel or "data/raw/telecom_functions.xlsx"
    lib_path = data_cfg.get(
        "function_library_path", "data/generated/v1.0/function_library.json"
    )

    if train_schema_path and test_schema_path:
        logger.info(f"Train schema: {train_schema_path}")
        logger.info(f"Test  schema: {test_schema_path}")
        if not Path(train_schema_path).exists() or not Path(test_schema_path).exists():
            logger.error("Provided train/test schema files do not exist.")
            sys.exit(1)

    else:
        library = None
        if single_schema_path and Path(single_schema_path).exists():
            library = load_function_schema(single_schema_path)
        elif Path(excel_path).exists():
            library = parse_telecom_functions(excel_path, output_path=lib_path)
        elif Path(lib_path).exists():
            with open(lib_path, encoding="utf-8") as fh:
                library = json.load(fh)
        else:
            logger.error(
                "No function source found!\n"
                "Provide one of:\n"
                "  --train-schema and --test-schema\n"
                "  --schema data/generated/v1.0/function_schema.json\n"
                "  --excel  data/raw/telecom_functions.xlsx\n"
                "  OR ensure data/generated/v1.0/function_library.json exists."
            )
            sys.exit(1)

        # Save the full library
        Path(lib_path).parent.mkdir(parents=True, exist_ok=True)
        with open(lib_path, "w", encoding="utf-8") as fh:
            json.dump(library, fh, indent=2, ensure_ascii=False)

        # Split into train / test schemas
        test_funcs = split_cfg.get("test_function_names")
        reserved_count = split_cfg.get("reserved_test_functions", 5)

        train_library, test_library = split_library(
            library,
            test_funcs=test_funcs,
            reserved_test_functions=reserved_count,
        )

        out_dir = Path(data_cfg.get("processed_dir", "data/generated/v1.0"))
        out_dir.mkdir(parents=True, exist_ok=True)
        train_schema_path = out_dir / "function_schema_train.json"
        test_schema_path = out_dir / "function_schema_test.json"

        with open(train_schema_path, "w", encoding="utf-8") as fh:
            json.dump(train_library, fh, indent=2, ensure_ascii=False)
        with open(test_schema_path, "w", encoding="utf-8") as fh:
            json.dump(test_library, fh, indent=2, ensure_ascii=False)

        logger.info(
            f"Schema split: {len(train_library)} train + {len(test_library)} test functions"
        )

    # ── Build full merged library for retrieval ────────────────────────────────
    if "library" in locals():
        full_library = library
    else:
        with open(train_schema_path, encoding="utf-8") as fh:
            train_lib = json.load(fh)
        with open(test_schema_path, encoding="utf-8") as fh:
            test_lib = json.load(fh)
        full_library = {**train_lib, **test_lib}

        # Save merged library if it doesn't exist yet
        if not Path(lib_path).exists():
            Path(lib_path).parent.mkdir(parents=True, exist_ok=True)
            with open(lib_path, "w", encoding="utf-8") as fh:
                json.dump(full_library, fh, indent=2, ensure_ascii=False)
            logger.info(f"Merged function library saved → {lib_path}")

    logger.info(f"Function library: {len(full_library)} functions total")

    # ── Step 2: Build argument values catalog ─────────────────────────────────
    arg_val_path = data_cfg.get(
        "argument_values_path", "data/generated/v1.0/argument_values.json"
    )
    if not Path(arg_val_path).exists():
        logger.info("Building argument values catalog...")
        import subprocess

        subprocess.run(
            [sys.executable, "scripts/build_argument_values.py"],
            check=True,
        )
    with open(arg_val_path, "r", encoding="utf-8") as fh:
        argument_values = json.load(fh)
    logger.info(f"Argument catalog: {len(argument_values)} parameter types")

    # ── Step 3: Build retrieval index ─────────────────────────────────────────
    index_dir = data_cfg.get(
        "retrieval_index_dir", "data/generated/v1.0/retrieval_index"
    )
    func_retriever = FunctionRetriever(
        function_library=full_library,
        method=ret_cfg.get("method", "hybrid"),
        encoder_model=ret_cfg.get(
            "encoder_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        index_dir=index_dir,
    )
    func_retriever.save(f"{index_dir}/retriever.pkl")
    logger.info(f"Retrieval index saved → {index_dir}/retriever.pkl")

    # ── Step 4: Generate synthetic dataset ────────────────────────────────────
    # Raw output paths (generator writes these)
    raw_train_path = data_cfg.get(
        "raw_train_path", "data/generated/v1.0/raw_train_dataset.jsonl"
    )
    raw_test_path = data_cfg.get(
        "raw_test_path", "data/generated/v1.0/raw_test_dataset.jsonl"
    )

    if not args.skip_generation:
        total = args.total or dg_cfg.get("total_samples", 2400)
        logger.info(
            f"Generating {total} samples via {dg_cfg.get('provider', 'openrouter')}..."
        )

        model_name = args.model or dg_cfg.get("model", "openai/gpt-oss-120b:free")
        generator = TelcoDatasetGenerator.from_schemas(
            train_schema_path=str(train_schema_path),
            test_schema_path=str(test_schema_path),
            provider=dg_cfg.get("provider", "openrouter"),
            model=model_name,
            api_key=os.getenv(dg_cfg.get("api_key_env", "OPENROUTER_API_KEY")),
            base_url=dg_cfg.get("base_url"),
            max_workers=dg_cfg.get("max_workers", 12),
            requests_per_minute=dg_cfg.get("requests_per_minute", 500),
            temperature=dg_cfg.get("temperature", 0.9),
            max_tokens=dg_cfg.get("max_tokens", 1024),
            seed=dg_cfg.get("seed", 42),
        )

        train_samples, test_samples = generator.generate(
            total=total,
            output_dir=data_cfg.get("processed_dir", "data/generated/v1.0"),
            workflow_distribution=dg_cfg.get("workflow_distribution"),
            train_split=dg_cfg.get("train_split", 0.8),
        )
        logger.info(
            f"Generated {len(train_samples)} train + {len(test_samples)} test samples"
        )
    else:
        logger.info("Skipping generation (--skip-generation)")
        if not args.skip_enrichment:
            missing = [
                p for p in [raw_train_path, raw_test_path] if not Path(p).exists()
            ]
            if missing:
                logger.error(
                    f"Raw dataset files not found: {missing}\n"
                    "Run without --skip-generation first, or also pass "
                    "--skip-enrichment."
                )
                sys.exit(1)

    # ── Step 5: Enrich dataset with argument values ────────────────────────────
    # Final output paths (what trainers load)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        final_train_path = str(out_dir / "train_dataset.jsonl")
        final_test_path = str(out_dir / "test_dataset.jsonl")
    else:
        final_train_path = data_cfg.get(
            "train_path", "data/generated/v1.0/train_dataset.jsonl"
        )
        final_test_path = data_cfg.get(
            "test_path", "data/generated/v1.0/test_dataset.jsonl"
        )

    if not args.skip_enrichment:
        val_retriever = ArgumentValueRetriever(
            argument_values=argument_values,
            top_k_values=ret_cfg.get("top_k_values", 5),
        )

        if Path(raw_train_path).exists():
            enrich_dataset_with_arg_values(
                raw_path=raw_train_path,
                output_path=final_train_path,
                val_retriever=val_retriever,
                function_library=full_library,
            )
        else:
            logger.warning(f"Raw train file not found, skipping: {raw_train_path}")

        if Path(raw_test_path).exists():
            enrich_dataset_with_arg_values(
                raw_path=raw_test_path,
                output_path=final_test_path,
                val_retriever=val_retriever,
                function_library=full_library,
            )
        else:
            logger.warning(f"Raw test file not found, skipping: {raw_test_path}")
    else:
        import shutil

        for raw, final in [
            (raw_train_path, final_train_path),
            (raw_test_path, final_test_path),
        ]:
            if Path(raw).exists() and not Path(final).exists():
                shutil.copy2(raw, final)
                logger.info(f"Copied {raw} → {final} (no enrichment)")

    # ── Summary ────────────────────────────────────────────────────────────────
    for label, path in [("train", final_train_path), ("test", final_test_path)]:
        if not Path(path).exists():
            logger.warning(f"{label} file not found: {path}")
            continue
        import jsonlines as _jl

        n = sum(1 for _ in _jl.open(path))
        logger.info(f"Done — {label}: {n} samples → {path}")


if __name__ == "__main__":
    main()
