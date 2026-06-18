#!/usr/bin/env python
"""
run_evaluation.py
─────────────────
Evaluate all trained models and generate comparative report.

Usage:
    python scripts/run_evaluation.py \
        --models outputs/rc_grpo_model outputs/awpo_model outputs/gvpo_model
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from src.data.retrieval import FunctionRetriever
from src.utils.sandbox import Sandbox
from src.evaluation.benchmark import evaluate_model
from src.evaluation.report_generator import generate_report
from src.utils.logging_utils import get_logger

logger = get_logger("run_evaluation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    data_cfg = cfg["data"]
    ret_cfg = cfg.get("retrieval", {})

    # ── Load function library ──────────────────────────────────────────────────
    with open(data_cfg["function_library_path"], encoding="utf-8") as fh:
        function_library = json.load(fh)

    # ── Load argument value catalog ────────────────────────────────────────────
    argument_values: dict | None = None
    arg_val_path = data_cfg.get("argument_values_path")
    if arg_val_path and Path(arg_val_path).exists():
        with open(arg_val_path, encoding="utf-8") as fh:
            argument_values = json.load(fh)
        logger.info(
            f"Argument values catalog loaded: {len(argument_values)} param types"
        )

    # ── Build function retriever ───────────────────────────────────────────────
    index_dir = data_cfg.get("retrieval_index_dir", "data/processed/retrieval_index")
    retriever_pkl = f"{index_dir}/retriever.pkl"

    if Path(retriever_pkl).exists():
        func_retriever = FunctionRetriever.load(retriever_pkl, function_library)
    else:
        func_retriever = FunctionRetriever(
            function_library=function_library,
            method=ret_cfg.get("method", "hybrid"),
            encoder_model=ret_cfg.get(
                "encoder_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
        )

    sandbox = Sandbox(function_library)

    # ── Evaluate each model ────────────────────────────────────────────────────
    all_results = []
    for model_path in args.models:
        tag = Path(model_path).name
        logger.info(f"Evaluating: {tag}")
        result = evaluate_model(
            model_path=model_path,
            test_dataset_path=data_cfg["test_path"],
            function_library=function_library,
            retriever=func_retriever,
            sandbox=sandbox,
            top_k=args.top_k,
            model_name_tag=tag,
            argument_values=argument_values,
        )
        all_results.append(result)

    # ── Generate report ────────────────────────────────────────────────────────
    generate_report(all_results, output_dir="outputs/evaluation_reports")
    logger.info("✓ Evaluation complete.")


if __name__ == "__main__":
    main()
