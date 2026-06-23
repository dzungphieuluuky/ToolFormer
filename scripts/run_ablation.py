#!/usr/bin/env python
"""
run_ablation.py — Controlled ablation studies for each algorithm.

Ablations run:
  RC-GRPO : without reward token injection
  AWPO    : without variance gating / without difficulty weighting / without reasoning reward
  GVPO    : without verification mask

Each ablation trains for a reduced number of steps and evaluates on the test set.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from src.algorithms.rc_grpo_trainer import train_rc_grpo
from src.algorithms.awpo_trainer import train_awpo
from src.algorithms.gvpo_trainer import train_gvpo
from src.data.retrieval import FunctionRetriever
from src.utils.sandbox import Sandbox
from src.evaluation.benchmark import evaluate_model
from src.evaluation.report_generator import generate_report
from src.utils.logging_utils import get_logger

logger = get_logger("run_ablation")

# ── Ablation definitions ──────────────────────────────────────────────────────

ABLATIONS = {
    # RC-GRPO
    "rc_grpo_no_reward_tokens": {
        "base": "config/base_config.yaml",
        "algo": "config/rc_grpo_config.yaml",
        "patch": {"rc_grpo": {"reward_token_injection": False}},
        "fn": train_rc_grpo,
        "output": "outputs/ablation_rc_grpo_no_tokens",
    },
    # AWPO
    "awpo_no_variance_gate": {
        "base": "config/base_config.yaml",
        "algo": "config/awpo_config.yaml",
        "patch": {"awpo": {"variance_std_threshold": 999.0}},  # gate never triggers
        "fn": train_awpo,
        "output": "outputs/ablation_awpo_no_variance",
    },
    "awpo_no_difficulty_weight": {
        "base": "config/base_config.yaml",
        "algo": "config/awpo_config.yaml",
        "patch": {"awpo": {"difficulty_weighting": False}},
        "fn": train_awpo,
        "output": "outputs/ablation_awpo_no_difficulty",
    },
    "awpo_no_reasoning": {
        "base": "config/base_config.yaml",
        "algo": "config/awpo_config.yaml",
        "patch": {"awpo": {"reasoning_weight": 0.0}},
        "fn": train_awpo,
        "output": "outputs/ablation_awpo_no_reasoning",
    },
    # GVPO
    "gvpo_no_mask": {
        "base": "config/base_config.yaml",
        "algo": "config/gvpo_config.yaml",
        "patch": {"gvpo": {"mask_failed_steps": False}},
        "fn": train_gvpo,
        "output": "outputs/ablation_gvpo_no_mask",
    },
}

SHORT_STEPS = 100  # ablation runs use fewer steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=list(ABLATIONS.keys()),
        help="Subset of ablation names to run",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, only evaluate existing checkpoints",
    )
    args = parser.parse_args()

    results = []

    for name in args.ablations:
        if name not in ABLATIONS:
            logger.warning(f"Unknown ablation: {name}, skipping.")
            continue

        spec = ABLATIONS[name]
        logger.info(f"\n{'=' * 60}\nRunning ablation: {name}\n{'=' * 60}")

        # Build config
        base = OmegaConf.load(spec["base"])
        algo = OmegaConf.load(spec["algo"])
        cfg  = OmegaConf.to_container(OmegaConf.merge(base, algo), resolve=True)

        # Apply ablation patches
        for section, overrides in spec["patch"].items():
            cfg.setdefault(section, {}).update(overrides)

        # Override output dir + reduce steps
        cfg.setdefault("training", {})["output_dir"] = spec["output"]
        cfg["training"]["max_steps"] = SHORT_STEPS
        cfg["training"]["save_steps"] = SHORT_STEPS

        if not args.eval_only:
            spec["fn"](cfg)

        # Evaluate
        data_cfg = cfg["data"]
        with open(data_cfg["function_library_path"]) as fh:
            function_library = json.load(fh)
        retriever = FunctionRetriever(function_library)
        sandbox = Sandbox(function_library)

        result = evaluate_model(
            model_path=spec["output"],
            test_dataset_path=data_cfg["test_path"],
            function_library =function_library,
            retriever        =retriever,
            sandbox          =sandbox,
            model_name_tag   =name,
        )
        results.append(result)

    if results:
        generate_report(results, output_dir="outputs/evaluation_reports/ablations")
    logger.info("✓ Ablation study complete.")


if __name__ == "__main__":
    main()
