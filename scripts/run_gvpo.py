#!/usr/bin/env python
"""run_gvpo.py — Train with GVPO algorithm."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from src.algorithms.gvpo_trainer import train_gvpo
from src.utils.logging_utils import get_logger

logger = get_logger("run_gvpo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config/gvpo_config.yaml")
    parser.add_argument("--base-config", default="config/base_config.yaml")
    args = parser.parse_args()

    base = OmegaConf.load(args.base_config)
    algo = OmegaConf.load(args.config)
    cfg  = OmegaConf.to_container(OmegaConf.merge(base, algo), resolve=True)

    logger.info("[GVPO] Config loaded. Starting training...")
    train_gvpo(cfg)


if __name__ == "__main__":
    main()