#!/usr/bin/env python
"""run_rc_grpo.py — Train with RC-GRPO algorithm."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from src.algorithms.rc_grpo_trainer import train_rc_grpo
from src.utils.logging_utils import get_logger

logger = get_logger("run_rc_grpo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/rc_grpo_config.yaml")
    parser.add_argument("--base-config", default="config/base_config.yaml")
    args = parser.parse_args()

    base = OmegaConf.load(args.base_config)
    algo = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(OmegaConf.merge(base, algo), resolve=True)

    logger.info("[RC-GRPO] Config loaded. Starting training...")
    train_rc_grpo(cfg)


if __name__ == "__main__":
    main()
