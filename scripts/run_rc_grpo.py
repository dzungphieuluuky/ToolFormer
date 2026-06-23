#!/usr/bin/env python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path as _Path
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from src.algorithms.base_trainer import (
    load_model,
    build_grpo_config,
    load_grpo_dataset,
    patch_tokenizer_for_custom_roles,
    build_trl_reward_functions,
)
from src.algorithms.rc_grpo_trainer import (
    RCGRPOTrainer,
    build_rctp_dataset_from_jsonl,
    RCTPDataset,
)
from src.data.excel_parser import load_function_library
from src.data.retrieval import FunctionRetriever
from src.utils.sandbox import Sandbox
from src.evaluation.benchmark import evaluate_model
from src.evaluation.report_generator import generate_report
from src.utils.logging_utils import get_logger

logger = get_logger("run_rc_grpo")

ALGORITHM = "rc_grpo"

DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FUNCTION_LIBRARY_PATH = DATA_DIR / "function_library.json"
ARGUMENT_VALUES_PATH = DATA_DIR / "argument_values.json"

TRAIN_CONFIG = {
    "model": {
        "name": "unsloth/Qwen3-4B-Base",
        "max_seq_length": 1024,
        "load_in_4bit": True,
        "fast_inference": True,
        "gpu_memory_utilization": 0.95,
    },
    "lora": {
        "r": 32,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
    },
    "training": {
        "output_dir": f"outputs/{ALGORITHM}_model",
        "learning_rate": 1e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_generations": 5,
        "max_steps": 50,
        "save_steps": 50,
        "logging_steps": 1,
        "max_grad_norm": 1.0,
        "report_to": "wandb",
        "seed": 3407,
    },
    "data": {
        "train_path": "data/processed/train_dataset.jsonl",
        "test_path": "data/processed/test_dataset.jsonl",
        "max_prompt_length": 1024,
        "max_completion_length": 512,
        "include_all_threshold": 10,
    },
    "grpo": {
        "temperature": 1.0,
        "epsilon": 0.2,
        "kl_coefficient": 0.1,
        "loss_type": "grpo",
        "mask_truncated_completions": True,
    },
    "rctp_ft": {
        "learning_rate": 1e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_epochs": 1,
        "output_dir": "outputs/rctp_ft_model",
        "failures_per_expert": 1,
    },
}


def main():
    logger.info("Loading function library...")
    function_library = load_function_library(str(FUNCTION_LIBRARY_PATH))
    print(f"Loaded {len(function_library)} functions")

    argument_values = None
    if ARGUMENT_VALUES_PATH.exists():
        with open(ARGUMENT_VALUES_PATH, "r") as f:
            argument_values = json.load(f)
        print(f"Loaded argument values catalog with {len(argument_values)} parameters")
    else:
        print("No argument_values.json; will skip argument values in prompts.")

    logger.info("Loading model...")
    model, tokenizer = load_model(TRAIN_CONFIG)

    print("Tokenizer special tokens:")
    print(f"  bos_token: {tokenizer.bos_token}")
    print(f"  eos_token: {tokenizer.eos_token}")
    print(f"  pad_token: {tokenizer.pad_token}")
    print(f"  additional_special_tokens: {tokenizer.additional_special_tokens}")

    # ── Stage 1: RCTP-FT ────────────────────────────────────────────────────
    high_reward_probability = 0.5

    if ALGORITHM == "rc_grpo":
        print("\n" + "=" * 70)
        print("STAGE 1: Reward-Conditioned Trajectory Policy (RCTP) Fine-tuning")
        print("=" * 70)

        rctp_cfg = TRAIN_CONFIG["rctp_ft"]

        rctp_trajectories = build_rctp_dataset_from_jsonl(
            jsonl_path=TRAIN_CONFIG["data"]["train_path"],
            function_library=function_library,
            argument_values_catalog=argument_values,
            failures_per_expert=rctp_cfg["failures_per_expert"],
            seed=TRAIN_CONFIG["training"]["seed"],
        )

        n_success = sum(1 for t in rctp_trajectories if t.reward == 1)
        n_total = len(rctp_trajectories)
        high_reward_probability = n_success / max(1, n_total)
        print(f"[Stage 1] p (high_reward_probability for Eq. 3) = {high_reward_probability:.4f} "
              f"({n_success}/{n_total} success)")

        rctp_dataset = RCTPDataset(rctp_trajectories, tokenizer,
                                   max_length=TRAIN_CONFIG["data"]["max_prompt_length"])
        rctp_loader = DataLoader(rctp_dataset, batch_size=rctp_cfg["batch_size"], shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=rctp_cfg["learning_rate"])
        accum_steps = rctp_cfg.get("gradient_accumulation_steps", 1)
        total_batches = len(rctp_loader)
        total_optim_steps = (total_batches + accum_steps - 1) // accum_steps
        num_training_steps = total_optim_steps * rctp_cfg["num_epochs"]

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_optim_steps),
            num_training_steps=num_training_steps,
        )

        model.train()
        for epoch in range(rctp_cfg["num_epochs"]):
            total_loss = 0.0
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(tqdm(rctp_loader, desc=f"RCTP-FT epoch {epoch + 1}/{rctp_cfg['num_epochs']}")):
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss = loss / accum_steps
                loss.backward()
                total_loss += loss.item() * accum_steps

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == total_batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = total_loss / total_batches
            print(f"[RCTP-FT] Epoch {epoch + 1}/{rctp_cfg['num_epochs']}  Loss: {avg_loss:.4f}")

        rctp_save_dir = Path(rctp_cfg["output_dir"])
        rctp_save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(rctp_save_dir))
        tokenizer.save_pretrained(str(rctp_save_dir))
        print(f"[Stage 1] RCTP-FT checkpoint saved -> {rctp_save_dir}")

    # ── Stage 2: RC-GRPO ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STAGE 2: {'RC-GRPO' if ALGORITHM == 'rc_grpo' else 'Vanilla GRPO baseline'}")
    print("=" * 70)

    dataset = load_grpo_dataset(
        TRAIN_CONFIG["data"]["train_path"],
        function_library,
        tokenizer,
        argument_values_catalog=argument_values,
    )

    grpo_args = build_grpo_config(TRAIN_CONFIG, output_dir=TRAIN_CONFIG["training"]["output_dir"])

    if ALGORITHM == "rc_grpo":
        from trl import GRPOTrainer
        trainer = RCGRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=build_trl_reward_functions("rc_grpo"),
            args=grpo_args,
            train_dataset=dataset,
            high_reward_probability=high_reward_probability,
        )
    else:
        from trl import GRPOTrainer
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=build_trl_reward_functions("grpo"),
            args=grpo_args,
            train_dataset=dataset,
        )

    print(f"Trainer initialized for {ALGORITHM}")

    custom_tags = ["<reasoning>", "</reasoning>", "<tool_call>", "</tool_call>",
                   "<|high_reward|>", "<|low_reward|>"]
    existing = set(tokenizer.all_special_tokens)
    from src.reward.rc_grpo_reward import REWARD_TOKENS
    conflicts = [tag for tag in custom_tags if tag in existing and tag not in REWARD_TOKENS]
    if conflicts:
        print(f"WARNING: custom tags collide with existing special tokens: {conflicts}")
    else:
        print("No conflicts detected with custom tags.")

    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # ── Save ─────────────────────────────────────────────────────────────────
    output_dir = grpo_args.output_dir
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # ── Evaluation ───────────────────────────────────────────────────────────
    test_dataset_path = TRAIN_CONFIG["data"]["test_path"]
    if Path(test_dataset_path).exists():
        retriever = FunctionRetriever(function_library, method="bm25")
        sandbox = Sandbox(function_library)

        eval_result = evaluate_model(
            model_path=output_dir,
            test_dataset_path=test_dataset_path,
            function_library=function_library,
            retriever=retriever,
            sandbox=sandbox,
            top_k=5,
            max_new_tokens=512,
            model_name_tag=ALGORITHM,
            argument_values=argument_values,
        )
        print("Evaluation result:", eval_result["aggregate"])
        generate_report([eval_result])
    else:
        print("Test dataset not found; skipping evaluation.")


if __name__ == "__main__":
    main()
