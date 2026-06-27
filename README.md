# ToolFormer

**Telecom Tool-Calling with Reinforcement Learning**

Fine-tuning Qwen3-4B for telecom function calling using RC-GRPO (Reward-Conditioned Group Relative Policy Optimization). Built on Unsloth + TRL. Pure research — no tests, no linter, no type checker, no CI, no git hooks.

---

## Quick Start

```bash
conda create -n unsloth_telco python=3.12
conda activate unsloth_telco
pip install -r requirements.txt
# Or restore from environment.yaml:
conda env create -f environment.yaml
```

Set optional API keys (not needed for local training):

```bash
export OPENROUTER_API_KEY=sk-...
export WANDB_API_KEY=...
export HF_TOKEN=...
```

Prepare the dataset (data pipeline) or use pre-built files under `data/generated/v1.0_k5/`:

```bash
python scripts/prepare_data.py
```

Open the notebook and run cells in order:

```bash
jupyter notebook Qwen3_(4B)_GRPO_ToolCalling.ipynb
```

Set `MODE` in Cell 23 to select training mode and run all cells.

---

## Entrypoints

| File | Role |
|------|------|
| `Qwen3_(4B)_GRPO_ToolCalling.ipynb` | **Sole implementation** — training, eval, inference, all inline (~10.5k lines, 30 cells). Does **not** import `scripts/` or `src/`. |
| `scripts/` | Standalone CLI data pipeline. Each script runs independently. Notebook loads pre-built datasets only. |
| `config/*.yaml` | Reference only. Notebook's hardcoded `TRAIN_CONFIG` dict wins when they diverge. |
| `AGENTS.md` | Quick-reference agent guide with gotchas, config tables, and verification commands. |
| `DATASET.md` | Dataset inventory, token statistics, workflow distributions, function retrieval breakdowns. |
| `PIPELINE.md` | Architectural walkthrough of the notebook. Reference for details not covered here. |

---

## Editing the Notebook

Do **not** edit the `.ipynb` file as raw JSON. Use the jupytext workflow:

```bash
# Convert to Markdown for editing
jupytext --to md Qwen3_(4B)_GRPO_ToolCalling.ipynb -o Qwen3_(4B)_GRPO_ToolCalling.md

# Edit Qwen3_(4B)_GRPO_ToolCalling.md freely

# Sync changes back to .ipynb
jupytext --sync Qwen3_(4B)_GRPO_ToolCalling.ipynb
```

The `.md` representation is git-friendly and makes diffs readable.

---

## Environment

| Requirement | Value |
|-------------|-------|
| Python | >=3.12 (`.python-version`) |
| Conda environment | `unsloth_telco` |
| CUDA | 12.1+ |
| Package management | `requirements.txt` + `uv.lock` (not `pyproject.toml` — zero runtime deps declared) |
| Base model | `unsloth/Qwen3-4B-Instruct-2507` (pre-quantized 4-bit at `models/Qwen3-4B-unsloth-bnb-4bit/`) |
| Runtime detection | Notebook auto-detects Colab, Kaggle, or local and adjusts paths |

---

## Training Modes

Set `MODE` in Cell 23 of the notebook:

| Mode | Trainer | Dataset | Description |
|------|---------|---------|-------------|
| `sft` | `SFTTrainer` | `sft_dataset.jsonl` | Supervised fine-tuning on expert demos |
| `rctp_ft` | `SFTTrainer` | `rctp_dataset.jsonl` | Stage 1: Reward-Conditioned Trajectory Policy FT |
| `grpo` | `GRPOTrainer` | `grpo_dataset.jsonl` | Vanilla GRPO baseline |
| `rc_grpo` | `RCGRPOTrainer` | `rcgrpo_dataset.jsonl` | Stage 2: RC-GRPO (RL with reward-token conditioning) |

Only SFT has trained weights: `outputs/sft_model/checkpoint-1335/`. GRPO and RC-GRPO output directories exist but are empty.

---

## Data

Active dataset: `data/generated/v1.0_k5/` (31 telecom functions, top-5 retrieval, per-sample argument values).

| File | Records |
|------|---------|
| `train_dataset_cleaned.jsonl` | 3,553 |
| `test_dataset_cleaned.jsonl` | 2,075 |
| `sft_dataset.jsonl` | 3,553 |
| `grpo_dataset.jsonl` | 3,553 |
| `rcgrpo_dataset.jsonl` | 3,553 |
| `rctp_dataset.jsonl` | 6,596 |
| `failures_dataset.jsonl` | 3,043 |

See `DATASET.md` for full inventory, token statistics, workflow distributions, and function retrieval breakdowns.

---

## Key Conventions

- **Script imports**: All `scripts/` use `sys.path.insert(0, ...)` to resolve project root, not package install.
- **Configuration**: Scripts load config via OmegaConf from `config/*.yaml`. The notebook uses a hardcoded `TRAIN_CONFIG` dict (source of truth).
- **Logging**: Scripts use Python logging. The notebook uses inline `print()` and a `get_logger()` helper. `src/utils/logging_utils.py` exists but is unused.
- **Reward token**: Injected into the **system message** (not user message) — both `_format_trajectory()` (RCTP-FT) and `inject_reward_token_into_prompt()` (RC-GRPO) target the system content before `<|im_end|>`.
- **GPU sharing**: `vllm_enable_sleep_mode=True` in `GRPOConfig` allows TRL's vLLM to share GPU with the main model by offloading during optimizer steps.
- **Data output**: Training runs write to `outputs/{mode}_model/`.

---

## Secrets

All optional:

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | LLM API for synthetic data generation |
| `WANDB_API_KEY` | Weights & Biases experiment tracking |
| `HF_TOKEN` | Hugging Face model downloads |

---

## License

Apache 2.0. See `LICENSE`.
