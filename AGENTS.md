# ToolFormer — Agent Guide

**Updated:** 2026-06-30 — complete rewrite with best-practice-first rule

## Execution Rules

### Sisyphus Execution Rule (ABSOLUTE)
Sisyphus is the ONLY agent allowed to execute tasks. He will execute every task directly and alone. No sub-agents, no delegation, no background workers. Every edit, command, grep, and read is done by Sisyphus himself.

### Best-Practice-First Rule (ABSOLUTE)
Before implementing ANY mechanism, function, or utility from a request, ALWAYS search the internet (web search + official docs) for best practices and methods that would yield the highest performance with the lowest failure rate. Do not default to naive implementations when proven patterns exist. Reference specific sources in your approach.

### Research-Before-Design Rule (ABSOLUTE)
For any ablation study, hyperparameter design, training configuration change, or algorithmic modification, the first action MUST be a web search for peer-reviewed papers, official documentation, and community benchmarks that inform the design. No code or configuration is written until research findings are synthesized and cited. This ensures all design decisions are grounded in established results (e.g., DAPO, GTPO, FG-ExPO, TRL docs) rather than guesswork.

## Project Overview

Telecom tool-calling RL project: fine-tuning Qwen3-4B for Vietnamese telecom function calling using RC-GRPO (Reward-Conditioned GRPO). Built on Unsloth + TRL. Pure research — **no tests, no linter, no type checker, no CI, no git hooks.**

The notebook (`Qwen3_(4B)_GRPO_ToolCalling.ipynb`) is the **sole implementation** — training, eval, inference all inline (~11k lines, 30 cells). It does NOT import `scripts/` at runtime; it loads pre-built datasets only.

## Quick Start

```bash
conda activate unsloth_telco
# Then open and run the notebook in Colab, Kaggle, or locally.
```

Secrets (set as env vars): `OPENCODE_API_KEY` (data gen), `OPENROUTER_API_KEY` (legacy fallback), `WANDB_API_KEY`, `HF_TOKEN`.

## Entrypoints

| File | Role |
|------|------|
| `Qwen3_(4B)_GRPO_ToolCalling.ipynb` | **Sole training implementation**. Edit via jupytext: `jupytext --sync Qwen3_(4B)_GRPO_ToolCalling.ipynb` |
| `scripts/` | Standalone CLI data pipeline. Each script runs independently. |
| `DATASET.md` | Dataset inventory, token stats, workflow distributions, retrieval figures. |
| `PIPELINE.md` | Architectural walkthrough of the notebook. |
| `README.md` | **Stale** — ignore it. |

## Training Modes

Set `MODE` in the notebook cell 23:

| Mode | Trainer | Dataset | Use |
|------|---------|---------|-----|
| `sft` | `SFTTrainer` | `sft_dataset.jsonl` | Supervised fine-tuning on expert demos |
| `rctp_ft` | `SFTTrainer` | `rctp_dataset.jsonl` | Stage 1: reward-conditioned trajectory policy |
| `grpo` | `GRPOTrainer` | `grpo_dataset_stage2.jsonl` | Vanilla GRPO baseline |
| `rc_grpo` | `RCGRPOTrainer` | `rcgrpo_dataset_stage2.jsonl` | Stage 2: RC-GRPO (RL with reward-token conditioning) |

GRPO/RC-GRPO auto-select `*_stage2.jsonl` with fallback to original files.

## Environment

- **Python >=3.12**, conda `unsloth_telco`, CUDA 12.1+
- `pyproject.toml` declares zero runtime deps — all deps in `requirements.txt` / `uv.lock`, installed by the notebook, not by `pip install -e .`
- Pre-quantized weights at `unsloth-Qwen3-4B-unsloth-bnb-4bit/`
- Notebook auto-detects Colab/Kaggle/local and adjusts paths

## Configuration: TRAIN_CONFIG is the Source of Truth

The notebook has a hardcoded `TRAIN_CONFIG` dict. The `config/*.yaml` files (for scripts via OmegaConf) **diverge** from the notebook. When they disagree, the notebook wins.

Key divergences (YAML → notebook):
- `num_generations: 8` → `5`
- `kl_coef: 0.04` → `0.1`
- `max_grad_norm: 0.1` → `1.0`
- `gpu_memory_utilization: 0.7` → `0.5`
- `args_threshold: 0.8` → exact binary match

## Critical Gotchas

### Unified `load_model()` — 4 branches
| Condition | What happens | Used by |
|-----------|-------------|---------|
| `adapter_model_path` set + `mode="train"` | Tokenizer from checkpoint → load base (GRPO patching) → resize → `PeftModel.from_pretrained` | GRPO/RC-GRPO resume |
| `adapter_model_path` set + `mode="inference"` | Tokenizer from checkpoint → load base → resize → `PeftModel.from_pretrained` → `for_inference()` | Evaluation from checkpoint |
| `adapter_model_path=None` + `mode="train"` | Tokenizer from base → load base → `patch_tokenizer_for_custom_roles` → `add_new_tokens` → `get_peft_model` (fresh LoRA) | SFT/RCTP-FT from scratch |
| `adapter_model_path=None` + `mode="inference"` | Tokenizer from base → load base → `for_inference()` | Base model benchmarking |

### Dual GPU memory pools
Two separate vLLM reservations that must be understood together:
- `gpu_memory_utilization`: Unsloth's engine. `0.3` when `UNSLOTH_VLLM_STANDBY=0` (SFT/RCTP-FT), `0.8` when `=1` (GRPO/RC-GRPO).
- `vllm_gpu_memory_utilization`: TRL's engine (in `GRPOConfig`). Hardcoded `0.3`.

With `UNSLOTH_VLLM_STANDBY=1` + `vllm_enable_sleep_mode=True`, Unsloth engine offloads to CPU on startup and TRL's engine offloads during optimizer steps. Changing either without understanding the split can OOM.

### `load_in_4bit` + `fast_inference` interaction
`fast_inference=True` (needed for GRPO vLLM) is compatible with `load_in_4bit=True`, but `max_lora_rank` can only be passed alongside `load_in_4bit` when `fast_inference=False`. The unified `load_model()` handles this.

### Reward token injection
Injected into the **system message** (not user message — departs from paper's Appendix B diagram). Both `_format_trajectory()` (RCTP-FT) and `inject_reward_token_into_prompt()` (RC-GRPO) target the system content before `<|im_end|>`.

### vLLM stop token
```python
SamplingParams(stop=["</tool_call>"], include_stop_str_in_output=True)
```

### Prompt truncation & generation tokens
- **Training**: `max_prompt_length=7680`, `max_completion_length=512`, `vllm_max_model_length=8192`
- **Data generation**: `max_tokens=8192` (in `generate_stage2_dataset.py` PROVIDER_CHAIN) — must match the model's full context window since each API call generates multiple samples per response
- Zero training prompts exceed limits (max actual: 5476 tokens)

## Data Pipeline (scripts/ — CLI only)

```
generate_stage2_dataset.py  → generate, clean, build GRPO/RC-GRPO via TelcoDatasetGenerator
  data_generator.py         → LLM API → synthetic (query, ground_truth) pairs
  validate_dataset.py       → quality checks → *_cleaned.jsonl
  clean_dataset.py          → dedup + validate + standardize schema
  generate_failures.py      → three-tier failure generation (LLM / heuristic / legacy)
  build_datasets.py         → gold + failures → all 4 dataset formats
  retrieval.py              → BM25/hybrid function retrieval (Vietnamese-aware)
```

### Key conventions
- All scripts import via `sys.path.insert(0, ...)` to resolve project root, not by package install.
- Scripts load config via OmegaConf from `config/` YAMLs; notebook uses hardcoded `TRAIN_CONFIG` dict.
- Logging: scripts use Python `logging`; notebook uses `print()` and `get_logger()`.
- No test framework, no type annotations enforced.

### Provider chain (for data generation)
`generate_stage2_dataset.py` tries providers in order: OpenRouter → OpenCode Zen. API keys from `OPENROUTER_API_KEY` / `OPENCODE_API_KEY` env vars. The script has built-in `_RateState` rate limiter + circuit breaker + checkpoint/resume.

### Default embedding model
`AITeamVN/Vietnamese_Embedding_v2` — cached locally in `.cache/encoders/`. FunctionRetriever auto-downloads and provides `cleanup_encoder()`.

## Active Dataset

`data/generated/v1.0_k5/` (31 telecom functions, top-5 retrieval, per-sample argument values).

| File | Records | Notes |
|------|---------|-------|
| `train_dataset_cleaned.jsonl` | 3,553 | Gold training set |
| `test_dataset_cleaned.jsonl` | 2,075 | Gold test set |
| `sft_dataset.jsonl` | 3,553 | SFT format |
| `grpo_dataset_stage2.jsonl` | — | Stage2 GRPO (generated) |
| `rcgrpo_dataset_stage2.jsonl` | — | Stage2 RC-GRPO (generated) |
| `rctp_dataset.jsonl` | 6,596 | RCTP trajectories |
| `failures_dataset.jsonl` | 3,043 | Failure trajectories |

Stage2 datasets must be generated via `python scripts/generate_stage2_dataset.py`. They are NOT checked in.

## Useful Commands

```bash
# Jupytext workflow (edit .md, sync to .ipynb)
jupytext --to md Qwen3_(4B)_GRPO_ToolCalling.ipynb -o Qwen3_(4B)_GRPO_ToolCalling.md
# Edit the .md, then:
jupytext --sync Qwen3_(4B)_GRPO_ToolCalling.ipynb

# Generate stage2 dataset
python scripts/generate_stage2_dataset.py

# Evaluate retriever with the default embedding model
python scripts/evaluate_retriever.py --methods bm25 hybrid 2>&1

# Inspect dataset stats
python scripts/inspect_dataset.py data/generated/v1.0_k5/train_dataset_cleaned.jsonl

# Validate dataset
python scripts/validate_dataset.py --data-dir data/generated/v1.0_k5

# Build all 4 training datasets (from cleaned gold + failures)
python scripts/build_datasets.py \
  --input-base data/generated/v1.0_k5/train_dataset_cleaned.jsonl \
  --input-failures data/generated/v1.0_k5/failures_dataset.jsonl \
  --output-dir data/generated/v1.0_k5 \
  --function-library data/generated/v1.0_k5/function_library.json \
  --argument-values data/generated/v1.0_k5/argument_values.json
```

## `opencode.json` Permission Constraints
- `python3 *`: allow | `git *`: allow (except `git push`: deny, `git commit`: ask)
- `rm -rf *`: ask | `sudo *`: deny
- `gh *delete*`, `gh org *`, `gh secret *`: deny
