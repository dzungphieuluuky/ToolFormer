# ToolFormer — Agent Guide

**Updated:** 2026-06-27 — End of session with unified `load_model()`

Telecom tool-calling RL project: fine-tuning Qwen3-4B on function calling using RC-GRPO (Reward-Conditioned GRPO). Built on Unsloth + TRL. Pure research — no tests, no linter, no type checker, no CI, no git hooks.

Codegraph is **indexed** — use `codegraph_explore` before reading source files.

## Entrypoints

| File | Role |
|------|------|
| `Qwen3_(4B)_GRPO_ToolCalling.ipynb` | **Sole implementation** — training, eval, inference, all inline (~10.5k lines, 30 cells). Does **not** import `scripts/` or `src/`. |
| `scripts/` | Standalone CLI data pipeline. Each script runs independently. Notebook loads pre-built datasets only. |
| `config/*.yaml` | Reference only. Notebook's hardcoded `TRAIN_CONFIG` dict wins when they diverge. |
| `PIPELINE.md` | Architectural walkthrough of the notebook. Reference for details not covered here. |
| `README.md` | **Stale** — references `run_awpo.py` / `run_gvpo.py` / `run_sft.py` / `run_rc_grpo.py` that were never implemented. Ignore it. |

## Quick start

```bash
conda activate unsloth_telco
# Then open and run the notebook in Colab, Kaggle, or locally.
# No other entrypoints exist for training.
```

Secrets (optional): `OPENROUTER_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN`.

## Environment

- **Python >=3.12**, conda `unsloth_telco`, CUDA 12.1+ (`.python-version`)
- `pyproject.toml` declares zero runtime deps — all deps in `requirements.txt` / `uv.lock`, installed by the notebook, not by `pip install -e .`
- `.venv/` exists locally; pre-quantized weights at `unsloth-Qwen3-4B-unsloth-bnb-4bit/`
- Notebook auto-detects Colab/Kaggle/local and adjusts paths (Kaggle overrides model path to local cached copy)

## Training (set `MODE` in the notebook)

| MODE | Trainer | Dataset | Description |
|------|---------|---------|-------------|
| `sft` | `SFTTrainer` | `sft_dataset.jsonl` | Supervised fine-tuning on expert demos |
| `rctp_ft` | `SFTTrainer` | `rctp_dataset.jsonl` | Stage 1: Reward-Conditioned Trajectory Policy FT |
| `grpo` | `GRPOTrainer` | `grpo_dataset.jsonl` | Vanilla GRPO baseline |
| `rc_grpo` | `RCGRPOTrainer` | `rcgrpo_dataset.jsonl` | Stage 2: RC-GRPO (RL with reward-token conditioning) |

Only SFT has trained weights: `outputs/sft_model/checkpoint-1335/`. GVPO / RC-GRPO directories exist but are empty.

## Model config (`TRAIN_CONFIG` in notebook — source of truth)

| Parameter | SFT | RCTP-FT | GRPO / RC-GRPO |
|-----------|-----|---------|----------------|
| Base model | `unsloth/Qwen3-4B-Instruct-2507` | same | same |
| LoRA rank | 16 | 16 | 16 |
| 4-bit | yes | yes | yes + `fast_inference=True` |
| Max seq len | **8192** | 8192 | 8192 |
| LR | 2e-5 | 2e-5 | 1e-6 |
| Batch / grad accum | 2 / 8 | 2 / 8 | 2 / 8 |
| Epochs | 3 | 3 | — (max_steps=50 test / 500 full) |
| Num generations | — | — | 5 |
| Optimizer | `adamw_8bit` | `adamw_8bit` | `adamw_8bit` |

Config YAMLs (`config/*.yaml`) diverge from the notebook. Notable: `num_generations: 8` (YAML) vs 5 (notebook), `kl_coef: 0.04` vs 0.1, `max_grad_norm: 0.1` vs 1.0, `gpu_memory_utilization: 0.7` vs 0.5, `args_threshold: 0.8` vs exact binary match. The notebook wins.

## Critical gotchas

### Unified `load_model()` — 4 branches
The notebook has one `load_model()` function that handles all loading. Its branching:

| Condition | What happens | Used by |
|-----------|-------------|---------|
| `adapter_model_path` set + `mode="train"` | Tokenizer from checkpoint → load base (internal GRPO patching) → resize → `load_adapter()` | GRPO/RC-GRPO resume |
| `adapter_model_path` set + `mode="inference"` | Tokenizer from checkpoint → load base → resize → `load_adapter()` → `for_inference()` | Evaluation from checkpoint |
| `adapter_model_path=None` + `mode="train"` | Tokenizer from base model → load base → `get_peft_model` (fresh LoRA) → `patch_tokenizer_for_custom_roles` → resize | SFT/RCTP-FT from scratch |
| `adapter_model_path=None` + `mode="inference"` | Tokenizer from base model → load base → `for_inference()` | Base model benchmarking |

### Dual GPU memory pools
The notebook manages **two separate vLLM memory reservations**:

| Parameter | Controls | Value logic |
|-----------|----------|------------|
| `gpu_memory_utilization` | Unsloth's vLLM engine (in `load_model()`) | `0.3` when `UNSLOTH_VLLM_STANDBY=0` (SFT/RCTP-FT), `0.8` when `=1` (GRPO/RC-GRPO) |
| `vllm_gpu_memory_utilization` | TRL's internal vLLM for generation (`GRPOConfig`) | `0.3` (hardcoded in `build_grpo_config()`) |

With `UNSLOTH_VLLM_STANDBY=1` + `vllm_enable_sleep_mode=True`, Unsloth engine offloads to CPU on startup and TRL's engine offloads during optimizer steps. Changing either value without understanding the split can OOM.

### `load_in_4bit` + `fast_inference` interaction
`fast_inference=True` (needed for GRPO vLLM) is compatible with `load_in_4bit=True`, but `max_lora_rank` can only be passed alongside `load_in_4bit` when `fast_inference=False`. The unified `load_model()` handles this: `max_lora_rank` is only added to kwargs when `adapter_model_path is None and not fast_inference and mode == "train"`.

### Reward token injection location
Injected into the **system message**, not the user message (departs from the paper's Appendix B diagram). Both `_format_trajectory()` (RCTP-FT) and `inject_reward_token_into_prompt()` (RC-GRPO) target the system content before `<|im_end|>`.

### vLLM stop token
```python
SamplingParams(
    stop=["</tool_call>"],
    include_stop_str_in_output=True,  # stop string INCLUDED in output
)
```

### Data paths
`TRAIN_CONFIG["data"]` points to `data/generated/v1.0/`. `v1.0_k5/` (top-5 functions, per-sample argument values) and `v2.0/` copies exist but are not referenced by the notebook.

### `src/utils/logging_utils.py`
Exists but **not used** by the notebook (inline logging instead).

## Data pipeline (scripts/ — CLI only)

```
prepare_data.py      → orchestrator (parses schema, generates data, enriches)
  data_generator.py  → LLM API → synthetic (query, ground_truth) pairs
  validate_dataset.py → quality checks → *_cleaned.jsonl (~11% dropped)
  generate_failures.py → three-tier failure generation (LLM / heuristic / legacy)
  build_datasets.py  → gold + failures → sft/grpo/rcgrpo/rctp dataset files
  retrieval.py       → BM25/hybrid function retrieval (Vietnamese-aware)
  value_catalog.py   → argument value lookup with scoring
```

Each script runs standalone. See `scripts/AGENTS.md` for details.

## Key conventions

- All `scripts/` import via `sys.path.insert(0, ...)` to resolve project root, not by package install
- Config loaded via OmegaConf from `config/` YAMLs (for scripts), but the notebook uses a hardcoded `TRAIN_CONFIG` dict
- Logging: scripts use Python logging; the notebook uses inline `print()` and a `get_logger()` helper
- Data is generated in `data/generated/v1.0/`; training runs output to `outputs/{mode}_model/`
- The `smart_truncate()` function exists as legacy (preserved, disabled) — 8192 max_seq_len fits all samples with compact per-sample argument values
- `vllm_enable_sleep_mode=True` is used in `GRPOConfig` to share GPU with the main model

## Quick verification commands

```bash
grep "vllm_gpu_memory_utilization" Qwen3_(4B)_GRPO_ToolCalling.ipynb
ls outputs/*/                            # check training outputs
ls data/generated/v1.0/*.jsonl | head   # check dataset files
python scripts/build_datasets.py --input-base data/generated/v1.0/train_dataset_cleaned.jsonl
python scripts/validate_dataset.py --input data/generated/v1.0/train_dataset.jsonl
python scripts/inspect_dataset.py data/generated/v1.0/train_dataset_cleaned.jsonl
```
