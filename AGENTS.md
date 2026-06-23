# ToolFormer — Agent Guide

Telecom tool-calling RL project: fine-tuning Qwen3-4B on function calling with GRPO variants (RC-GRPO, AWPO, GVPO). Built on Unsloth + TRL.

## Single Source of Truth

`Qwen3_(4B)_GRPO_ToolCalling.ipynb` (repo root) is authoritative over the extracted `src/` and `scripts/` code. If code in scripts conflicts with the notebook, the notebook wins.

## Quick Start Commands

```bash
# Data pipeline (schema → synthetic data → enriched dataset)
python scripts/prepare_data.py
python scripts/prepare_data.py --skip-generation   # re-enrich existing raw files
python scripts/prepare_data.py --excel data/raw/telecom_functions.xlsx

# SFT warm-start
python scripts/run_sft.py
python scripts/run_sft.py --rctp   # RCTP-FT (RC-GRPO Phase 1)

# RL training
python scripts/run_rc_grpo.py

# Evaluate
python scripts/run_evaluation.py --models outputs/rc_grpo_model outputs/sft_model
python scripts/run_ablation.py

# Validate
bash scripts/validate_datasets.sh
python scripts/validate_dataset.py --train ... --test ...
```

> **Note:** `run_awpo.py` and `run_gvpo.py` do not exist yet (referenced in README but unimplemented).

## Architecture

### Source Layout
```
src/
  algorithms/     base_trainer.py (model loading, dataset formatting, GRPO config)
                  rc_grpo_trainer.py (RCTP-FT, RC-GRPO loss, RCGRPOTrainer TRL subclass)
  reward/         base_reward.py (format/function/argument/composite rewards + extract_call)
                  rc_grpo_reward.py (binary reward, reward tokens)
  data/           dataset_generator.py (LLM-driven synthetic data)
                  retrieval.py (BM25/hybrid function retriever + argument value retriever)
                  excel_parser.py
  evaluation/     benchmark.py / metrics.py / report_generator.py
  utils/          model_utils.py / sandbox.py / logging_utils.py
                  vietnamese_normalizer.py / value_catalog.py
```

### Data Pipeline Order
1. Function schema → split into `function_schema_train.json` / `function_schema_test.json` (5 functions held out for test)
2. Build `argument_values.json` (`scripts/build_argument_values.py`)
3. Build retrieval index (`data/processed/retrieval_index/`)
4. Generate ~2400 synthetic samples via LLM → `raw_train_dataset.jsonl` + `raw_test_dataset.jsonl`
5. Enrich with argument values → `train_dataset.jsonl` + `test_dataset.jsonl` (what trainers load)

### Training Pipeline (RC-GRPO)
- **Stage 1 (RCTP-FT):** Build synthetic success+failure trajectories from the training dataset, fine-tune Qwen3-4B with `<|high_reward|>` / `<|low_reward|>` conditioning tokens prepended to system prompt. Both stages are combined in `run_rc_grpo.py`.
- **Stage 2 (RC-GRPO):** Reward-conditioned rollout sampling (Eq. 3), group-normalized advantages (Eq. 6), clipped surrogate + KL penalty (Eq. 8-9).

### Workflow Distribution
- `single_call` 60%, `parallel` 20%, `sequential` 15%, `abstention` 5%

## Model Config (Notebook Source of Truth)

```yaml
model:
  name: "unsloth/Qwen3-4B-Base"
  max_seq_length: 1024             # base_config.yaml says 2048; notebook says 1024
  load_in_4bit: true
  fast_inference: true
  gpu_memory_utilization: 0.95
lora:
  r: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  use_gradient_checkpointing: "unsloth"
training:                         # RL stage (Table 10)
  learning_rate: 1e-6
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  num_generations: 5               # group size G
  max_steps: 50-500                # notebook:50 for test, config:500 for full
  max_grad_norm: 1.0
grpo:
  temperature: 1.0
  epsilon: 0.2                     # clip ratio
  kl_coefficient: 0.1
  loss_type: "grpo"
rctp_ft:                          # SFT stage (Table 11)
  learning_rate: 1e-5
  batch_size: 4
  num_epochs: 1
  failures_per_expert: 1           # 1:1 success:failure ratio
```

## Key Implementation Details

### Prerequisite before any RL training
```python
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)   # must be called before model loading
```

### Conversation Format
- Uses `im_start`/`im_end` tokens (custom chat template in `base_trainer.py`)
- Distinct `retriever` role for function descriptions + argument values (not folded into system/user)
- Output: `<reasoning>...</reasoning>` and `<tool_call>{"function":...,"arguments":{...}}</tool_call>` tags
- Multi-call: multiple `<tool_call>` blocks in one response
- vLLM sampling stops at `</tool_call>` but includes the stop string in output

### Reward Tokens
- `<|high_reward|>` and `<|low_reward|>` registered as `additional_special_tokens`
- For RCTP-FT: prepended to system prompt content (not a separate message)
- For RC-GRPO Stage 2: injected as `[Reward Goal: <|high_reward|>]` in the user message

### ground_truth Handling
- Stored as a **JSON string** per sample (not a dict) to avoid pyarrow struct-schema inference issues (`ArrowInvalid` fix)
- All consumers must use `_parse_gt()` / `normalize_ground_truth()` which handles both string and dict forms

### RC-GRPO Binary Reward
- Based on `compute_action_coverage_reward()`: checks every gold call has a matching agent call (function name match + parameter-for-parameter string comparison)
- Returns 0 or 1 (no partial credit)

### Sandbox (Execution Reward)
- Mock executor: validates required params exist, checks constraints (min/max/enum), returns `{status, result}`
- Used for execution reward component (10% weight in composite reward)
- Does not actually call APIs — purely schema validation

### Retrieval
- `FunctionRetriever`: BM25 or hybrid (BM25 + sentence embedding), Vietnamese-aware (normalize → expand synonyms → tokenize)
- `ArgumentValueRetriever`: matches query tokens against value catalogs, extracts dates/data_level from query patterns
- Dataset samples come with pre-computed `retrieved_functions` + `retrieved_argument_values`; real retriever fallback at training time

### Costs / Defaults (from base_config.yaml)
- Dataset generation: OpenRouter (`openai/gpt-oss-20b:free`), 2400 samples, 8 workers, 500 RPM
- Evaluation cost estimate: $0.0002/1k tokens

## Environment
- Python ≥3.12 (`.python-version`), conda env `unsloth_telco`
- CUDA 12.1+ required (4-bit QLoRA, vLLM)
- `.venv/` exists locally (Windows Python at `.venv/Scripts/python.exe`)
- Secrets: `OPENROUTER_API_KEY` (dataset gen), `WANDB_API_KEY`, `HF_TOKEN`
- `uv` is the package manager used in the notebook
- No pre-commit hooks, no linter config, no type checker configured
