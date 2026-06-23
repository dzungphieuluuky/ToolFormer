# ToolFormer — Agent Guide

Telecom tool-calling RL project: fine-tuning Qwen3-4B on function calling using RC-GRPO (Reward-Conditioned GRPO). Built on Unsloth + TRL.

## Single Source of Truth

`Qwen3_(4B)_GRPO_ToolCalling.ipynb` (repo root) is the **only** implementation. All `src/` and `scripts/` directories have been deleted — the notebook contains every function, class, and training loop inline.

Config YAMLs under `config/` are **reference only**; the notebook uses a hardcoded `TRAIN_CONFIG` dict. If they conflict, the notebook wins.

## Quick Start

```bash
conda activate unsloth_telco
# Then open and run the notebook in Colab, Kaggle, or locally.
# No other entrypoints exist.
```

The notebook is environment-aware and auto-detects:
- **Colab:** `uv pip install`, clone repo from GitHub
- **Kaggle:** `pip --no-index --find-links=/kaggle/input/toolformer-wheels/`, skip git clone, read data from `/kaggle/input/toolformer-data/`
- **Local:** regular `pip install`, skip git clone

## Data Pipeline (pre-computed)

`data/processed/` contains everything trainers load. No re-generation needed:

```
data/processed/
  train_dataset.jsonl         3293 samples — what trainers load
  test_dataset.jsonl          1907 samples — what evaluation loads
  function_schema_train.json   Function defs for training (train split)
  function_schema_test.json    5 held-out function defs for test
  function_library.json        Combined train + test function library
  argument_values.json         Value catalog for retrieval enrichment
  raw_train_dataset.jsonl      Pre-enrichment generator output (optional)
  raw_test_dataset.jsonl       Pre-enrichment generator output (optional)
  retrieval_index/             BM25/hybrid search index (optional at train time)
```

Data pipeline order (historical, all done):
1. Function schema → split train/test (5 functions held out)
2. Build `argument_values.json`
3. Build retrieval index
4. Generate ~2400 synthetic samples via LLM (OpenRouter, `openai/gpt-oss-20b:free`)
5. Enrich with argument values → final JSONL files

**Raw CSV data** (unused after enrichment) at `data/raw/` (gitignored).

## Dataset Schema

`train_dataset.jsonl` / `test_dataset.jsonl` — each line a JSON object:

| Field | Type | Notes |
|---|---|---|
| `query` | str | User query |
| `ground_truth` | dict | `{calls: [...], workflow: "...", reasoning: "..."}` |
| `retrieved_functions` | list[str] | Top-5 pre-retrieved function names |
| `retrieved_argument_values` | dict | `{"param_name": [{code, label, group, score, alt_label}, ...]}` |
| `workflow_type` | str | `single_call` (59%), `parallel` (21%), `sequential` (15%), `abstention` (5%) |
| `id` | str | Unique ID |
| `function_name` | str | Gold function name |

**ground_truth critical detail:** In the raw JSONL this is a dict. When loaded into a `datasets.Dataset` via `load_grpo_dataset()`, `format_sample_for_grpo()` serialises it to JSON string via `json.dumps()` to avoid pyarrow `ArrowInvalid` (heterogeneous argument value types). All reward functions parse it back via `_parse_gt()` which handles both string and dict.

## Training Pipeline (RC-GRPO, two stages)

### Stage 1 — RCTP-FT (SFT-style)
- `build_rctp_dataset_from_jsonl()` reads JSONL, builds 1:1 expert:failure trajectories
- For each sample: expert response from `ground_truth.calls` (R=1), synthesized corrupted response (R=0)
- Skips abstention (no tool_call to learn)
- `RCTPDataset` injects `[Reward Goal: <|high_reward|>]` or `[Reward Goal: <|low_reward|>]` into the **first user message** (Appendix B), then tokenizes
- Standard CE training loop (AdamW, linear warmup, gradient clipping)
- Checkpoint becomes `π_ref` for Stage 2

### Stage 2 — RC-GRPO (RL via TRL)
- `load_grpo_dataset()` reads JSONL, builds prompts via `build_messages_for_grpo()`, serialises `ground_truth` to JSON string
- `RCGRPOTrainer` (subclasses TRL's `GRPOTrainer`):
  - For each group of G prompts (G=`num_generations`): samples reward tokens per Eq. 3 (`P_sample(r)` with `p = high_reward_probability` from Stage 1 stats)
  - Injects `[Reward Goal: <token>]` before `<|im_start|>assistant\n` in the prompt string
  - vLLM generates G completions
  - Reward funcs evaluate: `rc_grpo_reward_func` (binary action coverage, Eq. 5) + `rc_grpo_format_func` (tag/JSON format shaping)
  - Group-normalized advantages using `ddof=0` (population std, Eq. 6 exact)
  - TRL's GRPO loss (clipped surrogate + KL penalty, Eq. 8-9)

## Conversation Format

Custom chat template (`patch_tokenizer_for_custom_roles`) with `im_start`/`im_end` tokens:

```
system:  SYSTEM_PROMPT (telecom ops rules + output format)
user:    query (from JSONL)
retriever: function descriptions + argument value catalog (NOT folded into system/user)
assistant: <reasoning>...</reasoning>\n<tool_call>{"function":"...","arguments":{...}}</tool_call>
```

Multiple `<tool_call>` blocks for sequential/parallel workflows. vLLM sampling stops at `</tool_call>` but includes it in output.

## Reward Tokens

| Token | Purpose |
|---|---|
| `<|high_reward|>` | Condition generation toward success (R=1) |
| `<|low_reward|>` | Condition generation toward failure (R=0) |

Registered as `additional_special_tokens`, embedding resized via `model.resize_token_embeddings()`. In Stage 1 injected into user message; in Stage 2 injected into prompt string before assistant marker.

## Binary Reward Logic

`compute_action_coverage_reward(agent_calls, gold_calls)` → `0` or `1`:
- Every gold call must have a matching agent call (function match + per-parameter string comparison)
- No partial credit
- Abstention: agent produces no calls ↔ gold calls empty → reward 1

**Reward functions used by TRL:**
- `build_trl_reward_functions("rc_grpo")` → `[rc_grpo_reward_func, rc_grpo_format_func]`
- `build_trl_reward_functions("grpo")` → `[function_reward, argument_reward, format_reward]`

## Retrieval

- `FunctionRetriever`: BM25 or hybrid (BM25 + sentence embeddings), Vietnamese-aware (normalize → synonym expansion → tokenization)
- `ArgumentValueRetriever`: matches query tokens against value catalogs
- Dataset samples carry pre-computed `retrieved_functions` + `retrieved_argument_values`; real retriever falls back at training time if needed

## Model Config (Notebook Values — Source of Truth)

| Parameter | RCTP-FT (Stage 1) | RC-GRPO (Stage 2) |
|---|---|---|
| Base model | `unsloth/Qwen3-4B-Base` | same |
| LoRA rank | 16 | 16 (some cells: 32) |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | same |
| 4-bit | yes | yes |
| Max seq length | 1024 | 1024 (notebook); 2048 (config YAML) |
| Learning rate | 1e-5 | 1e-6 |
| Batch size | 4 | 1 (per device) |
| Grad accum | 1 | 4 |
| Num generations (G) | — | 5–8 |
| Max steps | — | 50 (test) / 500 (full) |
| Epsilon (clip) | — | 0.2 (symmetric); config says 0.28 asymmetric |
| KL coef | — | 0.1 (notebook); 0.04 (config YAML) |

**Important:** The notebook has two conflicting `load_model` definitions (fixed by keeping only the one that calls `resize_token_embeddings`). Do not add a second definition.

## Environment

- **Python ≥3.12** (`.python-version`), conda env `unsloth_telco`
- **CUDA 12.1+** required (4-bit QLoRA via Unsloth, vLLM)
- `.venv/` exists locally (Windows Python at `.venv/Scripts/python.exe`)
- `uv` is the notebook's package manager (on Colab)
- **Secrets:** `OPENROUTER_API_KEY` (dataset gen), `WANDB_API_KEY` (optional logging), `HF_TOKEN` (optional hub)
- **No linter, no type checker, no pre-commit, no test framework** — pure research code

## Gotchas

- `PatchFastRL("GRPO", FastLanguageModel)` **must** be called before model loading
- Output model directories are **empty** — no trained checkpoints exist yet
- Config YAMLs use different hyperparameters from the notebook (e.g., `max_seq_length: 2048` vs notebook's `1024`). Trust the notebook.
- `run_awpo.py` and `run_gvpo.py` are referenced in README but were never implemented
- When evaluating, inject `[Reward Goal: <|high_reward|>]` at inference time (Sec 4.1) — model was trained conditioned on this
- The `PIPELINE.md` file at root is empty
