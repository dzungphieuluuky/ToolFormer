# ToolFormer — Agent Guide

Telecom tool-calling RL project: fine-tuning Qwen3-4B on function calling using RC-GRPO (Reward-Conditioned GRPO). Built on Unsloth + TRL.

## Single Source of Truth

`Qwen3_(4B)_GRPO_ToolCalling.ipynb` (repo root) is the **only** implementation — fully **standalone** with zero external Python file imports. It contains every function, class, and training loop inline.

The `scripts/` directory contains **standalone CLI data pipeline tools** (generation, validation, retrieval, dataset building). The notebook loads only pre-built dataset files from `data/generated/`; it does **not** import or depend on any `scripts/` module at runtime.

Config YAMLs under `config/` are **reference only**; the notebook uses a hardcoded `TRAIN_CONFIG` dict. If they conflict, the notebook wins.

## Quick Start

```bash
conda activate unsloth_telco
# Then open and run the notebook in Colab, Kaggle, or locally.
# No other entrypoints exist for training.
```

The notebook is environment-aware and auto-detects:
- **Colab:** `uv pip install`, clone repo from GitHub
- **Kaggle:** `pip --no-index --find-links=/kaggle/input/toolformer-wheels/`, skip git clone, read data from `/kaggle/input/toolformer-data/`
- **Local:** regular `pip install`, skip git clone

## Data Pipeline (pre-computed, two versions)

Data lives under `data/generated/` with two versioned copies (`v1.0` / `v2.0`). The notebook's `TRAIN_CONFIG.data` dict points to `data/generated/v1.0/`. All dataset files are pre-built; no re-generation needed.

```
data/generated/v1.0/
  # ── Gold data (source) ──────────────────────────────
  train_dataset.jsonl              ~3.3k samples — original generator output
  train_dataset_cleaned.jsonl      ~3.3k samples — validated/filtered
  test_dataset.jsonl               ~1.9k samples — original generator output
  test_dataset_cleaned.jsonl       ~1.9k samples — validated/filtered

  # ── Schemas & catalogs ──────────────────────────────
  function_schema_train.json       Function defs for training (train split)
  function_schema_test.json        5 held-out function defs for test
  function_library.json            Combined train + test function library
  argument_values.json             Value catalog for retrieval enrichment

  # ── Failure samples (via generate_failures.py) ─────
  failures_dataset.jsonl           Synthesised failure trajectories (per-sample corruptions)

  # ── Pre-built dataset files (via build_datasets.py) ─
  sft_dataset.jsonl     (~75 MB)  — gold only, full-text format → SFTTrainer
  grpo_dataset.jsonl    (~76 MB)  — gold only, prompt+ground_truth → GRPOTrainer
  rcgrpo_dataset.jsonl  (~76 MB)  — gold only, identical format to grpo → RC-GRPO
  rctp_dataset.jsonl    (~142 MB) — gold+failure trajectories → RCTP-FT

  # ── Index ───────────────────────────────────────────
  retrieval_index/                 BM25/hybrid search index (optional at train time)
```

Data pipeline order (historical; run once to produce these files):
1. **Generate gold** → `scripts/data_generator.py` prompts LLM API for synthetic (query, ground_truth) pairs
2. **Validate** → `scripts/validate_dataset.py` quality-checks; outputs `*_cleaned.jsonl`
3. **Build catalogs** → `scripts/value_catalog.py` / `argument_values.json`
4. **Build retrieval index** → `scripts/retrieval.py` (BM25/hybrid, Vietnamese-aware)
5. **Generate failures** → `scripts/generate_failures.py` → `failures_dataset.jsonl` (three-tier: LLM / heuristic / legacy)
6. **Build all training datasets** → `scripts/build_datasets.py` → `sft_dataset.jsonl`, `grpo_dataset.jsonl`, `rcgrpo_dataset.jsonl`, `rctp_dataset.jsonl`
7. **Enrich** → Each dataset builder embeds `retrieved_functions` + `retrieved_argument_values` in each record

**Raw CSV data** (unused after enrichment) at `data/raw/` (gitignored).

## Notebook Training Modes

The notebook uses a `MODE` variable to select which training block runs:

| MODE | Trainer | Dataset loaded | Description |
|------|---------|----------------|-------------|
| `sft` | `SFTTrainer` | `sft_dataset.jsonl` | Supervised fine-tuning on expert demonstrations |
| `rctp_ft` | `SFTTrainer` | `rctp_dataset.jsonl` | Stage 1: Reward-Conditioned Trajectory Policy FT |
| `grpo` | `GRPOTrainer` | `grpo_dataset.jsonl` | Vanilla GRPO baseline (no reward tokens) |
| `rc_grpo` | `RCGRPOTrainer` | `rcgrpo_dataset.jsonl` | Stage 2: Reward-Conditioned GRPO |

All four modes load from the corresponding pre-built dataset under `data/generated/v1.0/`.

## Dataset Schemas

### Base schema (`train_dataset.jsonl` / `test_dataset_cleaned.jsonl`)

| Field | Type | Notes |
|---|---|---|
| `query` | str | User query |
| `ground_truth` | dict | `{calls: [...], workflow: "...", reasoning: "..."}` |
| `retrieved_functions` | list[str] | Top-5 pre-retrieved function names |
| `retrieved_argument_values` | dict | `{"param_name": [{code, label, group, score, alt_label}, ...]}` |
| `workflow_type` | str | `single_call` (59%), `parallel` (21%), `sequential` (15%), `abstention` (5%) |
| `id` | str | Unique ID |
| `function_name` | str | Gold function name |

### Pre-built SFT schema (`sft_dataset.jsonl`)

| Field | Type | Notes |
|---|---|---|
| `text` | str | Full chat-formatted text including system+user+retriever+assistant (all turns). No ground_truth needed at train time — the assistant response is the completion. |

### Pre-built GRPO/RC-GRPO schema (`grpo_dataset.jsonl` / `rcgrpo_dataset.jsonl`)

| Field | Type | Notes |
|---|---|---|
| `prompt` | str | Chat-formatted prompt ending with `<\|im_start\|>assistant\n` — does NOT include the assistant response |
| `ground_truth` | str | JSON-serialised dict (string, not object) — serialised to avoid pyarrow `ArrowInvalid` on heterogeneous arg types |
| `query` | str | Original user query (carried through for logging) |
| `workflow_type` | str | `single_call`, `parallel`, `sequential`, `abstention` |

### Pre-built RCTP schema (`rctp_dataset.jsonl`)

| Field | Type | Notes |
|---|---|---|
| `prompt_messages` | list[dict] | Chat messages (system, user, retriever) **including** `[Reward Goal: <token>]` injected in the system message |
| `response_text` | str | The full assistant response (expert or failure) |
| `reward` | int | `1` for expert (success) trajectory, `0` for failure trajectory |

**ground_truth critical detail:** In base JSONL this is a dict. When serialised to string for GRPO datasets, all reward functions parse it back via `_parse_gt()` which handles both string and dict.

## Failure Generation (`scripts/generate_failures.py`)

Three-tier hybrid approach (tried in order):

1. **Tier 1 (LLM):** `TelcoDatasetGenerator._call_failure()` via OpenRouter/other provider. Returns up to 4 failure candidates per sample.
2. **Tier 2 (heuristic):** Catalog-aware heuristic — picks wrong function from `retrieved_functions`, alternates values from `argument_values`, drops non-required args, or hallucinates fake params.
3. **Tier 3 (legacy):** Original `__WRONG__` prefix behaviour as last resort.

Failure types: `wrong_function`, `wrong_value`, `missing_arg`, `hallucinated_arg`.

```bash
python scripts/generate_failures.py \
    --input data/generated/v1.0/train_dataset_cleaned.jsonl \
    --output data/generated/v1.0/failures_dataset.jsonl \
    --failures-per-sample 1 \
    --provider openrouter --model openai/gpt-oss-120b:free
```

## Dataset Builder (`scripts/build_datasets.py`)

Consumes gold data + failures to produce all four pre-built dataset files:

```bash
python scripts/build_datasets.py \
    --input-base data/generated/v1.0/train_dataset_cleaned.jsonl \
    --input-failures data/generated/v1.0/failures_dataset.jsonl \
    --output-dir data/generated/v1.0
```

What it builds:
- **SFT**: gold only, `build_messages_for_sft()` → full-text with `apply_chat_template()` and `add_generation_prompt=False`
- **GRPO**: gold only, `build_messages_for_grpo()` → serialised prompt + ground_truth string
- **RC-GRPO**: identical to GRPO (same function, separate file for convenience)
- **RCTP**: gold + failures, `build_rctp_dataset()` → 1:1 expert:failure `Trajectory` objects with reward-token injection in the **system message**

## Training Pipeline (RC-GRPO, two stages)

### Stage 1 — RCTP-FT (SFT-style) [notebook cell, MODE="rctp_ft"]
- Loads `rctp_dataset.jsonl` (pre-built)
- Each record has `prompt_messages` (with `[Reward Goal: <token>]` injected in **system message**) + `response_text` + `reward` label
- `_format_trajectory()` appends the assistant response and tokenizes via the custom chat template
- Trained with `SFTTrainer`, `completion_only_loss=True`, `SFTConfig` (AdamW, warmup, gradient clipping)
- Computes `high_reward_probability` = empirical success rate from the dataset (Eq. 3)
- Checkpoint becomes `π_ref` for Stage 2

### Stage 2 — RC-GRPO (RL via TRL) [notebook cell, MODE="rc_grpo"]
- Loads `rcgrpo_dataset.jsonl` via `load_jsonl()` → `Dataset.from_list()`
- `RCGRPOTrainer` (subclasses TRL's `GRPOTrainer`):
  - Overrides `_generate_and_score_completions`: for each group of G prompts, samples reward tokens per Eq. 3 (`P_sample(r)` with `p = high_reward_probability` from Stage 1 stats)
  - Injects `[Reward Goal: <token>]` into the **system message** (via `inject_reward_token_into_prompt()`, which targets the system content before `<|im_end|>`)
  - vLLM generates G completions
  - Reward funcs evaluate: `rc_grpo_reward_func` (binary action coverage, Eq. 5) + `rc_grpo_format_func` (tag/JSON format shaping)
  - Group-normalized advantages using `ddof=0` (population std, Eq. 6 exact)
  - TRL's GRPO loss (clipped surrogate + KL penalty, Eq. 8-9)

### SFT mode [notebook cell, MODE="sft"]
- Loads `sft_dataset.jsonl` pre-built full-text
- `SFTTrainer` with `completion_only_loss=True`
- Standard CE on the assistant portion only

### Vanilla GRPO mode [notebook cell, MODE="grpo"]
- Loads `grpo_dataset.jsonl`
- Standard `GRPOTrainer` (no reward-token conditioning)
- Uses `function_reward`, `argument_reward`, `format_reward`

## Conversation Format

Custom chat template (`patch_tokenizer_for_custom_roles`) with `im_start`/`im_end` tokens:

```
system:  SYSTEM_PROMPT (telecom ops rules + output format)
[Reward Goal: <|high_reward|>]   ← injected only for RCTP-FT / RC-GRPO (in system message)
user:    query (from JSONL)
retriever: function descriptions + argument value catalog (NOT folded into system/user)
assistant: <reasoning>...</reasoning>\n<tool_call>{"function":"...","arguments":{...}}</tool_call>
```

Multiple `<tool_call>` blocks for sequential/parallel workflows. vLLM sampling stops at `</tool_call>` but includes it in output.

**Note on paper vs code difference:** The paper (Appendix B) diagrams the reward token appended to the **first user message**, but the actual implementation injects it into the **system message** (both in `_format_trajectory` for RCTP-FT and `inject_reward_token_into_prompt` for RC-GRPO). This is a deliberate adaptation to keep the user query clean for retrieval matching.

## Reward Tokens

| Token | Purpose |
|---|---|
| `<|high_reward|>` | Condition generation toward success (R=1) |
| `<|low_reward|>` | Condition generation toward failure (R=0) |

Registered as `additional_special_tokens`, embedding resized via `model.resize_token_embeddings()`. Injected into the **system message** (both stages), not the user message.

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
- `ArgumentValueRetriever`: matches query tokens against value catalogs. Date / data_level extracted via regex (no catalog).
- Dataset samples carry pre-computed `retrieved_functions` + `retrieved_argument_values`; real retriever falls back at training time if needed

## Model Config (Notebook Values — Source of Truth)

| Parameter | SFT | RCTP-FT (Stage 1) | GRPO / RC-GRPO (Stage 2) |
|---|---|---|---|
| Base model | `unsloth/Qwen3-4B-unsloth-bnb-4bit` | same | same |
| LoRA rank | 16 | 16 | 16 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | same | same |
| 4-bit quantisation | yes | yes | yes (4bit, fast_inference=True) |
| Max seq length | 2048 | 2048 | 2048 |
| Learning rate | 2e-5 | 2e-5 | 1e-6 |
| Batch size | 1 | 2 | 1 (per device) |
| Grad accum | 8 | 4 | 4 |
| Num epochs | 3 | 3 | — |
| Num generations (G) | — | — | 5 |
| Max steps | — | — | 50 (test) / 500 (full) |
| LR scheduler | linear warmup | linear warmup | cosine warmup |
| Epsilon (clip) | — | — | 0.2 (symmetric) |
| KL coef | — | — | 0.1 |
| Optimizer | adamw_8bit | adamw_8bit | adamw_8bit |

## Evaluation

The notebook evaluation cell (after training, MODE dependent):
1. Loads the trained model via `load_model_for_inference()`
2. Reads `test_dataset_cleaned.jsonl`
3. For each sample: retrieves functions (uses pre-computed if available, else BM25 fallback)
4. For `rc_grpo` MODE: injects `[Reward Goal: <|high_reward|>]` at inference time (Sec 4.1)
5. Generates response (temperature=0.0, greedy)
6. Computes 7 metrics: `function_selection_accuracy`, `argument_accuracy`, `schema_validity`, `execution_success_rate`, `task_success_rate`, `hallucinated_call_rate`, `abstention_accuracy`
7. Outputs aggregate table + CSV + JSON + plots (bar/radar)

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
- Config YAMLs use different hyperparameters from the notebook. Trust the notebook.
- `run_awpo.py` and `run_gvpo.py` are referenced in README but were never implemented
- When evaluating, inject `[Reward Goal: <|high_reward|>]` at inference time (Sec 4.1) — model was trained conditioned on this
- The `PIPELINE.md` file at root is empty
- Reward-token injection targets the **system message**, not the user message (departs from the paper's Appendix B diagram). This is intentional to preserve user query cleanliness for retrieval.
