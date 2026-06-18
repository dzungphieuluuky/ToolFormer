# Telco Agent RL — Complete System Roadmap

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Pipeline](#3-data-pipeline)
   - 3.1 [Function Schema Files](#31-function-schema-files)
   - 3.2 [Argument Value Catalog](#32-argument-value-catalog)
   - 3.3 [Dataset Generation](#33-dataset-generation)
   - 3.4 [Retrieval System](#34-retrieval-system)
   - 3.5 [Dataset Enrichment](#35-dataset-enrichment)
   - 3.6 [Final Dataset Format](#36-final-dataset-format)
4. [Conversation Format](#4-conversation-format)
   - 4.1 [Tag Architecture](#41-tag-architecture)
   - 4.2 [Prompt Construction](#42-prompt-construction)
   - 4.3 [SFT vs GRPO Format](#43-sft-vs-grpo-format)
5. [Training Pipeline](#5-training-pipeline)
   - 5.1 [Phase 0 — Environment Setup](#51-phase-0--environment-setup)
   - 5.2 [Phase 1 — SFT Warm-Start](#52-phase-1--sft-warm-start)
   - 5.3 [Phase 2 — RL Fine-Tuning](#53-phase-2--rl-fine-tuning)
6. [RL Algorithms](#6-rl-algorithms)
   - 6.1 [RC-GRPO](#61-rc-grpo)
   - 6.2 [AWPO](#62-awpo)
   - 6.3 [GVPO](#63-gvpo)
7. [Reward System](#7-reward-system)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Full Execution Order](#9-full-execution-order)
10. [Data Flow Diagram](#10-data-flow-diagram)
11. [File Responsibility Map](#11-file-responsibility-map)
12. [Config Key Reference](#12-config-key-reference)

---

## 1. System Overview

This project fine-tunes a **Qwen3-4B Small Language Model (SLM)** for accurate
function calling in Vietnamese telecom network operations. Given a natural-language
query from a network engineer, the model must select the correct API function and
fill in all required arguments with the right values.

Three reinforcement-learning algorithms are trained and compared:

| Algorithm | Core Innovation | Paper |
|---|---|---|
| **RC-GRPO** | Reward-conditioned token injection to solve vanishing advantage | arxiv 2602.03025 |
| **AWPO** | Variance-aware gated advantage blending + difficulty weighting | arxiv 2512.19126 |
| **GVPO** | Step-level process-verifiable advantage shaping per `<tool_call>` block | OpenReview RY47Tq0VsV |

**Base framework:** Unsloth `Qwen3-(4B)-GRPO` notebook +
TRL `GRPOTrainer` + vLLM rollout generation + 4-bit QLoRA.

**Language:** Queries are in Vietnamese. Function schemas and argument
value catalogs are bilingual (Vietnamese labels, English/code values).

---

## 2. Repository Structure

```
telco-agent-rl/
│
├── config/
│   ├── base_config.yaml           # shared hyperparameters for all runs
│   ├── rc_grpo_config.yaml        # RC-GRPO specific overrides
│   ├── awpo_config.yaml           # AWPO specific overrides
│   └── gvpo_config.yaml           # GVPO specific overrides
│
├── data/
│   ├── raw/
│   │   └── telecom_functions.xlsx  # optional: mentor-provided Excel
│   ├── processed/
│   │   ├── function_schema_train.json   # train function schemas (N-5 funcs)
│   │   ├── function_schema_test.json    # test function schemas (5 held-out)
│   │   ├── function_library.json        # merged train+test (for retriever)
│   │   ├── argument_values.json         # KPI/location/tech/provider catalogs
│   │   ├── raw_train_dataset.jsonl      # generator output (before enrichment)
│   │   ├── raw_test_dataset.jsonl       # generator output (before enrichment)
│   │   ├── train_dataset.jsonl          # final enriched training set
│   │   ├── test_dataset.jsonl           # final enriched test set
│   │   └── retrieval_index/
│   │       ├── func_embeddings.npy      # sentence-transformer embeddings
│   │       └── retriever.pkl            # serialised FunctionRetriever
│   └── synthetic/
│       └── generation_scripts/          # data synthesis code
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── excel_parser.py              # Excel → function_library.json
│   │   ├── dataset_generator.py         # LLM API → raw JSONL samples
│   │   └── retrieval.py                 # FunctionRetriever + ArgumentValueRetriever
│   │                                    # + TelcoRetriever + RetrievalResult
│   ├── reward/
│   │   ├── __init__.py
│   │   ├── base_reward.py               # extract_call, schema_valid, args_accuracy, ...
│   │   ├── rc_grpo_reward.py            # binary reward + TRL wrapper
│   │   ├── awpo_reward.py               # outcome + reasoning + variance gate
│   │   └── gvpo_reward.py               # process_reward_step + shaping term φ
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base_trainer.py              # load_model, build_grpo_config,
│   │   │                                # prompt builders, dataset loaders
│   │   ├── rc_grpo_trainer.py           # RCGRPOTrainer + train_rc_grpo()
│   │   ├── awpo_trainer.py              # AWPOTrainer + train_awpo()
│   │   └── gvpo_trainer.py              # GVPOTrainer + train_gvpo()
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py                 # evaluate_model()
│   │   ├── metrics.py                   # 9 metrics + aggregate_metrics()
│   │   └── report_generator.py          # CSV + bar chart + radar chart
│   └── utils/
│       ├── __init__.py
│       ├── logging_utils.py             # Rich-formatted logger
│       ├── model_utils.py               # load/save/merge LoRA helpers
│       └── sandbox.py                   # safe tool-call execution
│
├── scripts/
│   ├── build_argument_values.py         # writes argument_values.json
│   ├── prepare_data.py                  # full data pipeline orchestrator
│   ├── run_sft.py                       # SFT warm-start (+ --rctp flag)
│   ├── run_rc_grpo.py                   # RC-GRPO RL training
│   ├── run_awpo.py                      # AWPO RL training
│   ├── run_gvpo.py                      # GVPO RL training
│   ├── run_evaluation.py                # eval all models → report
│   ├── run_ablation.py                  # controlled ablation studies
│   └── verify_data_format.py            # print formatted samples for QA
│
├── outputs/
│   ├── sft_model/
│   ├── rctp_sft_model/                  # RC-GRPO Phase 1 checkpoint
│   ├── rc_grpo_model/
│   ├── awpo_model/
│   ├── gvpo_model/
│   └── evaluation_reports/
│       ├── metrics_summary.csv
│       ├── full_results.json
│       ├── bar_comparison.png
│       ├── radar_comparison.png
│       └── ablations/
│
├── notebooks/
│   ├── 01_explore_functions.ipynb
│   ├── 02_generate_dataset.ipynb
│   └── 03_evaluation_analysis.ipynb
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 3. Data Pipeline

### 3.1 Function Schema Files

**Source options (in priority order):**

```
Option A:  --train-schema + --test-schema   → already split, use directly
Option B:  --schema function_schema.json    → single file, split automatically
Option C:  --excel telecom_functions.xlsx   → Excel → parse → split
Option D:  function_library.json exists     → load and split
```

**Excel parser** (`src/data/excel_parser.py`):
- Reads `telecom_functions.xlsx` with columns:
  `function_name`, `description`, `parameters` (JSON),
  `example_queries`, `domain_info`, `constraints`, `tags`
- Writes `function_library.json`:
  ```json
  {
    "SPEEDTEST_PROVINCE": {
      "name": "SPEEDTEST_PROVINCE",
      "description": "Get speedtest results by province",
      "parameters": {
        "location_code": {"type": "string", "required": true, "description": "..."},
        "tech_type":     {"type": "string", "required": true, "description": "..."}
      },
      "examples": ["Xem tốc độ mạng tại TP.HCM"],
      "domain":   {"category": "performance"},
      "constraints": {},
      "tags": ["speedtest", "4G"]
    }
  }
  ```

**Train/test split** (`prepare_data.py → split_library()`):
- `reserved_test_functions: 5` (configurable via `dataset_generation.split`)
- OR explicit list via `test_function_names: [FUNC_A, FUNC_B, ...]`
- Writes:
  - `function_schema_train.json` — N-5 functions for generating train samples
  - `function_schema_test.json`  — 5 held-out functions for test samples
  - `function_library.json`      — merged (N functions, used by retriever)

---

### 3.2 Argument Value Catalog

**Script:** `scripts/build_argument_values.py`
**Output:** `data/processed/argument_values.json`

Maps **parameter semantic types** → list of valid values with codes and labels:

| Parameter Type | Example Values |
|---|---|
| `location_code` | `HCM → Thành phố Hồ Chí Minh`, `VNM → Việt Nam`, `KV1 → Khu vực miền Bắc` |
| `tech_type` | `2G`, `3G`, `4G`, `5G` |
| `network_provider` | `viettel`, `vinaphone`, `mobifone` |
| `kpi_code` | `call_drop_rate`, `data_traffic_volume`, `coverage_percentage_4g` |
| `data_level` | `day`, `week`, `month`, `year` |
| `location_type` | `NTMN → Nông thôn miền núi`, `TTLL`, `TPTW` |
| `speedtest_provider` | `ookla`, `nperf`, `internal` |
| `alarm_type` | `CRITICAL`, `MAJOR`, `MINOR`, `WARNING` |

**Parameter name matching** is handled by suffix/prefix lookup:
- `source_location_code` → matches `location_code` catalog
- `province_code` → matches `location_code` catalog

---

### 3.3 Dataset Generation

**Script:** `src/data/dataset_generator.py`
**Class:** `TelcoDatasetGenerator`

#### Provider strategy

| Provider | Mechanism | Notes |
|---|---|---|
| `openrouter` | `ThreadPoolExecutor` + OpenAI SDK + `default_headers` | Default; `:floor` suffix for cheapest routing |
| `openai` | `ThreadPoolExecutor` + OpenAI SDK | |
| `anthropic` | `ThreadPoolExecutor` + Anthropic SDK | |
| `together` | `ThreadPoolExecutor` + OpenAI SDK (Together base URL) | |
| `google` | `ThreadPoolExecutor` + Google GenAI SDK | |
| `mistral` | `ThreadPoolExecutor` + Mistral SDK | |

All providers share the same `_APIClient.complete()` call site with:
- `tenacity` retry (3 attempts, exponential back-off, `reraise=True`)
- Verbose error printing before each retry (HTTP status + body)

#### Workflow distribution

```
60%  single_call   — one function call resolves the query
20%  parallel      — multiple simultaneous function calls
15%  sequential    — chained calls (output of step N feeds step N+1)
 5%  abstention    — model should refuse (missing info / out of scope)
```

#### Generation flow

```
1. Plan tasks: (workflow_type, function_names[], n_samples)
2. ThreadPoolExecutor fires tasks concurrently
3. Each task → _APIClient.complete(SYSTEM_PROMPT, template) → raw text
4. _parse_json_list() strips markdown fences, extracts JSON array
5. Per-workflow parser creates DataSample dataclass instances
6. _simulate_retrieval() fills retrieved_functions with ground-truth fn
   + (k-1) random distractors (k=5 default)
7. Save raw_train_dataset.jsonl + raw_test_dataset.jsonl
```

#### SYSTEM_PROMPT to the generator LLM

The generator system prompt instructs the LLM to:
- Output **valid JSON only**, **queries in Vietnamese**
- Use **only the provided KPI codes** (12 codes)
- Use **only the provided location codes** (VNM, KV1-KV3, HNI, HCM, ...)
- Use **only the provided unit codes** (VCS, IDC, VTM, VTNet, VTT, ...)

#### Output sample (raw, pre-enrichment)

```json
{
  "id": "190d5a3e-b85c-49ef-9ac2-9890b96e62ca",
  "query": "Show me the average distance between stations in the NTMN area for June 2026.",
  "workflow_type": "single_call",
  "function_name": "REGIONAL_STATION_INFO",
  "ground_truth": {
    "function": "REGIONAL_STATION_INFO",
    "arguments": {
      "location_code": "VNM",
      "location_type": "NTMN",
      "from_date": "2026-06-01",
      "to_date": "2026-06-30",
      "query_type": "station_distance"
    },
    "workflow": "single_call",
    "reasoning": "User asks for station_distance, location_type NTMN, whole month."
  },
  "retrieved_functions": ["KE_HOACH_TRIEN_KHAI", "REGIONAL_STATION_INFO",
                          "TOP_TRAM_MIN", "TOP_CELL_MAX", "TOP_TRAM_MAX"],
  "split": "train"
}
```

---

### 3.4 Retrieval System

**File:** `src/data/retrieval.py`

Three classes work together:

#### `FunctionRetriever`

Finds the top-k most relevant **function names** for a query.

| Method | Mechanism |
|---|---|
| `bm25` | `BM25Okapi` on tokenised function name + description + param names |
| `embedding` | `sentence-transformers/all-MiniLM-L6-v2` cosine similarity |
| `hybrid` | `0.4 × BM25_normalised + 0.6 × embedding_normalised` |

- Embeddings are cached to `data/processed/retrieval_index/func_embeddings.npy`
- Full retriever serialised to `retrieval_index/retriever.pkl`

#### `ArgumentValueRetriever`

For each parameter in the retrieved functions, finds relevant values from the catalog.

**Scoring rules (per catalog entry):**

```
1.0   exact label match in query (normalised, diacritics removed)
0.9   alt_label match in query
0.8+  code appears as substring in query (bonus for longer/more specific codes)
0.7   multiple overlapping tokens between query and label
0.35  single overlapping token
0.0   no match
```

Vietnamese normalisation:
- Unicode NFKD decomposition → remove combining characters
- `đ/ð → d`
- Lowercase + collapse whitespace

Parameter name lookup uses suffix matching:
`source_location_code` → catalog key `location_code`

#### `TelcoRetriever`

Orchestrates both retrievers into a single `RetrievalResult`:

```python
result = telco_retriever.retrieve(query, function_library, k=5)
result.function_names    # ["SPEEDTEST_PROVINCE", "TOP_CELL_MAX", ...]
result.argument_values   # {"location_code": [ValueMatch(code="HCM", ...)], ...}
```

---

### 3.5 Dataset Enrichment

**Function:** `prepare_data.py → enrich_dataset_with_arg_values()`

Reads `raw_train_dataset.jsonl`, runs `ArgumentValueRetriever` on each sample,
adds a `retrieved_argument_values` field, writes `train_dataset.jsonl`.

```json
"retrieved_argument_values": {
  "location_code": [
    {"code": "HCM", "label": "Thành phố Hồ Chí Minh",
     "group": "Tỉnh/Thành phố", "score": 0.9, "alt_label": ""}
  ],
  "tech_type": [
    {"code": "4G", "label": "4G LTE", "group": "technology", "score": 0.8, "alt_label": ""}
  ]
}
```

This pre-computation means **zero extra retrieval cost at training time** —
the loaders simply deserialise the stored dicts.

---

### 3.6 Final Dataset Format

Every sample in `train_dataset.jsonl` / `test_dataset.jsonl` has:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | UUID4 |
| `query` | `str` | Vietnamese natural-language query |
| `workflow_type` | `str` | `single_call` / `parallel` / `sequential` / `abstention` |
| `function_name` | `str` | Primary function name (or `"none"` for abstention) |
| `ground_truth` | `dict` | `{function, arguments, workflow, reasoning, [calls]}` |
| `retrieved_functions` | `list[str]` | Top-5 simulated retrieval results |
| `retrieved_argument_values` | `dict` | Pre-computed argument value matches |
| `split` | `str` | `"train"` or `"test"` |

---

## 4. Conversation Format

### 4.1 Tag Architecture

The system uses **5 structured XML-style tags** to separate concerns:

```
<|im_start|>system                 → model persona, task instructions, output format rules
<|im_start|><user>                 → raw user query (untouched)
<|im_start|>retriever              → context injected by the retrieval system (NOT the user)
<|im_start|>assistant <reasoning>  → model's chain-of-thought (model generates this)
<|im_start|>assistant <tool_call>  → model's final function call JSON (model generates this)
```

This separation is intentional:
- `<retriever>` teaches the model that function schemas and argument hints
  come from an external system, not from the user
- `<reasoning>` is rewarded separately (reasoning quality score)
- `<tool_call>` is the verifiable output scored by all reward functions

### 4.2 Prompt Construction

**`src/algorithms/base_trainer.py`** contains all prompt builders:

```
SYSTEM_PROMPT
    ↓
build_retriever_block(function_names, function_library, argument_values)
    │
    ├── build_function_description(func_name, schema)
    │     → "### FUNC_NAME\nDescription: ...\nParameters:\n  - ..."
    │
    └── build_argument_values_block(argument_values)
          → "## Relevant Argument Values\n### location_code\n  - HCM → ..."
    ↓
build_messages_for_grpo(query, function_names, function_library, argument_values)
    → [{"role": "system", ...}, {"role": "user", "<user>query</user><retriever>...</retriever>"}]
    ↓
tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    → final prompt string (model generates from here)
```

**Complete prompt example** for the sample query:

```
<|im_start|>system
You are a telecom network operations assistant. You will receive a user
query, a list of available functions with their parameter schemas, and
relevant argument values retrieved from a knowledge base.

REQUIRED OUTPUT FORMAT:
<reasoning>
Step-by-step analysis:
- What the user is asking for
- Which function best matches and why
- Which argument values from the retriever match the query
- Final argument assignments
</reasoning>
<tool_call>
{"function": "<function_name>", "arguments": {"<param>": "<value>"}}
</tool_call>
<|im_end|>

<|im_start|>user
Tôi cần xem tốc độ tải xuống và tải lên của mạng 4G Viettel
tại TP.HCM trong ngày 15/06/2026
<|im_end|>

<|im_start|>retriever
## Available Functions

### SPEEDTEST_PROVINCE
Description: Get speedtest results by province
Parameters:
  - location_code [string] (required): Province code
  - tech_type [string] (required): Network technology (2G/3G/4G/5G)
  - from_date [string] (required): Start date YYYY-MM-DD
  - to_date [string] (required): End date YYYY-MM-DD
  - network_provider [string] (required): Telecom provider
  - data_level [string] (required): Aggregation level
  - speedtest_provider [string] (optional): Speedtest vendor

### TOP_CELL_MIN
Description: Get bottom-performing cells by KPI
Parameters:
  ...

## Relevant Argument Values

### location_code
  - HCM → Thành phố Hồ Chí Minh  [Tỉnh/Thành phố]
  - VNM → Việt Nam  [Toàn quốc]

### tech_type
  - 4G → 4G LTE  [technology]
  - 2G → 2G GSM  [technology]

### network_provider
  - viettel → Viettel  [provider]
  - vinaphone → VinaPhone (VNPT)  [provider]

### data_level
  - day → Ngày (daily)  [aggregation]

### speedtest_provider
  - ookla → Ookla Speedtest  [speedtest]
<|im_end|>

<|im_start|>assistant
```

**Model generates from here:**

```
<reasoning>
The user asks for download and upload speed of Viettel 4G network in
TP.HCM on 15/06/2026. From the retriever:
- location_code: HCM matches "TP.HCM"
- tech_type: 4G matches "mạng 4G"
- network_provider: viettel matches "Viettel"
- from_date and to_date: both "2026-06-15" (single day)
- data_level: day (single day query)
- speedtest_provider: ookla (standard default)
SPEEDTEST_PROVINCE is the correct function.
</reasoning>
<tool_call>
{"function": "SPEEDTEST_PROVINCE", "arguments": {
  "tech_type": "4G",
  "data_level": "day",
  "speedtest_provider": "ookla",
  "location_code": "HCM",
  "to_date": "2026-06-15",
  "from_date": "2026-06-15",
  "network_provider": "viettel"
}}
</tool_call>
```

### 4.3 SFT vs GRPO Format

| Aspect | SFT (`format_sample_for_sft`) | GRPO (`format_sample_for_grpo`) |
|---|---|---|
| Messages | `[system, user, assistant]` | `[system, user]` |
| `add_generation_prompt` | `False` (full conversation) | `True` (model generates) |
| Ground truth in prompt | ✅ Yes (assistant message) | ❌ No (reward function only) |
| Dataset columns | `text` | `prompt`, `ground_truth`, `query`, `workflow_type` |
| Used by | `SFTTrainer` | `GRPOTrainer` |

---

## 5. Training Pipeline

### 5.1 Phase 0 — Environment Setup

```bash
conda create -n unsloth_telco python=3.12
conda activate unsloth_telco
pip install -r requirements.txt

# Set API key for dataset generation
export OPENROUTER_API_KEY="sk-or-..."

# Run full data pipeline
python scripts/build_argument_values.py
python scripts/prepare_data.py --schema data/processed/function_schema.json

# Verify format
python scripts/verify_data_format.py --mode grpo --n 2
```

**`prepare_data.py` execution order:**

```
Step 1  Load / parse function schemas
          → function_schema_train.json  (N-5 functions)
          → function_schema_test.json   (5 held-out functions)
          → function_library.json       (merged, all N functions)

Step 2  Build argument value catalog
          → argument_values.json
          (auto-runs build_argument_values.py if file missing)

Step 3  Build retrieval index
          → retrieval_index/func_embeddings.npy
          → retrieval_index/retriever.pkl

Step 4  Generate synthetic dataset via LLM API
          → raw_train_dataset.jsonl    (~2135 samples, train functions)
          → raw_test_dataset.jsonl     (~265 samples, test functions)

Step 5  Enrich with argument values
          → train_dataset.jsonl        (final, with retrieved_argument_values)
          → test_dataset.jsonl         (final, with retrieved_argument_values)
```

### 5.2 Phase 1 — SFT Warm-Start

**Purpose:** Teach the model the output format (`<reasoning>` + `<tool_call>`)
before RL training. Without this, GRPO rollouts produce malformed outputs
and rewards are always 0.

**Two SFT modes:**

#### Standard SFT
```bash
python scripts/run_sft.py --config config/base_config.yaml
```
- Trains on complete conversations `[system, user, assistant]`
- Assistant response = `<reasoning>{gt.reasoning}</reasoning>\n<tool_call>{gt.call}</tool_call>`
- Output: `outputs/sft_model/`

#### RCTP-FT (RC-GRPO Phase 1)
```bash
python scripts/run_sft.py --config config/base_config.yaml --rctp
```
- Same as SFT but prepends `<|high_reward|>` or `<|low_reward|>` to the system prompt
- High-quality trajectories (valid function + arguments) → `<|high_reward|>`
- Abstention / missing arguments → `<|low_reward|>`
- Teaches the model to **condition output quality on reward token**
- Required before RC-GRPO Phase 2
- Output: `outputs/rctp_sft_model/`

**SFT config** (`config/base_config.yaml → sft:`):
```yaml
sft:
  num_train_epochs: 1
  learning_rate: 2.0e-4
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
```

### 5.3 Phase 2 — RL Fine-Tuning

All three algorithms use the same base:

**Model base:** `unsloth/qwen3-4b-unsloth-bnb-4bit`
**LoRA:** r=16, α=16, dropout=0, all projection matrices
**Optimiser:** AdamW 8-bit, cosine LR schedule
**Rollouts:** vLLM (`fast_inference=True`), 8 generations per prompt
**Stop token:** `</tool_call>`

```bash
python scripts/run_rc_grpo.py
python scripts/run_awpo.py
python scripts/run_gvpo.py
```

Each script:
1. Merges base config + algorithm config via `OmegaConf.merge()`
2. Calls `load_grpo_dataset()` → formats prompts using `build_messages_for_grpo()`
3. Instantiates the algorithm-specific trainer
4. Trains with `trainer.train()`
5. Saves model + tokenizer

---

## 6. RL Algorithms

### 6.1 RC-GRPO

**Paper:** arxiv 2602.03025
**Problem solved:** Vanishing advantage — when all rollouts in a group get
the same reward (all 0 or all 1), the normalised advantage `A_i = R_i − μ_g ≈ 0`
and training stalls.

**Two-phase approach:**

```
Phase 1 (RCTP-FT):
  Fine-tune on mixed trajectories where system prompt starts with
  <|high_reward|> for R=1 trajectories and <|low_reward|> for R=0.
  Objective: min_θ Σ −log π_θ(a | prompt_with_reward_token)
  Effect: model learns to produce DIFFERENT quality outputs depending
          on which reward token it sees.

Phase 2 (RC-GRPO RL):
  During rollout generation, randomly assign reward tokens:
    ~50% of rollouts per group → conditioned on <|high_reward|>
    ~50% of rollouts per group → conditioned on <|low_reward|>
  Because the model learned to condition on these tokens,
  high-token rollouts tend to get R=1, low-token rollouts tend to get R=0.
  → Within-group reward variance is RESTORED → non-zero advantages.
```

**Reward function:** Binary — `1.0` if all gates pass, else `0.0`:
```
Gate 1: schema_valid (parseable <tool_call> block)
Gate 2: func_selection_ok (correct function name)
Gate 3: args_accuracy ≥ 0.8 (at least 80% of arguments correct)
Gate 4: execution_success (sandbox, optional)
```

**Key config (`rc_grpo_config.yaml`):**
```yaml
rc_grpo:
  high_fraction: 0.5        # 50/50 token split per group
  args_threshold: 0.8
  high_reward_token: "<|high_reward|>"
  low_reward_token:  "<|low_reward|>"
```

**What makes it different from vanilla GRPO:**
The innovation is entirely in the **rollout sampling strategy**, not the loss
function. The PPO-clip loss is unchanged.

---

### 6.2 AWPO

**Paper:** arxiv 2512.19126
**Problem solved:** Pure outcome rewards miss reasoning quality;
mixing them naively destabilises training.

**Algorithm (replaces `compute_advantages` in `AWPOTrainer`):**

For each rollout group:

```
Step 1 — Outcome reward per response
  R_out_i = 0.4 × func_selection_ok
           + 0.3 × args_accuracy
           + 0.3 × execution_success

Step 2 — Reasoning reward per response (Eq. 23)
  R_reason_i = reasoning_quality(response)
              (CoT length score + structured-step bonus)

Step 3 — Variance-aware gate (binary, per group)
  σ²_out = Var(R_out_i for i in group)
  g = 0   if σ²_out > τ  → outcome already informative; skip reasoning
  g = 1   if σ²_out ≤ τ  → outcome low-variance; admit reasoning signal

Step 4 — Difficulty weight (per group, Eq. 22)
  μ_out = mean(R_out)
  w = 4 × μ_out × (1 − μ_out)
  (inverted-U: peaks at μ=0.5, zero at μ=0 or μ=1)

Step 5 — Mixed reward per response
  r_mix_i = (1−g) × R_out_i + g × R_reason_i

Step 6 — Normalised weighted advantage
  A_i = w × (r_mix_i − μ_mix) / (σ_mix + ε)
```

**Adaptive clipping (in `compute_loss`):**
When gate is open (`g=1`), reasoning rewards are higher-variance,
so the PPO clip window widens:
```
ε_effective = ε × (1 + g × δ)   where δ = 0.2
```

**Key config (`awpo_config.yaml`):**
```yaml
awpo:
  tau: 0.5                  # variance gate threshold
  adaptive_clip_delta: 0.2  # clip expansion when gate is open
  outcome_weights:
    func_selection: 0.4
    args_accuracy:  0.3
    execution:      0.3
```

---

### 6.3 GVPO

**Paper:** OpenReview RY47Tq0VsV (ICLR 2026)
**Full title:** "Group Verification-based Policy Optimization for Interactive Coding Agents"
**Problem solved:** GRPO broadcasts the same advantage to ALL tokens in a
response, giving equal gradient weight to correct and incorrect function calls
within the same trajectory — inaccurate credit assignment.

**Algorithm (reshapes advantage tensor before PPO-clip in `GVPOTrainer`):**

```
Standard GRPO advantage (flat):
  Â_i = (R_out_i − μ_g) / (σ_g + ε)   ← same for every token in response i

GVPO shaped advantage at step t:
  Ã_i,t = Â_i + b × φ(s_t, a_t)

where:
  b           = shaping coefficient (default 1.0, b=0 degrades to GRPO)
  φ(s_t, a_t) = r_proc(t) − mean_t(r_proc)   ← zero-mean within trajectory

  r_proc(t) ∈ {0,1}:
    1.0 if the t-th <tool_call> block passes ALL checks:
        schema valid + correct function + args_accuracy ≥ 0.8 + execution OK
    0.0 otherwise
```

**Token assignment:**
- Tokens inside `<tool_call>` block t → `Â_i + b × φ(t)` (shaped)
- Tokens inside `<reasoning>` block  → `Â_i` (outcome only)
- All other tokens                   → `Â_i` (outcome only)

**Token span mapping** uses `tokenizer(response, return_offsets_mapping=True)`
to go from character positions of `<tool_call>...</tool_call>` → token indices.

**Zero-mean property ensures:** the process shaping term never shifts the
overall advantage baseline — it only redistributes credit within a trajectory.

**Key config (`gvpo_config.yaml`):**
```yaml
gvpo:
  shaping_coeff: 1.0    # b; set 0 to degrade to vanilla GRPO
  args_threshold: 0.8   # threshold for process_reward_step = 1
```

---

## 7. Reward System

**File:** `src/reward/`

All reward functions follow the TRL `reward_funcs=` interface:
```python
def reward_func(completions: list[str], ground_truth: list[dict], **kwargs) -> list[float]
```

### Base components (`base_reward.py`)

| Function | Returns | Used by |
|---|---|---|
| `extract_call(response)` | Parsed `<tool_call>` dict or `None` | All |
| `extract_all_calls(response)` | List of all `<tool_call>` dicts | GVPO |
| `extract_reasoning(response)` | Text inside `<reasoning>` | AWPO |
| `schema_valid(response)` | `1.0` / `0.0` | RC-GRPO, GVPO |
| `func_selection_ok(response, expected)` | `1.0` / `0.0` | All |
| `args_accuracy(response, expected_args)` | `[0.0, 1.0]` fraction | All |
| `reasoning_quality(response)` | `[0.0, 1.0]` heuristic | AWPO |
| `format_reward(completions)` | `[0.0, 0.5, 1.0]` tag presence | All (secondary) |

### Reward function summary

```
RC-GRPO:  rc_grpo_reward_func    → binary {0, 1}
          rc_grpo_format_func    → format quality [0, 1]

AWPO:     awpo_reward_func       → continuous outcome R_out ∈ [0, 1]
          format_reward          → format quality [0, 1]
          (AWPO advantage uses reasoning_quality internally in compute_advantages)

GVPO:     gvpo_reward_func       → binary outcome R_out(τ) ∈ {0, 1}
          format_reward          → format quality [0, 1]
          (process shaping φ computed in GVPOTrainer._shape_advantages)
```

### Tag format (both input and output)

```
Output format the model must produce:
  <reasoning>...</reasoning>
  <tool_call>...</tool_call>

Backward compatibility: <call>...</call> also accepted by extract_call()
```

---

## 8. Evaluation Framework

**Scripts:** `scripts/run_evaluation.py`, `scripts/run_ablation.py`
**Files:** `src/evaluation/`

### 9 Metrics

| # | Metric | Description |
|---|---|---|
| 1 | `function_selection_accuracy` | Did the model call the correct function? |
| 2 | `argument_accuracy` | Fraction of arguments correctly filled |
| 3 | `schema_validity` | Is the `<tool_call>` block valid JSON? |
| 4 | `execution_success_rate` | Does the call run without sandbox errors? |
| 5 | `task_success_rate` | All of: correct function + args≥0.8 + execution OK |
| 6 | `hallucinated_call_rate` | Called a function not in the library (↓ better) |
| 7 | `abstention_accuracy` | Correctly refused when no function was appropriate |
| 8 | `latency_ms` | Inference time in milliseconds (↓ better) |
| 9 | `cost_per_query_usd` | Estimated token cost (↓ better) |

### Evaluation flow

```
For each model in [rc_grpo_model, awpo_model, gvpo_model]:
  For each sample in test_dataset.jsonl:
    1. Load pre-computed retrieved_functions (controlled experiment)
       OR run live FunctionRetriever (realistic scenario)
    2. Build prompt using SAME build_messages_for_grpo() as training
    3. generate_response() → measure latency
    4. compute_all_metrics(response, ground_truth, sandbox, latency, cost)
  aggregate_metrics() → per-model summary dict

generate_report([rc_grpo_results, awpo_results, gvpo_results])
  → metrics_summary.csv
  → full_results.json
  → bar_comparison.png   (5 core metrics bar chart)
  → radar_comparison.png (6 metrics radar chart)
```

### Ablation studies (`run_ablation.py`)

Each ablation disables one component, trains for 100 steps, evaluates:

| Ablation | What is disabled |
|---|---|
| `rc_grpo_no_reward_tokens` | `reward_token_injection: False` |
| `awpo_no_variance_gate` | `tau: 999.0` (gate never opens) |
| `awpo_no_difficulty_weight` | `difficulty_weighting: False` |
| `awpo_no_reasoning` | `reasoning_weight: 0.0` |
| `gvpo_no_mask` (process shaping) | `shaping_coeff: 0.0` (degrades to GRPO) |

---

## 9. Full Execution Order

```bash
# ── 0. One-time setup ─────────────────────────────────────────────────────
conda activate unsloth_telco
export OPENROUTER_API_KEY="sk-or-..."

# ── 1. Build argument value catalog ──────────────────────────────────────
python scripts/build_argument_values.py
# Output: data/processed/argument_values.json

# ── 2. Full data pipeline ─────────────────────────────────────────────────
python scripts/prepare_data.py \
    --schema data/processed/function_schema.json \
    --total 2400
# Outputs: function_schema_train.json, function_schema_test.json,
#          function_library.json, retrieval_index/,
#          raw_train_dataset.jsonl, raw_test_dataset.jsonl,
#          train_dataset.jsonl (enriched), test_dataset.jsonl (enriched)

# ── 3. Verify format ──────────────────────────────────────────────────────
python scripts/verify_data_format.py --mode grpo --n 2
python scripts/verify_data_format.py --mode sft  --n 1

# ── 4. SFT warm-start ─────────────────────────────────────────────────────
# Standard SFT (for AWPO and GVPO):
python scripts/run_sft.py
# Output: outputs/sft_model/

# RCTP-FT (for RC-GRPO Phase 1):
python scripts/run_sft.py --rctp
# Output: outputs/rctp_sft_model/

# ── 5. RL training ────────────────────────────────────────────────────────
python scripts/run_rc_grpo.py --config config/rc_grpo_config.yaml
# Output: outputs/rc_grpo_model/

python scripts/run_awpo.py --config config/awpo_config.yaml
# Output: outputs/awpo_model/

python scripts/run_gvpo.py --config config/gvpo_config.yaml
# Output: outputs/gvpo_model/

# ── 6. Evaluation ─────────────────────────────────────────────────────────
python scripts/run_evaluation.py \
    --models outputs/rc_grpo_model outputs/awpo_model outputs/gvpo_model
# Output: outputs/evaluation_reports/

# ── 7. Ablation studies ───────────────────────────────────────────────────
python scripts/run_ablation.py
# Output: outputs/evaluation_reports/ablations/
```

---

## 10. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PREPARATION                              │
│                                                                         │
│  function_schema.json ──→ split_library() ──→ function_schema_train.json│
│                                          └──→ function_schema_test.json  │
│                                          └──→ function_library.json      │
│                                                        │                │
│  argument_values.py ──→ argument_values.json           │                │
│                                │                       │                │
│                                ▼                       ▼                │
│                     ArgumentValueRetriever      FunctionRetriever       │
│                                │                       │                │
│                                └──────────┬────────────┘                │
│                                           │                             │
│  LLM API (OpenRouter) ──→ raw_train.jsonl │                             │
│                       └──→ raw_test.jsonl  │                             │
│                                           │                             │
│                                     enrich() ─────────────────────────→│
│                                           │                             │
│                             train_dataset.jsonl (final)                 │
│                             test_dataset.jsonl  (final)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                             │
│                                                                         │
│  train_dataset.jsonl                                                    │
│       │                                                                 │
│       ├─→ load_sft_dataset() ─→ SFTTrainer ─→ sft_model/               │
│       │    (format_sample_for_sft: full conversation)                   │
│       │                                                                 │
│       └─→ load_grpo_dataset() ─→ GRPOTrainer (base)                    │
│            (format_sample_for_grpo: prompt only)                        │
│                    │                                                    │
│                    ├─→ RCGRPOTrainer ─→ rc_grpo_model/                 │
│                    │    reward_funcs: [rc_grpo_reward_func,             │
│                    │                   rc_grpo_format_func]             │
│                    │    override: inject_diverse_reward_tokens()         │
│                    │                                                    │
│                    ├─→ AWPOTrainer ─→ awpo_model/                      │
│                    │    reward_funcs: [awpo_reward_func, format_reward] │
│                    │    override: compute_advantages() (AWPO algorithm) │
│                    │             compute_loss() (adaptive clip)         │
│                    │                                                    │
│                    └─→ GVPOTrainer ─→ gvpo_model/                      │
│                         reward_funcs: [gvpo_reward_func, format_reward] │
│                         override: compute_loss() (process shaping)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION & REPORTING                          │
│                                                                         │
│  test_dataset.jsonl                                                     │
│       │                                                                 │
│       └─→ evaluate_model() × 3 models                                  │
│                │                                                        │
│                ├─→ build_messages_for_grpo() (same prompt as training)  │
│                ├─→ generate_response()                                  │
│                └─→ compute_all_metrics() (9 metrics)                    │
│                                │                                        │
│                        generate_report()                                │
│                                │                                        │
│                    ┌───────────┼───────────┐                            │
│                    ▼           ▼           ▼                            │
│            metrics_summary  bar_chart  radar_chart                      │
│                .csv           .png        .png                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. File Responsibility Map

| File | Sole Responsibility |
|---|---|
| `build_argument_values.py` | Define and serialise all valid argument value catalogs |
| `prepare_data.py` | Orchestrate the 5-step data pipeline end-to-end |
| `excel_parser.py` | Parse `.xlsx` → `function_library.json` |
| `dataset_generator.py` | Call LLM API → raw `DataSample` instances |
| `retrieval.py` | `FunctionRetriever`, `ArgumentValueRetriever`, `TelcoRetriever` |
| `base_reward.py` | Shared extraction and scoring primitives |
| `rc_grpo_reward.py` | Binary gate reward + RCTP reward token constants |
| `awpo_reward.py` | Outcome components + variance gate + difficulty weight math |
| `gvpo_reward.py` | `process_reward_step` + `compute_process_shaping` + token span mapping |
| `base_trainer.py` | System prompt, all prompt builders, model loading, GRPOConfig, dataset loaders |
| `rc_grpo_trainer.py` | `RCGRPOTrainer` (token registration) + `inject_diverse_reward_tokens` |
| `awpo_trainer.py` | `AWPOTrainer.compute_advantages` + `compute_loss` (adaptive clip) |
| `gvpo_trainer.py` | `GVPOTrainer.compute_loss` (advantage reshaping) + `_shape_advantages` |
| `sandbox.py` | Safe mock execution of tool calls; validates required params + constraints |
| `benchmark.py` | Run inference on test set; measure all 9 metrics per sample |
| `metrics.py` | Compute and aggregate the 9 evaluation metrics |
| `report_generator.py` | Write CSV, bar chart, radar chart from evaluation results |
| `run_sft.py` | SFT and RCTP-FT training entry point |
| `run_rc_grpo.py` | RC-GRPO RL training entry point |
| `run_awpo.py` | AWPO RL training entry point |
| `run_gvpo.py` | GVPO RL training entry point |
| `run_evaluation.py` | Evaluate all models → generate comparative report |
| `run_ablation.py` | Run controlled ablations for each algorithm |
| `verify_data_format.py` | Print formatted samples for visual QA |

---

## 12. Config Key Reference

### `data:` section

| Key | Description |
|---|---|
| `processed_dir` | Root output directory for all processed files |
| `train_schema_path` | Path to train function schemas JSON |
| `test_schema_path` | Path to test function schemas JSON |
| `function_library_path` | Merged train+test library (used by retriever + trainer) |
| `argument_values_path` | Argument value catalog JSON |
| `raw_train_path` | Generator output before enrichment |
| `raw_test_path` | Generator output before enrichment |
| `train_path` | Final enriched training dataset (trainers load this) |
| `test_path` | Final enriched test dataset (evaluation loads this) |
| `retrieval_index_dir` | Directory for embedding cache + retriever pickle |
| `max_prompt_length` | Max tokens for the [system+user] prompt |
| `max_completion_length` | Max tokens the model generates |

### `dataset_generation:` section

| Key | Description |
|---|---|
| `provider` | `openrouter` / `openai` / `anthropic` / `together` / `google` / `mistral` |
| `model` | Model name for the chosen provider |
| `api_key_env` | Environment variable name holding the API key |
| `total_samples` | Target number of training samples to generate |
| `train_split` | Fraction of generated samples assigned to train split |
| `split.reserved_test_functions` | How many functions to hold out for testing |
| `split.test_function_names` | Explicit list of test function names (or `null` for random) |
| `workflow_distribution` | Ratios for `single_call`, `parallel`, `sequential`, `abstention` |

### `retrieval:` section

| Key | Description |
|---|---|
| `method` | `hybrid` / `bm25` / `embedding` |
| `top_k` | Number of functions to retrieve per query |
| `top_k_values` | Number of argument values to retrieve per parameter |
| `encoder_model` | Sentence transformer model for embedding retrieval |

### Algorithm-specific sections

| Section | Key | Description |
|---|---|---|
| `rc_grpo:` | `high_fraction` | Fraction of rollouts using `<\|high_reward\|>` token |
| `rc_grpo:` | `args_threshold` | Minimum args accuracy for binary reward = 1 |
| `awpo:` | `tau` | Variance gate threshold σ² |
| `awpo:` | `adaptive_clip_delta` | Clip expansion δ when reasoning gate opens |
| `awpo:` | `outcome_weights` | Component weights for func_selection, args_accuracy, execution |
| `gvpo:` | `shaping_coeff` | Process shaping coefficient b (0 = vanilla GRPO) |
| `gvpo:` | `args_threshold` | Minimum args accuracy for process reward = 1 per step |
```