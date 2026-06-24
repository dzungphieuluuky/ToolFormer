# ToolFormer Training & Inference Pipeline

This document describes how `Qwen3_(4B)_GRPO_ToolCalling.ipynb` works end-to-end: environment setup, data preparation, training (4 modes), evaluation, and inference.

---

## 1. Overview

The notebook fine-tunes **Qwen3-4B** for telecom tool-calling using Reinforcement Learning. It implements the **RC-GRPO** algorithm (Reward-Conditioned GRPO) from the paper, with vanilla GRPO and supervised fine-tuning (SFT) as baselines.

**Single file, no imports from `src/`.** Every function, class, and training loop lives inline in the notebook (30 cells, ~10,500 lines).

### Supported Training Modes

| Mode | Cell | Description |
|------|------|-------------|
| `sft` | 24 | Supervised Fine-Tuning on expert demonstrations |
| `rctp_ft` | 25 | Stage 1: Reward-Conditioned Trajectory Policy FT (SFT-style) |
| `rc_grpo` | 26 | Stage 2: RC-GRPO (RL with reward-token conditioning) |
| `grpo` | 26 | Vanilla GRPO baseline (no reward-token conditioning) |

Set `MODE = "sft"` in Cell 23 to select.

---

## 2. Environment & Setup (Cells 4–9)

The notebook auto-detects its runtime environment:

```
configure_environment_paths() → ENV_NAME ∈ {"colab", "kaggle", "local"}
```

| Environment | Package Install | Data Mount |
|-------------|----------------|------------|
| **Colab** | `uv pip install` from PyPI | Clones repo from GitHub |
| **Kaggle** | `pip --no-index` from `/kaggle/input/toolformer-wheels/` | Datasets already mounted |
| **Local** | Regular `pip install` | Uses local filesystem |

Key secrets loaded (optional): `OPENROUTER_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN`.

---

## 3. Core Data Structures (Cells 14–18)

### 3.1 Function Schema (`function_schema_train.json` / `test.json`)

Defines available telecom functions with their parameters, types, and descriptions. 5 functions are held out for the test set to measure generalization.

### 3.2 Function Library (`function_library.json`)

Combined train + test schemas, used by the retriever and reward functions.

### 3.3 Argument Values (`argument_values.json`)

A catalog of valid argument values (KPIs, locations, dates) organized by parameter name. Used for retrieval-augmented generation:

```
ValueCatalog:
  param_name → [CatalogEntry(code, label, group, alt_labels, score)]
```

### 3.4 Dataset Files (under `data/processed/`)

| File | Contents |
|------|----------|
| `train_dataset.jsonl` | 3293 training samples with pre-computed retrievals |
| `test_dataset.jsonl` | 1907 test samples (includes 5 held-out functions) |
| `raw_train/test_dataset.jsonl` | Pre-enrichment generator output |

**Dataset schema per JSONL line:**

| Field | Type | Notes |
|-------|------|-------|
| `query` | str | User query (telecom operations) |
| `ground_truth` | dict | `{calls: [...], workflow: "...", reasoning: "..."}` |
| `retrieved_functions` | list[str] | Top-5 pre-retrieved function names |
| `retrieved_argument_values` | dict | Pre-computed argument value matches |
| `workflow_type` | str | `single_call` (59%), `parallel` (21%), `sequential` (15%), `abstention` (5%) |
| `id` | str | Unique sample ID |
| `function_name` | str | Gold function name |

### 3.5 Trajectory Dataclass (Cell 18)

```python
@dataclass
class Trajectory:
    prompt_messages: list[dict]    # Chat messages (system + user)
    response_text: str             # Gold assistant response
    reward: int                    # 1 = success, 0 = failure
    reward_token: str              # "<|high_reward|>" or "<|low_reward|>"
    gold_function: str | None
    gold_arguments: dict | None
    gold_calls: list[dict] | None
    workflow_type: str
    function_name: str | None
    id: str | None
```

---

## 4. Retrieval System (Cell 15)

### 4.1 Function Retriever

```python
FunctionRetriever(method="bm25" | "hybrid")
```

- **BM25**: Keyword-based retrieval using `rank_bm25`
- **Hybrid**: BM25 + sentence embeddings (weighted combination)
- Vietnamese-aware: normalization → synonym expansion → tokenization

### 4.2 Argument Value Retriever

```python
ArgumentValueRetriever(argument_values_catalog)
```

Maps query tokens to parameter values via token overlap scoring. Returns the top-N matching values for each parameter.

### 4.3 TelcoRetriever (Composite)

```python
TelcoRetriever.build(function_library, argument_values_catalog, method="hybrid")
```

Returns a `RetrievalResult` with:
- `function_names`: matched functions
- `argument_values`: matched argument values per parameter

---

## 5. Conversation Format & Chat Template (Cell 20)

### 5.1 Custom Roles

The tokenizer is patched with custom roles:

```
system, user, assistant, retriever
```

Plus reward tokens `<|high_reward|>` and `<|low_reward|>` registered as additional special tokens.

### 5.2 Message Structure

```
[system]:   SYSTEM_PROMPT (telecom rules + tool definitions)
[retriever]: Retrieved function schemas + argument value catalog
[user]:     Query from dataset
[assistant]: <reasoning>...</reasoning>
              <tool_call>{"function": "...", "arguments": {...}}</tool_call>
```

Multiple `<tool_call>` blocks for sequential/parallel workflows. Abstention uses `<tool_call>null</tool_call>`.

### 5.3 SYSTEM_PROMPT

Contains the telecom operations assistant prompt with:
- Task instructions (reason then call)
- Required output format (`<reasoning>` + `<tool_call>`)
- Strict rules (no hallucinated functions, abstention protocol)
- Examples of single, sequential, and parallel tool call formats

### 5.4 Prompt Building Functions

| Function | For | Returns |
|----------|-----|---------|
| `build_messages_for_grpo()` | GRPO/RC-GRPO | Message list with retriever block + system + user |
| `build_messages_for_sft()` | SFT | Message list with system + user (+ assistant gold) |
| `format_sample_for_grpo()` | Dataset loader | `{"prompt": str, "ground_truth": json_str, "query": str, "workflow_type": str}` |
| `format_sample_for_sft()` | Dataset loader | `{"text": str}` (full chat template) |

---

## 6. Model Loading & LoRA (Cell 20)

```python
def load_model(config):
    # 1. PatchFastRL("GRPO", FastLanguageModel)  — required for GRPO training
    # 2. FastLanguageModel.from_pretrained(...)    — load 4-bit base model
    # 3. FastLanguageModel.get_peft_model(...)     — attach LoRA adapters
    # 4. patch_tokenizer_for_custom_roles(...)     — register reward tokens
    # 5. model.resize_token_embeddings(...)        — resize for new tokens
```

### Key Parameters

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen3-4B-unsloth-bnb-4bit` |
| Max seq length | 2048 |
| 4-bit | Yes (bnb, pre-quantized) |
| LoRA rank | 32 (configurable, TRAIN_CONFIG) |
| LoRA alpha | `lora_rank * 2` (= 64) |
| LoRA dropout | 0.0 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Gradient checkpointing | `"unsloth"` |

---

## 7. Training Modes

### 7.1 SFT — Supervised Fine-Tuning (Cell 24)

**When:** `MODE == "sft"`

```
load_sft_dataset() → Dataset with {"text": chat_formatted_string}
       ↓
SFTTrainer(model, tokenizer, train_dataset=dataset)
       ↓
trainer.train() → save model
```

- Trains on full expert trajectories (query → reasoning + tool call)
- Dataset is formatted as full chat text via `tokenizer.apply_chat_template()`
- No reward tokens involved

**Config** (from `TRAIN_CONFIG["sft"]`):
- LR: 2e-5, batch: 2, epochs: 3, max_seq: 2048
- Gradient accumulation: 4, warmup: 0.1

### 7.2 RCTP-FT — Stage 1 (Cell 25)

**When:** `MODE in ("rctp_ft", "rc_grpo")`

Reward-Conditioned Trajectory Policy Fine-Tuning (SFT-style with reward token injection).

```
build_rctp_dataset_from_jsonl() → List[Trajectory]
  (Creates 1:1 expert:corrupted pairs)
       ↓
For each trajectory:
  Inject [Reward Goal: <|high_reward|>] or [Reward Goal: <|low_reward|>]
  into the system message (Appendix B)
       ↓
Format via _format_trajectory() → {"text": chat_string}
       ↓
SFTTrainer(model, tokenizer, train_dataset=dataset)
```

Key details:
- Each expert sample gets one corrupted (failure) counterpart
- Corruption: permutes arguments, wrong function names, wrong workflow ordering
- Reward token is injected into the system message (`[Reward Goal: <token>]`)
- **The checkpoint becomes `π_ref` for Stage 2** (Algorithm 1, line 9)
- `high_reward_probability` is computed from the empirical dataset success rate

**Config** (from `TRAIN_CONFIG["rctp_ft"]`):
- LR: 1e-5, batch: 4, epochs: 1

### 7.3 RC-GRPO — Stage 2 RL (Cell 26)

**When:** `MODE == "rc_grpo"`

```
load_grpo_dataset() → Dataset with {"prompt": str, "ground_truth": json_str, ...}
       ↓
RCGRPOTrainer(
    model,           # π_θ (with LoRA)
    tokenizer,
    reward_funcs=[rc_grpo_reward_func, rc_grpo_format_func],
    train_dataset=dataset,
    high_reward_probability=p,    # from Stage 1 stats (Eq. 3)
)
```

**RCGRPOTrainer** (Cell 19, subclass of TRL's `GRPOTrainer`):

1. **Reward token sampling** (Eq. 3): For each group of G prompts, sample reward tokens with probability `p` for `HIGH`, `1-p` for `LOW`
2. **Token injection**: Insert `[Reward Goal: <token>]` before `<|im_start|>assistant\n` in the prompt string
3. **Generation**: vLLM generates G completions per prompt
4. **Evaluation**: Two reward functions:
   - `rc_grpo_reward_func` — binary action coverage (all gold calls must match, Eq. 5)
   - `rc_grpo_format_func` — format shaping (valid JSON, reasoning tags present)
5. **Advantage normalization**: Group-normalized with `ddof=0` (population std, Eq. 6)
6. **Loss**: Clipped surrogate + KL penalty (Eq. 8–9, TRL's GRPO loss)

**Config** (from `TRAIN_CONFIG["training"]` + `["grpo"]`):
- LR: 1e-6, G=5 generations, symmetric epsilon=0.2, KL beta=0.1
- Max prompt: 1536, max completion: 512
- vLLM sampling: temp=1.0, top_p=0.95, min_p=0.05, stop at `</tool_call>`

### 7.4 Vanilla GRPO Baseline (Cell 26)

**When:** `MODE == "grpo"`

Same dataset as RC-GRPO, but:
- Uses `GRPOTrainer` (no reward-token conditioning)
- Reward functions: `function_reward`, `argument_reward`, `format_reward` (three separate signals)
- No `high_reward_probability` parameter

---

## 8. Reward Functions (Cells 17–19)

### 8.1 Binary Reward (RC-GRPO)

```python
compute_action_coverage_reward(agent_calls, gold_calls) → 0 or 1
```

- Every gold call must have a matching agent call (function name match + per-parameter string comparison)
- No partial credit
- Abstention: agent produces no calls ↔ gold calls empty → reward 1

### 8.2 Continuous Rewards (Vanilla GRPO)

| Function | Measures |
|----------|----------|
| `function_reward` | Correct function selection |
| `argument_reward` | Correct argument values |
| `format_reward` | Valid JSON + reasoning tag presence |

### 8.3 Format Shaping Rewards

- `rc_grpo_format_func`: Ensures `<reasoning>` and `<tool_call>` tags are present
- `format_reward`: General format compliance

---

## 9. Evaluation (Cell 28)

After training, runs `evaluate_model()` on the test set:

```python
evaluate_model(
    model_path=MODE_OUTPUT_DIR,
    test_dataset_path=TRAIN_CONFIG["data"]["test_path"],
    function_library=function_library,
    retriever=retriever,
    sandbox=Sandbox(function_library),  # executes tool calls in a safe sandbox
    top_k=5,
    max_new_tokens=512,
    model_name_tag=MODE,
    condition_on_high_reward=(MODE == "rc_grpo"),  # inject [Reward Goal: <|high_reward|>]
    argument_values=argument_values_catalog,
)
```

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| Function Selection Accuracy | Correct function picked |
| Argument Accuracy | Correct argument values used |
| Schema Validity | Output is valid JSON + follows schema |
| Execution Success Rate | Tool call executed without error |
| Task Success Rate | Full task completed correctly |
| Hallucinated Call Rate | Rate of calling non-existent functions |
| Abstention Accuracy | Correct abstention on unanswerable queries |
| Latency (ms) | Response generation time |
| Cost / Query (USD) | Estimated API cost |

Output: tabulated results + generated report.

---

## 10. Inference (Cell 10)

```python
def load_model_for_inference(
    model_path: str,
    base_model_name: str = "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    # Load a saved checkpoing for inference (no LoRA config needed)
    model, tokenizer = FastLanguageModel.from_pretrained(...)
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_response(
    model, tokenizer,
    query: str,
    retrieved_functions: list[str],
    function_library: dict,
    argument_values: dict | None = None,
    include_all_threshold: int = 10,
    max_new_tokens: int = 512,
    condition_on_high_reward: bool = False,
) -> str:
    # 1. Build messages via build_messages_for_grpo()
    # 2. Optionally inject [Reward Goal: <|high_reward|>]
    # 3. Apply chat template → prompt string
    # 4. model.generate() with vLLM
    # 5. Decode and return
```

---

## 11. Cleanup (Cell 29)

```python
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()
```

---

## 12. File Layout Reference

```
Qwen3_(4B)_GRPO_ToolCalling.ipynb   ← Single source of truth
config/
  base_config.yaml                   ← Reference only (notebook may differ)
data/
  processed/
    train_dataset.jsonl              ← 3293 training samples
    test_dataset.jsonl               ← 1907 test samples
    function_schema_train.json       ← Function defs (train split)
    function_schema_test.json        ← 5 held-out function defs (test)
    function_library.json            ← Combined function library
    argument_values.json             ← Value catalog for retrieval
    retrieval_index/                 ← BM25/hybrid search index
    raw_train/test_dataset.jsonl     ← Pre-enrichment generator output
```

---

## 13. Quick Start

```bash
conda activate unsloth_telco
# Open the notebook in Colab, Kaggle, or locally.
# Set MODE in Cell 23 and run all.

# For inference-only on an existing checkpoint:
python3 -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('outputs/sft_model')
FastLanguageModel.for_inference(model)
# ... generate responses
"
```
