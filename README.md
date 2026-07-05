# ToolFormer

**Telecom Tool-Calling with Reinforcement Learning — Qwen3-4B Fine-tuning for Vietnamese Telecom Function Calling**

Fine-tuning Qwen3-4B for Vietnamese telecom function calling using RC-GRPO (Reward-Conditioned Group Relative Policy Optimization) and 5 other RL algorithms. Built on Unsloth + TRL. Targets GPU T4 16GB deployment.

Pure research — no tests, no linter, no type checker, no CI, no git hooks.

---

## Key Results

| Metric | SFT Baseline | RC-GRPO (KL=0) | Improvement |
|--------|-------------|----------------|-------------|
| Func Selection | 0.275 | **0.615** | **+123.6%** |
| Task Success | 0.242 | **0.489** | **+102.1%** |
| Schema Validity | 0.244 | **0.749** | +206.6% |
| Execution Success | – | **0.729** | – |

**Ablation findings:**
- KL penalty removal is the **sole driver** of improvement (KL=0 → +53.8% vs KL=0.1)
- DAPO, CISPO at KL=0 produce **100% identical predictions** to GRPO — loss type is irrelevant at KL=0
- `num_generations=16` is sufficient for 4B model (ng8=ng16 same MD5; ng32 within noise floor)
- **Noise floor**: ~10% per-sample disagreement — any difference <10% is statistical noise
- **Retrieval**: BM25-only for inference (0.46s total, Hit@5=85%), Hybrid+F2LLM for offline data generation (Hit@5=88.6%)

---

## Quick Start

```bash
conda create -n unsloth_telco python=3.12
conda activate unsloth_telco
pip install pip install -r requirements.txt
```

Set optional API keys (needed for data generation only):

```bash
export OPENROUTER_API_KEY=sk-...
export OPENCODE_API_KEY=sk-...
export WANDB_API_KEY=...
export HF_TOKEN=...
```

**Open notebook in Jupyter** (`notebooks/` — each `.ipynb` has a paired `.md` for git-friendly editing):

```bash
# Training pipeline (data gen → SFT → RL)
jupyter notebook notebooks/qwen-toolcaller-training.ipynb

# Evaluation (load checkpoints → inference → metrics)
jupyter notebook notebooks/qwen-toolcaller-evaluation.ipynb
```

**Or run evaluation from CLI:**

```bash
python notebooks/qwen-toolcaller-evaluation.py \
    --checkpoint outputs/train_ckpts/rctp_ft_rcgrpo_KL0_numgen16 \
    --output-dir outputs/evaluation_reports/rctp_ft_rcgrpo_KL0_numgen16
```

---

## Entrypoints

| File | Role |
|------|------|
| `notebooks/qwen-toolcaller-training.ipynb` | Training notebook — data gen, SFT, RCTP-FT, GRPO, RC-GRPO, DAPO, CISPO |
| `notebooks/qwen-toolcaller-evaluation.ipynb` | Evaluation notebook — load checkpoints, run inference, produce metrics |
| `scripts/` | Standalone CLI data pipeline (18 scripts, ~6k lines). Each script runs independently. |
| `utils/export_report_docx.py` | Generate final Word report with proper tables (872 lines, 6 formatted tables) |
| `config/ablations/` | Ablation hyperparameter YAMLs (beta, num_generations, p values) |
| `AGENTS.md` | Quick-reference agent guide with gotchas, config tables, verification commands |
| `DATASET.md` | Dataset inventory, token statistics, workflow distributions |
| `REPORT.md` | Full Vietnamese research report (~50 pages) |

---

## Project Structure

```
ToolFormer/
├── notebooks/                         # Training + evaluation (jupytext-paired)
│   ├── qwen-toolcaller-training.ipynb # ~7k lines, 15 cells
│   ├── qwen-toolcaller-training.md
│   ├── qwen-toolcaller-evaluation.ipynb # ~4k lines, 10 cells
│   └── qwen-toolcaller-evaluation.md
├── scripts/                           # Data pipeline (CLI only)
│   ├── prepare_data.py                # Excel schema → internal format
│   ├── data_generator.py              # LLM API → synthetic (query, ground_truth)
│   ├── validate_dataset.py            # Quality checks → *_cleaned.jsonl
│   ├── clean_dataset.py               # Dedup + validate + standardize schema
│   ├── generate_failures.py           # 3-tier failure generation (LLM/heuristic/legacy)
│   ├── build_datasets.py              # Gold + failures → 4 dataset formats
│   ├── retrieval.py                   # BM25/Hybrid function retriever + argument value retriever
│   ├── evaluate_retriever.py          # Benchmark retrieval methods
│   ├── rerank_functions.py            # Cross-encoder reranking for retrieved functions
│   ├── vietnamese_normalizer.py       # NFKD normalization, synonym expansion
│   ├── verify_token_counts.py         # Token auditing utility
│   ├── inspect_dataset.py             # Dataset statistics viewer
│   ├── nb_to_md.py                    # Notebook-to-markdown converter
│   ├── value_catalog.py               # Argument value catalog management
│   └── excel_parser.py                # Excel schema parser
├── config/                            # Hyperparameter configs
│   └── ablations/                     # YAMLs for ablation studies
├── outputs/
│   ├── train_ckpts/                   # Trained checkpoint directories
│   │   ├── sft_model/                  # SFT baseline
│   │   ├── rctp_ft_model/              # RCTP-FT stage 1
│   │   ├── sft_grpo_model/             # SFT → GRPO
│   │   ├── sft_rcgrpo_model/           # SFT → RC-GRPO
│   │   ├── rctp_ft_grpo_model/         # RCTP-FT → GRPO
│   │   ├── rctp_ft_rcgrpo_model/       # RCTP-FT → RC-GRPO (KL=0.1)
│   │   ├── rctp_ft_rcgrpo_KL0_numgen16/ # RC-GRPO KL=0, G=16 (best)
│   │   ├── rctp_ft_rcgrpo_KL0_numgen8/  # RC-GRPO KL=0, G=8
│   │   ├── rctp_ft_rcgrpo_KL0_numgen32/ # RC-GRPO KL=0, G=32
│   │   ├── rctp_ft_rcgrpo_KL0_numgen16_dapo/   # DAPO KL=0, G=16
│   │   ├── rctp_ft_rcgrpo_KL0_numgen16_cispo/  # CISPO KL=0, G=16
│   │   ├── rctp_ft_rcgrpo_KL0_numgen16_cispo_highrewardprob=0.2/
│   │   ├── rctp_ft_rcgrpo_KL0_numgen16_cispo_highrewardprob=0.8/
│   │   ├── rctp_rcgrpo_KL0_numgen16_cispo_maxstep1000/ # CISPO KL=0, 1000 steps
│   │   └── ... (zipped checkpoints also available)
│   └── evaluation_reports/            # Per-checkpoint metric reports
│       ├── Base/                       # Untrained base model
│       ├── sft/                        # SFT baseline
│       ├── rctp_ft/                    # RCTP-FT
│       ├── rctp_ft_grpo/              # RCTP-FT → GRPO
│       ├── rctp_ft_rcgrpo/            # RCTP-FT → RC-GRPO (KL=0.1)
│       ├── rctp_ft_rcgrpo_KL0_numgen16/ # Best result
│       ├── rctp_ft_rcgrpo_KL0_numgen8/
│       ├── rctp_ft_rcgrpo_KL0_numgen32/
│       ├── rctp_ft_rcgrpo_KL0_numgen16_dapo/
│       ├── rctp_ft_rcgrpo_KL0_numgen16_cispo/
│       ├── rctp_ft_rcgrpo_KL0_numgen16_cispo_highrewardprob0.2/
│       ├── rctp_ft_rcgrpo_KL0_numgen16_cispo_highrewardprob0.8/
│       ├── rctp_rcgrpo_KL0_numgen16_cispo_maxstep1000/
│       ├── AITeamVN/                    # Retriever benchmark: Vietnamese-Embedding-v2
│       ├── BAAI/                        # Retriever benchmark: bge-m3
│       └── codefuse-ai/                 # Retriever benchmark: F2LLM-v2-0.6B
├── docs/
│   ├── REPORT.md                       # Full research report (Vietnamese)
│   ├── BASE-EVALUATION.md              # Base model benchmark
│   ├── KL-DIVERGENCE-EVAL.md           # KL divergence ablation results
│   ├── LOSS-TYPE-EVAL.md               # Loss type comparison (GRPO vs DAPO vs CISPO)
│   ├── NUM-GENERATIONS-EVAL.md         # num_generations ablation results
│   ├── RETRIEVAL-EVAL.md               # Retrieval method benchmark (BM25 vs 3 embeddings)
│   ├── DATASET.md                      # Dataset inventory & stats
│   ├── Bieu-mau-bao-cao.md             # Report template
│   └── grpo-loss-variants-survey.md    # Survey of GRPO loss variants
├── data/generated/v1.0_k5/            # Active dataset (31 functions, top-5 retrieval)
│   ├── train_dataset_cleaned.jsonl     # 3,553 gold training samples
│   ├── test_dataset_cleaned.jsonl      # 2,075 gold test samples
│   ├── sft_dataset.jsonl               # SFT format (3,553)
│   ├── grpo_dataset.jsonl              # GRPO format (3,553)
│   ├── rcgrpo_dataset.jsonl            # RC-GRPO format (3,553)
│   ├── rctp_dataset.jsonl              # RCTP trajectories (6,596)
│   ├── failures_dataset.jsonl          # Failure trajectories (3,043)
│   ├── function_library.json           # 31 function schemas
│   └── argument_values.json            # Argument value catalog
├── utils/
│   └── export_report_docx.py           # Word report generator (python-docx, 6 formatted tables)
├── qwen-models/                        # Base model files (gitignored, downloaded on first run)
│   ├── Qwen3-4B-Instruct-2507/
│   └── Qwen3-4B-unsloth-bnb-4bit/
├── AGENTS.md                           # Agent quick-reference
├── CHANGELOG.md                        # Session-by-session change log
└── ablation-studies-02072026.md        # Ablation study session notes
```

---

## RL Algorithms Implemented

| Algorithm | Abbr. | Description | Status |
|-----------|-------|-------------|--------|
| **Reward-Conditioned GRPO** | RC-GRPO | Reward token injected into system prompt; token-level conditioning | ✅ Trained & evaluated |
| **Group Relative Policy Optimization** | GRPO | Vanilla GRPO baseline (DeepSeekMath) | ✅ Trained & evaluated |
| **Dynamic Sampling Policy Optimization** | DAPO | No KL divergence, clipping-based | ✅ Trained & evaluated |
| **CISPO** (Cost-sensitive Importance Sampling PO) | CISPO | Importance-sampled version of DAPO | ✅ Trained & evaluated |
| **Gradient-corrected Trusted Policy Optimization** | GTPO | Conflict-aware gradient correction with lambda masks | 🔧 Code only |
| **MMR-GRPO** | MMR-GRPO | Diversity-aware reward reweighting from hidden states | 🔧 Code only |
| **Adaptive Virtual Sample Policy Optimization** | AVSPO | Virtual sample generation, collapse detection via ACR | 🔧 Code only |

---

## Data Pipeline

```
Excel Schema ──► prepare_data.py ──► data_generator.py (7+ LLM providers)
                      │
                      ▼
              validate_dataset.py + clean_dataset.py
                      │
                      ▼
              generate_failures.py (3-tier: LLM / heuristic / legacy)
                      │
                      ▼
              build_datasets.py ──► SFT / GRPO / RC-GRPO / RCTP
```

### Provider Chain (for data generation)
`data_generator.py` tries providers in order: OpenRouter → OpenCode Zen. API keys from `OPENROUTER_API_KEY` / `OPENCODE_API_KEY` env vars. Built-in `_RateState` rate limiter + circuit breaker + checkpoint/resume.

### Retrieval System
- **Argument Value Retriever**: Deterministic 7-level matching (exact → normalized → alias → bigram Jaccard), zero embedding dependency
- **Function Retriever**: BM25Okapi + Vietnamese normalization + synonym expansion; optional hybrid with embedding models
- **Default embedding**: `AITeamVN/Vietnamese_Embedding_v2` (cached in `.cache/encoders/`)

### Dual-Retrieval Strategy
| Phase | Method | Purpose |
|-------|--------|---------|
| **Offline** (data gen) | **Hybrid + F2LLM-v2-0.6B** | Max Hit@5 (88.6%), R@10 (93.0%) for clean training signals |
| **Online** (inference) | **BM25-only** | 0.22ms/sample, 0 GPU RAM, Hit@5=85.1% — economical & fast |

---

## Training Modes

Set `MODE` in the training notebook. Modes auto-select the appropriate dataset.

| Mode | Trainer | Dataset | Description |
|------|---------|---------|-------------|
| `sft` | `SFTTrainer` | `sft_dataset.jsonl` | Supervised fine-tuning on expert demos |
| `rctp_ft` | `SFTTrainer` | `rctp_dataset.jsonl` | Stage 1: Reward-conditioned trajectory policy FT |
| `grpo` | `GRPOTrainer` | `grpo_dataset.jsonl` | Vanilla GRPO baseline |
| `rc_grpo` | `RCGRPOTrainer` | `rcgrpo_dataset.jsonl` | Stage 2: RC-GRPO |
| `dapo` | `RCGRPOTrainer` | `rcgrpo_dataset.jsonl` | DAPO loss (no KL) |
| `cispo` | `RCGRPOTrainer` | `rcgrpo_dataset.jsonl` | CISPO loss |

---

## Evaluation Reports

Each checkpoint generates a full evaluation report with:
- 9 metrics (Func Accuracy, Param Accuracy, Schema Validity, Exec Success, Task Success, Hallucination Rate, Rejection Accuracy, Latency)
- Per-sample disagreement analysis (noise floor estimation)
- Metric summaries in CSV, JSON, Markdown
- Radar and bar charts

**Ablation evaluation docs:**

| Doc | What |
|-----|------|
| `docs/KL-DIVERGENCE-EVAL.md` | KL=0 vs KL=0.1 (6 metrics, per-sample analysis) |
| `docs/LOSS-TYPE-EVAL.md` | GRPO vs DAPO vs CISPO at KL=0 |
| `docs/NUM-GENERATIONS-EVAL.md` | G=8 vs G=16 vs G=32 |
| `docs/RETRIEVAL-EVAL.md` | BM25 vs 3 embedding models (F2LLM, bge-m3, VN-Embedding-v2) |
| `docs/BASE-EVALUATION.md` | Untrained base model benchmark |

---

## Key Conventions

- **Script imports**: All `scripts/` use `sys.path.insert(0, ...)` to resolve project root
- **Configuration**: Notebook has hardcoded `TRAIN_CONFIG` dict (source of truth); `config/*.yaml` is reference-only
- **Reward token**: Injected into **system message** (not user message) — both `_format_trajectory()` (RCTP-FT) and `inject_reward_token_into_prompt()` (RC-GRPO) target system content before `<|im_end|>`
- **GPU sharing**: `vllm_enable_sleep_mode=True` allows TRL's vLLM to share GPU via offloading
- **Dual GPU memory pools**: `gpu_memory_utilization` (Unsloth) + `vllm_gpu_memory_utilization` (TRL) — must be understood together to avoid OOM
- **Data output**: Training runs write to `outputs/train_ckpts/{mode}_model/`
- **Notebook editing**: Edit `.md` files, then `jupytext --sync` to `.ipynb`
- **Report generation**: `python3 utils/export_report_docx.py` → `VDT_2026_Report_ToolFormer_draft.docx`

---

## Environment

| Requirement | Value |
|-------------|-------|
| Python | >=3.12 (`.python-version`) |
| Conda environment | `unsloth_telco` |
| CUDA | 12.1+ |
| Package management | `requirements.txt` + `uv.lock` |
| Base model | `unsloth/Qwen3-4B-Instruct-2507` (4-bit quantized) |
| GPU target | NVIDIA T4 16GB |
| Runtime detection | Notebook auto-detects Colab, Kaggle, local |

### Kaggle Library Versions (source of truth)

| Library | Version |
|---------|---------|
| unsloth | 2026.3.17 |
| torch | 2.10.0+cu128 |
| transformers | 4.57.6 |
| trl | 0.24.0 |
| peft | 0.18.1 |
| vllm | 0.18.0 |
| datasets | 4.3.0 |
| bitsandbytes | 0.49.2 |

---

## Secrets

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | LLM API for synthetic data generation (legacy fallback) |
| `OPENCODE_API_KEY` | Primary LLM API for data generation |
| `WANDB_API_KEY` | Weights & Biases experiment tracking |
| `HF_TOKEN` | Hugging Face model downloads |

---

## License

Apache 2.0. See `LICENSE`.
