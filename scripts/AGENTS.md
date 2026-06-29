# scripts/ — Data Pipeline

**Generated:** 2026-06-29
**Branch:** main
**Commit:** a253f9a

Standalone CLI data pipeline tools for the ToolFormer project (generation, enrichment, validation, retrieval). All scripts run independently — the notebook at root does **not** import or depend on any of them at runtime (it loads only pre-built datasets from `data/generated/`).

## Pipeline Order

1. **Schema** → `prepare_data.py` orchestrates everything: parses Excel or loads function schema, splits train/test, builds retrieval index, generates synthetic data, enriches with argument values.
2. **Generate gold** → `data_generator.py` prompts an LLM API for synthetic (query, ground_truth) pairs. Configurable workflow distribution (single_call 65%, parallel 25%, abstention 10%).
3. **Validate** → `validate_dataset.py` runs quality checks (enum violations, missing args, schema conformance). Produces `*_cleaned.jsonl` (~11% dropped).
4. **Clean** → `clean_dataset.py` batch-fixes or filters common dataset issues.
5. **Generate failures** → `generate_failures.py` creates failure trajectories (three-tier: LLM / heuristic / legacy). Produces `failures_dataset.jsonl`.
6. **Build training datasets** → `build_datasets.py` consumes gold + failures to build `sft_dataset.jsonl`, `grpo_dataset.jsonl`, `rcgrpo_dataset.jsonl`, `rctp_dataset.jsonl`.
7. **Retrieve** → `retrieval.py` provides `FunctionRetriever` (BM25/hybrid, Vietnamese-aware) and `ArgumentValueRetriever`.
8. **Enrich** → `value_catalog.py` builds argument value catalogs; `vietnamese_normalizer.py` provides Vietnamese text normalization for retrieval.

## Utilities

| File | Lines | Purpose |
|------|-------|---------|
| `excel_parser.py` | 115 | Parse Excel function sheets into schema JSON |
| `inspect_dataset.py` | 274 | CLI inspection and stats for JSONL datasets |
| `nb_to_md.py` | 233 | Convert Jupyter notebook to research-writeup Markdown |
| `prepare_data.py` | 427 | **Main orchestrator** — calls all other scripts |
| `vietnamese_normalizer.py` | 106 | Vietnamese text normalization for retrieval |
| `rerank_functions.py` | 148 | Cross-encoder reranking of retrieved functions |
| `verify_token_counts.py` | 97 | Verify dataset token counts against model limits |

## Running

```bash
conda activate unsloth_telco
python scripts/prepare_data.py --config config/base_config.yaml
python scripts/validate_dataset.py --input data/generated/v1.0/train_dataset.jsonl
python scripts/inspect_dataset.py data/generated/v1.0/train_dataset_cleaned.jsonl
```

## Where to Look

| Task | File |
|------|------|
| Generate synthetic samples | `data_generator.py` |
| Validate/clean dataset | `validate_dataset.py`, `clean_dataset.py` |
| Build retrieval index | `retrieval.py` |
| Rerank retrieved functions | `rerank_functions.py` |
| Verify token counts | `verify_token_counts.py` |
| Build training datasets | `build_datasets.py` |
| Generate failures | `generate_failures.py` |
| Parse functions from Excel | `excel_parser.py` |
| Inspect dataset stats | `inspect_dataset.py` |
| Prepare data (end-to-end) | `prepare_data.py` |

## Conventions

- All scripts import via `sys.path.insert(0, ...)` to resolve project root.
- Config loaded via OmegaConf from `config/` YAMLs.
- Logging via Python logging (`logging.getLogger()`).
- No test framework, no type annotations enforced.
- Data produced under `data/generated/v1.0/` (v2.0 copy exists but not referenced by notebook).
