# scripts/ — Data Pipeline

Standalone Python entry points for the ToolFormer data pipeline (generation, enrichment, validation, retrieval). All scripts can be run independently; the notebook at root can also call equivalent logic inline.

## Pipeline Order

1. **Schema** → `prepare_data.py` orchestrates everything: parses Excel or loads function schema, splits train/test, builds retrieval index, generates synthetic data, enriches with argument values.
2. **Generate** → `data_generator.py` prompts an LLM API for synthetic (query, ground_truth) pairs. Configurable workflow distribution (single_call 65%, parallel 25%, abstention 10%).
3. **Validate** → `validate_dataset.py` runs quality checks (enum violations, missing args, schema conformance). Produces `*_cleaned.jsonl` (~11% dropped).
4. **Clean** → `clean_dataset.py` batch-fixes or filters common dataset issues.
5. **Retrieve** → `retrieval.py` provides `FunctionRetriever` (BM25/hybrid, Vietnamese-aware) and `ArgumentValueRetriever`.
6. **Enrich** → `value_catalog.py` builds argument value catalogs; `vietnamese_normalizer.py` provides Vietnamese text normalization for retrieval.

## Utilities

| File | Lines | Purpose |
|------|-------|---------|
| `excel_parser.py` | 115 | Parse Excel function sheets into schema JSON |
| `inspect_dataset.py` | 274 | CLI inspection and stats for JSONL datasets |
| `nb_to_md.py` | 233 | Convert Jupyter notebook to research-writeup Markdown |
| `prepare_data.py` | 427 | **Main orchestrator** — calls all other scripts |

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
| Parse functions from Excel | `excel_parser.py` |
| Inspect dataset stats | `inspect_dataset.py` |
| Prepare data (end-to-end) | `prepare_data.py` |

## Conventions

- All scripts import via `sys.path.insert(0, ...)` to resolve project root.
- Config loaded via OmegaConf from `config/` YAMLs.
- Logging via `src.utils.logging_utils.get_logger()`.
- No test framework, no type annotations enforced.
