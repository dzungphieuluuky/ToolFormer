# Changelog

All notable changes to the ToolFormer project.

## [2026-06-27] — Session 1: Full 8K context, prompt truncation fix, PatchFastRL removal

### Added
- Cross-encoder reranking for retrieved functions (`scripts/rerank_functions.py`, `scripts/data_generator.py`) [24f4c9b]
- `scripts/verify_token_counts.py` utility for dataset token auditing [58b9580]

### Changed
- Notebook cleanup, remove dead code paths and trailing noise (`Qwen3_(4B)_GRPO_ToolCalling.ipynb`) [514b3f9]
- Bump `max_seq_length` to 8192, conditional `gpu_memory_utilization` (0.3 for SFT, 0.8 for GRPO), disable `smart_truncate`, fix `np.isnan` TypeError guards [6ebbb3e]
- Fix 4-bit loading, add `full_finetuning` mode, use `train_on_responses_only` for SFT, bump `logging_steps` to 5 [3f6657e]
- Minor trailing fix to notebook (completing earlier refactors) [edb1444]

### Fixed
- GRPO prompt truncation: `max_prompt_length` 3584→7680, `max_completion_length` 256→512, `vllm_max_model_length=8192`. Zero prompts now exceed limits (was 9.9% overflow). [13628d7]
- Per-sample `retrieved_argument_values` in `build_datasets.py` (was incorrectly using global catalog fallback) [58b9580]

### Removed
- Obsolete `PatchFastRL` calls — modern Unsloth handles GRPO patching internally with `fast_inference=True`. Removed imports, calls, and docstrings from notebook; updated `AGENTS.md`, `PIPELINE.md`, `scripts/AGENTS.md`. [f5641cd]

### Docs
- Add `models/` to `.gitignore` [71c9f31]
- New `README.md` (replaces stale deleted version) and jupytext-paired `Qwen3_(4B)_GRPO_ToolCalling.md` [2b9df3b]

---

### Commit Reference

| Hash | Description |
|------|-------------|
| 24f4c9b | Add cross-encoder reranking utility |
| 3f6657e | Fix 4-bit, add full_finetuning, train_on_responses_only |
| 58b9580 | Per-sample argument values, add verify_token_counts.py |
| 6ebbb3e | Bump max_seq_length to 8192, conditional gpu_memory_util |
| f5641cd | Remove PatchFastRL dependency |
| 13628d7 | Fix GRPO prompt truncation |
| edb1444 | Minor trailing notebook fix |
| 2b9df3b | Add README.md + jupytext paired .md |
| 71c9f31 | Update .gitignore for models/ |
| 514b3f9 | Notebook cleanup, latest version |
