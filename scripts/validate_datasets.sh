#!/usr/bin/env bash
#
# validate_datasets.sh
# ─────────────────────
# Convenience wrapper to validate both train and test datasets.
#
# Usage:
#   bash scripts/validate_datasets.sh                          # defaults from config
#   bash scripts/validate_datasets.sh --library <path>         # custom library path
#   bash scripts/validate_datasets.sh --summary-only           # skip per-sample details
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default paths (matching config/base_config.yaml)
TRAIN_PATH="${PROJECT_DIR}/data/processed/train_dataset.jsonl"
TEST_PATH="${PROJECT_DIR}/data/processed/test_dataset.jsonl"
TRAIN_SCHEMA="${PROJECT_DIR}/data/processed/function_schema_train.json"
TEST_SCHEMA="${PROJECT_DIR}/data/processed/function_schema_test.json"
LIBRARY="${PROJECT_DIR}/data/processed/function_library.json"

# Override from command line
SUMMARY_ONLY=false
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --library) LIBRARY="$2"; shift 2 ;;
    --train) TRAIN_PATH="$2"; shift 2 ;;
    --test) TEST_PATH="$2"; shift 2 ;;
    --train-schema) TRAIN_SCHEMA="$2"; shift 2 ;;
    --test-schema) TEST_SCHEMA="$2"; shift 2 ;;
    --summary-only) SUMMARY_ONLY=true; shift ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

# Validate files exist
for f in "$TRAIN_PATH" "$TEST_PATH" "$TRAIN_SCHEMA" "$TEST_SCHEMA"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: File not found: $f" >&2
    exit 1
  fi
done

if [[ ! -f "$LIBRARY" ]]; then
  echo "WARNING: Library not found at $LIBRARY (cross-split checks will be limited)" >&2
  LIBRARY=""
fi

PYTHON="python3"
# Try .venv Windows python first
if [[ -x "${PROJECT_DIR}/.venv/Scripts/python.exe" ]]; then
  PYTHON="${PROJECT_DIR}/.venv/Scripts/python.exe"
fi

SUMMARY_ARG=""
if $SUMMARY_ONLY; then
  SUMMARY_ARG="--summary-only"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Dataset Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Train:  ${TRAIN_PATH}"
echo "  Test:   ${TEST_PATH}"
echo "  Python: ${PYTHON}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ -n "$LIBRARY" ]]; then
  exec "$PYTHON" "$SCRIPT_DIR/validate_dataset.py" \
    --train "$TRAIN_PATH" \
    --train-schema "$TRAIN_SCHEMA" \
    --test "$TEST_PATH" \
    --test-schema "$TEST_SCHEMA" \
    --library "$LIBRARY" \
    $SUMMARY_ARG
else
  exec "$PYTHON" "$SCRIPT_DIR/validate_dataset.py" \
    --train "$TRAIN_PATH" \
    --train-schema "$TRAIN_SCHEMA" \
    --test "$TEST_PATH" \
    --test-schema "$TEST_SCHEMA" \
    $SUMMARY_ARG
fi
