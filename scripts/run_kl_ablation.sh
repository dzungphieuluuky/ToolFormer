#!/usr/bin/env bash
# =============================================================================
# KL Ablation Study — Stage 2 GRPO/RC-GRPO
#
# Runs all 4 arms sequentially via papermill.  Each arm sets KL_COEFFICIENT
# env var and trains for N_STEPS with the same TRAIN_CONFIG except for β.
#
# Prerequisites:
#   pip install papermill   (or: conda install papermill -c conda-forge)
#
# Usage:
#   export MODE=grpo              # or rc_grpo
#   export N_STEPS=500            # per arm
#   bash scripts/run_kl_ablation.sh
#
# Arms (see .omo/plans/kl-ablation-study.md for rationale):
#   A: β=0.0     — reference-free (DAPO-style, arXiv 2503.14476)
#   B: β=0.001   — minimal safety net (GRPO-Zero)
#   C: β=0.01    — moderate, best per arXiv 2512.07611
#   D: adaptive  — FG-ExPO accuracy-conditioned (arXiv 2605.11403)
#
# Output directories:
#   outputs/grpo_model_0.0/
#   outputs/grpo_model_0.001/
#   outputs/grpo_model_0.01/
#   outputs/grpo_model_adaptive/
# =============================================================================

set -euo pipefail

MODE="${MODE:-grpo}"
N_STEPS="${N_STEPS:-500}"
NOTEBOOK="Qwen3_(4B)_GRPO_ToolCalling.ipynb"

# Check prerequisites
if ! command -v papermill &> /dev/null; then
    echo "ERROR: papermill not found. Install with: pip install papermill"
    exit 1
fi
if [ ! -f "$NOTEBOOK" ]; then
    echo "ERROR: $NOTEBOOK not found in current directory"
    exit 1
fi

echo "============================================"
echo "KL Ablation Study"
echo "  Notebook:   $NOTEBOOK"
echo "  Mode:       $MODE"
echo "  Steps/arm:  $N_STEPS"
echo "============================================"

# Map KL_COEFFICIENT value → output dir suffix
run_arm() {
    local kl_val="$1"
    local label="$2"
    local output_suffix="${kl_val}"

    echo ""
    echo "─── Arm ${label}: β = ${kl_val} ───"
    echo "  Output: outputs/${MODE}_model_${output_suffix}/"

    KL_COEFFICIENT="${kl_val}" papermill \
        "${NOTEBOOK}" \
        "outputs/notebooks/kl_ablation_${MODE}_${output_suffix}.ipynb" \
        -p MODE "${MODE}" \
        -p KL_COEFFICIENT "${kl_val}" \
        --kernel python3 \
        --request-save-on-cell-execute \
        2>&1 | tail -5

    echo "  Done."
}

# Arm A: β = 0.0
run_arm "0.0" "A"
# Arm B: β = 0.001
run_arm "0.001" "B"
# Arm C: β = 0.01
run_arm "0.01" "C"
# Arm D: Adaptive (FG-ExPO)
run_arm "adaptive" "D"

echo ""
echo "============================================"
echo "All 4 arms completed."
echo "Results:"
echo "  outputs/${MODE}_model_0.0/"
echo "  outputs/${MODE}_model_0.001/"
echo "  outputs/${MODE}_model_0.01/"
echo "  outputs/${MODE}_model_adaptive/"
echo "============================================"
