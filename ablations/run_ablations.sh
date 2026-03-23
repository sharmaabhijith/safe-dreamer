#!/bin/bash
# =============================================================================
# Run all ablation studies for multimodal text encoder evaluation.
#
# Task: cheetah_run (distract_cheetah_run for distractor variants)
#
# Training ablations (train from scratch):
#   A2  FiLM Only (no TextGate)
#   A3  Gate Only (no FiLM, standard CNN + TextGate)
#   A4  Full Multimodal — our method (reference)
#   A5  CNN Baseline (reference)
#   F1  Difficulty Sweep (CNN + Multimodal at easy/medium/hard)
#   H3  Wider CNN (parameter-matched, depth=77)
#
# Text-swap eval ablations (eval-only, no retraining):
#   Run via:  bash ablations/run_text_swap_eval.sh
#   Tests: real_text, adversarial, nonsense, random_vector, zero_vector
#   on the trained distractor_multimodal checkpoint.
#
# Usage:
#   bash ablations/run_ablations.sh               # submit all training ablations
#   bash ablations/run_ablations.sh a4 a5         # submit specific ones
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- SLURM defaults ---------------------------------------------------------
PARTITION="long"
CPUS=2
MEM="64G"
GPU="a100-sxm4-40gb:1"
TIME="12:00:00"

cd "$PROJECT_DIR"
mkdir -p slurm_logs

# ---- Ablation definitions ---------------------------------------------------
# Format: "ablation_id  config_name  task_name  time_limit"
ALL_ABLATIONS=(
    # Component isolation (Section A)
    # "a2   dmc/ablations/a2_film_only         distract_cheetah_run  14:00:00"
    # "a3   dmc/ablations/a3_gate_only         distract_cheetah_run  14:00:00"
    # "a4   dmc/ablations/a4_full_multimodal   distract_cheetah_run  14:00:00"
    "a5   dmc/ablations/a5_cnn_baseline      distract_cheetah_run  14:00:00"

    # Difficulty sweep (Section F1) — CNN variants
    "f1_cnn_med   dmc/ablations/f1_cnn_medium         distract_cheetah_run  12:00:00"
    "f1_cnn_hard  dmc/ablations/f1_cnn_hard            distract_cheetah_run  12:00:00"

    # Difficulty sweep (Section F1) — Multimodal variants
    "f1_mm_med    dmc/ablations/f1_multimodal_medium   distract_cheetah_run  12:00:00"
    "f1_mm_hard   dmc/ablations/f1_multimodal_hard     distract_cheetah_run  12:00:00"

    # Parameter-matched CNN (Section H3)
    "h3   dmc/ablations/h3_wider_cnn         distract_cheetah_run  14:00:00"
)

# ---- Parse which ablations to run -------------------------------------------
if [[ $# -gt 0 ]]; then
    REQUESTED=("$@")
else
    REQUESTED=()
fi

# ---- Submit jobs ------------------------------------------------------------
SUBMITTED_IDS=()

for ABLATION_LINE in "${ALL_ABLATIONS[@]}"; do
    read -r AB_ID CONFIG TASK_NAME TIME_LIMIT <<< "$ABLATION_LINE"

    # If specific ablations requested, skip non-matching ones
    if [[ ${#REQUESTED[@]} -gt 0 ]]; then
        MATCH=0
        for req in "${REQUESTED[@]}"; do
            if [[ "$AB_ID" == "$req" ]]; then
                MATCH=1
                break
            fi
        done
        if [[ $MATCH -eq 0 ]]; then
            continue
        fi
    fi

    JOB_NAME="abl-${AB_ID}"

    JOB_ID=$(sbatch --parsable \
        --job-name="$JOB_NAME" \
        --partition="$PARTITION" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS" \
        --gres="gpu:${GPU}" \
        --mem="$MEM" \
        --time="$TIME_LIMIT" \
        --output="slurm_logs/%j_${JOB_NAME}-o.txt" \
        --error="slurm_logs/%j_${JOB_NAME}-e.txt" \
        --wrap="#!/bin/bash
. \"${PROJECT_DIR}/scripts/setup_env.sh\"
python -u train.py --config-name ${CONFIG} env.task=${TASK_NAME}
")

    echo "Submitted: ${JOB_NAME}  (job=${JOB_ID}, config=${CONFIG}, task=${TASK_NAME})"
    SUBMITTED_IDS+=("$JOB_ID")
done

if [[ ${#SUBMITTED_IDS[@]} -eq 0 ]]; then
    echo "No ablations submitted. Usage:"
    echo "  bash ablations/run_ablations.sh              # all ablations"
    echo "  bash ablations/run_ablations.sh a1 a4 a5     # specific ones"
    echo ""
    echo "Available ablation IDs:"
    for ABLATION_LINE in "${ALL_ABLATIONS[@]}"; do
        read -r AB_ID CONFIG TASK_NAME TIME_LIMIT <<< "$ABLATION_LINE"
        echo "  ${AB_ID}"
    done
    exit 1
fi

echo ""
echo "============================================="
echo "Submitted ${#SUBMITTED_IDS[@]} ablation jobs."
echo "Job IDs: ${SUBMITTED_IDS[*]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "After all jobs complete, generate plots with:"
echo "  python ablations/plot_ablation_results.py \\"
echo "    --base_logdir /nfs-stor/salem.lahlou/asharma/logdir/ablations \\"
echo "    --output_dir ablations/results"
echo ""
echo "For text-swap eval ablation (no retraining), run:"
echo "  bash ablations/run_text_swap_eval.sh"
echo ""
echo "For gate analysis (E3), run:"
echo "  python ablations/plot_gate_analysis.py \\"
echo "    --logdirs /nfs-stor/salem.lahlou/asharma/logdir/ablations/ablation_a4_full_multimodal/distract_cheetah_run \\"
echo "    --labels 'Full Multimodal' \\"
echo "    --output ablations/results/e3_gate_analysis.pdf"
echo "============================================="
