#!/bin/bash
# =============================================================================
# Text-swap ablation: evaluate trained multimodal model with different text
# at test time (no retraining required).
#
# Conditions:
#   real_text     — correct task descriptions (baseline)
#   adversarial   — semantically opposite (focus on background, ignore agent)
#   nonsense      — shuffled words (vocabulary stats, no semantics)
#   random_vector — bypass CLIP, feed a fixed random context vector
#   zero_vector   — all-zeros context vector (no text signal)
#
# Usage:
#   bash ablations/run_text_swap_eval.sh
#   bash ablations/run_text_swap_eval.sh /path/to/checkpoint.pt  # custom checkpoint
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default checkpoint (trained multimodal model on distract_cheetah_run)
CHECKPOINT="${1:-/nfs-stor/salem.lahlou/asharma/logdir/ablations/distractor_multimodal/latest.pt}"
NUM_EPISODES=100
OUTPUT_DIR="${PROJECT_DIR}/ablations/results/text_swap"

# SLURM settings
PARTITION="long"
CPUS=2
MEM="64G"
GPU="a100-sxm4-40gb:1"
TIME="4:00:00"

cd "$PROJECT_DIR"
mkdir -p slurm_logs

JOB_NAME="text-swap-eval"

JOB_ID=$(sbatch --parsable \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --gres="gpu:${GPU}" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="slurm_logs/%j_${JOB_NAME}-o.txt" \
    --error="slurm_logs/%j_${JOB_NAME}-e.txt" \
    --wrap="#!/bin/bash
. \"${PROJECT_DIR}/scripts/setup_env.sh\"
python -u ablations/eval_text_swap.py \
    --checkpoint ${CHECKPOINT} \
    --num_episodes ${NUM_EPISODES} \
    --output_dir ${OUTPUT_DIR} \
    --conditions real_text adversarial nonsense random_vector zero_vector
")

echo "Submitted: ${JOB_NAME}  (job=${JOB_ID})"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Episodes per condition: ${NUM_EPISODES}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Output:  cat slurm_logs/${JOB_ID}_${JOB_NAME}-o.txt"
echo "Results: ${OUTPUT_DIR}/text_swap_results.json"
