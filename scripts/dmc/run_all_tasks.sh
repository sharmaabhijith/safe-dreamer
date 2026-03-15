#!/bin/bash
# =============================================================================
# Submit all DMC tasks × all experiment variants as SLURM jobs.
#
# For each task, 4 variants run in parallel. The next task's jobs only start
# after all 4 variants of the previous task have finished (using --dependency).
#
# Tasks:  cheetah_run, hopper_hop, cartpole_swingup, finger_spin
# Variants: cnn, multimodal, distractor_cnn, distractor_multimodal
#
# Usage:
#   bash scripts/dmc/run_all_tasks.sh              # submit all tasks
#   bash scripts/dmc/run_all_tasks.sh cheetah_run   # submit one task only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Tasks to run -----------------------------------------------------------
if [[ $# -gt 0 ]]; then
    TASKS=("$@")
else
    TASKS=(
        cheetah_run
        hopper_hop
        cartpole_swingup
        finger_spin
    )
fi

# ---- Experiment variants ----------------------------------------------------
# Each entry: config_name  job_suffix  time_limit
VARIANTS=(
    "dmc/cnn                   cnn       24:00:00"
    "dmc/multimodal             mm        24:00:00"
    "dmc/distractor_cnn         dist-cnn  24:00:00"
    "dmc/distractor_multimodal  dist-mm   24:00:00"
)

# ---- SLURM defaults ---------------------------------------------------------
PARTITION="long"
CPUS=8
MEM="64G"
GPU="a100-sxm4-40gb:1"

cd "$PROJECT_DIR"
mkdir -p slurm_logs

PREV_JOB_IDS=""

for TASK in "${TASKS[@]}"; do
    CURRENT_JOB_IDS=()

    for VARIANT_LINE in "${VARIANTS[@]}"; do
        read -r CONFIG SUFFIX TIME <<< "$VARIANT_LINE"

        # Determine task prefix: distractor configs use "distract_", others use "dmc_"
        if [[ "$CONFIG" == *distractor* ]]; then
            TASK_NAME="distract_${TASK}"
        else
            TASK_NAME="dmc_${TASK}"
        fi

        JOB_NAME="${TASK}-${SUFFIX}"

        # Build dependency flag: wait for all previous task's jobs to finish
        DEP_FLAG=""
        if [[ -n "$PREV_JOB_IDS" ]]; then
            DEP_FLAG="--dependency=afterany:${PREV_JOB_IDS}"
        fi

        JOB_ID=$(sbatch --parsable \
            $DEP_FLAG \
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
            --wrap="
source \"${PROJECT_DIR}/scripts/setup_env.sh\"
python -u train.py --config-name ${CONFIG} env.task=${TASK_NAME}
")

        echo "Submitted: ${JOB_NAME}  (job=${JOB_ID}, config=${CONFIG}, task=${TASK_NAME})"
        CURRENT_JOB_IDS+=("$JOB_ID")
    done

    # Join current batch's job IDs with ":" for the next task's dependency
    PREV_JOB_IDS=$(IFS=:; echo "${CURRENT_JOB_IDS[*]}")
    echo "--- ${TASK}: 4 jobs submitted (${PREV_JOB_IDS}) ---"
    echo ""
done

echo "All jobs submitted. Tasks will run sequentially, 4 variants in parallel per task."
