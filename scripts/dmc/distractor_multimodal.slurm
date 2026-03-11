#!/bin/bash
#SBATCH --job-name=dmc-dist-mm
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%j_dmc-dist-mm-o.txt
#SBATCH --error=slurm_logs/%j_dmc-dist-mm-e.txt

mkdir -p slurm_logs
source "${SLURM_SUBMIT_DIR}/scripts/setup_env.sh"

python -u train.py --config-name dmc/distractor_multimodal
