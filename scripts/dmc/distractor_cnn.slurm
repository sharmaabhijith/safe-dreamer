#!/bin/bash
#SBATCH --job-name=dmc-dist-cnn
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j_dmc-dist-cnn-o.txt
#SBATCH --error=slurm_logs/%j_dmc-dist-cnn-e.txt

mkdir -p slurm_logs
source "${SLURM_SUBMIT_DIR}/scripts/setup_env.sh"

python -u train.py --config-name dmc/distractor_cnn
