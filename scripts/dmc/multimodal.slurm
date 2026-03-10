#!/bin/bash
#SBATCH --job-name=dmc-multimodal
#SBATCH --account=cscc-users
#SBATCH --partition=long
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%j_output.txt
#SBATCH --error=slurm_logs/%j_error.txt

mkdir -p slurm_logs
source "$(dirname "$0")/../setup_env.sh"

python -u train.py --config-name dmc/multimodal device=cuda:0
