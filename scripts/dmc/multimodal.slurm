#!/bin/bash
#SBATCH --job-name=dmc-multimodal
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%j_dmc-multimodal-o.txt
#SBATCH --error=slurm_logs/%j_dmc-multimodal-e.txt

mkdir -p slurm_logs
source "${SLURM_SUBMIT_DIR}/scripts/setup_env.sh"

python -u train.py --config-name dmc/multimodal
