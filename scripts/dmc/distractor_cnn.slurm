#!/bin/bash
#SBATCH --job-name=dmc-dist-cnn
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j_dmc-dist-cnn-o.txt
#SBATCH --error=slurm_logs/%j_dmc-dist-cnn-e.txt

mkdir -p slurm_logs
source "${SLURM_SUBMIT_DIR}/scripts/setup_env.sh"

python -u train.py --config-name dmc/distractor_cnn
