#!/bin/bash
#SBATCH --job-name=safe-dreamer
#SBATCH --account=cscc-users
#SBATCH --partition=long
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%j_output.txt
#SBATCH --error=slurm_logs/%j_error.txt

# Create log directory
mkdir -p slurm_logs

echo "Job started on $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Initialize conda (not available by default in SLURM batch jobs)
eval "$(/apps/local/anaconda3/bin/conda shell.bash hook)"
conda activate ashar_mbrl

# Set writable HuggingFace cache (default /l/users may not be writable)
export HF_HOME="/nfs-stor/salem.lahlou/.cache/huggingface"
export TRANSFORMERS_CACHE="/nfs-stor/salem.lahlou/.cache/huggingface/hub"
mkdir -p "$TRANSFORMERS_CACHE"

# Use EGL for headless GPU rendering with MuJoCo
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

# NVIDIA EGL setup: create a temporary vendor JSON since none exists on this cluster
NVIDIA_EGL_JSON=$(mktemp /tmp/nvidia_egl_vendor_XXXXXX.json)
cat > "$NVIDIA_EGL_JSON" << 'EJSON'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EJSON
export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_JSON"
echo "Created NVIDIA EGL vendor JSON: $NVIDIA_EGL_JSON"

# Ensure NVIDIA EGL libs are on the library path
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# --- EGL diagnostic (remove once confirmed working) ---
echo "=== EGL Diagnostic ==="
echo "MUJOCO_GL=$MUJOCO_GL"
echo "MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
ls -la /usr/lib/x86_64-linux-gnu/libEGL_nvidia* /lib/x86_64-linux-gnu/libEGL_nvidia* 2>&1 || echo "libEGL_nvidia not in standard paths"
ldconfig -p 2>/dev/null | grep -i "egl_nvidia" || echo "libEGL_nvidia not in ldconfig"
python -c "
import os
os.environ['MUJOCO_GL'] = 'egl'
from dm_control import suite
env = suite.load('walker', 'walk')
ts = env.reset()
img = env.physics.render(64, 64, camera_id=0)
print(f'EGL render OK: shape={img.shape}, dtype={img.dtype}')
" 2>&1 || echo "EGL render FAILED"
echo "======================"

# Run training — select experiment config via --config-name.
# Available configs: dmc/cnn, dmc/multimodal, dmc/distractor_cnn, dmc/distractor_multimodal
# Add any Hydra overrides after the config name, e.g.:
#   env.task=distract_cheetah_run env.difficulty=hard seed=42
python -u train.py --config-name dmc/distractor_cnn device=cuda:0

echo "Job finished on $(date)"
