#!/bin/bash
# Shared environment setup sourced by all SLURM training scripts.

eval "$(/apps/local/anaconda3/bin/conda shell.bash hook)"
conda activate ashar_mbrl

export HF_HOME="/nfs-stor/salem.lahlou/.cache/huggingface"
export TRANSFORMERS_CACHE="/nfs-stor/salem.lahlou/.cache/huggingface/hub"
mkdir -p "$TRANSFORMERS_CACHE"

# MuJoCo EGL rendering
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

# NVIDIA EGL vendor JSON (needed on clusters without system-level EGL config)
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
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
