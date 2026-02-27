# R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

This repository provides a PyTorch implementation of [R2-Dreamer][r2dreamer] (ICLR 2026), a computationally efficient world model that achieves high performance on continuous control benchmarks. It also includes an efficient PyTorch DreamerV3 reproduction that trains **~5x faster** than a widely used [codebase][dreamerv3-torch], along with other baselines. Selecting R2-Dreamer via the config provides an additional **~1.6x speedup** over this baseline.

## Instructions

Install dependencies (tested with Ubuntu 24.04 and Python 3.11):
```bash
# Installing via a virtual env like uv is recommended.
pip install -r requirements.txt
```

Run training on default settings:

```bash
python3 train.py logdir=./logdir/test
```

Monitoring results:
```bash
tensorboard --logdir ./logdir
```

Switching algorithms:

```bash
# Choose an algorithm via model.rep_loss:
# r2dreamer|dreamer|infonce|dreamerpro
python3 train.py model.rep_loss=r2dreamer
```

For easier code reading, inline tensor shape annotations are provided. See [`docs/tensor_shapes.md`](docs/tensor_shapes.md).


## Available Benchmarks
At the moment, the following benchmarks are available in this repository.

| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [Meta-World](https://github.com/Farama-Foundation/Metaworld) | Image | Continuous | 1M | Robotic manipulation with complex contact interactions.|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [DMC Subtle](envs/dmc_subtle.py) | Image | Continuous |1M| DeepMind Control Suite with tiny task-relevant objects. |
| [Atari 100k](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|

Use Hydra to select a benchmark and a specific task using `env` and `env.task`, respectively.

```bash
python3 train.py ... env=dmc_vision env.task=dmc_walker_walk
```

## Headless rendering

If you run MuJoCo-based environments (DMC / MetaWorld) on headless machines, you may need to set `MUJOCO_GL` for offscreen rendering. **Using EGL is recommended** as it accelerates rendering, leading to faster simulation throughput.

```bash
# For example, when using EGL (GPU)
export MUJOCO_GL=egl
# (optional) Choose which GPU EGL uses
export MUJOCO_EGL_DEVICE_ID=0
```

More details: [Working with MuJoCo-based environments](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)


## Adversarial Patch (Planning-Level Attack)

This repo includes an adversarial patch attack against the agent's world-model planning pipeline. A small localized image patch is optimized to minimize the agent's **imagined return** by differentiating through the encoder, RSSM posterior, and imagination rollout. This is a *planning-level* attack, not merely an encoder feature attack.

Reference: [Adversarial Attacks on World-Model-Based RL](https://proceedings.neurips.cc/paper_files/paper/2024/file/17af43527227c5c96db0f8d4c6aadc4e-Paper-Conference.pdf) (NeurIPS 2024)

### Quick Start

**1. Train the agent** (standard training):
```bash
python3 train.py logdir=./logdir/myrun env=dmc_vision env.task=dmc_walker_walk model.rep_loss=r2dreamer
```

**2. Train the adversarial patch** (all hyperparams in `configs/attack/adv_patch.yaml`):
```bash
python3 train_adv_patch.py agent_ckpt=./logdir/myrun/latest.pt
```

**3. Evaluate clean vs patched**:
```bash
python3 eval_adv_patch.py \
    agent_ckpt=./logdir/myrun/latest.pt \
    patch_path=./logdir/attack/<timestamp>/best_patch.pt
```

### Patch Placement Options

| Placement | Description |
|-----------|-------------|
| `bottom_center` | Bottom-center band (default) |
| `top_center` | Top-center band |
| `center` | Center of image |
| `fixed` | Custom position via `attack.x0`, `attack.y0` |
| `saliency` | Auto-selected via gradient saliency |

### Key Configuration (configs/attack/adv_patch.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_hw` | [16, 16] | Patch height and width |
| `placement` | bottom_center | Placement strategy |
| `eps` | null | L-inf bound (null = direct [0,1]) |
| `w_return` | 1.0 | Weight on imagined return loss |
| `w_kl` | 0.0 | Posterior KL divergence weight |
| `w_action` | 0.0 | Action shift weight |
| `w_tv` | 0.01 | Total variation regularization |
| `eot_translations` | 0 | EoT random translation samples |
| `steps` | 5000 | Optimization steps |
| `lr` | 0.05 | Learning rate |

### Files

| File | Purpose |
|------|---------|
| `attacks/adv_patch.py` | AdversarialPatch module + PlanningAttackLoss |
| `train_adv_patch.py` | Data collection + patch optimization script |
| `eval_adv_patch.py` | Clean vs patched evaluation script |
| `configs/attack/adv_patch.yaml` | Attack hyperparameters |
| `configs/train_adv_patch.yaml` | Top-level Hydra config for patch training |
| `configs/eval_adv_patch.yaml` | Top-level Hydra config for evaluation |
| `tests/test_adv_patch_apply.py` | Unit tests for patch mechanics |

### Running Tests
```bash
python -m pytest tests/test_adv_patch_apply.py -v
```


## Code formatting

If you want automatic formatting/basic checks before commits, you can enable `pre-commit`:

```bash
pip install pre-commit
# This sets up a pre-commit hook so that checks are run every time you commit
pre-commit install
# Manual pre-commit run on all files
pre-commit run --all-files
```

## Citation

If you find this code useful, please consider citing:

```bibtex
@inproceedings{
morihira2026rdreamer,
title={R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
author={Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Je2QqXrcQq}
}
```

[r2dreamer]: https://openreview.net/forum?id=Je2QqXrcQq&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)
[dreamerv3-torch]: https://github.com/NM512/dreamerv3-torch
