"""Evaluate a trained adversarial patch against a Dreamer agent.

Runs episodes with and without the patch and reports return statistics.

Usage:
    python3 attacks/eval_adv_patch.py \
        agent_ckpt=./logdir/myrun/latest.pt \
        patch_path=./attacks/logdir/best_patch.pt \
        env=dmc_vision env.task=dmc_walker_walk
"""

import atexit
import pathlib
import sys
import os
import warnings

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

# Add project root to path (this file lives in attacks/)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import tools
from dreamer import Dreamer
from envs import make_envs
from attacks.adv_patch import AdversarialPatch

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


def _setup_gpu(config):
    device = config.device
    if device.startswith("cuda"):
        gpu_id = device.split(":")[-1] if ":" in device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu_id
        OmegaConf.update(config, "device", "cuda:0")


def load_agent(config, obs_space, act_space, ckpt_path):
    agent = Dreamer(config.model, obs_space, act_space).to(config.device)
    if config.model.use_multimodal_encoder:
        task_name = config.env.task
        if task_name.startswith("dmc_"):
            task_name = task_name[4:]
        agent.set_task_name(task_name)
    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    # Filter out _frozen_* and _slow_value keys — clone_and_freeze() rebuilds
    # them with shared data pointers from the primary modules.
    state_dict = {k: v for k, v in ckpt["agent_state_dict"].items()
                  if not k.startswith(("_frozen_", "_slow_value"))}
    agent.load_state_dict(state_dict, strict=False)
    agent.clone_and_freeze()
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    return agent


class PatchedAgent:
    """Wrapper that applies an adversarial patch to observations before acting."""

    def __init__(self, agent, patch_module, enabled=True):
        self.agent = agent
        self.patch_module = patch_module
        self.enabled = enabled
        self.device = agent.device

    def get_initial_state(self, B):
        return self.agent.get_initial_state(B)

    @torch.no_grad()
    def act(self, obs, state, eval=True):
        if self.enabled and self.patch_module is not None:
            # obs["image"] is uint8 (B, H, W, C) from env
            img_float = obs["image"].float() / 255.0
            img_patched = self.patch_module.apply(img_float)
            # Convert back to uint8 for the agent's preprocess
            obs = obs.clone()
            obs["image"] = (img_patched * 255.0).clamp(0, 255).to(torch.uint8)
        return self.agent.act(obs, state, eval=eval)


def run_evaluation(patched_agent, envs, num_episodes, device, save_video_path=None):
    """Run evaluation episodes and return statistics.

    Returns dict with keys: returns, lengths, videos (if save_video_path).
    """
    B = envs.env_num
    done = torch.ones(B, dtype=torch.bool, device=device)
    once_done = torch.zeros(B, dtype=torch.bool, device=device)
    steps = torch.zeros(B, dtype=torch.int32, device=device)
    returns = torch.zeros(B, dtype=torch.float32, device=device)

    agent_state = patched_agent.get_initial_state(B)
    act = agent_state["prev_action"].clone()
    video_frames = []

    while not once_done.all():
        steps += ~done * ~once_done
        act_cpu = act.detach().to("cpu")
        done_cpu = done.detach().to("cpu")
        trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
        trans = trans_cpu.to(device, non_blocking=True)
        done = done_cpu.to(device)

        trans["action"] = act

        # Save video frames from first env
        if save_video_path is not None and not once_done[0]:
            video_frames.append(trans["image"][0].cpu())

        act, agent_state = patched_agent.act(trans, agent_state, eval=True)
        returns += trans["reward"][:, 0] * ~once_done
        once_done |= done

    result = {
        "returns": returns.cpu().numpy(),
        "lengths": steps.cpu().numpy(),
    }

    # Save video
    if save_video_path is not None and len(video_frames) > 0:
        try:
            video = torch.stack(video_frames).numpy()  # (T, H, W, C)
            save_video_grid(video, save_video_path)
            result["video_path"] = str(save_video_path)
        except Exception as e:
            print(f"Warning: Could not save video: {e}")

    return result


def save_video_grid(frames, path):
    """Save frames as a simple image grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_frames = min(len(frames), 32)
    cols = 8
    rows = (n_frames + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = [axes]
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i]
        if i < n_frames:
            ax.imshow(frames[i])
            ax.set_title(f"t={i}", fontsize=6)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


@hydra.main(version_base=None, config_path="configs", config_name="eval_adv_patch")
def main(config):
    _setup_gpu(config)
    tools.set_seed_everywhere(config.seed)
    device = torch.device(config.device)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Eval logdir: {logdir}")

    # Create environments
    print("Creating environments...")
    _, eval_envs, obs_space, act_space = make_envs(config.env)

    # Load agent
    ckpt_path = pathlib.Path(config.agent_ckpt).expanduser()
    print(f"Loading agent from {ckpt_path}")
    agent = load_agent(config, obs_space, act_space, ckpt_path)

    # Load patch
    patch_path = pathlib.Path(config.patch_path).expanduser()
    print(f"Loading patch from {patch_path}")
    patch_module = AdversarialPatch.load(patch_path, device=device)
    patch_module.eval()
    print(f"Patch: {patch_module.ph}x{patch_module.pw} at ({patch_module._x0}, {patch_module._y0})")

    num_episodes = config.env.eval_episode_num

    # Clean evaluation
    print(f"\n--- Clean evaluation ({num_episodes} episodes) ---")
    clean_agent = PatchedAgent(agent, patch_module, enabled=False)
    clean_results = run_evaluation(
        clean_agent, eval_envs, num_episodes, device,
        save_video_path=logdir / "clean_frames.png",
    )
    clean_mean = np.mean(clean_results["returns"])
    clean_std = np.std(clean_results["returns"])
    clean_len = np.mean(clean_results["lengths"])
    print(f"Clean return:  {clean_mean:.1f} +/- {clean_std:.1f}")
    print(f"Clean length:  {clean_len:.1f}")

    # Patched evaluation
    print(f"\n--- Patched evaluation ({num_episodes} episodes) ---")
    patched_agent = PatchedAgent(agent, patch_module, enabled=True)
    patched_results = run_evaluation(
        patched_agent, eval_envs, num_episodes, device,
        save_video_path=logdir / "patched_frames.png",
    )
    patched_mean = np.mean(patched_results["returns"])
    patched_std = np.std(patched_results["returns"])
    patched_len = np.mean(patched_results["lengths"])
    print(f"Patched return:  {patched_mean:.1f} +/- {patched_std:.1f}")
    print(f"Patched length:  {patched_len:.1f}")

    # Report
    drop = clean_mean - patched_mean
    pct_drop = (drop / max(abs(clean_mean), 1e-8)) * 100
    print(f"\n--- Results ---")
    print(f"Return drop:    {drop:.1f} ({pct_drop:.1f}%)")
    print(f"Clean:          {clean_mean:.1f} +/- {clean_std:.1f}")
    print(f"Patched:        {patched_mean:.1f} +/- {patched_std:.1f}")

    # Save metrics
    metrics = {
        "clean_return_mean": float(clean_mean),
        "clean_return_std": float(clean_std),
        "clean_length_mean": float(clean_len),
        "patched_return_mean": float(patched_mean),
        "patched_return_std": float(patched_std),
        "patched_length_mean": float(patched_len),
        "return_drop": float(drop),
        "return_drop_pct": float(pct_drop),
    }

    import json
    with open(logdir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {logdir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
