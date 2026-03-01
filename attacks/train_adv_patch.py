"""Train an adversarial patch against a trained Dreamer agent.

Loads a trained checkpoint, collects trajectories, and optimizes a localized
image patch to minimize the agent's imagined return through the world model.

Usage:
    python3 attacks/train_adv_patch.py \
        agent_ckpt=./logdir/myrun/latest.pt \
        env=dmc_vision env.task=dmc_walker_walk \
        attack.patch_hw=[16,16] attack.placement=bottom_center \
        attack.steps=5000 attack.lr=0.05
"""

import atexit
import pathlib
import sys
import os
import warnings

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.amp import autocast

# Add project root to path (this file lives in attacks/)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import tools
from dreamer import Dreamer
from envs import make_envs
from attacks.adv_patch import AdversarialPatch, PlanningAttackLoss, compute_saliency_placement

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


def _setup_gpu(config):
    """Restrict CUDA and MuJoCo EGL to the same physical GPU."""
    device = config.device
    if device.startswith("cuda"):
        gpu_id = device.split(":")[-1] if ":" in device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu_id
        OmegaConf.update(config, "device", "cuda:0")


def load_agent(config, obs_space, act_space, ckpt_path):
    """Instantiate a Dreamer agent and load its checkpoint."""
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
    # Freeze all agent parameters
    for p in agent.parameters():
        p.requires_grad_(False)
    return agent


def collect_trajectories(agent, envs, num_episodes, device):
    """Collect trajectories using the trained agent (no patch).

    Returns a list of episode dicts, each containing:
      - image: (T, H, W, C) uint8
      - action: (T, A) float32
      - reward: (T,) float32
      - is_first: (T,) bool
      - is_terminal: (T,) bool
    """
    episodes = []
    ep_count = 0
    B = envs.env_num
    done = torch.ones(B, dtype=torch.bool, device=device)
    agent_state = agent.get_initial_state(B)
    act = agent_state["prev_action"].clone()

    # Per-env buffers
    buffers = [{"image": [], "action": [], "reward": [], "is_first": [], "is_terminal": []} for _ in range(B)]

    while ep_count < num_episodes:
        act_cpu = act.detach().to("cpu")
        done_cpu = done.detach().to("cpu")
        trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
        trans = trans_cpu.to(device, non_blocking=True)
        done = done_cpu.to(device)

        trans["action"] = act
        act, agent_state = agent.act(trans, agent_state, eval=True)

        for i in range(B):
            if ep_count >= num_episodes:
                break
            buffers[i]["image"].append(trans["image"][i].cpu())
            buffers[i]["action"].append(trans["action"][i].cpu())
            buffers[i]["reward"].append(trans["reward"][i, 0].cpu())
            buffers[i]["is_first"].append(trans["is_first"][i, 0].cpu())
            buffers[i]["is_terminal"].append(trans["is_terminal"][i, 0].cpu())

            if done[i]:
                ep = {
                    "image": torch.stack(buffers[i]["image"]),
                    "action": torch.stack(buffers[i]["action"]),
                    "reward": torch.stack(buffers[i]["reward"]),
                    "is_first": torch.stack(buffers[i]["is_first"]),
                    "is_terminal": torch.stack(buffers[i]["is_terminal"]),
                }
                episodes.append(ep)
                ep_count += 1
                buffers[i] = {"image": [], "action": [], "reward": [], "is_first": [], "is_terminal": []}

    return episodes


class TrajectoryDataset:
    """Simple dataset that yields random windows from collected trajectories."""

    def __init__(self, episodes, context_len, device):
        self.episodes = episodes
        self.context_len = context_len
        self.device = device
        self._lengths = [ep["image"].shape[0] for ep in episodes]

    def sample_batch(self, batch_size):
        """Sample a batch of trajectory windows.

        Returns dict of:
            image: (B, T, H, W, C) float32 in [0, 1]
            action: (B, T, A)
            is_first: (B, T) bool
        """
        images, actions, is_firsts = [], [], []
        for _ in range(batch_size):
            ep_idx = np.random.randint(len(self.episodes))
            ep = self.episodes[ep_idx]
            ep_len = self._lengths[ep_idx]
            max_start = max(0, ep_len - self.context_len)
            start = np.random.randint(0, max_start + 1)
            end = start + self.context_len

            img = ep["image"][start:end].float() / 255.0
            act = ep["action"][start:end].float()
            is_f = ep["is_first"][start:end]

            # Pad if episode shorter than context_len
            T = img.shape[0]
            if T < self.context_len:
                pad_len = self.context_len - T
                img = F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_len))
                act = F.pad(act, (0, 0, 0, pad_len))
                is_f = F.pad(is_f, (0, pad_len), value=True)

            images.append(img)
            actions.append(act)
            is_firsts.append(is_f)

        batch = {
            "image": torch.stack(images).to(self.device),
            "action": torch.stack(actions).to(self.device),
            "is_first": torch.stack(is_firsts).to(self.device),
        }
        return batch


def save_visualization(patch_module, dataset, save_path):
    """Save patch visualization and example patched observations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Top row: the patch itself + mask
    patch_img = patch_module.get_patch().squeeze().detach().cpu().numpy()
    axes[0, 0].imshow(patch_img)
    axes[0, 0].set_title("Patch")
    axes[0, 0].axis("off")

    mask = patch_module.get_mask(patch_module.delta.device).squeeze().cpu().numpy()
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Mask")
    axes[0, 1].axis("off")

    # Sample some observations and show clean vs patched
    batch = dataset.sample_batch(3)
    images = batch["image"]  # (3, T, H, W, C)
    for i in range(min(3, images.shape[0])):
        clean = images[i, 0].cpu().numpy()
        patched = patch_module.apply(images[i:i+1, :1]).squeeze().detach().cpu().numpy()

        col = i + 2 if i < 2 else i
        axes[0, col].imshow(clean)
        axes[0, col].set_title(f"Clean {i}")
        axes[0, col].axis("off")

        axes[1, col].imshow(patched)
        axes[1, col].set_title(f"Patched {i}")
        axes[1, col].axis("off")

    # Hide unused axes
    for ax in axes.flat:
        if not ax.images and not ax.lines:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {save_path}")


@hydra.main(version_base=None, config_path="configs", config_name="train_adv_patch")
def main(config):
    _setup_gpu(config)
    tools.set_seed_everywhere(config.seed)
    device = torch.device(config.device)

    # HydraConfig.get().runtime.output_dir is the absolute path to the run dir,
    # even though Hydra has already cd'd into it. config.logdir is relative after the cd.
    logdir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Attack logdir: {logdir}")

    logger = tools.Logger(logdir)
    logger.log_hydra_config(config, name="attack_config", step=0)

    # Override env settings so make_envs creates only the environments we need:
    # - eval_episode_num controls the parallel eval envs used for collection
    # - env_num=1 avoids wasting resources on unused train envs
    OmegaConf.update(config, "env.eval_episode_num", config.attack.collect_episodes)
    OmegaConf.update(config, "env.env_num", 1)

    # Create environments
    print("Creating environments...")
    _, eval_envs, obs_space, act_space = make_envs(config.env)

    # Load agent
    ckpt_path = pathlib.Path(config.agent_ckpt).expanduser()
    print(f"Loading agent from {ckpt_path}")
    agent = load_agent(config, obs_space, act_space, ckpt_path)
    print("Agent loaded and frozen.")

    atk = config.attack

    # Collect trajectories
    print(f"Collecting {atk.collect_episodes} episodes...")
    episodes = collect_trajectories(agent, eval_envs, atk.collect_episodes, device)
    ep_returns = [ep["reward"].sum().item() for ep in episodes]
    mean_clean_return = np.mean(ep_returns)
    print(f"Collected {len(episodes)} episodes. Mean return: {mean_clean_return:.1f}")
    logger.scalar("attack/clean_return", mean_clean_return)
    logger.write(0)

    dataset = TrajectoryDataset(episodes, atk.context_len, device)

    # Create patch
    patch_hw = tuple(atk.patch_hw) if hasattr(atk.patch_hw, '__iter__') else (atk.patch_hw, atk.patch_hw)
    image_hw = tuple(config.env.size)

    # Optional saliency-based placement
    if atk.placement == "saliency":
        print("Computing saliency-based placement...")
        # Temporarily enable grads for saliency computation
        batch = dataset.sample_batch(atk.batch_size)
        initial = agent.rssm.initial(atk.batch_size)
        (y0, x0), saliency = compute_saliency_placement(
            agent, batch, patch_hw, image_hw,
            allowed_region=(image_hw[0] // 2, image_hw[0], 0, image_hw[1])  # bottom half
        )
        print(f"Saliency placement: y0={y0}, x0={x0}")
        placement = "fixed"
        x0_cfg, y0_cfg = x0, y0
    else:
        placement = atk.placement
        x0_cfg = atk.x0
        y0_cfg = atk.y0

    patch_module = AdversarialPatch(
        image_hw=image_hw,
        patch_hw=patch_hw,
        placement=placement,
        x0=x0_cfg,
        y0=y0_cfg,
        eps=atk.eps,
        eot_translations=atk.eot_translations,
        eot_max_shift=atk.eot_max_shift,
        eot_brightness=atk.eot_brightness,
        floor_mask_path=atk.floor_mask_path,
    ).to(device)

    print(f"Patch: {patch_hw} at ({patch_module._x0}, {patch_module._y0}), "
          f"eps={atk.eps}, placement={placement}")

    # Attack loss
    loss_fn = PlanningAttackLoss(
        w_return=atk.w_return,
        w_kl=atk.w_kl,
        w_action=atk.w_action,
        w_tv=atk.w_tv,
        w_l2=atk.w_l2,
        imag_horizon=atk.imag_horizon,
        context_len=atk.context_len,
        use_mode_actions=atk.use_mode_actions,
    )

    # Optimizer (only patch params)
    optimizer = torch.optim.Adam(patch_module.parameters(), lr=atk.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=atk.steps, eta_min=atk.lr * 0.1)

    # Training loop
    print(f"Starting patch optimization for {atk.steps} steps...")
    best_loss = float("inf")
    log_every = max(1, atk.steps // 50)

    for step in range(1, atk.steps + 1):
        patch_module.train()
        batch = dataset.sample_batch(atk.batch_size)
        initial = agent.rssm.initial(atk.batch_size)

        optimizer.zero_grad()

        # Forward with mixed precision for world model, but full precision for patch grads
        with autocast(device_type="cuda", dtype=torch.float16):
            loss, metrics = loss_fn(
                agent, patch_module,
                batch["image"], batch["action"], batch["is_first"],
                initial,
            )

        loss.backward()

        # Gradient clipping on patch
        torch.nn.utils.clip_grad_norm_(patch_module.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Project patch to valid range
        patch_module.project()

        if step % log_every == 0 or step == 1:
            grad_norm = patch_module.delta.grad.norm().item() if patch_module.delta.grad is not None else 0.0
            current_lr = scheduler.get_lr()[0]
            print(
                f"[Step {step:5d}/{atk.steps}] "
                f"loss={metrics['total_loss']:.4f} "
                f"imag_return={metrics['imagined_return']:.4f} "
                f"tv={metrics.get('tv_loss', 0):.4f} "
                f"grad_norm={grad_norm:.4f} "
                f"lr={current_lr:.6f}"
            )
            logger.scalar("attack/total_loss", metrics["total_loss"])
            logger.scalar("attack/imagined_return", metrics["imagined_return"])
            logger.scalar("attack/grad_norm", grad_norm)
            logger.scalar("attack/lr", current_lr)
            for key in ("tv_loss", "l2_loss", "kl_post", "action_shift"):
                if key in metrics:
                    logger.scalar(f"attack/{key}", metrics[key])
            # Log the patch image
            patch_img = patch_module.get_patch().squeeze(0).detach().cpu().numpy()  # (H, W, C)
            logger.image("attack/patch", patch_img.transpose(2, 0, 1))  # (C, H, W) for TB
            logger.write(step)

        if metrics["total_loss"] < best_loss:
            best_loss = metrics["total_loss"]
            patch_module.save(logdir / "best_patch.pt")

    # Save final patch
    patch_module.save(logdir / "final_patch.pt")
    print(f"Final patch saved to {logdir / 'final_patch.pt'}")
    print(f"Best patch saved to {logdir / 'best_patch.pt'} (loss={best_loss:.4f})")
    logger.scalar("attack/best_loss", best_loss)
    patch_img = patch_module.get_patch().squeeze(0).detach().cpu().numpy()
    logger.image("attack/final_patch", patch_img.transpose(2, 0, 1))
    logger.write(atk.steps)

    # Visualization
    patch_module.eval()
    save_visualization(patch_module, dataset, logdir / "patch_visualization.png")

    print("Patch training complete.")


if __name__ == "__main__":
    main()
