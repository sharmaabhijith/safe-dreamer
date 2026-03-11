"""Evaluate trained models with and without video background distractors.

Reads a config listing multiple checkpoints and evaluates each in two
conditions (clean and distractor background). Results are logged side-by-side
in TensorBoard for easy comparison.

All settings are specified in a YAML config file.

Usage:
    python eval.py --config configs/eval.yaml

    # View results:
    tensorboard --logdir eval_results/
"""

import os
import pathlib

# Force local HuggingFace cache BEFORE any library imports to avoid
# permission errors with shared caches (e.g. /shared/.cache/huggingface).
_hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_cache, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_hf_cache, "hub")

import argparse
import warnings

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

from utils import tools
from world_model.dreamer import Dreamer
from envs import make_eval_envs


def _setup_gpu(device_str):
    """Restrict CUDA to a single physical GPU, matching train.py behavior.

    On SLURM the scheduler already sets CUDA_VISIBLE_DEVICES, so we just
    read the first visible GPU for MuJoCo EGL and use cuda:0.
    """
    if device_str.startswith("cuda"):
        on_slurm = "SLURM_JOB_ID" in os.environ
        if on_slurm:
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            first_gpu = cvd.split(",")[0]
            os.environ.setdefault("MUJOCO_GL", "egl")
            os.environ["MUJOCO_EGL_DEVICE_ID"] = first_gpu
        else:
            gpu_id = device_str.split(":")[-1] if ":" in device_str else "0"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            os.environ.setdefault("MUJOCO_GL", "egl")
            os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu_id
        return "cuda:0"
    return device_str


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and reconstruct config from the .hydra directory."""
    ckpt_dir = pathlib.Path(checkpoint_path).parent
    hydra_config_path = ckpt_dir / ".hydra" / "config.yaml"
    if not hydra_config_path.exists():
        raise FileNotFoundError(
            f"Cannot find training config at {hydra_config_path}. "
            "Make sure the checkpoint directory has a .hydra/config.yaml."
        )
    with open(hydra_config_path) as f:
        raw = yaml.safe_load(f)

    # Remove keys that use Hydra-only resolvers (e.g. ${now:...}) which
    # OmegaConf cannot resolve outside of a Hydra run.
    raw.pop("logdir", None)
    raw.pop("hydra", None)

    # Override device BEFORE resolving so all ${device} references
    # (rssm.device, encoder.mlp.device, etc.) pick up the correct value.
    raw["device"] = device

    config = OmegaConf.create(raw)
    OmegaConf.resolve(config)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return config, ckpt


def build_agent(config, obs_space, act_space, ckpt, device):
    """Reconstruct the Dreamer agent and load checkpoint weights."""
    # Disable torch.compile for eval — we never call _cal_grad
    model_cfg = OmegaConf.create(OmegaConf.to_container(config.model))
    OmegaConf.update(model_cfg, "compile", False)

    agent = Dreamer(model_cfg, obs_space, act_space).to(device)

    if model_cfg.use_multimodal_encoder:
        task_name = config.env.task
        for prefix in ("distract_", "dmc_"):
            if task_name.startswith(prefix):
                task_name = task_name[len(prefix):]
                break
        agent.set_task_name(task_name)

    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    return agent


def set_eval_text(agent, text):
    """Override the evaluation text for the multimodal encoder."""
    if agent.use_multimodal_encoder and text:
        agent.encoder._eval_text = text
        if hasattr(agent, '_frozen_encoder'):
            agent._frozen_encoder._eval_text = text
        agent.encoder._cached_text = None
        agent.encoder._cached_ctx = None
        if hasattr(agent, '_frozen_encoder'):
            agent._frozen_encoder._cached_text = None
            agent._frozen_encoder._cached_ctx = None
        print(f"  Eval text: \"{text}\"")


def run_eval_episodes(agent, envs, num_episodes, device, record_video=True):
    """Run evaluation episodes and collect metrics.

    Returns:
        scores: list of episode returns
        lengths: list of episode lengths
        video: (T, H, W, C) uint8 array of first episode, or None
    """
    agent.eval()
    B = envs.env_num
    done = torch.ones(B, dtype=torch.bool, device=device)
    once_done = torch.zeros(B, dtype=torch.bool, device=device)
    steps = torch.zeros(B, dtype=torch.int32, device=device)
    returns = torch.zeros(B, dtype=torch.float32, device=device)
    agent_state = agent.get_initial_state(B)
    act = agent_state["prev_action"].clone()

    video_frames = []
    all_scores = []
    all_lengths = []
    episodes_done = 0

    while episodes_done < num_episodes:
        steps += ~done * ~once_done
        act_cpu = act.detach().to("cpu")
        done_cpu = done.detach().to("cpu")
        trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
        trans = trans_cpu.to(device, non_blocking=True)
        done = done_cpu.to(device)

        trans["action"] = act

        if record_video and "image" in trans and not once_done[0]:
            frame = trans["image"][0, 0].detach().cpu().numpy()
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            video_frames.append(frame)

        act, agent_state = agent.act(trans, agent_state, eval=True)
        returns += trans["reward"][:, 0] * ~once_done

        newly_done = done & ~once_done
        for i in range(B):
            if newly_done[i]:
                all_scores.append(returns[i].item())
                all_lengths.append(steps[i].item())
                episodes_done += 1
        once_done |= done

        if once_done.all() and episodes_done < num_episodes:
            done = torch.ones(B, dtype=torch.bool, device=device)
            once_done = torch.zeros(B, dtype=torch.bool, device=device)
            steps = torch.zeros(B, dtype=torch.int32, device=device)
            returns = torch.zeros(B, dtype=torch.float32, device=device)
            agent_state = agent.get_initial_state(B)
            act = agent_state["prev_action"].clone()

    video = np.stack(video_frames, axis=0) if video_frames else None
    return all_scores[:num_episodes], all_lengths[:num_episodes], video


def _video_to_tb(video, max_frames=200):
    """Convert video array to TensorBoard format (1, T, C, H, W) uint8 tensor."""
    v = np.asarray(video)
    # Ensure 4D: (T, H, W, C)
    if v.ndim == 3:
        v = np.stack([v, v, v], axis=-1)
    if v.shape[-1] == 1:
        v = np.repeat(v, 3, axis=-1)
    if v.dtype != np.uint8:
        v = np.clip(v, 0, 255).astype(np.uint8)
    v = v[:max_frames]
    # (T, H, W, C) -> (1, T, C, H, W) as torch tensor
    # Using a tensor avoids the moviepy/imageio GIF path in TensorBoard
    t = torch.from_numpy(v).permute(0, 3, 1, 2).unsqueeze(0)
    return t


def make_env_config(train_config, video_dir=None):
    """Create an env config from the training config, optionally with video_dir."""
    env_config = OmegaConf.create({
        "task": train_config.env.task,
        "action_repeat": train_config.env.action_repeat,
        "time_limit": train_config.env.time_limit,
        "size": list(train_config.env.size),
        "seed": train_config.env.seed,
        "device": train_config.device,
        "eval_episode_num": train_config.env.eval_episode_num,
        "env_num": train_config.env.eval_episode_num,
        "encoder": train_config.env.encoder,
        "decoder": train_config.env.decoder,
    })
    if video_dir:
        OmegaConf.update(env_config, "video_dir", video_dir)
    return env_config


def evaluate_model(model_entry, device, num_episodes, video_dir, logdir):
    """Evaluate a single model (clean + distractor) and log to TensorBoard."""
    checkpoint = model_entry["checkpoint"]
    model_name = model_entry.get("name", pathlib.Path(checkpoint).parent.name)
    text = model_entry.get("text", None)

    print("\n" + "#" * 70)
    print(f"# Model: {model_name}")
    print(f"# Checkpoint: {checkpoint}")
    if text:
        print(f"# Text: {text}")
    print("#" * 70)

    # Load checkpoint
    train_config, ckpt = load_checkpoint(checkpoint, device)
    is_multimodal = bool(train_config.model.use_multimodal_encoder)
    encoder_type = "multimodal" if is_multimodal else "cnn"
    print(f"  Encoder type: {encoder_type}")

    # Setup TensorBoard (one writer per model)
    tb_dir = pathlib.Path(logdir) / model_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    results = {"name": model_name, "encoder": encoder_type}

    # --------------------------------------------------
    # 1. Evaluate WITHOUT distractor (clean background)
    # --------------------------------------------------
    print(f"\n  [clean] Evaluating {num_episodes} episodes ...")
    env_config_clean = make_env_config(train_config, video_dir=None)
    clean_envs, obs_space, act_space = make_eval_envs(env_config_clean)
    agent = build_agent(train_config, obs_space, act_space, ckpt, device)
    if is_multimodal and text:
        set_eval_text(agent, text)

    clean_scores, clean_lengths, clean_video = run_eval_episodes(
        agent, clean_envs, num_episodes, device
    )
    clean_mean, clean_std = np.mean(clean_scores), np.std(clean_scores)
    clean_len_mean = np.mean(clean_lengths)
    print(f"  [clean] Score: {clean_mean:.2f} +/- {clean_std:.2f}  |  Length: {clean_len_mean:.1f}")

    writer.add_scalar("eval/clean_score_mean", clean_mean, 0)
    writer.add_scalar("eval/clean_score_std", clean_std, 0)
    writer.add_scalar("eval/clean_length_mean", clean_len_mean, 0)
    for i, s in enumerate(clean_scores):
        writer.add_scalar("eval/clean_score_per_episode", s, i)
    if clean_video is not None:
        # clean_video: (T, H, W, C) -> (1, T, C, H, W) for TensorBoard
        vid = _video_to_tb(clean_video)
        writer.add_video("eval/clean_video", vid, 0, fps=16)

    results["clean_mean"] = clean_mean
    results["clean_std"] = clean_std

    # Free agent memory before distractor eval
    del agent
    torch.cuda.empty_cache()

    # --------------------------------------------------
    # 2. Evaluate WITH distractor (video background)
    # --------------------------------------------------
    if video_dir and os.path.isdir(video_dir):
        print(f"\n  [distractor] Evaluating {num_episodes} episodes ...")
        env_config_dist = make_env_config(train_config, video_dir=video_dir)
        dist_envs, obs_space_d, act_space_d = make_eval_envs(env_config_dist)
        agent_dist = build_agent(train_config, obs_space_d, act_space_d, ckpt, device)
        if is_multimodal and text:
            set_eval_text(agent_dist, text)

        dist_scores, dist_lengths, dist_video = run_eval_episodes(
            agent_dist, dist_envs, num_episodes, device
        )
        dist_mean, dist_std = np.mean(dist_scores), np.std(dist_scores)
        dist_len_mean = np.mean(dist_lengths)
        print(f"  [distractor] Score: {dist_mean:.2f} +/- {dist_std:.2f}  |  Length: {dist_len_mean:.1f}")

        writer.add_scalar("eval/distractor_score_mean", dist_mean, 0)
        writer.add_scalar("eval/distractor_score_std", dist_std, 0)
        writer.add_scalar("eval/distractor_length_mean", dist_len_mean, 0)
        for i, s in enumerate(dist_scores):
            writer.add_scalar("eval/distractor_score_per_episode", s, i)
        if dist_video is not None:
            vid = _video_to_tb(dist_video)
            writer.add_video("eval/distractor_video", vid, 0, fps=16)

        drop = clean_mean - dist_mean
        drop_pct = (drop / max(clean_mean, 1e-8)) * 100
        writer.add_scalar("eval/performance_drop", drop, 0)
        writer.add_scalar("eval/performance_drop_pct", drop_pct, 0)
        writer.add_scalars("eval/score_comparison", {
            "clean": clean_mean, "distractor": dist_mean,
        }, 0)

        results["dist_mean"] = dist_mean
        results["dist_std"] = dist_std
        results["drop"] = drop
        results["drop_pct"] = drop_pct

        del agent_dist
        torch.cuda.empty_cache()
    else:
        if video_dir:
            print(f"\n  Warning: '{video_dir}' not found. Skipping distractor evaluation.")
        results["dist_mean"] = None

    # Log config text
    config_str = f"checkpoint: {checkpoint}\n"
    config_str += f"encoder: {encoder_type}\n"
    config_str += f"task: {train_config.env.task}\n"
    config_str += f"eval_episodes: {num_episodes}\n"
    if is_multimodal and text:
        config_str += f"text: {text}\n"
    writer.add_text("eval/config", f"```\n{config_str}```", 0)
    if is_multimodal and text:
        writer.add_text("eval/text_prompt", text, 0)

    writer.flush()
    writer.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models with distractor backgrounds"
    )
    parser.add_argument("--config", type=str, default="configs/eval.yaml",
                        help="Path to evaluation config file")
    args = parser.parse_args()

    # Load eval config — all settings come from the YAML
    with open(args.config) as f:
        eval_config = yaml.safe_load(f)

    device = _setup_gpu(eval_config.get("device", "cuda:0"))
    models = eval_config["models"]
    video_dir = eval_config.get("video_dir")
    num_episodes = eval_config.get("eval_episodes", 10)
    logdir = eval_config.get("logdir", "eval_results")
    seed = eval_config.get("seed", 0)

    tools.set_seed_everywhere(seed)

    print(f"Evaluating {len(models)} model(s)")
    print(f"  Episodes per condition: {num_episodes}")
    print(f"  Video dir: {video_dir}")
    print(f"  TensorBoard logdir: {logdir}")

    # Evaluate each model
    all_results = []
    for model_entry in models:
        result = evaluate_model(model_entry, device, num_episodes, video_dir, logdir)
        all_results.append(result)

    # ============================
    # Final comparison summary (2x2 matrix: encoder x background)
    # ============================
    print("\n" + "=" * 78)
    print("RESULTS  (encoder x background)")
    print("=" * 78)
    header = f"{'Model':<20} {'Encoder':<12} {'Clean BG':>14} {'Distractor BG':>14} {'Drop':>14}"
    print(header)
    print("-" * 78)
    for r in all_results:
        clean = f"{r['clean_mean']:.1f} +/- {r['clean_std']:.1f}"
        if r.get("dist_mean") is not None:
            dist = f"{r['dist_mean']:.1f} +/- {r['dist_std']:.1f}"
            drop = f"{r['drop']:.1f} ({r['drop_pct']:.1f}%)"
        else:
            dist = "N/A"
            drop = "N/A"
        print(f"{r['name']:<20} {r['encoder']:<12} {clean:>14} {dist:>14} {drop:>14}")
    print("-" * 78)

    # Cross-model comparison: which encoder is more robust?
    if len(all_results) >= 2 and all(r.get("dist_mean") is not None for r in all_results):
        drops = {r["name"]: r["drop_pct"] for r in all_results}
        best = min(drops, key=drops.get)
        print(f"\nMost robust to distractors: {best} (smallest drop: {drops[best]:.1f}%)")

    # Log cross-model comparison to a shared TensorBoard writer
    compare_dir = pathlib.Path(logdir) / "_comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    cw = SummaryWriter(log_dir=str(compare_dir))
    for i, r in enumerate(all_results):
        cw.add_scalar("compare/clean_score", r["clean_mean"], i)
        if r.get("dist_mean") is not None:
            cw.add_scalar("compare/distractor_score", r["dist_mean"], i)
            cw.add_scalar("compare/drop_pct", r["drop_pct"], i)
        cw.add_text("compare/model_name", r["name"], i)
    cw.flush()
    cw.close()

    print(f"\nResults saved. Run:  tensorboard --logdir {logdir}")


if __name__ == "__main__":
    main()
