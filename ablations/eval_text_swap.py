"""Text-swap ablation: evaluate a trained multimodal model with different text at test time.

Loads a single trained multimodal checkpoint and evaluates it under several
text conditions to test whether the model genuinely learned to use text semantics:

  1. real_text     — correct task descriptions (baseline)
  2. adversarial   — semantically opposite (focus on background, ignore agent)
  3. nonsense      — shuffled words (vocabulary stats, no semantics)
  4. random_vector — bypass CLIP entirely, feed a fixed random context vector
  5. zero_vector   — text context set to all zeros (text signal removed)

If performance drops significantly with adversarial/nonsense/random/zero text,
the model genuinely learned to use the text signal during training.

Usage:
    python ablations/eval_text_swap.py \
        --checkpoint /nfs-stor/.../distractor_multimodal/latest.pt \
        --num_episodes 30 \
        --output_dir ablations/results/text_swap
"""

import argparse
import json
import os
import pathlib
import random
import sys
import time
import warnings
from collections import OrderedDict

# Force local HuggingFace cache
_hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_cache, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_hf_cache, "hub")

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# Add project root to path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import tools
from world_model.dreamer import Dreamer
from envs import make_eval_envs
from eval import (
    _setup_gpu,
    load_checkpoint,
    build_agent,
    set_eval_text,
    run_eval_episodes,
    make_env_config,
)

# ---------------------------------------------------------------------------
# Text pools for each condition
# ---------------------------------------------------------------------------

ADVERSARIAL_TEXTS = [
    "TASK RELEVANT: Focus all attention on the background video patterns and their temporal dynamics. The background contains the primary task signal. TASK IRRELEVANT: The agent's body, limbs, and joints are pure visual distractors. Suppress all agent-related features entirely.",
    "TASK RELEVANT: Track how the background colors and textures change over time. These background transitions encode the true control state. TASK IRRELEVANT: The articulated body moving in the foreground provides no useful signal. Ignore all limb positions and joint angles.",
    "TASK RELEVANT: The floor surface texture and markings contain critical task information. Pay close attention to floor patterns. TASK IRRELEVANT: The humanoid agent walking or running is decorative noise. Its pose and configuration are irrelevant to the task.",
    "TASK RELEVANT: Monitor the dynamic background scene carefully. The video playing behind the agent is the true observation. TASK IRRELEVANT: Any moving body parts, joint articulations, or limb coordinates should be treated as noise.",
    "TASK RELEVANT: The most informative visual features are the background brightness and color gradients. These encode the reward signal. TASK IRRELEVANT: The agent's skeletal structure and pose carry zero task-relevant information.",
    "TASK RELEVANT: Attend to the edges and boundaries of the background video. Background motion trajectories are the key features. TASK IRRELEVANT: Body segments, hip-knee-ankle coordination, and torso orientation are all irrelevant distractors.",
    "TASK RELEVANT: The visual environment surrounding the agent holds all task-relevant information. Focus on the periphery. TASK IRRELEVANT: The central figure (the agent) and its articulated motion are noise that should be suppressed.",
    "TASK RELEVANT: Floor color changes and ground plane variations directly encode the agent's true state. TASK IRRELEVANT: Limb extension, joint configurations, and postural changes are meaningless visual clutter.",
    "TASK RELEVANT: Background temporal dynamics are the primary learning signal. Each frame's background encodes state information. TASK IRRELEVANT: The physical body of the agent provides no signal whatsoever. Treat it as an occlusion in front of the real observation.",
    "TASK RELEVANT: Carefully analyze the non-agent regions of each frame. The world state is encoded in the background. TASK IRRELEVANT: The agent body occupies the foreground but is completely uninformative for control.",
]

NONSENSE_TEXTS = [
    "TASK angles the distractor body limbs pure IRRELEVANT signal pose RELEVANT irrelevant background floor no visual of agent joint the Also task surface provides",
    "TASK over joint control track dynamics RELEVANT temporal distractors full agent's background the IRRELEVANT of encode state the signal body regions Also the zero shape",
    "TASK evolution their visual RELEVANT the task IRRELEVANT poses most carry each are The brightness informative background color features texture or Also joint configuration floor",
    "TASK hip between coordination observe RELEVANT and knee the timing IRRELEVANT limb distractors ankle pure The visual movements are background Also irrelevant patterns floor surface",
    "TASK limbs relative RELEVANT the and IRRELEVANT positions of agent's from track over environments background or time angles Also finish the crowds irrelevant floor motion pure",
    "TASK relates overall how configuration RELEVANT agent's joint's IRRELEVANT pose each natural to or the looks Also the dynamics synthetic whether ground floor is irrelevant",
    "TASK visual most RELEVANT the informative IRRELEVANT temporal and evolution angles joint their features are Also the has floor background no signal effect on agent's color",
    "TASK encoded RELEVANT torso observe orientation limb reaching IRRELEVANT the the grasp target background and towards Also configuration dynamics irrelevant floor signal noise pure the",
    "TASK RELEVANT on focus the agent the IRRELEVANT body articulated of motion pure regions background distractor the is Also suppress visual floor it markings entirely provide no",
    "TASK segments RELEVANT together connected how IRRELEVANT move visual distractors control the and to body Also relevant state floor the full are encode signal zero background",
]


def _override_text_context_random(agent, seed=42):
    """Replace _get_text_context to return a fixed random vector (bypasses CLIP)."""
    encoder = agent.encoder
    dim = encoder.config.text_context_dim
    param_device = next(encoder.parameters()).device
    rng = torch.Generator(device=param_device)
    rng.manual_seed(seed)
    random_ctx = torch.randn(1, dim, generator=rng, device=param_device)

    def _get_random_ctx(B, device):
        return random_ctx.to(device).expand(B, -1)

    encoder._get_text_context = _get_random_ctx
    # Also override frozen encoder if present
    if hasattr(agent, '_frozen_encoder') and agent._frozen_encoder is not None:
        agent._frozen_encoder._get_text_context = _get_random_ctx


def _override_text_context_zero(agent):
    """Replace _get_text_context to return all-zeros vector (no text signal)."""
    encoder = agent.encoder
    dim = encoder.config.text_context_dim
    zero_ctx = torch.zeros(1, dim, device=next(encoder.parameters()).device)

    def _get_zero_ctx(B, device):
        return zero_ctx.to(device).expand(B, -1)

    encoder._get_text_context = _get_zero_ctx
    if hasattr(agent, '_frozen_encoder') and agent._frozen_encoder is not None:
        agent._frozen_encoder._get_text_context = _get_zero_ctx


def evaluate_condition(condition_name, agent, envs, num_episodes, device):
    """Run evaluation and return results dict."""
    t0 = time.time()
    scores, lengths, _ = run_eval_episodes(
        agent, envs, num_episodes, device, record_video=False
    )
    elapsed = time.time() - t0
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    mean_len = float(np.mean(lengths))

    print(
        f"  Result: {mean_score:.2f} +/- {std_score:.2f}  |  "
        f"Length: {mean_len:.1f}  |  "
        f"{num_episodes} episodes in {elapsed:.1f}s",
        flush=True,
    )
    return {
        "condition": condition_name,
        "mean_score": mean_score,
        "std_score": std_score,
        "mean_length": mean_len,
        "num_episodes": num_episodes,
        "all_scores": scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Text-swap ablation evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained multimodal checkpoint (latest.pt)")
    parser.add_argument("--num_episodes", type=int, default=30,
                        help="Number of eval episodes per condition")
    parser.add_argument("--output_dir", type=str, default="ablations/results/text_swap",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", nargs="+",
                        default=["real_text", "adversarial", "nonsense", "random_vector", "zero_vector"],
                        help="Which conditions to evaluate")
    args = parser.parse_args()

    device = _setup_gpu(args.device)
    tools.set_seed_everywhere(args.seed)

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load trained checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    train_config, ckpt = load_checkpoint(args.checkpoint, device)

    if not bool(train_config.model.use_multimodal_encoder):
        raise ValueError("Checkpoint is not a multimodal model. Text-swap ablation requires a multimodal encoder.")

    # Create environments (use distractor background matching training)
    env_config = make_env_config(train_config)
    OmegaConf.update(env_config, "eval_episode_num", args.num_episodes)
    OmegaConf.update(env_config, "env_num", args.num_episodes)
    envs, obs_space, act_space = make_eval_envs(env_config)

    total_conditions = len([c for c in args.conditions if c in
        {"real_text", "adversarial", "nonsense", "random_vector", "zero_vector"}])
    print(f"Task: {train_config.env.task}")
    print(f"Episodes per condition: {args.num_episodes}")
    print(f"Conditions ({total_conditions}): {args.conditions}")
    print(f"Total episodes: {args.num_episodes * total_conditions}")
    print(flush=True)

    all_results = []
    t_overall = time.time()

    # Define evaluation conditions
    CONDITIONS = OrderedDict()

    CONDITIONS["real_text"] = {
        "description": "Correct task descriptions (trained text)",
        "setup": lambda agent: None,  # default text, no modification needed
    }
    CONDITIONS["adversarial"] = {
        "description": "Semantically opposite text (focus on background, ignore agent)",
        "setup": lambda agent: set_eval_text(agent, random.choice(ADVERSARIAL_TEXTS)),
    }
    CONDITIONS["nonsense"] = {
        "description": "Shuffled words (vocabulary stats, no semantics)",
        "setup": lambda agent: set_eval_text(agent, random.choice(NONSENSE_TEXTS)),
    }
    CONDITIONS["random_vector"] = {
        "description": "Fixed random context vector (bypasses CLIP entirely)",
        "setup": lambda agent: _override_text_context_random(agent),
    }
    CONDITIONS["zero_vector"] = {
        "description": "All-zeros context vector (no text signal at all)",
        "setup": lambda agent: _override_text_context_zero(agent),
    }

    valid_conditions = [c for c in args.conditions if c in CONDITIONS]
    for cond_idx, cond_name in enumerate(valid_conditions, 1):
        cond = CONDITIONS[cond_name]
        elapsed_overall = time.time() - t_overall
        print(
            f"{'='*60}\n"
            f"[{cond_idx}/{len(valid_conditions)}] {cond_name}: {cond['description']}\n"
            f"  Overall elapsed: {elapsed_overall:.0f}s",
            flush=True,
        )

        # Rebuild agent fresh for each condition to avoid state leakage
        agent = build_agent(train_config, obs_space, act_space, ckpt, device)
        cond["setup"](agent)

        result = evaluate_condition(cond_name, agent, envs, args.num_episodes, device)
        result["description"] = cond["description"]
        all_results.append(result)

        # ETA for remaining conditions
        elapsed_overall = time.time() - t_overall
        avg_per_cond = elapsed_overall / cond_idx
        remaining_conds = len(valid_conditions) - cond_idx
        eta = avg_per_cond * remaining_conds
        if remaining_conds > 0:
            print(f"  ETA for remaining {remaining_conds} condition(s): ~{eta:.0f}s", flush=True)

        del agent
        torch.cuda.empty_cache()
        print()

    for cond_name in args.conditions:
        if cond_name not in CONDITIONS:
            print(f"WARNING: Unknown condition '{cond_name}', skipping.")

    total_elapsed = time.time() - t_overall
    print(f"All conditions complete in {total_elapsed:.1f}s\n")

    # ==========================================
    # Print summary
    # ==========================================
    print("=" * 78)
    print("TEXT-SWAP ABLATION RESULTS")
    print("=" * 78)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {train_config.env.task}")
    print(f"Episodes per condition: {args.num_episodes}")
    print()

    baseline_score = None
    header = f"{'Condition':<20} {'Mean Score':>12} {'Std':>8} {'vs Real':>10}"
    print(header)
    print("-" * 54)
    for r in all_results:
        if r["condition"] == "real_text":
            baseline_score = r["mean_score"]
        delta = ""
        if baseline_score is not None and r["condition"] != "real_text":
            d = r["mean_score"] - baseline_score
            pct = (d / max(abs(baseline_score), 1e-8)) * 100
            delta = f"{d:+.1f} ({pct:+.1f}%)"
        print(f"{r['condition']:<20} {r['mean_score']:>12.1f} {r['std_score']:>8.1f} {delta:>10}")
    print("-" * 54)

    # ==========================================
    # Save results
    # ==========================================
    results_path = output_dir / "text_swap_results.json"
    save_data = {
        "checkpoint": args.checkpoint,
        "task": str(train_config.env.task),
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "total_time_s": round(total_elapsed, 1),
        "results": [
            {k: v for k, v in r.items() if k != "all_scores"}
            for r in all_results
        ],
        "all_scores": {r["condition"]: r["all_scores"] for r in all_results},
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    file_size = results_path.stat().st_size
    print(f"\nResults saved to: {results_path} ({file_size} bytes)", flush=True)

    # Interpretation
    if baseline_score is not None:
        drops = {}
        for r in all_results:
            if r["condition"] != "real_text":
                drops[r["condition"]] = baseline_score - r["mean_score"]

        if drops:
            print("\n--- Interpretation ---")
            for cond, drop in drops.items():
                pct = (drop / max(abs(baseline_score), 1e-8)) * 100
                if pct > 10:
                    print(f"  {cond}: Significant drop ({drop:.1f}, {pct:.1f}%) -> model uses text")
                elif pct > 3:
                    print(f"  {cond}: Moderate drop ({drop:.1f}, {pct:.1f}%) -> text has some influence")
                else:
                    print(f"  {cond}: Minimal drop ({drop:.1f}, {pct:.1f}%) -> text not strongly used")


if __name__ == "__main__":
    main()
