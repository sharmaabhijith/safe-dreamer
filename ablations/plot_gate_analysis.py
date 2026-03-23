"""E3: Plot text gate value evolution over training.

Reads TensorBoard event files from experiments that use the multimodal encoder
and plots how text_gate_mean evolves over training steps. This shows how the
model learns to use text over time.

Usage:
    python ablations/plot_gate_analysis.py --logdir /path/to/logdir

    # Compare multiple runs:
    python ablations/plot_gate_analysis.py \
        --logdirs /path/to/a4_full_multimodal \
        --labels "Full Multimodal" \
        --output ablations/results/e3_gate_analysis.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
})


def read_tensorboard_scalar(logdir, tag):
    """Read a scalar tag from TensorBoard event files."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(logdir))
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        # Try with train/ prefix
        tag_with_prefix = f"train/{tag}"
        if tag_with_prefix not in ea.Tags().get("scalars", []):
            available = ea.Tags().get("scalars", [])
            gate_tags = [t for t in available if "gate" in t.lower()]
            raise ValueError(
                f"Tag '{tag}' not found. Gate-related tags: {gate_tags}\n"
                f"All available: {available[:20]}..."
            )
        tag = tag_with_prefix

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values, weight=0.9):
    """Exponential moving average for smoother curves."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def plot_gate_analysis(logdirs, labels, output_path, smooth_weight=0.9):
    """Plot text_gate_mean over training for multiple experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(logdirs), 3)))

    # Plot 1: Gate mean over training
    ax = axes[0]
    for i, (logdir, label) in enumerate(zip(logdirs, labels)):
        try:
            steps, values = read_tensorboard_scalar(logdir, "encoder/text_gate_mean")
            ax.plot(steps, smooth(values, smooth_weight), color=colors[i],
                    label=label, linewidth=2)
            ax.plot(steps, values, color=colors[i], alpha=0.15, linewidth=0.5)
        except Exception as e:
            print(f"Warning: could not read gate_mean from {logdir}: {e}")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Text Gate Mean")
    ax.set_title("(a) Text Gate Value Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.047, color="gray", linestyle="--", alpha=0.5, label="Init (sigmoid(-3))")

    # Plot 2: Eval score over training
    ax = axes[1]
    for i, (logdir, label) in enumerate(zip(logdirs, labels)):
        try:
            steps, values = read_tensorboard_scalar(logdir, "episode/eval_score")
            ax.plot(steps, smooth(values, smooth_weight), color=colors[i],
                    label=label, linewidth=2)
            ax.plot(steps, values, color=colors[i], alpha=0.15, linewidth=0.5)
        except Exception as e:
            print(f"Warning: could not read eval_score from {logdir}: {e}")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Score")
    ax.set_title("(b) Evaluation Score Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved gate analysis plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E3: Plot text gate value evolution over training")
    parser.add_argument("--logdirs", nargs="+", required=True,
                        help="TensorBoard log directories to plot")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each logdir (defaults to directory names)")
    parser.add_argument("--output", type=str,
                        default="ablations/results/e3_gate_analysis.pdf",
                        help="Output file path")
    parser.add_argument("--smooth", type=float, default=0.9,
                        help="EMA smoothing weight (0=no smoothing, 1=max)")
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [Path(d).name for d in args.logdirs]

    assert len(args.logdirs) == len(args.labels), "Must have same number of logdirs and labels"

    plot_gate_analysis(args.logdirs, args.labels, args.output, args.smooth)


if __name__ == "__main__":
    main()
