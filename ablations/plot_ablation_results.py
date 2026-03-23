"""Plot ablation study results.

Reads TensorBoard event files from training ablation runs and produces:
  1. Component ablation bar chart (A2-A5)
  2. Distractor difficulty sweep line plot (F1)
  3. Parameter-matched comparison bar chart (H3 vs A4 vs A5)
  4. Training curves for all ablations
  5. Summary table printed to console and saved as CSV

Text content ablations (adversarial, nonsense, random, zero) are now
eval-time only — see ablations/eval_text_swap.py.

Usage:
    python ablations/plot_ablation_results.py \
        --base_logdir /nfs-stor/salem.lahlou/asharma/logdir/ablations \
        --output_dir ablations/results
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})


def read_final_eval_score(logdir, last_n=5):
    """Read final eval score (mean of last N evaluations) from TensorBoard."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logdir = Path(logdir)
    if not logdir.exists():
        return None, None

    ea = EventAccumulator(str(logdir))
    ea.Reload()

    tag = "episode/eval_score"
    if tag not in ea.Tags().get("scalars", []):
        return None, None

    events = ea.Scalars(tag)
    if len(events) < last_n:
        last_n = len(events)
    values = np.array([e.value for e in events[-last_n:]])
    return float(np.mean(values)), float(np.std(values))


def read_eval_curve(logdir):
    """Read full eval score curve from TensorBoard."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logdir = Path(logdir)
    if not logdir.exists():
        return None, None

    ea = EventAccumulator(str(logdir))
    ea.Reload()

    tag = "episode/eval_score"
    if tag not in ea.Tags().get("scalars", []):
        return None, None

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values, weight=0.9):
    """Exponential moving average."""
    s = np.zeros_like(values)
    s[0] = values[0]
    for i in range(1, len(values)):
        s[i] = weight * s[i - 1] + (1 - weight) * values[i]
    return s


def plot_component_ablation(results, output_dir):
    """Bar chart for component isolation ablations (A2-A5)."""
    ablations = [
        ("A5: CNN Baseline", "ablation_a5_cnn_baseline"),
        ("A3: Gate Only\n(No FiLM)", "ablation_a3_gate_only"),
        ("A2: FiLM Only\n(No Gate)", "ablation_a2_film_only"),
        ("A4: Full Multimodal\n(Ours)", "ablation_a4_full_multimodal"),
    ]

    labels, means, stds = [], [], []
    for label, key in ablations:
        if key in results and results[key][0] is not None:
            labels.append(label)
            means.append(results[key][0])
            stds.append(results[key][1])

    if not labels:
        print("No data for component ablation plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728", "#9467bd", "#2ca02c", "#1f77b4"][:len(labels)]
    bars = ax.bar(range(len(labels)), means, yerr=stds, color=colors,
                  edgecolor="black", linewidth=0.8, capsize=5, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Final Eval Score")
    ax.set_title("Component Isolation Ablation — Distract Cheetah Run")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Add value labels on bars
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "component_ablation.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'component_ablation.pdf'}")
    plt.close(fig)


def plot_text_content_ablation(results, output_dir):
    """Placeholder: text content ablations are now eval-time only.

    See ablations/eval_text_swap.py for text-swap evaluation results.
    """
    print("Text content ablation is now eval-time only (see eval_text_swap.py).")


def plot_difficulty_sweep(results, output_dir):
    """Line plot for distractor difficulty sweep (F1)."""
    difficulties = ["easy", "medium", "hard"]

    cnn_means, cnn_stds = [], []
    mm_means, mm_stds = [], []

    cnn_keys = ["ablation_a5_cnn_baseline", "ablation_f1_cnn_medium", "ablation_f1_cnn_hard"]
    mm_keys = ["ablation_a4_full_multimodal", "ablation_f1_multimodal_medium", "ablation_f1_multimodal_hard"]

    for key in cnn_keys:
        if key in results and results[key][0] is not None:
            cnn_means.append(results[key][0])
            cnn_stds.append(results[key][1])
        else:
            cnn_means.append(0)
            cnn_stds.append(0)

    for key in mm_keys:
        if key in results and results[key][0] is not None:
            mm_means.append(results[key][0])
            mm_stds.append(results[key][1])
        else:
            mm_means.append(0)
            mm_stds.append(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(difficulties))

    ax.errorbar(x, cnn_means, yerr=cnn_stds, marker="o", markersize=8,
                linewidth=2, capsize=5, color="#d62728", label="CNN Baseline")
    ax.errorbar(x, mm_means, yerr=mm_stds, marker="s", markersize=8,
                linewidth=2, capsize=5, color="#1f77b4", label="Multimodal (Ours)")

    # Shade the gap
    ax.fill_between(x, cnn_means, mm_means, alpha=0.15, color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties], fontsize=12)
    ax.set_xlabel("Distractor Difficulty")
    ax.set_ylabel("Final Eval Score")
    ax.set_title("Distractor Difficulty Sweep — Cheetah Run")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "difficulty_sweep.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'difficulty_sweep.pdf'}")
    plt.close(fig)


def plot_parameter_comparison(results, output_dir):
    """Bar chart for parameter-matched comparison (H3 vs A4 vs A5)."""
    ablations = [
        ("A5: CNN Baseline\n(220K params)", "ablation_a5_cnn_baseline"),
        ("H3: Wider CNN\n(~5M params)", "ablation_h3_wider_cnn"),
        ("A4: Multimodal\n(~5M params)", "ablation_a4_full_multimodal"),
    ]

    labels, means, stds = [], [], []
    for label, key in ablations:
        if key in results and results[key][0] is not None:
            labels.append(label)
            means.append(results[key][0])
            stds.append(results[key][1])

    if not labels:
        print("No data for parameter comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#d62728", "#ff7f0e", "#1f77b4"][:len(labels)]
    bars = ax.bar(range(len(labels)), means, yerr=stds, color=colors,
                  edgecolor="black", linewidth=0.8, capsize=5, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Final Eval Score")
    ax.set_title("Parameter-Matched Comparison — Distract Cheetah Run")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "parameter_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'parameter_comparison.pdf'}")
    plt.close(fig)


def plot_training_curves(results_dirs, output_dir):
    """Plot training curves for all ablations on one figure."""
    ablations = [
        ("A5: CNN Baseline", "ablation_a5_cnn_baseline", "#d62728", "-"),
        ("A3: Gate Only", "ablation_a3_gate_only", "#9467bd", "--"),
        ("A2: FiLM Only", "ablation_a2_film_only", "#2ca02c", "--"),
        ("H3: Wider CNN", "ablation_h3_wider_cnn", "#bcbd22", "-."),
        ("A4: Full Multimodal (Ours)", "ablation_a4_full_multimodal", "#1f77b4", "-"),
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    for label, key, color, ls in ablations:
        if key in results_dirs:
            steps, values = read_eval_curve(results_dirs[key])
            if steps is not None:
                ax.plot(steps, smooth(values, 0.9), color=color, linestyle=ls,
                        label=label, linewidth=2)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Score")
    ax.set_title("Training Curves — Distract Cheetah Run Ablations")
    ax.legend(loc="lower right", ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'training_curves.pdf'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--base_logdir", type=str, required=True,
                        help="Base logdir containing experiment subdirectories")
    parser.add_argument("--task", type=str, default="distract_cheetah_run",
                        help="Task subdirectory inside each experiment folder")
    parser.add_argument("--output_dir", type=str, default="ablations/results",
                        help="Output directory for plots and tables")
    parser.add_argument("--last_n", type=int, default=5,
                        help="Average last N evaluations for final score")
    args = parser.parse_args()

    base_logdir = Path(args.base_logdir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expected experiment directories
    experiment_names = [
        "ablation_a2_film_only",
        "ablation_a3_gate_only",
        "ablation_a4_full_multimodal",
        "ablation_a5_cnn_baseline",
        "ablation_f1_cnn_medium",
        "ablation_f1_cnn_hard",
        "ablation_f1_multimodal_medium",
        "ablation_f1_multimodal_hard",
        "ablation_h3_wider_cnn",
    ]

    # Read results
    results = {}
    results_dirs = {}
    for name in experiment_names:
        logdir = base_logdir / name / args.task
        mean, std = read_final_eval_score(logdir, args.last_n)
        results[name] = (mean, std)
        results_dirs[name] = logdir
        status = f"{mean:.1f} +/- {std:.1f}" if mean is not None else "NOT FOUND"
        print(f"  {name:45s} {status}")

    print()

    # Generate plots
    plot_component_ablation(results, output_dir)
    plot_text_content_ablation(results, output_dir)
    plot_difficulty_sweep(results, output_dir)
    plot_parameter_comparison(results, output_dir)
    plot_training_curves(results_dirs, output_dir)

    # Save summary CSV
    csv_path = output_dir / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ablation", "Experiment", "FiLM", "TextGate", "Text Type", "Mean Score", "Std Score"])
        rows = [
            ("A5", "ablation_a5_cnn_baseline", "No", "No", "None"),
            ("A3", "ablation_a3_gate_only", "No", "Yes", "Real CLIP"),
            ("A2", "ablation_a2_film_only", "Yes", "No", "Real CLIP"),
            ("H3", "ablation_h3_wider_cnn", "No", "No", "None (wider)"),
            ("A4", "ablation_a4_full_multimodal", "Yes", "Yes", "Real CLIP"),
            ("F1-CNN-med", "ablation_f1_cnn_medium", "No", "No", "None"),
            ("F1-CNN-hard", "ablation_f1_cnn_hard", "No", "No", "None"),
            ("F1-MM-med", "ablation_f1_multimodal_medium", "Yes", "Yes", "Real CLIP"),
            ("F1-MM-hard", "ablation_f1_multimodal_hard", "Yes", "Yes", "Real CLIP"),
        ]
        for ablation_id, key, film, gate, text_type in rows:
            mean, std = results.get(key, (None, None))
            writer.writerow([
                ablation_id, key, film, gate, text_type,
                f"{mean:.1f}" if mean is not None else "N/A",
                f"{std:.1f}" if std is not None else "N/A",
            ])

    print(f"Saved: {csv_path}")
    print("\nDone! All plots saved to", output_dir)


if __name__ == "__main__":
    main()
