"""
plotting.py — STAGE 7: Trade-Off Visualisations

Produces publication-style matplotlib figures comparing the three models
across key metrics.  Answers the core research question:
  "How does reward design affect clarity, completeness, and over-optimization?"

Figures saved to outputs/figures/:
  1. readability_vs_completeness.png   — scatter: readability × completeness
  2. reward_vs_completeness.png        — scatter: simplicity reward × completeness
  3. readability_vs_repetition.png     — scatter: readability × repetition
  4. metric_bar_chart.png              — grouped bar chart of key metrics
  5. ppo_training_curves.png           — reward over PPO steps (both runs)
  6. radar_chart.png                   — radar / spider chart of normalised metrics
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, FIGURES_DIR, REPORT_DIR, print_config

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
MODEL_COLORS = {
    "baseline":        "#4C72B0",
    "ppo_simplicity":  "#DD8452",
    "ppo_balanced":    "#55A868",
}
MODEL_MARKERS = {
    "baseline":        "o",
    "ppo_simplicity":  "s",
    "ppo_balanced":    "^",
}
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def _save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path}")


# ═════════════════════════════════════════════
#  1.  READABILITY  vs  COMPLETENESS  (scatter)
# ═════════════════════════════════════════════

def plot_readability_vs_completeness(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, grp in df.groupby("model"):
        ax.scatter(
            grp["readability"], grp["completeness"],
            label=model, alpha=0.75, s=60,
            color=MODEL_COLORS.get(model, "gray"),
            marker=MODEL_MARKERS.get(model, "o"),
        )
    ax.set_xlabel("Readability (Flesch Reading Ease)")
    ax.set_ylabel("Completeness (keyword coverage)")
    ax.set_title("Readability vs Completeness — Trade-Off View")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "readability_vs_completeness.png")


# ═════════════════════════════════════════════
#  2.  REWARD  vs  COMPLETENESS
# ═════════════════════════════════════════════

def plot_reward_vs_completeness(df):
    reward_col = "simplicity_heuristic" if "simplicity_heuristic" in df.columns else "readability"
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, grp in df.groupby("model"):
        ax.scatter(
            grp[reward_col], grp["completeness"],
            label=model, alpha=0.75, s=60,
            color=MODEL_COLORS.get(model, "gray"),
            marker=MODEL_MARKERS.get(model, "o"),
        )
    ax.set_xlabel("Simplicity Reward (heuristic)")
    ax.set_ylabel("Completeness")
    ax.set_title("Simplicity Reward vs Completeness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "reward_vs_completeness.png")


# ═════════════════════════════════════════════
#  3.  READABILITY  vs  REPETITION
# ═════════════════════════════════════════════

def plot_readability_vs_repetition(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, grp in df.groupby("model"):
        ax.scatter(
            grp["readability"], grp["repetition"],
            label=model, alpha=0.75, s=60,
            color=MODEL_COLORS.get(model, "gray"),
            marker=MODEL_MARKERS.get(model, "o"),
        )
    ax.set_xlabel("Readability (Flesch Reading Ease)")
    ax.set_ylabel("Repetition (n-gram ratio)")
    ax.set_title("Readability vs Repetition")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "readability_vs_repetition.png")


# ═════════════════════════════════════════════
#  4.  GROUPED BAR CHART  of key metrics
# ═════════════════════════════════════════════

def plot_metric_bar_chart(summary_df):
    metrics = ["readability", "completeness", "response_length", "repetition", "lexical_diversity"]
    available = [m for m in metrics if m in summary_df.columns]
    models = summary_df["model"].tolist()

    x = np.arange(len(available))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models):
        row = summary_df[summary_df["model"] == model].iloc[0]
        vals = [row[m] for m in available]
        # Normalise response_length for visual comparability
        norm_vals = []
        for m, v in zip(available, vals):
            if m == "response_length":
                norm_vals.append(v / 100.0)  # scale down
            elif m == "readability":
                norm_vals.append(v / 100.0)  # scale to ~[0,1]
            else:
                norm_vals.append(v)
        ax.bar(
            x + i * width, norm_vals, width,
            label=model, color=MODEL_COLORS.get(model, "gray"),
        )

    ax.set_xticks(x + width)
    labels_display = [m.replace("_", " ").title() for m in available]
    labels_display = [l + " (/100)" if "Readability" in l else l for l in labels_display]
    labels_display = [l + " (/100)" if "Response Length" in l else l for l in labels_display]
    ax.set_xticklabels(labels_display, rotation=15, ha="right")
    ax.set_ylabel("Normalised Value")
    ax.set_title("Key Metrics Comparison (normalised)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "metric_bar_chart.png")


# ═════════════════════════════════════════════
#  5.  PPO TRAINING CURVES
# ═════════════════════════════════════════════

def plot_ppo_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for idx, (label, color) in enumerate([
        ("ppo_simplicity", MODEL_COLORS["ppo_simplicity"]),
        ("ppo_balanced",   MODEL_COLORS["ppo_balanced"]),
    ]):
        path = os.path.join(OUTPUT_DIR, f"{label}_log.csv")
        if not os.path.exists(path):
            axes[idx].set_title(f"{label} (no data)")
            continue
        log = pd.read_csv(path)
        axes[idx].plot(log["step"], log["mean_reward"], color=color, marker="o", markersize=4)
        axes[idx].set_xlabel("PPO Step")
        axes[idx].set_ylabel("Mean Reward")
        axes[idx].set_title(f"Training Reward — {label}")
        axes[idx].grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "ppo_training_curves.png")


# ═════════════════════════════════════════════
#  6.  RADAR / SPIDER CHART
# ═════════════════════════════════════════════

def plot_radar_chart(summary_df):
    categories = ["readability", "completeness", "lexical_diversity"]
    available = [c for c in categories if c in summary_df.columns]
    if len(available) < 3:
        print("  [skip] radar chart — not enough metrics")
        return

    N = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for _, row in summary_df.iterrows():
        model = row["model"]
        vals = []
        for c in available:
            v = row[c]
            # Normalise readability to [0, 1]
            if c == "readability":
                v = v / 100.0
            vals.append(v)
        vals += vals[:1]
        ax.plot(angles, vals, marker="o", label=model,
                color=MODEL_COLORS.get(model, "gray"))
        ax.fill(angles, vals, alpha=0.1, color=MODEL_COLORS.get(model, "gray"))

    ax.set_thetagrids(np.degrees(angles[:-1]), [c.replace("_", " ").title() for c in available])
    ax.set_title("Model Comparison — Radar", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save_fig(fig, "radar_chart.png")


# ═════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════

def run_plotting():
    print_config()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load evaluation data
    full_path = os.path.join(OUTPUT_DIR, "eval_full.csv")
    summary_path = os.path.join(OUTPUT_DIR, "eval_metrics_summary.csv")

    if not os.path.exists(full_path):
        print("[Stage 7] eval_full.csv not found. Run evaluate.py first.")
        return
    if not os.path.exists(summary_path):
        print("[Stage 7] eval_metrics_summary.csv not found. Run evaluate.py first.")
        return

    df = pd.read_csv(full_path)
    summary_df = pd.read_csv(summary_path)

    print("[Stage 7] Generating plots ...\n")

    plot_readability_vs_completeness(df)
    plot_reward_vs_completeness(df)
    plot_readability_vs_repetition(df)
    plot_metric_bar_chart(summary_df)
    plot_ppo_training_curves()
    plot_radar_chart(summary_df)

    print(f"\n[Stage 7] All plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run_plotting()
