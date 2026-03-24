"""Stage 7 — generate trade-off visualisations comparing the three models."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, FIGURES_DIR

MODEL_COLORS = {
    "baseline":       "#4C72B0",
    "ppo_simplicity": "#DD8452",
    "ppo_balanced":   "#55A868",
}
MODEL_MARKERS = {
    "baseline":       "o",
    "ppo_simplicity": "s",
    "ppo_balanced":   "^",
}
plt.rcParams.update({
    "figure.dpi":    130,
    "font.size":     10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def _save_fig(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, name), bbox_inches="tight")
    plt.close(fig)


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


def plot_reward_vs_completeness(df):
    reward_col = "simplicity_heuristic" if "simplicity_heuristic" in df.columns else "readability"  # fallback if eval skipped heuristics
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


def plot_metric_bar_chart(summary_df):
    metrics   = ["readability", "completeness", "response_length", "repetition", "lexical_diversity"]
    available = [m for m in metrics if m in summary_df.columns]
    models    = summary_df["model"].tolist()

    x     = np.arange(len(available))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models):
        row  = summary_df[summary_df["model"] == model].iloc[0]
        norm_vals = []
        for m in available:
            v = row[m]
            # Scale readability and response_length into roughly [0,1] for visual comparability
            if m in ("readability", "response_length"):
                v /= 100.0
            norm_vals.append(v)
        ax.bar(x + i * width, norm_vals, width, label=model, color=MODEL_COLORS.get(model, "gray"))

    ax.set_xticks(x + width)
    labels_display = [m.replace("_", " ").title() for m in available]
    labels_display = [l + " (/100)" if l in ("Readability", "Response Length") else l
                      for l in labels_display]
    ax.set_xticklabels(labels_display, rotation=15, ha="right")
    ax.set_ylabel("Normalised Value")
    ax.set_title("Key Metrics Comparison (normalised)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "metric_bar_chart.png")


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


def plot_radar_chart(summary_df):
    categories = ["readability", "completeness", "lexical_diversity"]
    available  = [c for c in categories if c in summary_df.columns]
    if len(available) < 3:
        return

    N      = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for _, row in summary_df.iterrows():
        model = row["model"]
        vals  = [row[c] / 100.0 if c == "readability" else row[c] for c in available]
        vals += vals[:1]
        ax.plot(angles, vals, marker="o", label=model, color=MODEL_COLORS.get(model, "gray"))
        ax.fill(angles, vals, alpha=0.1, color=MODEL_COLORS.get(model, "gray"))

    ax.set_thetagrids(np.degrees(angles[:-1]), [c.replace("_", " ").title() for c in available])
    ax.set_title("Model Comparison — Radar", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save_fig(fig, "radar_chart.png")


def run_plotting():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    full_path    = os.path.join(OUTPUT_DIR, "eval_full.csv")
    summary_path = os.path.join(OUTPUT_DIR, "eval_metrics_summary.csv")

    if not os.path.exists(full_path) or not os.path.exists(summary_path):
        return

    df         = pd.read_csv(full_path)
    summary_df = pd.read_csv(summary_path)

    plot_readability_vs_completeness(df)
    plot_reward_vs_completeness(df)
    plot_readability_vs_repetition(df)
    plot_metric_bar_chart(summary_df)
    plot_ppo_training_curves()
    plot_radar_chart(summary_df)


if __name__ == "__main__":
    run_plotting()
