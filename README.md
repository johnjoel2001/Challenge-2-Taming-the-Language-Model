# Reward Design Matters: A Multi-Objective RLHF Study of Clarity, Completeness, and Over-Optimization in Small Language Models

## Project Overview

This project implements a **complete end-to-end RLHF pipeline** using Hugging Face TRL on a small language model (**GPT-2, 124M parameters**). The goal is to align the model to generate beginner-friendly explanations of technical topics — and then **systematically compare** how different reward designs lead to different behaviours, trade-offs, and failure modes.

### Core Research Question

> *How does reward design affect clarity, completeness, and over-optimization in RLHF for beginner-friendly technical explanations?*

### Three Models Compared

| # | Model | Description |
|---|-------|-------------|
| 1 | **Baseline** | Pretrained GPT-2 (124M), no RLHF |
| 2 | **Simplicity-PPO** | RLHF-aligned to maximise readability & brevity |
| 3 | **Balanced-PPO** | RLHF-aligned to balance readability, completeness & low repetition |

### Key Findings

- Different reward functions produce **measurably different model behaviours** after just 8 PPO steps.
- **Over-optimization is observable** even with minimal training — the balanced model triggered 16 misalignment flags.
- **Multi-objective rewards are not automatically safer** in low-data regimes — competing objectives can cause erratic optimisation.
- RLHF **cannot inject knowledge** the base model doesn't have — completeness stayed at ~20% across all models.
- **Reward design matters as much as PPO itself.**

---

## Repository Structure

```
├── data/                          # Generated preference datasets
│   ├── candidates.csv             # 48 candidate responses with scores
│   ├── preference_simplicity.csv  # 12 simplicity preference pairs
│   └── preference_balanced.csv    # 12 balanced preference pairs
├── src/
│   ├── __init__.py
│   ├── config.py                  # Central configuration & hyperparameters
│   ├── prompts.py                 # Prompt lists & keyword mappings
│   ├── utils.py                   # Metrics, scoring, I/O helpers
│   ├── generate_baseline.py       # Stage 1 — baseline generation
│   ├── create_preference_data.py  # Stage 2 — preference dataset
│   ├── train_reward_model_simplicity.py  # Stage 3A — simplicity RM
│   ├── train_reward_model_balanced.py    # Stage 3B — balanced RM
│   ├── train_ppo_simplicity.py    # Stage 4A — PPO with simplicity RM
│   ├── train_ppo_balanced.py      # Stage 4B — PPO with balanced RM
│   ├── evaluate.py                # Stage 5 — comparative evaluation
│   ├── misalignment_analysis.py   # Stage 6 — failure-mode detection
│   └── plotting.py                # Stage 7 — trade-off visualisations
├── notebooks/
│   └── pipeline_demo.ipynb        # End-to-end walkthrough notebook
├── outputs/                       # All generated outputs
│   ├── models/                    # Saved reward models & PPO models
│   ├── figures/                   # 6 matplotlib visualisations
│   ├── eval_metrics_summary.csv   # Per-model metric comparison
│   ├── eval_side_by_side.csv      # Side-by-side model responses
│   ├── eval_full.csv              # 48 rows of per-prompt evaluation
│   ├── flagged_examples.csv       # Misalignment-flagged outputs
│   └── misalignment_summary.json  # Failure counts per model
├── reports/
│   ├── final_report.md            # Full research report with actual results
│   └── misalignment_report.md     # Auto-generated failure-mode report
├── requirements.txt
└── README.md                      # This file
```

---

## Environment Setup

### Prerequisites
- Python 3.9+
- pip
- macOS (Apple Silicon MPS supported), Linux, or Windows

### Installation

```bash
# Navigate to the project directory
cd "Challenge 2"

# (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# Install dependencies
python3 -m pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch>=2.0` | PyTorch (CUDA, MPS, or CPU) |
| `transformers>=4.35,<5.0` | Hugging Face model loading & generation |
| `trl>=0.7,<0.10` | PPOTrainer & AutoModelForCausalLMWithValueHead |
| `datasets>=2.14` | Hugging Face dataset format |
| `accelerate>=0.24` | Device placement & mixed precision |
| `textstat>=0.7.3` | Flesch Reading Ease computation |
| `pandas>=2.0` | Data manipulation |
| `numpy>=1.24` | Numerical operations |
| `matplotlib>=3.7` | Plotting |
| `scikit-learn>=1.3` | Utilities |

---

## How to Run

### Quick Mode (default, ~5–8 minutes on Apple Silicon)

Quick mode uses minimal epochs and steps for fast end-to-end execution.

```bash
export RLHF_MODE=quick   # optional, "quick" is the default
```

### Extended Mode (~20–40 minutes)

```bash
export RLHF_MODE=extended
```

### Step-by-Step Execution

Run each stage from the **project root directory**:

```bash
# Stage 1 — Generate baseline outputs & metrics
python3 -m src.generate_baseline

# Stage 2 — Create preference datasets (simplicity + balanced)
python3 -m src.create_preference_data

# Stage 3A — Train simplicity reward model
python3 -m src.train_reward_model_simplicity

# Stage 3B — Train balanced reward model
python3 -m src.train_reward_model_balanced

# Stage 4A — PPO training with simplicity reward
python3 -m src.train_ppo_simplicity

# Stage 4B — PPO training with balanced reward
python3 -m src.train_ppo_balanced

# Stage 5 — Evaluate all three models
python3 -m src.evaluate

# Stage 6 — Misalignment & over-optimization analysis
python3 -m src.misalignment_analysis

# Stage 7 — Generate trade-off plots
python3 -m src.plotting
```

### Run Everything at Once

```bash
python3 -m src.generate_baseline && \
python3 -m src.create_preference_data && \
python3 -m src.train_reward_model_simplicity && \
python3 -m src.train_reward_model_balanced && \
python3 -m src.train_ppo_simplicity && \
python3 -m src.train_ppo_balanced && \
python3 -m src.evaluate && \
python3 -m src.misalignment_analysis && \
python3 -m src.plotting
```

### Notebook

Open `notebooks/pipeline_demo.ipynb` in Jupyter for an interactive walkthrough.

---

## Results Summary

### Evaluation Metrics (GPT-2, seed 42, quick mode)

| Metric | Baseline | Simplicity-PPO | Balanced-PPO |
|--------|----------|----------------|--------------|
| Readability (FRE) | 43.46 | 32.19 | 30.53 |
| Response Length | 116.1 | 110.4 | 83.1 |
| Lexical Diversity | 0.947 | 0.937 | 0.772 |
| Completeness | 0.214 | 0.194 | 0.202 |
| Repetition | 0.0 | 0.0 | 0.0 |
| Pairwise Overlap | 0.138 | 0.125 | 0.106 |

### Misalignment Flags

| Model | Over-shortening | Completeness Loss | Low Diversity | Reward Hacking | **Total** |
|-------|:-:|:-:|:-:|:-:|:-:|
| Baseline | 0 | 6 | 0 | 1 | **7** |
| Simplicity-PPO | 0 | 6 | 0 | 0 | **6** |
| Balanced-PPO | 3 | 6 | 3 | 4 | **16** |

See `reports/final_report.md` for the full analysis with example outputs.

---

## Generated Outputs

After a full run you will find:

| Path | Description |
|------|-------------|
| `outputs/baseline_outputs.csv` | 16 baseline responses + per-prompt metrics |
| `outputs/baseline_metrics.json` | Aggregated baseline metrics |
| `data/candidates.csv` | 48 candidate responses with heuristic scores |
| `data/preference_*.csv` | Preference pairs (12 each) |
| `outputs/models/reward_model_*/` | Trained reward models (weights + tokenizer) |
| `outputs/ppo_*_log.csv` | PPO training logs (reward, length, KL per step) |
| `outputs/ppo_*_samples.csv` | Sample generations from each PPO step |
| `outputs/models/ppo_*/` | PPO-aligned models |
| `outputs/eval_side_by_side.csv` | Side-by-side responses for all 16 prompts |
| `outputs/eval_metrics_summary.csv` | Metric comparison table (3 rows) |
| `outputs/eval_full.csv` | Full evaluation data (48 rows) |
| `outputs/flagged_examples.csv` | All misalignment-flagged outputs |
| `outputs/misalignment_summary.json` | Failure counts per model |
| `reports/misalignment_report.md` | Auto-generated failure-mode report |
| `outputs/figures/readability_vs_completeness.png` | Trade-off scatter |
| `outputs/figures/reward_vs_completeness.png` | Reward-hacking zone |
| `outputs/figures/readability_vs_repetition.png` | Readability vs. repetition |
| `outputs/figures/metric_bar_chart.png` | Grouped metric comparison |
| `outputs/figures/ppo_training_curves.png` | PPO reward/length over steps |
| `outputs/figures/radar_chart.png` | Multi-dimensional radar chart |

---

## Key Metrics Explained

| Metric | Description | Range |
|--------|-------------|-------|
| **Readability** | Flesch Reading Ease | 0–120 (higher = simpler) |
| **Completeness** | Fraction of expected keywords present | 0–1 |
| **Repetition** | Repeated bigram ratio | 0–1 (lower = better) |
| **Lexical Diversity** | Type-token ratio (unique/total words) | 0–1 (higher = more diverse) |
| **Response Length** | Word count | varies |
| **Pairwise Overlap** | Jaccard similarity across a model's outputs | 0–1 (high = template collapse) |

---

## Configuration

Edit `src/config.py` or set the `RLHF_MODE` environment variable:

| Parameter | Quick | Extended |
|-----------|-------|----------|
| Base model | `gpt2` (124M) | `gpt2` (124M) |
| RM epochs | 2 | 5 |
| PPO steps | 8 | 24 |
| PPO batch size | 4 | 4 |
| Candidates per prompt | 4 | 6 |
| Max new tokens (gen) | 120 | 180 |

To use a different base model, change `BASE_MODEL_NAME` in `src/config.py`.

---

## Reproducibility

- All scripts use `SEED = 42` via `set_seed()`.
- Device auto-detection: CUDA → MPS → CPU.
- Outputs are deterministic given the same seed, mode, and hardware.

---

## License

This project is for educational and research purposes.
