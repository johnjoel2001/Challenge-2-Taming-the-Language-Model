"""
config.py — Central configuration for the RLHF research project.

Contains all hyperparameters, paths, and mode settings.
Two modes:
  - "quick"    : minimal epochs/steps for fast smoke-testing (~2-5 min)
  - "extended" : longer training for more meaningful results (~15-30 min on CPU)
"""

import os
import torch

# ─────────────────────────────────────────────
# PROJECT ROOT (auto-detected from this file)
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Ensure directories exist
for d in [DATA_DIR, OUTPUT_DIR, REPORT_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# MODE: "quick" or "extended"
# Set via environment variable or change default here.
# ─────────────────────────────────────────────
MODE = os.environ.get("RLHF_MODE", "quick")  # "quick" | "extended"

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
BASE_MODEL_NAME = "gpt2"  # GPT-2 small (124M params)

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ─────────────────────────────────────────────
# RANDOM SEED
# ─────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────
# GENERATION SETTINGS
# ─────────────────────────────────────────────
GENERATION_KWARGS = dict(
    max_new_tokens=120 if MODE == "quick" else 180,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    repetition_penalty=1.2,
)

# ─────────────────────────────────────────────
# REWARD MODEL TRAINING
# ─────────────────────────────────────────────
REWARD_MODEL_EPOCHS = 2 if MODE == "quick" else 5
REWARD_MODEL_LR = 2e-5
REWARD_MODEL_BATCH_SIZE = 4
REWARD_MODEL_MAX_LEN = 256

# ─────────────────────────────────────────────
# PPO TRAINING
# ─────────────────────────────────────────────
PPO_EPOCHS = 2 if MODE == "quick" else 6
PPO_STEPS = 8 if MODE == "quick" else 24
PPO_BATCH_SIZE = 4
PPO_MINI_BATCH_SIZE = 2
PPO_LR = 1.41e-5
PPO_MAX_NEW_TOKENS = 100 if MODE == "quick" else 150

# ─────────────────────────────────────────────
# PREFERENCE DATA
# ─────────────────────────────────────────────
NUM_CANDIDATES_PER_PROMPT = 4 if MODE == "quick" else 6

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
EVAL_MAX_NEW_TOKENS = 150

# ─────────────────────────────────────────────
# PRINTING HELPER
# ─────────────────────────────────────────────
def print_config():
    """Pretty-print current configuration."""
    print("=" * 55)
    print("  RLHF Research Project — Configuration")
    print("=" * 55)
    print(f"  Mode          : {MODE}")
    print(f"  Base model    : {BASE_MODEL_NAME}")
    print(f"  Device        : {DEVICE}")
    print(f"  Seed          : {SEED}")
    print(f"  RM epochs     : {REWARD_MODEL_EPOCHS}")
    print(f"  PPO steps     : {PPO_STEPS}")
    print(f"  Project root  : {PROJECT_ROOT}")
    print("=" * 55)


if __name__ == "__main__":
    print_config()
