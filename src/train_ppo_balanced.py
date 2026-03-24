"""Stage 4B — PPO fine-tuning with the balanced reward model.

Identical training loop as Stage 4A; only the reward model differs.
"""

import os
from src.config import MODELS_DIR
from src.train_ppo_simplicity import run_ppo_training


def run():
    rm_dir   = os.path.join(MODELS_DIR, "reward_model_balanced")
    save_dir = os.path.join(MODELS_DIR, "ppo_balanced")
    return run_ppo_training(rm_dir, save_dir, "ppo_balanced", label="balanced")


if __name__ == "__main__":
    run()
