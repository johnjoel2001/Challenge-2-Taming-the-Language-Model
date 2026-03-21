"""
train_ppo_balanced.py — STAGE 4B: PPO Training with Balanced Reward

Uses TRL's PPOTrainer to fine-tune the base language model using the
balanced reward model as the scoring function.

Outputs:
  - outputs/models/ppo_balanced/           (final model)
  - outputs/ppo_balanced_log.csv           (training log)
  - outputs/ppo_balanced_samples.csv       (sample generations per step)
"""

import os
from src.config import MODELS_DIR
from src.train_ppo_simplicity import run_ppo_training


def run():
    rm_dir = os.path.join(MODELS_DIR, "reward_model_balanced")
    save_dir = os.path.join(MODELS_DIR, "ppo_balanced")
    return run_ppo_training(rm_dir, save_dir, "ppo_balanced", label="balanced")


if __name__ == "__main__":
    run()
