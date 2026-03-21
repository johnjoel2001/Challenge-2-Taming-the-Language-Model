"""
train_reward_model_balanced.py — STAGE 3B: Train Balanced Reward Model

Trains a reward model that learns to prefer responses balancing readability,
completeness, appropriate length, and low repetition.

Input : data/preference_balanced.csv
Output: outputs/models/reward_model_balanced/
"""

import os
from src.config import DATA_DIR, MODELS_DIR
from src.train_reward_model_simplicity import train_reward_model


def run():
    csv_path = os.path.join(DATA_DIR, "preference_balanced.csv")
    save_dir = os.path.join(MODELS_DIR, "reward_model_balanced")
    return train_reward_model(csv_path, save_dir, label="balanced")


if __name__ == "__main__":
    run()
