"""Stage 3B — Train the balanced reward model.

Identical architecture and training loop as Stage 3A; only the preference
data differs (balanced heuristic scores instead of simplicity scores).
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
