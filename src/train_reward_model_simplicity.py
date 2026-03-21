"""
train_reward_model_simplicity.py — STAGE 3A: Train Simplicity Reward Model

Trains a lightweight reward model that learns to prefer simpler, shorter,
more readable responses.  The model is a DistilGPT-2 backbone with a
single scalar-valued head.

Input : data/preference_simplicity.csv
Output: outputs/models/reward_model_simplicity/
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from src.config import (
    BASE_MODEL_NAME, DEVICE, SEED,
    REWARD_MODEL_EPOCHS, REWARD_MODEL_LR,
    REWARD_MODEL_BATCH_SIZE, REWARD_MODEL_MAX_LEN,
    DATA_DIR, MODELS_DIR, OUTPUT_DIR, print_config,
)
from src.utils import set_seed, save_json


# ═════════════════════════════════════════════
#  REWARD MODEL ARCHITECTURE
# ═════════════════════════════════════════════

class RewardModel(nn.Module):
    """
    Reward model: transformer backbone → mean pooling → linear head → scalar.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size  # 768 for distilgpt2
        self.reward_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state               # (B, T, H)
        # Mean-pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()              # (B, T, 1)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        reward = self.reward_head(pooled).squeeze(-1)            # (B,)
        return reward


# ═════════════════════════════════════════════
#  PREFERENCE DATASET
# ═════════════════════════════════════════════

class PreferencePairDataset(Dataset):
    """Yields (chosen_input_ids, chosen_mask, rejected_input_ids, rejected_mask)."""
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Ensure all text fields are strings (guard against NaN from CSV)
        self.prompts = [str(x) if not isinstance(x, str) else x for x in df["prompt"].tolist()]
        self.chosen = [str(x) if not isinstance(x, str) else x for x in df["chosen_response"].tolist()]
        self.rejected = [str(x) if not isinstance(x, str) else x for x in df["rejected_response"].tolist()]

    def __len__(self):
        return len(self.prompts)

    def _encode(self, prompt, response):
        text = prompt + " " + response
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        c_ids, c_mask = self._encode(self.prompts[idx], self.chosen[idx])
        r_ids, r_mask = self._encode(self.prompts[idx], self.rejected[idx])
        return c_ids, c_mask, r_ids, r_mask


# ═════════════════════════════════════════════
#  TRAINING LOOP
# ═════════════════════════════════════════════

def train_reward_model(csv_path, save_dir, label="simplicity"):
    """Train the reward model on preference pairs and save it."""
    print_config()
    set_seed(SEED)

    # ── Data ─────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"[RM-{label}] Loaded {len(df)} preference pairs from {csv_path}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = PreferencePairDataset(df, tokenizer, REWARD_MODEL_MAX_LEN)
    loader = DataLoader(dataset, batch_size=REWARD_MODEL_BATCH_SIZE, shuffle=True)

    # ── Model ────────────────────────────────────────────
    model = RewardModel(BASE_MODEL_NAME).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=REWARD_MODEL_LR)

    # ── Train ────────────────────────────────────────────
    history = {"epoch": [], "loss": []}
    for epoch in range(1, REWARD_MODEL_EPOCHS + 1):
        model.train()
        epoch_losses = []
        for batch in loader:
            c_ids, c_mask, r_ids, r_mask = [t.to(DEVICE) for t in batch]
            r_chosen = model(c_ids, c_mask)
            r_rejected = model(r_ids, r_mask)
            # Pairwise ranking loss: chosen should score higher
            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-8).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        print(f"  Epoch {epoch}/{REWARD_MODEL_EPOCHS}  loss={avg_loss:.4f}")

    # ── Save ─────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "reward_model.pt"))
    tokenizer.save_pretrained(save_dir)
    save_json(history, os.path.join(save_dir, "training_history.json"))
    print(f"[RM-{label}] Model saved to {save_dir}")
    return model, history


def run():
    csv_path = os.path.join(DATA_DIR, "preference_simplicity.csv")
    save_dir = os.path.join(MODELS_DIR, "reward_model_simplicity")
    return train_reward_model(csv_path, save_dir, label="simplicity")


if __name__ == "__main__":
    run()
