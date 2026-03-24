"""Stage 3A — Train the simplicity reward model."""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

from src.config import (
    BASE_MODEL_NAME, DEVICE, SEED,
    REWARD_MODEL_EPOCHS, REWARD_MODEL_LR,
    REWARD_MODEL_BATCH_SIZE, REWARD_MODEL_MAX_LEN,
    REWARD_MODEL_WARMUP_STEPS, REWARD_MODEL_GRAD_ACCUM,
    REWARD_MODEL_WEIGHT_DECAY, REWARD_MODEL_EARLY_STOP_PATIENCE,
    DATA_DIR, MODELS_DIR,
)
from src.utils import set_seed, save_json


class RewardModel(nn.Module):
    """GPT-2 encoder with mean-pool and a 3-layer MLP head that outputs a scalar reward."""

    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, H)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.reward_head(pooled).squeeze(-1)  # (B,)


class PreferencePairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.prompts   = [str(x) for x in df["prompt"].tolist()]
        self.chosen    = [str(x) for x in df["chosen_response"].tolist()]
        self.rejected  = [str(x) for x in df["rejected_response"].tolist()]

    def __len__(self):
        return len(self.prompts)

    def _encode(self, prompt, response):
        enc = self.tokenizer(
            prompt + " " + response,
            truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        c_ids, c_mask = self._encode(self.prompts[idx], self.chosen[idx])
        r_ids, r_mask = self._encode(self.prompts[idx], self.rejected[idx])
        return c_ids, c_mask, r_ids, r_mask


def _linear_schedule(optimizer, num_warmup_steps, num_total_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_total_steps - step) /
                   float(max(1, num_total_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)


def train_reward_model(csv_path, save_dir, label="simplicity"):
    set_seed(SEED)

    df = pd.read_csv(csv_path)
    print(f"[Stage 3] Training {label} reward model on {len(df)} preference pairs")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 80/20 train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
    print(f"[Stage 3] Train: {len(train_df)} pairs  |  Val: {len(val_df)} pairs")

    train_loader = DataLoader(
        PreferencePairDataset(train_df, tokenizer, REWARD_MODEL_MAX_LEN),
        batch_size=REWARD_MODEL_BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        PreferencePairDataset(val_df, tokenizer, REWARD_MODEL_MAX_LEN),
        batch_size=REWARD_MODEL_BATCH_SIZE, shuffle=False,
    )

    model     = RewardModel(BASE_MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=REWARD_MODEL_LR,
                                  weight_decay=REWARD_MODEL_WEIGHT_DECAY)

    total_update_steps = (len(train_loader) * REWARD_MODEL_EPOCHS) // REWARD_MODEL_GRAD_ACCUM
    scheduler = _linear_schedule(optimizer, REWARD_MODEL_WARMUP_STEPS, total_update_steps)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    best_val_loss   = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, REWARD_MODEL_EPOCHS + 1):
        # train
        model.train()
        train_losses = []
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            c_ids, c_mask, r_ids, r_mask = [t.to(DEVICE) for t in batch]
            r_chosen   = model(c_ids, c_mask)
            r_rejected = model(r_ids, r_mask)
            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-8).mean()
            (loss / REWARD_MODEL_GRAD_ACCUM).backward()
            train_losses.append(loss.item())

            if (batch_idx + 1) % REWARD_MODEL_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # flush leftover gradients if batches don't divide evenly
        if len(train_loader) % REWARD_MODEL_GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train = float(np.mean(train_losses))

        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                c_ids, c_mask, r_ids, r_mask = [t.to(DEVICE) for t in batch]
                r_chosen   = model(c_ids, c_mask)
                r_rejected = model(r_ids, r_mask)
                loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-8).mean()
                val_losses.append(loss.item())

        avg_val = float(np.mean(val_losses)) if val_losses else float("inf")
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["lr"].append(current_lr)

        print(f"[Stage 3] {label}  epoch {epoch:>3}/{REWARD_MODEL_EPOCHS}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  lr={current_lr:.2e}")

        # keep best checkpoint
        os.makedirs(save_dir, exist_ok=True)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "reward_model.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= REWARD_MODEL_EARLY_STOP_PATIENCE:
                print(f"[Stage 3] Early stopping: val loss no improvement for "
                      f"{REWARD_MODEL_EARLY_STOP_PATIENCE} epochs")
                break

    # restore best weights before returning
    model.load_state_dict(torch.load(os.path.join(save_dir, "reward_model.pt"),
                                     map_location=DEVICE, weights_only=True))
    tokenizer.save_pretrained(save_dir)
    save_json(history, os.path.join(save_dir, "training_history.json"))
    print(f"[Stage 3] {label} reward model saved to {save_dir}")
    return model, history


def run():
    csv_path = os.path.join(DATA_DIR, "preference_simplicity.csv")
    save_dir = os.path.join(MODELS_DIR, "reward_model_simplicity")
    return train_reward_model(csv_path, save_dir, label="simplicity")


if __name__ == "__main__":
    run()
