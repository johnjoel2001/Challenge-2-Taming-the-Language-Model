"""
train_ppo_simplicity.py — STAGE 4A: PPO Training with Simplicity Reward

Uses TRL's PPOTrainer to fine-tune the base language model using the
simplicity reward model as the scoring function.

Outputs:
  - outputs/models/ppo_simplicity/           (final model)
  - outputs/ppo_simplicity_log.csv           (training log)
  - outputs/ppo_simplicity_samples.csv       (sample generations per step)
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from src.config import (
    BASE_MODEL_NAME, DEVICE, SEED,
    PPO_EPOCHS, PPO_STEPS, PPO_BATCH_SIZE, PPO_MINI_BATCH_SIZE,
    PPO_LR, PPO_MAX_NEW_TOKENS,
    MODELS_DIR, OUTPUT_DIR, print_config,
)
from src.prompts import TRAIN_PROMPTS
from src.utils import set_seed, save_csv, save_json
from src.train_reward_model_simplicity import RewardModel


# ═════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════

def load_reward_model(rm_dir, device):
    """Load a trained RewardModel from disk."""
    rm = RewardModel(BASE_MODEL_NAME).to(device)
    rm.load_state_dict(torch.load(
        os.path.join(rm_dir, "reward_model.pt"),
        map_location=device, weights_only=True,
    ))
    rm.eval()
    tokenizer = AutoTokenizer.from_pretrained(rm_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return rm, tokenizer


def score_responses(rm, rm_tokenizer, prompts, responses, device, max_len=256):
    """Score a batch of (prompt, response) pairs with the reward model."""
    rewards = []
    for p, r in zip(prompts, responses):
        text = p + " " + r
        enc = rm_tokenizer(
            text, truncation=True, max_length=max_len,
            padding="max_length", return_tensors="pt",
        )
        with torch.no_grad():
            score = rm(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            )
        rewards.append(score.item())
    return rewards


# ═════════════════════════════════════════════
#  PPO TRAINING FUNCTION  (shared by both scripts)
# ═════════════════════════════════════════════

def run_ppo_training(reward_model_dir, save_model_dir, log_prefix, label="simplicity"):
    """
    Generic PPO training loop.
    - reward_model_dir: path to the trained reward model folder
    - save_model_dir:   where to save the final PPO model
    - log_prefix:       prefix for log/sample CSVs
    - label:            human-readable label
    """
    print_config()
    set_seed(SEED)

    # ── Tokenizer & LM ──────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Do NOT manually .to(DEVICE) — let TRL/accelerate handle placement
    model = AutoModelForCausalLMWithValueHead.from_pretrained(BASE_MODEL_NAME)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(BASE_MODEL_NAME)

    # ── Reward model (kept on CPU for simplicity) ────────
    rm, rm_tok = load_reward_model(reward_model_dir, "cpu")

    # ── PPO config ───────────────────────────────────────
    ppo_config = PPOConfig(
        learning_rate=PPO_LR,
        batch_size=PPO_BATCH_SIZE,
        mini_batch_size=PPO_MINI_BATCH_SIZE,
        ppo_epochs=PPO_EPOCHS,
        log_with=None,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    # Derive actual device from the model after PPOTrainer wraps it
    ppo_device = next(ppo_trainer.model.parameters()).device

    # ── Training loop ────────────────────────────────────
    log_rows = []
    sample_rows = []
    prompts_cycle = TRAIN_PROMPTS * ((PPO_STEPS * PPO_BATCH_SIZE // len(TRAIN_PROMPTS)) + 1)

    print(f"\n[PPO-{label}] Starting PPO training for {PPO_STEPS} steps ...\n")

    for step in range(PPO_STEPS):
        # Sample a mini-batch of prompts
        batch_prompts = prompts_cycle[
            step * PPO_BATCH_SIZE : (step + 1) * PPO_BATCH_SIZE
        ]
        if len(batch_prompts) == 0:
            batch_prompts = TRAIN_PROMPTS[:PPO_BATCH_SIZE]

        # Tokenize prompts — place on same device as the PPO model
        query_tensors = [
            tokenizer.encode(p, return_tensors="pt").squeeze(0).to(ppo_device)
            for p in batch_prompts
        ]

        # Generate responses
        response_tensors = []
        for qt in query_tensors:
            try:
                gen = ppo_trainer.generate(
                    qt, max_new_tokens=PPO_MAX_NEW_TOKENS,
                    do_sample=True, top_k=50, top_p=0.95,
                )
            except TypeError:
                # Fallback for TRL versions with different generate() signature
                gen = ppo_trainer.generate(qt, max_new_tokens=PPO_MAX_NEW_TOKENS)
            # gen may include the query; keep only new tokens
            if gen.dim() == 2:
                gen = gen.squeeze(0)
            resp_t = gen[qt.shape[0]:]
            # Guard against empty response
            if resp_t.numel() == 0:
                resp_t = torch.tensor([tokenizer.eos_token_id], device=ppo_device)
            response_tensors.append(resp_t)

        # Decode
        response_texts = [
            tokenizer.decode(rt, skip_special_tokens=True) for rt in response_tensors
        ]

        # Score with reward model (on CPU)
        reward_values = score_responses(
            rm, rm_tok, batch_prompts, response_texts, "cpu"
        )
        # Rewards must be on same device as model for ppo_trainer.step()
        reward_tensors = [torch.tensor(r, dtype=torch.float32).to(ppo_device) for r in reward_values]

        # PPO update
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

        # Logging
        mean_reward = np.mean(reward_values)
        mean_len = np.mean([len(r.split()) for r in response_texts])
        # Extract KL divergence — key name varies across TRL versions
        kl = 0.0
        for kl_key in ["objective/kl", "ppo/mean_non_score_reward", "ppo/kl"]:
            if kl_key in stats:
                kl = stats[kl_key]
                break

        log_rows.append({
            "step": step,
            "mean_reward": mean_reward,
            "mean_response_length": mean_len,
            "kl": float(kl) if isinstance(kl, (int, float)) else 0.0,
        })

        # Save a sample from each step
        sample_rows.append({
            "step": step,
            "prompt": batch_prompts[0],
            "response": response_texts[0],
            "reward": reward_values[0],
        })

        print(f"  Step {step+1}/{PPO_STEPS}  "
              f"reward={mean_reward:.4f}  len={mean_len:.0f}  kl={kl}")

    # ── Save model & logs ────────────────────────────────
    os.makedirs(save_model_dir, exist_ok=True)
    model.save_pretrained(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)

    save_csv(pd.DataFrame(log_rows), os.path.join(OUTPUT_DIR, f"{log_prefix}_log.csv"))
    save_csv(pd.DataFrame(sample_rows), os.path.join(OUTPUT_DIR, f"{log_prefix}_samples.csv"))
    print(f"\n[PPO-{label}] Training complete. Model saved to {save_model_dir}")
    return log_rows


def run():
    rm_dir = os.path.join(MODELS_DIR, "reward_model_simplicity")
    save_dir = os.path.join(MODELS_DIR, "ppo_simplicity")
    return run_ppo_training(rm_dir, save_dir, "ppo_simplicity", label="simplicity")


if __name__ == "__main__":
    run()
