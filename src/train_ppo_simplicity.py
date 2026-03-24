"""Stage 4A — PPO fine-tuning with the simplicity reward model."""

import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from src.config import (
    BASE_MODEL_NAME, DEVICE, SEED,
    PPO_EPOCHS, PPO_STEPS, PPO_BATCH_SIZE, PPO_MINI_BATCH_SIZE,
    PPO_LR, PPO_MAX_NEW_TOKENS, PPO_GRAD_ACCUM_STEPS,
    PPO_GAMMA, PPO_GAE_LAMBDA, PPO_EPS_CLIP, PPO_VALUE_EPS_CLIP,
    PPO_INIT_KL_COEF, PPO_TARGET_KL, PPO_VF_COEF, PPO_MAX_GRAD_NORM,
    MODELS_DIR, OUTPUT_DIR,
    NUM_TRAINING_PROMPTS,
)
from src.prompts import TRAIN_PROMPTS
from src.utils import set_seed, save_csv
from src.train_reward_model_simplicity import RewardModel


def load_reward_model(rm_dir, device):
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
    rewards = []
    for p, r in zip(prompts, responses):
        enc = rm_tokenizer(
            p + " " + r,
            truncation=True, max_length=max_len,
            padding="max_length", return_tensors="pt",
        )
        with torch.no_grad():
            score = rm(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        rewards.append(score.item())
    return rewards


def run_ppo_training(reward_model_dir, save_model_dir, log_prefix, label="simplicity"):
    set_seed(SEED)

    print(f"[Stage 4] Starting {label} PPO: {PPO_STEPS} steps, batch {PPO_BATCH_SIZE}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model     = AutoModelForCausalLMWithValueHead.from_pretrained(BASE_MODEL_NAME)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(BASE_MODEL_NAME)
    ref_model.eval()  # reference must stay frozen throughout

    # reward model stays on CPU so the two policy models can use the full GPU
    rm, rm_tok = load_reward_model(reward_model_dir, "cpu")

    ppo_config = PPOConfig(
        learning_rate=PPO_LR,
        batch_size=PPO_BATCH_SIZE,
        mini_batch_size=PPO_MINI_BATCH_SIZE,
        ppo_epochs=PPO_EPOCHS,
        gamma=PPO_GAMMA,
        lam=PPO_GAE_LAMBDA,
        cliprange=PPO_EPS_CLIP,
        cliprange_value=PPO_VALUE_EPS_CLIP,
        vf_coef=PPO_VF_COEF,
        adap_kl_ctrl=True,
        init_kl_coef=PPO_INIT_KL_COEF,
        target=PPO_TARGET_KL,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        gradient_accumulation_steps=PPO_GRAD_ACCUM_STEPS,
        log_with=None,
        seed=SEED,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    ppo_device = next(ppo_trainer.model.parameters()).device

    train_prompts  = TRAIN_PROMPTS[:NUM_TRAINING_PROMPTS]
    prompts_cycle  = train_prompts * ((PPO_STEPS * PPO_BATCH_SIZE // len(train_prompts)) + 1)

    log_rows, sample_rows = [], []

    for step in range(PPO_STEPS):
        batch_prompts = prompts_cycle[step * PPO_BATCH_SIZE : (step + 1) * PPO_BATCH_SIZE]
        if not batch_prompts:
            batch_prompts = train_prompts[:PPO_BATCH_SIZE]

        query_tensors = [
            tokenizer.encode(p, return_tensors="pt").squeeze(0).to(ppo_device)
            for p in batch_prompts
        ]

        response_tensors = []
        for qt in query_tensors:
            gen = ppo_trainer.generate(
                qt,
                max_new_tokens=PPO_MAX_NEW_TOKENS,
                min_length=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.85,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if gen.dim() == 2:
                gen = gen.squeeze(0)
            resp_t = gen[qt.shape[0]:]
            if resp_t.numel() == 0:
                resp_t = torch.tensor([tokenizer.eos_token_id], device=ppo_device)
            response_tensors.append(resp_t)

        response_texts = [tokenizer.decode(rt, skip_special_tokens=True)
                          for rt in response_tensors]

        reward_values  = score_responses(rm, rm_tok, batch_prompts, response_texts, "cpu")
        reward_tensors = [torch.tensor(r, dtype=torch.float32).to(ppo_device)
                          for r in reward_values]

        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

        mean_reward = float(np.mean(reward_values))
        mean_len    = float(np.mean([len(r.split()) for r in response_texts]))
        kl = 0.0
        for kl_key in ["objective/kl", "objective/kl_dist", "ppo/mean_non_score_reward",
                        "ppo/kl", "env/kl", "kl"]:
            if kl_key in stats:
                kl = stats[kl_key]
                break

        log_rows.append({
            "step": step,
            "mean_reward": mean_reward,
            "mean_response_length": mean_len,
            "kl": float(kl) if isinstance(kl, (int, float)) else 0.0,
        })

        # log every 10 steps
        if step % 10 == 0 or step == PPO_STEPS - 1:
            sample_rows.append({
                "step":     step,
                "prompt":   batch_prompts[0],
                "response": response_texts[0],
                "reward":   reward_values[0],
            })
            print(f"[Stage 4] {label}  step {step:>3}/{PPO_STEPS}  "
                  f"reward={mean_reward:.4f}  len={mean_len:.1f}  kl={float(kl):.4f}")

    os.makedirs(save_model_dir, exist_ok=True)
    model.save_pretrained(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)

    save_csv(pd.DataFrame(log_rows),    os.path.join(OUTPUT_DIR, f"{log_prefix}_log.csv"))
    save_csv(pd.DataFrame(sample_rows), os.path.join(OUTPUT_DIR, f"{log_prefix}_samples.csv"))
    print(f"[Stage 4] {label} PPO model saved to {save_model_dir}")
    return log_rows


def run():
    rm_dir   = os.path.join(MODELS_DIR, "reward_model_simplicity")
    save_dir = os.path.join(MODELS_DIR, "ppo_simplicity")
    return run_ppo_training(rm_dir, save_dir, "ppo_simplicity", label="simplicity")


if __name__ == "__main__":
    run()
