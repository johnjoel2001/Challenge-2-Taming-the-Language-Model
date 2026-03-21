"""
create_preference_data.py — STAGE 2: Preference Dataset Construction

For each training prompt:
  1. Generate multiple candidate responses from the base model.
  2. Score every candidate with BOTH the simplicity and balanced heuristics.
  3. Build preference pairs (chosen vs rejected) for each reward type.
  4. Save as CSV + Hugging Face Dataset objects.

Outputs:
  - data/preference_simplicity.csv
  - data/preference_balanced.csv
  - data/candidates.csv          (all candidates with scores)
"""

import os
import sys
import pandas as pd
from datasets import Dataset

from src.config import (
    BASE_MODEL_NAME, DEVICE, SEED, GENERATION_KWARGS,
    DATA_DIR, NUM_CANDIDATES_PER_PROMPT, print_config,
)
from src.prompts import TRAIN_PROMPTS
from src.utils import (
    set_seed, simplicity_reward, balanced_reward,
    compute_all_metrics, save_csv,
)
from src.generate_baseline import load_base_model


def generate_candidates(tokenizer, model, prompts, n_candidates):
    """Generate n_candidates responses per prompt with sampling."""
    all_candidates = []
    gen_kwargs = dict(GENERATION_KWARGS)  # copy
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        for j in range(n_candidates):
            output_ids = model.generate(input_ids, **gen_kwargs)
            full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            resp = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
            # Compute heuristic rewards
            s_reward = simplicity_reward(resp, prompt)
            b_reward = balanced_reward(resp, prompt)
            metrics = compute_all_metrics(resp, prompt)
            all_candidates.append({
                "prompt": prompt,
                "candidate_id": j,
                "response": resp,
                "simplicity_reward": s_reward,
                "balanced_reward": b_reward,
                **metrics,
            })
        print(f"  [{i+1}/{len(prompts)}] Generated {n_candidates} candidates for: {prompt[:45]}...")
    return pd.DataFrame(all_candidates)


def build_preference_pairs(candidates_df, reward_col, reward_type_label):
    """
    For each prompt, pick the best and worst candidate under `reward_col`
    to form a preference pair.  Returns a list of dicts.
    """
    pairs = []
    for prompt, grp in candidates_df.groupby("prompt"):
        grp_sorted = grp.sort_values(reward_col, ascending=False)
        if len(grp_sorted) < 2:
            continue
        chosen = grp_sorted.iloc[0]
        rejected = grp_sorted.iloc[-1]
        # Skip if scores are identical (no meaningful preference)
        if chosen[reward_col] == rejected[reward_col]:
            rejected = grp_sorted.iloc[1] if len(grp_sorted) > 2 else rejected
        pairs.append({
            "prompt": prompt,
            "chosen_response": chosen["response"],
            "rejected_response": rejected["response"],
            "chosen_score": chosen[reward_col],
            "rejected_score": rejected[reward_col],
            "reward_type": reward_type_label,
        })
    return pairs


def run_create_preference_data():
    """Full Stage 2 pipeline."""
    print_config()
    set_seed(SEED)

    # ── Load model and generate candidates ───────────────
    tokenizer, model = load_base_model()
    print(f"[Stage 2] Generating {NUM_CANDIDATES_PER_PROMPT} candidates per prompt...")
    cand_df = generate_candidates(
        tokenizer, model, TRAIN_PROMPTS, NUM_CANDIDATES_PER_PROMPT
    )
    save_csv(cand_df, os.path.join(DATA_DIR, "candidates.csv"))

    # ── Build preference pairs ───────────────────────────
    print("[Stage 2] Building simplicity preference pairs...")
    simp_pairs = build_preference_pairs(cand_df, "simplicity_reward", "simplicity")
    simp_df = pd.DataFrame(simp_pairs)
    save_csv(simp_df, os.path.join(DATA_DIR, "preference_simplicity.csv"))

    print("[Stage 2] Building balanced preference pairs...")
    bal_pairs = build_preference_pairs(cand_df, "balanced_reward", "balanced")
    bal_df = pd.DataFrame(bal_pairs)
    save_csv(bal_df, os.path.join(DATA_DIR, "preference_balanced.csv"))

    # ── Save as Hugging Face Datasets ────────────────────
    ds_simp = Dataset.from_pandas(simp_df)
    ds_bal = Dataset.from_pandas(bal_df)
    ds_simp.save_to_disk(os.path.join(DATA_DIR, "hf_preference_simplicity"))
    ds_bal.save_to_disk(os.path.join(DATA_DIR, "hf_preference_balanced"))
    print("[Stage 2] HuggingFace datasets saved.")

    print(f"\n[Stage 2] Done — {len(simp_df)} simplicity pairs, {len(bal_df)} balanced pairs.")
    return simp_df, bal_df


if __name__ == "__main__":
    run_create_preference_data()
