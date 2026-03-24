"""Stage 2 — Generate candidates and build preference datasets for both reward types."""

import os
import pandas as pd
import numpy as np
from datasets import Dataset

from src.config import (
    BASE_MODEL_NAME, DEVICE, SEED,
    NUM_TRAINING_PROMPTS, NUM_CANDIDATES_PER_PROMPT,
    CANDIDATE_GENERATION_VARIANTS, DATA_DIR,
    MIN_PREFERENCE_SCORE_GAP,
)
from src.prompts import TRAIN_PROMPTS
from src.utils import (
    set_seed, simplicity_reward, balanced_reward,
    compute_all_metrics, save_csv, save_json,
)
from src.generate_baseline import load_base_model


def generate_diverse_candidates(tokenizer, model, prompts, n_candidates):
    """Generate n_candidates per prompt, cycling through sampling variants for variety."""
    all_candidates = []
    for prompt_idx, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        for cand_idx in range(n_candidates):
            variant = CANDIDATE_GENERATION_VARIANTS[cand_idx % len(CANDIDATE_GENERATION_VARIANTS)]
            gen_kwargs = {
                "max_new_tokens": 180,
                "do_sample": True,
                "top_k": 50,
                "repetition_penalty": 1.2,
                **variant,
            }
            output_ids = model.generate(input_ids, **gen_kwargs)
            full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            resp = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
            all_candidates.append({
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "candidate_idx": cand_idx,
                "temperature": variant["temperature"],
                "top_p": variant["top_p"],
                "response": resp,
                "simplicity_reward": simplicity_reward(resp, prompt),
                "balanced_reward":   balanced_reward(resp, prompt),
                **compute_all_metrics(resp, prompt),
            })
    return pd.DataFrame(all_candidates)


def build_preference_pairs(candidates_df, reward_col, reward_type_label,
                           min_score_gap=MIN_PREFERENCE_SCORE_GAP):
    """Rank candidates per prompt and pair any two with a score gap above the threshold."""
    pairs = []
    for prompt, grp in candidates_df.groupby("prompt"):
        grp_sorted = grp.sort_values(reward_col, ascending=False).reset_index(drop=True)
        if len(grp_sorted) < 2:
            continue
        # All pairs with a sufficient gap, not just top vs bottom.
        for i in range(len(grp_sorted)):
            for j in range(i + 1, len(grp_sorted)):
                gap = float(grp_sorted.loc[i, reward_col]) - float(grp_sorted.loc[j, reward_col])
                if gap >= min_score_gap:
                    pairs.append({
                        "prompt":            prompt,
                        "chosen_response":   grp_sorted.loc[i, "response"],
                        "rejected_response": grp_sorted.loc[j, "response"],
                        "chosen_score":      float(grp_sorted.loc[i, reward_col]),
                        "rejected_score":    float(grp_sorted.loc[j, reward_col]),
                        "score_gap":         gap,
                        "reward_type":       reward_type_label,
                    })
    return pairs


def run_create_preference_data():
    set_seed(SEED)

    tokenizer, model = load_base_model()
    model.eval()

    prompts = TRAIN_PROMPTS[:NUM_TRAINING_PROMPTS]
    print(f"[Stage 2] Generating {len(prompts)} prompts x {NUM_CANDIDATES_PER_PROMPT} candidates "
          f"= {len(prompts) * NUM_CANDIDATES_PER_PROMPT} total")

    cand_df = generate_diverse_candidates(tokenizer, model, prompts, NUM_CANDIDATES_PER_PROMPT)
    save_csv(cand_df, os.path.join(DATA_DIR, "candidates.csv"))
    print(f"[Stage 2] Candidates generated: {len(cand_df)}")

    # Score distribution check
    for col in ("simplicity_reward", "balanced_reward"):
        vals = cand_df[col]
        print(f"[Stage 2] {col}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")


    simp_pairs = build_preference_pairs(cand_df, "simplicity_reward", "simplicity")
    simp_df    = pd.DataFrame(simp_pairs)
    save_csv(simp_df, os.path.join(DATA_DIR, "preference_simplicity.csv"))
    print(f"[Stage 2] Simplicity pairs: {len(simp_df)}")


    bal_pairs = build_preference_pairs(cand_df, "balanced_reward", "balanced")
    bal_df    = pd.DataFrame(bal_pairs)
    save_csv(bal_df, os.path.join(DATA_DIR, "preference_balanced.csv"))
    print(f"[Stage 2] Balanced pairs:   {len(bal_df)}")

    # HuggingFace arrow format for downstream loaders
    Dataset.from_pandas(simp_df).save_to_disk(os.path.join(DATA_DIR, "hf_preference_simplicity"))
    Dataset.from_pandas(bal_df).save_to_disk(os.path.join(DATA_DIR, "hf_preference_balanced"))

    # Summary
    save_json({
        "total_candidates":  len(cand_df),
        "simplicity_pairs":  len(simp_df),
        "balanced_pairs":    len(bal_df),
        "unique_prompts":    int(cand_df["prompt"].nunique()),
        "candidates_per_prompt": NUM_CANDIDATES_PER_PROMPT,
    }, os.path.join(DATA_DIR, "preference_data_summary.json"))

    print("[Stage 2] Done.")
    return simp_df, bal_df


if __name__ == "__main__":
    run_create_preference_data()
