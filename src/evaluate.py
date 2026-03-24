"""Stage 5 — generate responses from all three models and compare metrics."""

import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

from src.config import (
    BASE_MODEL_NAME, DEVICE, SEED, EVAL_MAX_NEW_TOKENS,
    GENERATION_KWARGS, MODELS_DIR, OUTPUT_DIR, NUM_EVAL_PROMPTS,
)
from src.prompts import EVAL_PROMPTS as _ALL_EVAL_PROMPTS
EVAL_PROMPTS = _ALL_EVAL_PROMPTS[:NUM_EVAL_PROMPTS]
from src.utils import (
    set_seed, compute_all_metrics, simplicity_reward, balanced_reward,
    pairwise_overlap, save_csv, save_json,
)
from src.train_ppo_simplicity import load_reward_model, score_responses


def _gen_kwargs():
    kw = dict(GENERATION_KWARGS)
    kw["max_new_tokens"] = EVAL_MAX_NEW_TOKENS
    return kw


def load_baseline():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(DEVICE)
    m.config.pad_token_id = tok.eos_token_id
    m.eval()
    return tok, m, "baseline"


def load_ppo_model(model_dir, label):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLMWithValueHead.from_pretrained(model_dir).to(DEVICE)
    m.eval()
    return tok, m, label


def generate_with_model(tokenizer, model, prompts, gen_kwargs):
    """Generate one response per prompt. Handles both standard and value-head models."""
    responses = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            try:
                out = model.generate(ids, **gen_kwargs)
            except AttributeError:
                # Some TRL versions require going through the underlying pretrained model
                out = model.pretrained_model.generate(ids, **gen_kwargs)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        responses.append(text[len(prompt):].strip() if text.startswith(prompt) else text.strip())
    return responses


def run_evaluation():
    set_seed(SEED)
    gk = _gen_kwargs()

    models_info = []

    tok_b, m_b, lbl_b = load_baseline()
    models_info.append((tok_b, m_b, lbl_b))

    ppo_simp_dir = os.path.join(MODELS_DIR, "ppo_simplicity")
    if os.path.isdir(ppo_simp_dir):
        models_info.append(load_ppo_model(ppo_simp_dir, "ppo_simplicity"))

    ppo_bal_dir = os.path.join(MODELS_DIR, "ppo_balanced")
    if os.path.isdir(ppo_bal_dir):
        models_info.append(load_ppo_model(ppo_bal_dir, "ppo_balanced"))

    # Load reward models to produce learned scores alongside heuristic scores
    rm_simp_dir = os.path.join(MODELS_DIR, "reward_model_simplicity")
    rm_bal_dir  = os.path.join(MODELS_DIR, "reward_model_balanced")
    rm_simp, rm_simp_tok = (None, None)
    rm_bal,  rm_bal_tok  = (None, None)
    if os.path.isdir(rm_simp_dir):
        rm_simp, rm_simp_tok = load_reward_model(rm_simp_dir, DEVICE)
    if os.path.isdir(rm_bal_dir):
        rm_bal, rm_bal_tok = load_reward_model(rm_bal_dir, DEVICE)

    all_rows = []
    side_by_side = {p: {"prompt": p} for p in EVAL_PROMPTS}

    for tok, mdl, label in models_info:
        responses = generate_with_model(tok, mdl, EVAL_PROMPTS, gk)

        for prompt, resp in zip(EVAL_PROMPTS, responses):
            metrics = compute_all_metrics(resp, prompt)
            metrics["simplicity_heuristic"] = simplicity_reward(resp, prompt)
            metrics["balanced_heuristic"]   = balanced_reward(resp, prompt)
            if rm_simp is not None:
                metrics["rm_simplicity_score"] = score_responses(
                    rm_simp, rm_simp_tok, [prompt], [resp], DEVICE)[0]
            if rm_bal is not None:
                metrics["rm_balanced_score"] = score_responses(
                    rm_bal, rm_bal_tok, [prompt], [resp], DEVICE)[0]
            all_rows.append({"model": label, "prompt": prompt, "response": resp, **metrics})
            side_by_side[prompt][f"response_{label}"] = resp

    full_df = pd.DataFrame(all_rows)

    save_csv(pd.DataFrame(side_by_side.values()), os.path.join(OUTPUT_DIR, "eval_side_by_side.csv"))

    metric_cols = [
        "readability", "avg_sentence_length", "avg_word_length",
        "response_length", "lexical_diversity", "completeness", "repetition",
        "simplicity_heuristic", "balanced_heuristic",
    ]
    if rm_simp is not None:
        metric_cols.append("rm_simplicity_score")
    if rm_bal is not None:
        metric_cols.append("rm_balanced_score")

    summary_rows = []
    for label, grp in full_df.groupby("model"):
        row = {"model": label}
        for col in metric_cols:
            if col in grp.columns:
                row[col] = round(float(grp[col].mean()), 4)
        row["pairwise_overlap"] = round(pairwise_overlap(grp["response"].tolist()), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, os.path.join(OUTPUT_DIR, "eval_metrics_summary.csv"))
    save_csv(full_df,    os.path.join(OUTPUT_DIR, "eval_full.csv"))
    save_json(summary_rows, os.path.join(OUTPUT_DIR, "eval_metrics_summary.json"))
    return full_df, summary_df


if __name__ == "__main__":
    run_evaluation()
