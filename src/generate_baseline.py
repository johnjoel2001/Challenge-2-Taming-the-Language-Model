"""
generate_baseline.py — STAGE 1: Baseline Model Generation

Loads the pretrained model (distilgpt2), generates responses for all
evaluation prompts, computes baseline metrics, and saves everything
for later comparison.

Outputs:
  - outputs/baseline_outputs.csv
  - outputs/baseline_metrics.json
"""

import os
import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import (
    BASE_MODEL_NAME, DEVICE, SEED, GENERATION_KWARGS,
    OUTPUT_DIR, print_config,
)
from src.prompts import EVAL_PROMPTS
from src.utils import set_seed, compute_all_metrics, save_csv, save_json


def load_base_model():
    """Load tokenizer and pretrained model."""
    print(f"[Stage 1] Loading base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(DEVICE)
    # distilgpt2 has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return tokenizer, model


def generate_responses(tokenizer, model, prompts, gen_kwargs=None):
    """Generate one response per prompt."""
    if gen_kwargs is None:
        gen_kwargs = GENERATION_KWARGS
    responses = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        output_ids = model.generate(input_ids, **gen_kwargs)
        # Decode only the new tokens
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Strip the prompt prefix to isolate the response
        response = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()
        responses.append(response)
        print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...")
    return responses


def run_baseline():
    """Full Stage 1 pipeline."""
    print_config()
    set_seed(SEED)

    tokenizer, model = load_base_model()
    responses = generate_responses(tokenizer, model, EVAL_PROMPTS)

    # ── Build results DataFrame ──────────────────────────
    rows = []
    for prompt, resp in zip(EVAL_PROMPTS, responses):
        metrics = compute_all_metrics(resp, prompt)
        rows.append({"prompt": prompt, "response": resp, **metrics})
    df = pd.DataFrame(rows)

    # ── Aggregate metrics ────────────────────────────────
    metric_cols = [
        "readability", "avg_sentence_length", "avg_word_length",
        "response_length", "lexical_diversity", "completeness", "repetition",
    ]
    summary = {col: float(df[col].mean()) for col in metric_cols}
    summary["model"] = "baseline"

    # ── Save ─────────────────────────────────────────────
    save_csv(df, os.path.join(OUTPUT_DIR, "baseline_outputs.csv"))
    save_json(summary, os.path.join(OUTPUT_DIR, "baseline_metrics.json"))

    print("\n[Stage 1] Baseline generation complete.")
    print(f"  Avg readability  : {summary['readability']:.2f}")
    print(f"  Avg completeness : {summary['completeness']:.2f}")
    print(f"  Avg resp. length : {summary['response_length']:.1f} words")
    return df, summary


if __name__ == "__main__":
    run_baseline()
