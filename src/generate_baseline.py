"""Stage 1 — generate baseline responses and compute metrics."""

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import (
    BASE_MODEL_NAME, DEVICE, SEED, GENERATION_KWARGS,
    OUTPUT_DIR, NUM_EVAL_PROMPTS,
)
from src.prompts import EVAL_PROMPTS
from src.utils import set_seed, compute_all_metrics, save_csv, save_json


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(DEVICE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return tokenizer, model


def generate_responses(tokenizer, model, prompts, gen_kwargs=None):
    if gen_kwargs is None:
        gen_kwargs = GENERATION_KWARGS
    responses = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        output_ids = model.generate(input_ids, **gen_kwargs)
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()
        responses.append(response)
    return responses


def run_baseline():
    set_seed(SEED)

    tokenizer, model = load_base_model()
    prompts = EVAL_PROMPTS[:NUM_EVAL_PROMPTS]
    responses = generate_responses(tokenizer, model, prompts)

    rows = []
    for prompt, resp in zip(prompts, responses):
        rows.append({"prompt": prompt, "response": resp, **compute_all_metrics(resp, prompt)})
    df = pd.DataFrame(rows)

    metric_cols = [
        "readability", "avg_sentence_length", "avg_word_length",
        "response_length", "lexical_diversity", "completeness", "repetition",
    ]
    summary = {col: float(df[col].mean()) for col in metric_cols}
    summary["model"] = "baseline"

    save_csv(df, os.path.join(OUTPUT_DIR, "baseline_outputs.csv"))
    save_json(summary, os.path.join(OUTPUT_DIR, "baseline_metrics.json"))
    return df, summary


if __name__ == "__main__":
    run_baseline()
