"""
utils.py — Shared utility functions for metrics, scoring, and I/O.

Provides:
  - Readability, completeness, repetition, lexical-diversity scoring
  - Simple heuristic reward functions (simplicity & balanced)
  - CSV / JSON save helpers
  - Seed-setting helper
"""

import os
import re
import json
import math
import random
import collections

import numpy as np
import pandas as pd
import torch
import textstat

from src.prompts import get_keywords_for_prompt


# ─────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    """Set random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════
#  TEXT METRICS
# ═════════════════════════════════════════════

def readability_score(text: str) -> float:
    """Flesch Reading Ease (higher = easier). Clamp to [0, 120]."""
    if not text.strip():
        return 0.0
    score = textstat.flesch_reading_ease(text)
    return max(0.0, min(120.0, score))


def avg_sentence_length(text: str) -> float:
    """Average number of words per sentence."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    return np.mean([len(s.split()) for s in sentences])


def avg_word_length(text: str) -> float:
    """Average character-length of words."""
    words = text.split()
    if not words:
        return 0.0
    return np.mean([len(w) for w in words])


def response_length(text: str) -> int:
    """Word count."""
    return len(text.split())


def lexical_diversity(text: str) -> float:
    """Type-token ratio (unique words / total words). Range [0, 1]."""
    words = [w.lower() for w in re.findall(r'\w+', text)]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def completeness_score(text: str, prompt: str) -> float:
    """Fraction of expected keywords found in the response. Range [0, 1]."""
    keywords = get_keywords_for_prompt(prompt)
    if not keywords:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return hits / len(keywords)


def repetition_score(text: str, n: int = 3) -> float:
    """
    Fraction of repeated n-grams. Higher = more repetitive.
    Range [0, 1].
    """
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = collections.Counter(ngrams)
    total = len(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / total if total > 0 else 0.0


def pairwise_overlap(texts: list) -> float:
    """Average Jaccard similarity between all pairs of texts."""
    # Coerce non-string entries (e.g. NaN) to empty string
    texts = [str(t) if not isinstance(t, str) else t for t in texts]
    if len(texts) < 2:
        return 0.0
    sims = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            set_a = set(texts[i].lower().split())
            set_b = set(texts[j].lower().split())
            if not set_a and not set_b:
                sims.append(1.0)
            elif not set_a or not set_b:
                sims.append(0.0)
            else:
                sims.append(len(set_a & set_b) / len(set_a | set_b))
    return float(np.mean(sims))


# ═════════════════════════════════════════════
#  COMPOSITE METRICS DICT  (one response)
# ═════════════════════════════════════════════

def compute_all_metrics(text: str, prompt: str) -> dict:
    """Return a dict of all scalar metrics for a single (prompt, response) pair."""
    return {
        "readability": readability_score(text),
        "avg_sentence_length": avg_sentence_length(text),
        "avg_word_length": avg_word_length(text),
        "response_length": response_length(text),
        "lexical_diversity": lexical_diversity(text),
        "completeness": completeness_score(text, prompt),
        "repetition": repetition_score(text),
    }


# ═════════════════════════════════════════════
#  HEURISTIC REWARD FUNCTIONS
#  Used to create the preference dataset and
#  as reference scorers in evaluation.
# ═════════════════════════════════════════════

def simplicity_reward(text: str, prompt: str) -> float:
    """
    Heuristic reward that favours short, easy-to-read responses.
    Components (all normalised roughly to [0, 1]):
      +  readability / 120
      +  1 / (1 + word_count / 40)   (shorter is better)
      -  avg_word_length / 10         (penalise long words)
    """
    read = readability_score(text) / 120.0
    brevity = 1.0 / (1.0 + response_length(text) / 40.0)
    word_pen = avg_word_length(text) / 10.0
    return 0.5 * read + 0.35 * brevity - 0.15 * word_pen


def balanced_reward(text: str, prompt: str) -> float:
    """
    Heuristic reward that balances readability, completeness, and penalises
    repetition and extreme shortness.
    Components:
      +  readability / 120
      +  completeness
      -  repetition
      -  brevity penalty (too short is bad)
      -  avg_word_length penalty (keep it accessible)
    """
    read = readability_score(text) / 120.0
    comp = completeness_score(text, prompt)
    rep = repetition_score(text)
    wlen = response_length(text)
    # Penalise very short responses (< 15 words) and very long (> 120)
    length_bonus = 0.0
    if wlen < 15:
        length_bonus = -0.3
    elif 15 <= wlen <= 120:
        length_bonus = 0.2
    else:
        length_bonus = 0.0
    word_pen = avg_word_length(text) / 10.0
    return (0.30 * read
            + 0.30 * comp
            + length_bonus
            - 0.15 * rep
            - 0.10 * word_pen)


# ═════════════════════════════════════════════
#  I/O HELPERS
# ═════════════════════════════════════════════

def save_csv(df: pd.DataFrame, path: str):
    """Save a DataFrame to CSV, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [saved] {path}  ({len(df)} rows)")


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types when serialising to JSON."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def save_json(obj, path: str):
    """Save a JSON-serialisable object to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)
    print(f"  [saved] {path}")


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
