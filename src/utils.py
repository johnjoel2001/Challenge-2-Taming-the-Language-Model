import os
import re
import json
import collections

import numpy as np
import pandas as pd
import torch
import textstat

from src.prompts import get_keywords_for_prompt


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def readability_score(text: str) -> float:
    """Flesch Reading Ease, clamped to [0, 120]."""
    if not text.strip():
        return 0.0
    return max(0.0, min(120.0, textstat.flesch_reading_ease(text)))


def avg_sentence_length(text: str) -> float:
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    return float(np.mean([len(s.split()) for s in sentences]))


def avg_word_length(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return float(np.mean([len(w) for w in words]))


def response_length(text: str) -> int:
    return len(text.split())


def lexical_diversity(text: str) -> float:
    """Type-token ratio."""
    words = [w.lower() for w in re.findall(r'\w+', text)]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def completeness_score(text: str, prompt: str) -> float:
    """Fraction of expected keywords present in the response."""
    keywords = get_keywords_for_prompt(prompt)
    if not keywords:
        return 1.0
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower) / len(keywords)


def repetition_score(text: str, n: int = 3) -> float:
    """Fraction of repeated n-grams — higher means more repetitive."""
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = collections.Counter(ngrams)
    total = len(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / total if total > 0 else 0.0


def pairwise_overlap(texts: list) -> float:
    """Average Jaccard similarity across all pairs of texts."""
    texts = [str(t) if not isinstance(t, str) else t for t in texts]
    if len(texts) < 2:
        return 0.0
    sims = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a = set(texts[i].lower().split())
            b = set(texts[j].lower().split())
            if not a and not b:
                sims.append(1.0)
            elif not a or not b:
                sims.append(0.0)
            else:
                sims.append(len(a & b) / len(a | b))
    return float(np.mean(sims))


def compute_all_metrics(text: str, prompt: str) -> dict:
    return {
        "readability":        readability_score(text),
        "avg_sentence_length": avg_sentence_length(text),
        "avg_word_length":    avg_word_length(text),
        "response_length":    response_length(text),
        "lexical_diversity":  lexical_diversity(text),
        "completeness":       completeness_score(text, prompt),
        "repetition":         repetition_score(text),
    }


def simplicity_reward(text: str, prompt: str) -> float:
    """
    Favours short, easy-to-read responses.
      + readability / 120
      + 1 / (1 + word_count / 40)   shorter is better
      - avg_word_length / 10         penalise long words
    """
    read    = readability_score(text) / 120.0
    brevity = 1.0 / (1.0 + response_length(text) / 40.0)
    wpen    = avg_word_length(text) / 10.0
    return 0.5 * read + 0.35 * brevity - 0.15 * wpen


def balanced_reward(text: str, prompt: str) -> float:
    """
    Balances readability and completeness, penalises repetition and extremes in length.
    Length bonus: -0.3 if < 15 words, +0.2 if 15-120 words, 0.0 otherwise.
    """
    read = readability_score(text) / 120.0
    comp = completeness_score(text, prompt)
    rep  = repetition_score(text)
    wlen = response_length(text)

    if wlen < 15:
        length_bonus = -0.3
    elif wlen <= 120:
        length_bonus = 0.2
    else:
        length_bonus = 0.0

    wpen = avg_word_length(text) / 10.0
    return (0.30 * read + 0.30 * comp + length_bonus - 0.15 * rep - 0.10 * wpen)


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


class _NumpyEncoder(json.JSONEncoder):
    """Makes numpy scalars and arrays JSON-serialisable."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
