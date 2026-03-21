"""
prompts.py — Prompt definitions and keyword mappings for completeness scoring.

Contains:
  - TRAIN_PROMPTS : used during preference-data creation and PPO training
  - EVAL_PROMPTS  : held-out prompts used only for evaluation
  - KEYWORD_MAP   : expected concept keywords for each prompt (completeness)
"""

# ─────────────────────────────────────────────
# TRAINING PROMPTS  (used in Stages 2-4)
# ─────────────────────────────────────────────
TRAIN_PROMPTS = [
    "Explain machine learning in simple words.",
    "What is a neural network for a beginner?",
    "Explain reinforcement learning simply.",
    "What is blockchain in plain English?",
    "Explain cloud computing to a school student.",
    "What is overfitting in machine learning?",
    "Explain an API in simple terms.",
    "What is a database?",
    "Explain gradient descent simply.",
    "What is a transformer model?",
    "What is supervised learning?",
    "Explain backpropagation in easy language.",
]

# ─────────────────────────────────────────────
# EVALUATION / HELD-OUT PROMPTS  (used in Stages 5-7)
# ─────────────────────────────────────────────
EVAL_PROMPTS = [
    "Explain machine learning in simple words.",
    "What is a neural network for a beginner?",
    "Explain reinforcement learning simply.",
    "What is blockchain in plain English?",
    "Explain cloud computing to a school student.",
    "What is overfitting in machine learning?",
    "Explain an API in simple terms.",
    "What is a database?",
    "Explain gradient descent simply.",
    "What is a transformer model?",
    "What is supervised learning?",
    "Explain backpropagation in easy language.",
    "What is natural language processing?",
    "Explain recursion to a beginner.",
    "What is an algorithm in simple words?",
    "Explain deep learning simply.",
]

# ─────────────────────────────────────────────
# KEYWORD MAP  — expected concepts per prompt
# Used to compute a completeness score.
# Keys are matched via substring against the prompt.
# ─────────────────────────────────────────────
KEYWORD_MAP = {
    "machine learning": [
        "data", "learn", "pattern", "predict", "algorithm", "train", "model",
    ],
    "neural network": [
        "layer", "neuron", "data", "learn", "pattern", "weight", "input", "output",
    ],
    "reinforcement learning": [
        "agent", "reward", "action", "environment", "learn", "policy", "trial",
    ],
    "blockchain": [
        "block", "chain", "transaction", "secure", "record", "decentralize", "ledger",
    ],
    "cloud computing": [
        "internet", "server", "storage", "access", "remote", "service", "data",
    ],
    "overfitting": [
        "train", "data", "generalize", "memorize", "error", "test", "model", "noise",
    ],
    "api": [
        "interface", "request", "data", "service", "communicate", "application", "send",
    ],
    "database": [
        "data", "store", "table", "query", "record", "organize", "retrieve",
    ],
    "gradient descent": [
        "loss", "minimum", "step", "update", "slope", "optimize", "parameter",
    ],
    "transformer": [
        "attention", "sequence", "token", "language", "model", "context", "parallel",
    ],
    "supervised learning": [
        "label", "data", "train", "predict", "input", "output", "example",
    ],
    "backpropagation": [
        "error", "gradient", "weight", "update", "layer", "backward", "learn",
    ],
    "natural language processing": [
        "text", "language", "understand", "word", "sentence", "computer", "meaning",
    ],
    "recursion": [
        "function", "call", "itself", "base", "case", "repeat", "problem",
    ],
    "algorithm": [
        "step", "instruction", "problem", "solve", "input", "output", "process",
    ],
    "deep learning": [
        "layer", "neural", "data", "learn", "pattern", "model", "network",
    ],
}


def get_keywords_for_prompt(prompt: str) -> list[str]:
    """Return the keyword list that best matches the given prompt (case-insensitive)."""
    prompt_lower = prompt.lower()
    for key, keywords in KEYWORD_MAP.items():
        if key in prompt_lower:
            return keywords
    # Fallback: return generic keywords
    return ["data", "learn", "model", "example"]
