# Training prompts — domain-diverse, concept-grounded
# 70 prompts covering ML fundamentals, NLP, RL, systems, algorithms, and maths
TRAIN_PROMPTS = [
    # Machine Learning Fundamentals
    "Explain machine learning in simple words.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain overfitting in machine learning with an example.",
    "What is regularization and why is it important in ML?",
    "Explain cross-validation in simple terms.",
    "What is a confusion matrix and why do we use it?",
    "Explain gradient descent and how it finds optimal weights.",
    "What is feature engineering and why does it matter in ML?",
    "Explain the bias-variance tradeoff.",
    "What is ensemble learning and when should you use it?",

    # Neural Networks and Deep Learning
    "What is a neural network for a beginner?",
    "Explain backpropagation in easy language.",
    "What is an activation function and why do we need them?",
    "Explain convolutional neural networks for image processing.",
    "What is a recurrent neural network and when do we use it?",
    "Explain batch normalization in neural networks.",
    "What is dropout and how does it prevent overfitting?",
    "Explain the difference between RNN and LSTM.",
    "What is a transformer model and how does attention work?",
    "Explain word embeddings and why they are useful.",

    # Reinforcement Learning
    "Explain reinforcement learning simply.",
    "What is a Markov Decision Process?",
    "Explain the explore-exploit tradeoff in RL.",
    "What is Q-learning and how does it work?",
    "Explain policy gradient methods in reinforcement learning.",
    "What is the difference between on-policy and off-policy learning?",
    "Explain reward shaping in reinforcement learning.",
    "What is a value function in reinforcement learning?",

    # Natural Language Processing
    "What is natural language processing and what are its applications?",
    "Explain tokenization in NLP.",
    "What is a language model and how does it predict text?",
    "Explain the bag-of-words model in NLP.",
    "What is sentiment analysis and how do you approach it?",
    "Explain named entity recognition and its applications.",
    "What is machine translation and what are the main challenges?",
    "Explain text classification and common approaches.",
    "What is semantic similarity and how do you measure it?",
    "Explain sequence-to-sequence models for NLP tasks.",

    # Algorithms and Data Structures
    "Explain recursion to a beginner.",
    "What is an algorithm in simple words?",
    "Explain the difference between arrays and linked lists.",
    "What is a hash table and why is it useful?",
    "Explain binary search and when you should use it.",
    "What is sorting and what are common sorting algorithms?",
    "Explain dynamic programming with a simple example.",
    "What is graph traversal and what are its applications?",
    "Explain Big O notation and why it matters.",
    "What is a tree data structure and what are its types?",

    # Software Engineering and Systems
    "What is blockchain and how does it work?",
    "Explain cloud computing to a school student.",
    "What is an API and what is it used for?",
    "Explain microservices architecture advantages.",
    "What is version control and why is it important?",
    "Explain database normalization.",
    "What is a distributed system and what are its challenges?",
    "Explain caching and when you should use it.",

    # Mathematics and Fundamentals
    "Explain linear algebra and its importance in ML.",
    "What is probability and why does it matter in data science?",
    "Explain Bayes theorem with a simple example.",
    "What is calculus and how is it used in machine learning?",
    "Explain matrices and matrix multiplication.",
    "What is eigenvalue decomposition and when is it useful?",
    "Explain principal component analysis simply.",
    "What is statistical significance and hypothesis testing?",

    # Applications
    "How do you build a recommendation system?",
    "Explain time series forecasting and its challenges.",
    "What is anomaly detection and how do you approach it?",
    "Explain computer vision and its main applications.",
    "What is knowledge distillation in deep learning?",
    "Explain transfer learning and fine-tuning.",
    "What is a database?",
    "What is supervised learning?",
    "What is deep learning?",
    "What is overfitting in machine learning?",
]

# Held-out evaluation prompts — completely separate distribution
EVAL_PROMPTS = [
    "Explain how neural networks learn from data.",
    "What is the curse of dimensionality?",
    "Explain the perceptron algorithm.",
    "What is a kernel method in machine learning?",
    "Explain decision trees and how they make splits.",
    "What is support vector machines and how do they work?",
    "Explain zero-shot and few-shot learning.",
    "What is self-attention and how does it work?",
    "Explain hyperparameter tuning and common strategies.",
    "What is data augmentation and why is it useful?",
    "Explain clustering and common algorithms.",
    "What is dimensionality reduction and when to use it?",
    "Explain the concept of generalization in machine learning.",
    "What is meta-learning and how does it work?",
    "Explain causal inference versus correlation.",
]

# Expected concept keywords per prompt — used for completeness scoring.
# Key is matched via substring against the prompt (case-insensitive).
KEYWORD_MAP = {
    "machine learning": [
        "data", "learn", "pattern", "predict", "algorithm", "train", "model",
        "accuracy", "performance", "task", "example",
    ],
    "supervised": [
        "label", "ground truth", "feedback", "train", "supervised", "input", "output",
    ],
    "unsupervised": [
        "cluster", "pattern", "unlabeled", "discover", "structure", "unsupervised",
    ],
    "overfitting": [
        "train", "test", "generalize", "memorize", "overfit", "performance", "data",
    ],
    "regularization": [
        "penalty", "weight", "constraint", "error", "prevent", "overfit", "smooth",
    ],
    "cross-validation": [
        "fold", "split", "validation", "train", "test", "data", "evaluate",
    ],
    "confusion matrix": [
        "true positive", "false positive", "true negative", "false negative", "classification",
    ],
    "gradient descent": [
        "optimization", "gradient", "step", "weight", "learn", "minimize", "loss",
    ],
    "feature engineering": [
        "feature", "input", "represent", "extract", "create", "transform", "data",
    ],
    "bias-variance": [
        "error", "underfit", "overfit", "complex", "simple", "tradeoff", "bias", "variance",
    ],
    "bias variance": [
        "error", "underfit", "overfit", "complex", "simple", "tradeoff", "bias", "variance",
    ],
    "ensemble": [
        "combine", "multiple", "model", "vote", "average", "improve", "prediction",
    ],
    "neural network": [
        "layer", "neuron", "node", "weight", "activation", "compute", "deep",
    ],
    "backpropagation": [
        "gradient", "backward", "chain rule", "error", "weight", "update", "derivative",
    ],
    "activation": [
        "function", "relu", "sigmoid", "tanh", "nonlinear", "output", "layer",
    ],
    "convolutional": [
        "filter", "kernel", "conv", "image", "spatial", "feature", "map",
    ],
    "recurrent": [
        "rnn", "lstm", "sequence", "time", "memory", "hidden", "state",
    ],
    "transformer": [
        "attention", "self-attention", "query", "key", "value", "token", "sequence",
    ],
    "reinforcement learning": [
        "agent", "reward", "action", "environment", "policy", "state", "episode",
    ],
    "markov": [
        "state", "transition", "probability", "memoryless", "process", "decision",
    ],
    "explore-exploit": [
        "exploration", "exploitation", "tradeoff", "balance", "strategy", "reward",
    ],
    "explore exploit": [
        "exploration", "exploitation", "tradeoff", "balance", "strategy", "reward",
    ],
    "q-learning": [
        "q-table", "bellman", "value", "action", "state", "reward", "learn",
    ],
    "policy gradient": [
        "policy", "gradient", "actor", "critic", "reward", "parameterize", "network",
    ],
    "natural language": [
        "language", "text", "word", "sentence", "meaning", "process", "nlp",
    ],
    "tokenization": [
        "token", "word", "split", "text", "sequence", "vocabulary", "subword",
    ],
    "language model": [
        "predict", "probability", "text", "token", "sequence", "likelihood", "next",
    ],
    "sentiment": [
        "positive", "negative", "sentiment", "opinion", "emotion", "classify", "text",
    ],
    "blockchain": [
        "block", "chain", "transaction", "hash", "ledger", "secure", "distributed",
    ],
    "cloud computing": [
        "server", "online", "internet", "service", "compute", "storage", "data",
    ],
    "api": [
        "interface", "request", "response", "endpoint", "communication", "software",
    ],
    "database": [
        "data", "store", "query", "table", "relation", "schema", "organize",
    ],
    "recursion": [
        "function", "call", "base case", "self", "recursive", "stack", "termination",
    ],
    "algorithm": [
        "step", "procedure", "problem", "efficient", "solution", "solve", "compute",
    ],
    "deep learning": [
        "layer", "neural", "data", "learn", "pattern", "model", "network",
    ],
    "principal component": [
        "variance", "reduce", "dimension", "eigenvector", "projection", "compress",
    ],
    "transfer learning": [
        "pretrain", "finetune", "domain", "knowledge", "adapt", "model", "task",
    ],
    "knowledge distillation": [
        "teacher", "student", "compress", "smaller", "model", "transfer", "soft",
    ],
}


def get_keywords_for_prompt(prompt: str) -> list:
    """Return keyword list for completeness scoring."""
    prompt_lower = prompt.lower()
    for key, words in KEYWORD_MAP.items():
        if key in prompt_lower:
            return words
    return ["data", "learn", "model", "example"]
