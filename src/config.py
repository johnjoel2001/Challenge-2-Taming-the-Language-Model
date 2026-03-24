"""Central config. RLHF_MODE=extended (default) or quick."""

import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

for d in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

MODE = os.environ.get("RLHF_MODE", "extended")

BASE_MODEL_NAME = "gpt2"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SEED = 42

# --- Candidate generation ---
NUM_TRAINING_PROMPTS      = 50 if MODE == "extended" else 12
NUM_EVAL_PROMPTS          = 15 if MODE == "extended" else 4
NUM_CANDIDATES_PER_PROMPT =  8 if MODE == "extended" else 3

# Four sampling configs cycled across candidates to get varied outputs from the same model.
CANDIDATE_GENERATION_VARIANTS = [
    {"temperature": 0.70, "top_p": 0.90},  # more deterministic
    {"temperature": 0.85, "top_p": 0.95},  # default
    {"temperature": 1.00, "top_p": 0.95},  # exploratory
    {"temperature": 1.20, "top_p": 0.98},  # very exploratory
]

GENERATION_KWARGS = dict(
    max_new_tokens=180 if MODE == "extended" else 120,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.85,
    repetition_penalty=1.2,
)

# --- Reward model training ---
REWARD_MODEL_EPOCHS     = 10 if MODE == "extended" else 2
REWARD_MODEL_LR         = 2e-5
REWARD_MODEL_BATCH_SIZE =  8 if MODE == "extended" else 4
REWARD_MODEL_MAX_LEN    = 256
REWARD_MODEL_WARMUP_STEPS = 50 if MODE == "extended" else 0
REWARD_MODEL_GRAD_ACCUM =  2 if MODE == "extended" else 1
REWARD_MODEL_WEIGHT_DECAY = 0.01
REWARD_MODEL_EARLY_STOP_PATIENCE = 3

# --- PPO training ---
PPO_STEPS           = 30
PPO_EPOCHS          =   1   # 1 inner epoch prevents policy-ref divergence in single-turn LM
PPO_BATCH_SIZE      =  16 if MODE == "extended" else 4
PPO_MINI_BATCH_SIZE =   4 if MODE == "extended" else 2
PPO_LR              = 1.41e-5
PPO_MAX_NEW_TOKENS  = 180 if MODE == "extended" else 100
PPO_GRAD_ACCUM_STEPS = 2 if MODE == "extended" else 1
PPO_WARMUP_STEPS    = 20 if MODE == "extended" else 0

# PPO hyperparameters
PPO_GAMMA         = 1.0    # correct for single-turn LM: no discounting across tokens
PPO_GAE_LAMBDA    = 0.95
PPO_EPS_CLIP      = 0.10   # tighter clipping for stability
PPO_VALUE_EPS_CLIP = 0.10
PPO_INIT_KL_COEF  = 0.2    # initial KL penalty coefficient
PPO_TARGET_KL     = 6.0    # adaptive KL controller target (TRL default)
PPO_VF_COEF       = 0.1    # value function loss coefficient (TRL default)
PPO_MAX_GRAD_NORM = 0.5

# --- Evaluation ---
EVAL_MAX_NEW_TOKENS = 180

# --- Preference data splits ---
TRAIN_SPLIT = 0.80
VAL_SPLIT   = 0.10
TEST_SPLIT  = 0.10

MIN_PREFERENCE_SCORE_GAP = 0.05  # pairs closer than this get dropped

if __name__ == "__main__":
    print(f"Mode:   {MODE}")
    print(f"Device: {DEVICE}")
    print(f"Reward model: {REWARD_MODEL_EPOCHS} epochs, batch {REWARD_MODEL_BATCH_SIZE}")
    print(f"PPO: {PPO_STEPS} steps, batch {PPO_BATCH_SIZE}")
    print(f"Candidates: {NUM_TRAINING_PROMPTS} prompts x {NUM_CANDIDATES_PER_PROMPT} = "
          f"{NUM_TRAINING_PROMPTS * NUM_CANDIDATES_PER_PROMPT} total")
