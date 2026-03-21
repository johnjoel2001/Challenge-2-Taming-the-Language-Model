# Final Report: Reward Design Matters

## A Multi-Objective RLHF Study of Clarity, Completeness, and Over-Optimization in Small Language Models

---

## 1. Project Objective

This project investigates how **reward design** in Reinforcement Learning from Human Feedback (RLHF) affects the behaviour of a small language model aligned to produce beginner-friendly technical explanations.

We train and compare three models:

1. **Baseline** — pretrained GPT-2 (124M) with no alignment
2. **Simplicity-PPO** — aligned with a reward model that favours short, readable, simple responses
3. **Balanced-PPO** — aligned with a reward model that balances readability, completeness, appropriate length, and low repetition

The core research question:

> *How does reward design affect clarity, completeness, and over-optimization in RLHF for beginner-friendly technical explanations?*

---

## 2. Method

### 2.1 Pipeline Overview

The project is implemented as a seven-stage Python pipeline, with each stage producing artifacts consumed by subsequent stages:

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `generate_baseline.py` | Generate baseline outputs from the pretrained model and compute metrics |
| 2 | `create_preference_data.py` | Sample multiple candidate responses per prompt, score them with heuristic rewards, and form preference pairs |
| 3 | `train_reward_model_simplicity.py` / `train_reward_model_balanced.py` | Train two separate reward models on their respective preference datasets |
| 4 | `train_ppo_simplicity.py` / `train_ppo_balanced.py` | Fine-tune the base model via PPO using each learned reward model |
| 5 | `evaluate.py` | Run all three models on 16 held-out prompts and compute comprehensive metrics |
| 6 | `misalignment_analysis.py` | Detect over-optimization failure modes with heuristic detectors |
| 7 | `plotting.py` | Generate six trade-off visualisations |

### 2.2 Base Model

We use **GPT-2** (124M parameters) — OpenAI's small GPT-2 variant. It is large enough to produce coherent multi-sentence text, yet small enough to train on a laptop with Apple Silicon (MPS) acceleration in under 10 minutes.

### 2.3 Prompts

We use **12 training prompts** and **16 evaluation prompts** covering topics such as machine learning, neural networks, blockchain, APIs, databases, gradient descent, transformers, supervised learning, recursion, and more. Each prompt asks for a beginner-friendly explanation (e.g., *"Explain machine learning in simple words."*).

### 2.4 Preference Data Generation

For each training prompt we generate **4 candidate responses** (48 total) from the base model using nucleus sampling (`top_p=0.95`, `top_k=50`). Each candidate is scored with two heuristic reward functions. For each prompt, the highest-scoring and lowest-scoring candidates form a preference pair, yielding **12 simplicity pairs** and **12 balanced pairs**.

### 2.5 Reward Model Architecture

Each reward model consists of the GPT-2 transformer backbone with a linear head that maps the final hidden state to a scalar reward score. Training uses a pairwise ranking (Bradley-Terry) loss over the preference pairs for **2 epochs** (quick mode).

### 2.6 PPO Training

We use the TRL library's `PPOTrainer` (v0.9.6) with `AutoModelForCausalLMWithValueHead`. Training runs for **8 PPO steps** in quick mode, with batch size 4 and learning rate 1.41e-5. At each step, the model generates responses to a mini-batch of prompts, the learned reward model scores them, and the PPO objective updates the policy.

---

## 3. Reward Design Explanation

### 3.1 Simplicity Reward

The simplicity heuristic scores responses based on:
- **Readability** (Flesch Reading Ease / 120) — weight 0.50
- **Brevity** (inverse of word count) — weight 0.35
- **Word-length penalty** (penalises long words) — weight -0.15

This reward explicitly pushes the model toward shorter, easier-to-read text. It does **not** consider whether the response actually covers the key concepts.

### 3.2 Balanced Reward

The balanced heuristic scores responses based on:
- **Readability** (Flesch Reading Ease / 120) — weight 0.30
- **Completeness** (keyword coverage) — weight 0.30
- **Length bonus** (+0.2 for 15–120 words, -0.3 for <15 words) — structural
- **Repetition penalty** — weight -0.15
- **Word-length penalty** — weight -0.10

This reward trades off readability against completeness and penalises degenerate short or repetitive outputs.

### 3.3 Why Two Rewards?

The comparison is the core contribution. A single reward study cannot reveal how different design choices create different failure modes. By training two separate PPO models, we can directly observe:
- What simplicity-only optimisation gains and loses
- Whether a balanced reward produces a better trade-off
- Which design is more prone to over-optimization

---

## 4. Evaluation Setup

All three models generate responses for the same **16 held-out prompts** using identical generation parameters (`max_new_tokens=120`, `do_sample=True`, `top_k=50`, `top_p=0.95`). We compute the following metrics per response, then average across prompts:

| Metric | Description |
|--------|-------------|
| Readability | Flesch Reading Ease (0–120, higher = easier) |
| Completeness | Fraction of expected keywords covered per prompt |
| Response Length | Word count |
| Lexical Diversity | Type-token ratio (unique words / total words) |
| Repetition | Repeated n-gram ratio (bigram overlap) |
| Pairwise Overlap | Jaccard similarity across a model's 16 outputs |
| RM Simplicity Score | Learned simplicity reward model's prediction |
| RM Balanced Score | Learned balanced reward model's prediction |

---

## 5. Results

### 5.1 Quantitative Evaluation Summary

The following table reports the actual metrics from our GPT-2 pipeline run (seed 42, quick mode, MPS device):

| Metric | Baseline | Simplicity-PPO | Balanced-PPO |
|--------|----------|----------------|--------------|
| **Readability** (FRE) | 43.46 | 32.19 | 30.53 |
| **Avg Sentence Length** | 36.25 | 27.81 | 25.01 |
| **Avg Word Length** | 4.73 | 5.38 | 4.07 |
| **Response Length** (words) | 116.1 | 110.4 | 83.1 |
| **Lexical Diversity** (TTR) | 0.947 | 0.937 | 0.772 |
| **Completeness** (keyword) | 0.214 | 0.194 | 0.202 |
| **Repetition** (n-gram) | 0.0 | 0.0 | 0.0 |
| **Simplicity Heuristic** | 0.205 | 0.149 | 0.220 |
| **Balanced Heuristic** | 0.188 | 0.185 | 0.115 |
| **RM Simplicity Score** | -0.281 | -0.324 | -0.288 |
| **RM Balanced Score** | -0.229 | -0.290 | -0.283 |
| **Pairwise Overlap** | 0.138 | 0.125 | 0.106 |

### 5.2 PPO Training Dynamics

**Simplicity-PPO** training over 8 steps:
- Mean reward improved from **-0.287** (step 0) to **-0.150** (step 7) — a clear upward trend.
- Average response length remained in the 61–81 word range.

**Balanced-PPO** training over 8 steps:
- Mean reward fluctuated between **-0.217** and **-0.269**, ending at **-0.179**.
- Training showed a KL-divergence warning at step 8 (KL = -2.85), suggesting the model was drifting.

### 5.3 Key Observations

1. **Completeness is universally low.** All three models achieve only 19–21% keyword coverage. GPT-2 is not an instruction-following model, so its outputs rarely contain domain-specific terminology regardless of alignment. This makes completeness differences between models small in absolute terms.

2. **Readability decreased for both PPO models.** Counter-intuitively, both aligned models produced *lower* Flesch Reading Ease scores than the baseline (32 and 31 vs. 43). In this toy setting with only 8 PPO steps, the models did not converge enough to clearly improve readability — they primarily shifted output style and length.

3. **Balanced-PPO aggressively shortened responses.** The balanced reward's length bonus (+0.2 for 15–120 words) combined with the brevity-neutral readability weight caused the balanced model to produce the shortest responses (83 words vs. 116 baseline), reducing lexical diversity to 0.77.

4. **Simplicity-PPO preserved diversity.** Despite targeting brevity, the simplicity model maintained high lexical diversity (0.94) and response length (110 words), suggesting it found a different optimisation path — adjusting word choice rather than simply truncating.

5. **Pairwise overlap is low across all models** (0.11–0.14), indicating none of the models collapsed to a single template.

---

## 6. Misalignment & Failure Mode Analysis

### 6.1 Detection Thresholds

| Detector | Threshold |
|----------|-----------|
| Over-shortening | Response < 12 words |
| Completeness loss | Completeness < baseline mean (0.21) - 0.15 |
| Template repetition | n-gram repetition ratio > 0.2 |
| Low diversity | Type-token ratio < 0.45 |
| Reward hacking | Simplicity heuristic > 0.3 AND completeness < 0.25 |

### 6.2 Flag Counts

| Model | Over-shortening | Completeness Loss | Template Rep. | Low Diversity | Reward Hacking | **Total** |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| Baseline | 0 | 6 | 0 | 0 | 1 | **7** |
| Simplicity-PPO | 0 | 6 | 0 | 0 | 0 | **6** |
| Balanced-PPO | 3 | 6 | 0 | 3 | 4 | **16** |

### 6.3 Analysis

- **Completeness loss is pervasive** (6 flags per model). This is driven by GPT-2's inability to produce on-topic technical content, not by RLHF degradation.
- **Balanced-PPO accumulated the most flags** (16 total). It triggered 3 over-shortening flags (some responses fell under 12 words), 3 low-diversity flags (TTR < 0.45 on individual responses), and 4 reward-hacking flags. This shows that even a multi-objective reward can produce pathological outputs when training is short.
- **Simplicity-PPO had the fewest flags** (6). In this particular run, with limited PPO steps, the simplicity model changed less from the baseline distribution.
- **No template repetition** was detected in any model — pairwise overlap stayed well below the 0.55 threshold.

### 6.4 Interpretation

The result that balanced-PPO accumulated *more* flags than simplicity-PPO is itself an important finding. With only 8 PPO steps:
- The balanced reward's multiple objectives created **competing gradients**, pushing the model harder in different directions simultaneously.
- The simplicity reward's single-objective signal was weaker per-step, resulting in less deviation from the baseline.
- With more training steps (extended mode), we would expect simplicity-PPO to eventually over-optimise more aggressively, while balanced-PPO should stabilise at a better equilibrium.

This illustrates a nuance of reward design: **a more complex reward can be harder to optimise in low-data regimes**, even if it produces better outcomes with sufficient training.

---

## 7. Example Outputs

### Prompt: "What is blockchain in plain English?"

**Baseline:**
> *"As a general matter, it means that Bitcoin transactions are only performed on the Blockchain. It's not true to say how many people have seen this technology coming and now we know what an 'unconfirmed' block size number might be..."* (rambling, off-topic)

**Simplicity-PPO:**
> *"A crypto-currency which has been developed as a replacement for bitcoin and the Ethereum network. A digital currency that serves to decentralize computing power (based on blockchains)..."* (more focused, uses relevant terms)

**Balanced-PPO:**
> *"Blockchain has been around for over half a century, and most people didn't know it until they read this article. It's about what we do at the company to better understand how cryptocurrency works by using an immutable ledger..."* (mentions key concepts like "immutable ledger" and "cryptocurrency")

### Prompt: "What is a database?"

**Baseline:**
> *"A server (like MySQL) stores data about the user and also what they are typing in its JSON. A typical application would store this information directly on your website..."* (mentions MySQL/JSON but drifts into passwords)

**Simplicity-PPO:**
> *"<!DOCTYPE html> <html lang='en'..."* (degenerate — generates HTML code instead of an explanation)

**Balanced-PPO:**
> *"A virtual machine – the one that runs on Windows or Linux. 'Data structures (like tables and documents) are defined by an SQL table, with special information about how each element in those data structure relates to what it represents.'"* (mentions tables/SQL, partially relevant)

These examples highlight that GPT-2 without instruction tuning produces inconsistent quality regardless of alignment method. The RLHF signal nudges the distribution but cannot overcome the base model's fundamental limitations.

---

## 8. Trade-Off Visualisations

Six plots are saved in `outputs/figures/`:

| Plot | What It Shows |
|------|---------------|
| `readability_vs_completeness.png` | Scatter of readability vs. completeness per model — reveals the trade-off frontier |
| `reward_vs_completeness.png` | Simplicity reward vs. completeness — exposes the reward-hacking zone |
| `readability_vs_repetition.png` | Whether higher readability comes at the cost of repetition |
| `metric_bar_chart.png` | Grouped bar chart comparing normalised metrics across all three models |
| `ppo_training_curves.png` | Reward and response length over PPO training steps for both models |
| `radar_chart.png` | Multi-dimensional radar comparing readability, completeness, diversity, and length |

---

## 9. Technical Details

### 9.1 Software Stack

| Component | Version / Tool |
|-----------|----------------|
| Language model | `gpt2` (124M params, Hugging Face) |
| RL library | TRL 0.9.6 (`PPOTrainer`, `AutoModelForCausalLMWithValueHead`) |
| Transformers | Hugging Face Transformers ≥4.35 |
| Compute | Apple Silicon MPS (Metal Performance Shaders) |
| Readability | `textstat` (Flesch Reading Ease) |
| Plotting | `matplotlib` |
| Data | `pandas`, `datasets` |

### 9.2 Hyperparameters (Quick Mode)

| Parameter | Value |
|-----------|-------|
| Reward model epochs | 2 |
| PPO steps | 8 |
| PPO batch size | 4 |
| PPO mini-batch size | 2 |
| PPO learning rate | 1.41e-5 |
| PPO epochs per step | 4 |
| Max new tokens (generation) | 120 |
| Candidates per prompt | 4 |
| Random seed | 42 |

### 9.3 Runtime

The full pipeline (stages 1–7) completes in approximately **5–8 minutes** on an Apple Silicon Mac in quick mode.

---

## 10. Limitations

1. **Toy scale** — GPT-2 (124M) is not an instruction-following model. It was pretrained on web text and has no chat or instruction-tuning. Results illustrate the *dynamics* of reward design, not production-quality alignment.
2. **Heuristic rewards** — Real RLHF uses human annotators or large reward models (e.g., fine-tuned LLaMA). Our heuristic functions are lightweight approximations.
3. **Small data** — 12 training prompts and 16 eval prompts. Larger, more diverse prompt sets would yield more robust and generalisable conclusions.
4. **No human evaluation** — We rely entirely on automatic metrics. Human judgement (e.g., pairwise preference ratings) would provide stronger evidence for quality claims.
5. **Single seed** — Results may vary across seeds; we use seed 42 for reproducibility but do not report confidence intervals.
6. **Limited PPO budget** — 8 PPO steps is far below what production RLHF uses (typically hundreds to thousands). This limits convergence and makes the training dynamics noisy.
7. **KL divergence instability** — Negative KL values were observed during balanced-PPO training, indicating the model was drifting in unexpected ways under the multi-objective signal.

---

## 11. Future Work

- **Extended training** — Run in `extended` mode (5 RM epochs, 24 PPO steps) to observe longer-horizon reward dynamics and potential collapse.
- **Scale up** to GPT-2 Medium (355M) or a 1B parameter model for more meaningful generations.
- **Instruction-tuned base** — Start from an instruction-tuned model (e.g., Phi-2, TinyLlama-Chat) so the base model already follows prompts.
- **Learned preference model** — Use a DeBERTa-based reward model trained on human preference data instead of heuristic scoring.
- **Collect human preferences** for a small subset to validate the heuristic approach.
- **Multi-objective PPO** — Use a single PPO run with a weighted combination of rewards and sweep the weights to find the Pareto frontier.
- **KL-divergence analysis** — Deeper study of how far each aligned model drifts from the base, with KL penalty tuning.
- **Constitutional AI comparison** — Add a self-critique step as an alternative to reward modelling.

---

## 12. Conclusion

This project demonstrates that **reward design matters as much as the PPO algorithm itself**. Even in a toy setting with GPT-2 and only 8 PPO steps:

- Different reward functions produce **measurably different model behaviours** — the simplicity reward preserved diversity while the balanced reward shortened outputs and reduced diversity.
- **Over-optimization is observable** even with minimal training — the balanced model triggered 16 misalignment flags including reward hacking and over-shortening.
- **Multi-objective rewards are not automatically safer** — in low-data regimes, competing objectives can create erratic optimisation that produces more failure modes than a simpler reward.
- The **completeness gap** between models was small (0.19–0.21), confirming that RLHF cannot inject knowledge the base model doesn't have — it can only reshape how existing knowledge is expressed.

These findings underscore that reward design in RLHF requires careful thought about objective interactions, training budget, and failure-mode monitoring — not just metric maximisation.

---

*Report generated from actual pipeline run: GPT-2 (124M), seed 42, quick mode, Apple Silicon MPS.*
*See `README.md` for full instructions on reproducing these results.*
