# 02 · Fine-tuning & GRPO
**When to read**: after baseline Top-1 stalls, scaling experiments, RL-based training

> 李宏毅 ML 2025 — Lec 7 (Correct Fine-tuning), Lec 8 (Reasoning), HW5 (Fine-tune is powerful), HW6 (Forgetting), HW7 (RLHF/GRPO)

---

## When to Move Beyond Basic Causal LM

The causal LM objective teaches **distributional plausibility** (what step usually follows).
It does NOT directly teach **rule compliance** (what step is *required* by the process grammar).

Symptom that you need more: Top-1 > 0.5 but Task 3 anomaly F1 < 0.6.
The model predicts plausible steps but doesn't detect rule violations.

**Decision tree:**
```
Top-1 < 0.4 after 3000 steps → fix architecture/data first (see 01-architecture-training.md)
Top-1 ≥ 0.4 but F1 < 0.6    → GRPO with rule-based reward
Top-1 ≥ 0.4 and F1 ≥ 0.6    → scaling experiment (larger model)
```

---

## Catastrophic Forgetting (李宏毅 HW6)

Any further fine-tuning after the initial causal LM run risks forgetting the base sequence understanding.

**Three mitigation strategies ranked by effort:**

### 1. Experience Replay (easiest, recommended)
Mix 10-20% of the original training sequences into any further fine-tuning batch:
```python
# In training loop: for every 5 GRPO samples, add 1 replay sample
REPLAY_RATIO = 0.2
replay_pool = load_sequences("data/raw/training_data/")
```

### 2. Low learning rate
When fine-tuning further, use LR ≤ 1/10 of original:
```yaml
# For GRPO fine-tune after causal LM
training:
  lr: 5.0e-5  # was 5e-4 in gpt2_finetune_v1
```

### 3. LoRA (Parameter-Efficient Fine-Tuning)
If catastrophic forgetting is severe, add LoRA adapters instead of updating all weights:
```python
from peft import get_peft_model, LoraConfig
config = LoraConfig(r=8, target_modules=["c_attn"], lora_alpha=16)
model = get_peft_model(model, config)
# Only 0.3% of params are trainable; base model stays frozen
```

---

## GRPO (Group Relative Policy Optimization)

> From 李宏毅 HW7 (RLHF) + DeepSeek-R1 paper

**Key idea**: Instead of training a separate critic model (expensive), generate a *group* of completions for the same prompt and use the group average as the baseline.

```
advantage_i = (reward_i - mean(reward_group)) / std(reward_group)
```

This eliminates the critic model entirely — perfect for a 36-hour hackathon.

### Applying GRPO to Track 1

The rule violations in `generation_rules.md` give us a free reward function:

```python
from scripts.generate_sequences import validate_sequence  # organizer's validator

def compute_reward(generated_steps: list[str], family: str) -> float:
    """Binary reward: 1.0 if valid, 0.0 if any rule violated."""
    violations = validate_sequence(generated_steps, family)
    return 1.0 if not violations else 0.0
```

### GRPO training loop sketch

```python
G = 8  # group size (completions per prompt)
for partial_seq, family in training_prompts:
    # 1. Generate G completions
    completions = [complete_sequence(model, tok, partial_seq) for _ in range(G)]
    
    # 2. Score each with rule-based reward
    rewards = [compute_reward(c, family) for c in completions]
    
    # 3. Group-relative advantage
    mean_r, std_r = np.mean(rewards), np.std(rewards) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]
    
    # 4. Policy gradient loss (with KL penalty to prevent forgetting)
    loss = -sum(adv * log_prob(model, partial_seq, comp)
                for adv, comp in zip(advantages, completions))
    loss += kl_penalty * kl_divergence(model, ref_model, partial_seq)
    
    loss.backward()
    optimizer.step()
```

### GRPO hyperparameters

| Parameter | Value | Why |
|---|---|---|
| Group size G | 8 | Balance diversity vs. compute |
| KL penalty β | 0.01–0.1 | Prevent reward hacking / forgetting |
| LR | 5e-5 | Much lower than causal LM phase |
| Replay ratio | 0.2 | Keep causal LM ability intact |

### What GRPO improves

- **Task 2 (completion)**: Generated sequences will respect more process rules
- **Task 3 (anomaly)**: Model perplexity on rule-violating sequences becomes higher (better discrimination)
- **Does NOT help**: Task 1 Top-1 accuracy (that's a prediction task, not generation quality)

---

## Scaling Experiment

If time allows, compare three model sizes (李宏毅 Lec 8 — scaling laws):

| Config | n_embd | n_layer | Params | Expected benefit |
|---|---|---|---|---|
| Small (current) | 256 | 6 | ~2.5M | Fast iteration |
| Medium | 512 | 8 | ~12M | +5–10% Top-1 |
| Large | 768 | 12 | ~30M | Diminishing returns at 3K seq |

Use same `configs/exp/gpt2_finetune_v1.yaml` but override:
```yaml
run_name: gpt2_medium_v1
model:
  n_embd: 512
  n_layer: 8
  n_head: 8
```

**Key insight from 李宏毅**: with only 3,000 training sequences, models larger than ~12M params will overfit. Use `eval/top1` on val set to detect overfitting (val top1 stops improving while train loss keeps decreasing).
