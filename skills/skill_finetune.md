# Skill: GRPO Fine-tuning

Trigger: "fine-tune", "GRPO", "RL training", "rule-based reward", "forgetting", Gate 1 passed.

## Prerequisites
- [ ] Gate 1 approved (GPT-2 baseline Top-1 ≥ 0.40)
- [ ] Checkpoint at `data/oof/gpt2_finetune_v1/`
- [ ] `data/raw/training_data/generation_rules.md` read (understand 10 forbidden rules)

## When to use GRPO vs. scaling

| Situation | Action |
|---|---|
| Top-1 ≥ 0.4 but F1 (anomaly) < 0.6 | Use GRPO with rule-based reward |
| Top-1 stalls < 0.4 after 3000 steps | Fix architecture/data first |
| Top-1 ≥ 0.4, F1 ≥ 0.6, time left | Scaling experiment (n_embd=512) |

## GRPO config additions
```yaml
# Add to gpt2_finetune_v1.yaml for GRPO phase
run_name: gpt2_grpo_v1
training:
  lr: 5.0e-5          # 10× lower than causal LM phase
  steps: 1000
  grpo_group_size: 8
  grpo_kl_penalty: 0.05
  replay_ratio: 0.2   # 20% original seqs to prevent forgetting
```

## GRPO reward function
```python
# Use organizer's validator as reward signal
from scripts.generate_sequences import validate_sequence  # organizer's script

def compute_reward(generated_steps, family):
    violations = validate_sequence(generated_steps, family)
    return 1.0 if not violations else 0.0
```

## Catastrophic forgetting prevention
Always use replay_ratio ≥ 0.2. Monitor `eval/top1` — if it drops > 10% from baseline, stop GRPO.

## Gate 2 signal
If last 3 GRPO runs improved F1 by < 0.02 each → stop, move to submission generation.

See `process-sequence-modeling/references/02-finetuning-grpo.md` for full GRPO implementation.
