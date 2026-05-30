---
name: process-sequence-modeling
description: |
  Knowledge base for training transformer sequence models on industrial process data
  and solving the three Track 1 submission tasks.
  Distilled from 李宏毅 ML 2025 (Lec 4, 6, 7, 8 + HW3–7) and competition materials.

  Trigger on: GPT-2 transformer sequence-model process-steps tokenizer causal-LM
  next-step-prediction sequence-completion anomaly-detection perplexity
  GRPO rule-based-reward RL-training catastrophic-forgetting
  Top-1 MRR edit-distance F1 ROC-AUC
  industrial semiconductor IC IGBT MOSFET generation_rules
  train.py submit.py nextstep.csv completion.csv anomaly.csv
metadata:
  source: 李宏毅 ML 2025 (Lec 4 6 7 8, HW3–7) + Infineon Track 1 briefing
  competition: Zero One Hack 2026, Vienna, Track 1 Industrial AI
---

# process-sequence-modeling

## Seven Core Rules

1. **One token = one step name** — never split "DEPOSIT GATE OXIDE" into sub-words; the entire string is one vocabulary item
2. **Causal LM is the only training objective** — cross-entropy on next token; do NOT use masked LM (BERT-style)
3. **PAD positions must be masked** — `labels=-100` on padding, `attention_mask=0`; otherwise model wastes capacity predicting PAD
4. **Validate against generation_rules.md before submitting** — use `scripts/generate_sequences.py --validate`; rule violations disqualify outputs
5. **Top-1 is your gate metric** — monitor it every eval; if it plateaus before reaching 0.4, change something
6. **Anomaly threshold = median perplexity on eval set** — never hardcode a number; calibrate at inference time
7. **Submission format is strict** — pipe-separated steps in one column; wrong format = zero score regardless of model quality

---

## 36-Hour Competition SOP

```
Data arrives
  │
  ▼
① Verify tokenizer builds correctly (vocab ~120 steps expected)
  └─ uv run python src/train.py configs/exp/gpt2_dummy.yaml (sanity check)
  │
  ▼
② Run gpt2_finetune_v1 — 3 000 steps (A100, ~20 min)
  └─ Monitor: train/loss should decrease; eval/top1 should reach ≥0.4 by step 1000
  │
  ├─ Top-1 < 0.2 at step 500? → lower LR (5e-4 → 2e-4) or increase batch_size
  ├─ Loss not decreasing? → check attention_mask is passed; check labels=-100 on PAD
  │
  ▼
③ If Top-1 stalls: GRPO with rule-based reward (see 02-finetuning-grpo.md)
  │
  ▼
④ Generate submission files
  └─ uv run python scripts/submit.py --eval-valid ... --eval-anomaly ... --output-dir submission/
  │
  ▼
⑤ Validate locally with eval_metrics.py before handing to organizers
```

---

## Quick Reference

| File | Purpose |
|---|---|
| `src/train.py` | Training entry point |
| `src/data/process_loader.py` | Tokenizer + CSV loader |
| `src/models/process_lm.py` | predict_next_step / complete_sequence / anomaly_score |
| `src/evaluation/process_metrics.py` | Top-K, MRR, edit distance, F1, ROC-AUC |
| `scripts/submit.py` | Generate nextstep.csv / completion.csv / anomaly.csv |
| `scripts/generate_sequences.py` | Augment training data + validate sequences |
| `data/raw/training_data/generation_rules.md` | Process grammar + forbidden patterns |
| `configs/exp/gpt2_finetune_v1.yaml` | Full training config |
| `configs/exp/gpt2_dummy.yaml` | 200-step smoke test |

---

## Reference Files

| Topic | File | When to read |
|---|---|---|
| Architecture & training | `references/01-architecture-training.md` | Setting up model, debugging loss |
| Fine-tuning & GRPO | `references/02-finetuning-grpo.md` | After baseline, scaling, RL training |
| Evaluation & submission | `references/03-evaluation-submission.md` | Generating and validating outputs |
| Data strategy | `references/04-data-strategy.md` | Augmentation, loader issues |
