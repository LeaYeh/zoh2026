# Skill: Evaluation — Track 1 Metrics

Trigger: "evaluate", "metrics", "Top-1", "MRR", "edit distance", "F1", "ROC-AUC", "anomaly", any post-training review.

## Primary metric: eval/top1 (Task 1 Top-1 Accuracy)
Higher is better. Fraction of val samples where the true next step is the model's top prediction.

## Three tasks, three metric groups

| Task | Primary metric | Secondary |
|---|---|---|
| 1 Next-Step | Top-1 Accuracy | Top-3, Top-5, MRR |
| 2 Completion | Normalized Edit Distance ↓ | Exact Match Rate, Token Accuracy |
| 3 Anomaly | F1 | ROC-AUC, Binary Accuracy |

## Using process_metrics.py
```python
from src.evaluation.process_metrics import (
    evaluate_next_step, evaluate_completion, evaluate_anomaly
)

# Task 1
t1 = evaluate_next_step(ranked_preds, targets)
print(f"Top-1: {t1.top1:.4f}  MRR: {t1.mrr:.4f}")

# Task 3
from src.models.process_lm import anomaly_score
scores = [anomaly_score(model, tok, seq) for seq in val_seqs]
t3 = evaluate_anomaly(scores, labels)
print(f"F1: {t3.f1:.4f}  ROC-AUC: {t3.roc_auc:.4f}")
```

## Anomaly checklist (after every run)
- [ ] Top-1 < 0.40? → model not learning structure, check architecture
- [ ] Edit distance > 0.60? → completions are random, check EOS handling
- [ ] F1 < 0.55? → anomaly detection barely better than random, check threshold calibration
- [ ] Top-1 improving but F1 degrading? → model learns plausibility but loses rule-awareness — use GRPO
- [ ] Any NaN in predictions? → check tokenizer decode, check `max_new_steps` is large enough

## Metric targets
| Metric | Baseline (random) | Good | Excellent |
|---|---|---|---|
| Top-1 | ~0.8% | >0.40 | >0.65 |
| MRR | ~0.02 | >0.50 | >0.70 |
| Edit Distance ↓ | ~1.0 | <0.30 | <0.15 |
| F1 | ~0.50 | >0.65 | >0.80 |
| ROC-AUC | 0.50 | >0.75 | >0.90 |

See `process-sequence-modeling/references/03-evaluation-submission.md` for submission format details.
