# 03 · Evaluation & Submission
**When to read**: generating submission files, interpreting metrics, validating outputs

> 李宏毅 ML 2025 — Lec 7 (evaluation best practices), HW5 (task-specific metrics)

---

## Three Submission Tasks

### Task 1 — Next-Step Prediction

**Input**: partial sequence  
**Output**: top-5 most likely next steps (ranked)  
**Metrics**: Top-1/3/5 Accuracy, MRR

```python
# src/models/process_lm.py
preds = predict_next_step(model, tokenizer, partial_steps, top_k=5)
# returns: [("MEASURE THICKNESS", 0.32), ("CLEAN SURFACE", 0.18), ...]
```

**Output CSV format** (`nextstep.csv`):
```
EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5
valid_0001, MEASURE THICKNESS, CLEAN SURFACE, DEPOSIT GATE OXIDE, ..., ...
```

**MRR intuition**: if the correct answer is at rank 3, this sample contributes `1/3 = 0.33` to MRR.
A model with Top-1=0.5 and MRR=0.6 means it often has the answer in rank 2 even when not rank 1.

---

### Task 2 — Sequence Completion

**Input**: partial sequence (cut at 60% or 80%)  
**Output**: predicted remaining steps only (not the partial input)  
**Metrics**: Exact Match Rate, Normalized Edit Distance (↓), Token Accuracy

```python
completed = complete_sequence(model, tokenizer, partial_steps, max_new_steps=200)
# returns only the NEW steps after the cut point
```

**Output CSV format** (`completion.csv`):
```
EXAMPLE_ID, PREDICTED_SEQUENCE
valid_0001, MEASURE THICKNESS|CLEAN SURFACE|DEPOSIT GATE OXIDE|...|SHIP LOT
```

**Important**: predict only the steps AFTER the cut point. Do not repeat the partial sequence.

**Normalized Edit Distance**: Levenshtein distance / max(len_prediction, len_ground_truth). Range [0, 1]. Lower is better. A perfect match = 0.

**Common failure mode**: model generates `<EOS>` too early → short completions → high edit distance. Fix: increase `max_new_steps` or check EOS token ID is correct.

---

### Task 3 — Anomaly Detection

**Input**: complete sequence (may contain rule violations)  
**Output**: IS_VALID (0/1), SCORE (probability valid, range [0,1]), PREDICTED_RULE (optional)  
**Metrics**: Binary Accuracy, Precision, Recall, F1, ROC-AUC, Rule Attribution Accuracy

```python
score = anomaly_score(model, tokenizer, steps)
# returns: perplexity (higher = more anomalous)
```

**Threshold calibration** (critical — do NOT hardcode):
```python
import statistics
all_scores = [anomaly_score(model, tok, seq) for seq in eval_sequences]
threshold = statistics.median(all_scores)  # splits population in half
is_valid = 1 if score < threshold else 0
validity_prob = 1.0 - min(score / max(all_scores), 1.0)  # normalise to [0,1]
```

**Output CSV format** (`anomaly.csv`):
```
EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE
valid_0001,   1, 0.92,
forbidden_042, 0, 0.08, RULE_DEP_NO_CLEAN
```

`IS_VALID`: required. `SCORE`: used for AUC. `PREDICTED_RULE`: optional but boosts score.

**10 Rule IDs** (from `data/raw/training_data/generation_rules.md` Section 3):
Inspect the rules file for exact IDs — they follow the pattern `RULE_<TYPE>_<DESCRIPTION>`.

---

## Running Submission Generation

```bash
uv run python scripts/submit.py \
  --eval-valid   data/raw/eval_input_valid.csv \
  --eval-anomaly data/raw/eval_input_anomaly.csv \
  --output-dir   submission/

# Run single task only (faster for iteration)
uv run python scripts/submit.py \
  --eval-valid   data/raw/eval_input_valid.csv \
  --eval-anomaly data/raw/eval_input_anomaly.csv \
  --output-dir   submission/ \
  --tasks 3
```

---

## Self-Evaluation Before Submitting

The organizer provides `eval_metrics.py` with no external dependencies:

```bash
# Score Task 1
python data/raw/training_data/../eval_metrics.py \
  --task nextstep \
  --ground-truth <ground_truth.csv> \
  --predictions submission/nextstep.csv

# Score Task 3  
python data/raw/training_data/../eval_metrics.py \
  --task anomaly \
  --ground-truth <ground_truth.csv> \
  --predictions submission/anomaly.csv
```

Note: ground truth files are not available during competition. Use the internal val set for relative comparisons.

---

## Internal Evaluation (During Training)

`src/evaluation/process_metrics.py` provides all metrics. Use the val split of training data:

```python
from src.evaluation.process_metrics import (
    evaluate_next_step, evaluate_completion, evaluate_anomaly
)
# ranked_preds: list of top-5 step name lists
# targets: list of true next step names
t1 = evaluate_next_step(ranked_preds, targets)
print(f"Top-1: {t1.top1:.4f}  Top-3: {t1.top3:.4f}  MRR: {t1.mrr:.4f}")
```

---

## Metric Targets (Competition Context)

| Metric | Baseline (random) | Good | Excellent |
|---|---|---|---|
| Task 1 Top-1 | ~0.8% (1/124) | >0.40 | >0.65 |
| Task 1 MRR | ~0.02 | >0.50 | >0.70 |
| Task 2 Edit Dist ↓ | ~1.0 | <0.30 | <0.15 |
| Task 2 Token Acc | ~0.8% | >0.50 | >0.70 |
| Task 3 F1 | ~0.50 | >0.65 | >0.80 |
| Task 3 ROC-AUC | 0.50 | >0.75 | >0.90 |

Baseline = uniform random prediction over 124 tokens. Even a 200-step model easily beats random.
