# ZOH 2026 — Track #1 Industrial AI: Process Sequence Modeling

**Competition**: Zero One Hack · Vienna · May 29–31, 2026 · 36 hours · 64× NVIDIA A100  
**Track**: #1 — Industrial AI (Infineon) · *Models that learn how semiconductor manufacturing processes unfold*

---

## Model Hierarchy

Three models, in order of complexity:

| Model | Branch | GPU needed | Task 1 Top-1 | Task 2 Token Acc | Task 3 ROC-AUC |
|---|---|---|---|---|---|
| **Bigram** | `feat/track1-industrial` | No | 68.3% | 6.2% | 0.996¹ |
| **GPT-2 zero-shot** | `feat/gpt2-baseline` | No | 13.3% | — | 0.57¹ |
| **GPT-2 fine-tuned** | TBD (Leonardo) | Yes (A100) | target | target | — |
| **Rule checker** | `main` | No | — | — | ~1.0² |

¹ Measured on synthetic swap anomalies. Real eval uses the 10 forbidden rules — see ADR-010.  
² `validate_sequence()` is deterministic; Task 3 is effectively solved — see ADR-010.

---

## Quick Start

```bash
# Install dependencies
uv sync

# Verify environment
uv run python scripts/verify_setup.py
```

---

## Bigram Baseline (`feat/track1-industrial`)

Family-conditioned bigram model. No GPU, no training loop — fit is a single scan of sequences.

```bash
git checkout feat/track1-industrial
```

**Run full evaluation (Tasks 1, 2, 3):**

```bash
uv run python scripts/eval_bigram.py
```

**Options:**

```bash
uv run python scripts/eval_bigram.py \
  --data-dir data/raw/training_data \
  --val-ratio 0.1 \
  --seed 42
```

**Expected output:**

```
Train: 2709  Val: 302

── Task 1: Next-Step Prediction ─────────────────────────────
  Samples : 302
  Top-1   : 0.6830  (68.3%)
  Top-3   : 0.8017  (80.2%)
  Top-5   : 0.8543  (85.4%)
  MRR     : 0.8120

── Task 2: Sequence Completion ──────────────────────────────
  Token Accuracy   : 0.0620
  Exact Match Rate : 0.0000
  Norm. Edit Dist  : 0.8241

── Task 3: Anomaly Detection (synthetic anomalies) ──────────
  ROC-AUC  : 0.9960
  F1       : 0.9400
```

**Key numbers to know:**
- Task 1 is strong because the manufacturing grammar is near-deterministic (H = 0.89–0.95 bits per family)
- Task 2 is weak (6.2%) due to error cascading: one wrong step propagates through greedy completion
- Task 3 ROC-AUC 0.996 is on synthetic swap anomalies, not the real 10-rule eval (see Rule Checker below)

---

## GPT-2 Zero-Shot Baseline (`feat/gpt2-baseline`)

Pre-trained `gpt2` (124M, WebText) with constrained decoding over the 198-step vocabulary. No fine-tuning.

```bash
git checkout feat/gpt2-baseline
```

**Run evaluation (Tasks 1 and 3 only — Task 2 is too slow for zero-shot):**

```bash
uv run python scripts/eval_gpt2_zeroshot.py
```

**Options:**

```bash
uv run python scripts/eval_gpt2_zeroshot.py \
  --n-task1 300 \      # prediction points to sample (default 300; use 30 for quick check)
  --n-task3 100 \      # anomaly sequences to score
  --device cpu \       # or cuda if available
  --seed 42
```

> **Note:** Task 1 takes ~11 s/sample on CPU. Use `--n-task1 30` for a quick smoke test (~3 min).

**Expected output (n-task1=30):**

```
── Task 1: Next-Step Prediction ─────────────────────────────
  Samples : 30
  Top-1   : 0.1333  (13.3%)
  Top-3   : 0.1333  (13.3%)
  MRR     : 0.1333

── Task 3: Anomaly Detection (synthetic anomalies) ──────────
  ROC-AUC  : 0.5732
```

**What this tells you:** GPT-2 zero-shot is the lower bound. It has no domain knowledge so sequence perplexity cannot distinguish process rule violations from valid sequences. After fine-tuning on A100, expect Task 1 to exceed the bigram 68.3%.

---

## Task 3: Rule Checker (`main`)

All 10 forbidden rules are window-based or global ordering constraints — no statistical model can detect them. Use the deterministic rule checker instead.

```bash
# Validate a sequence directly
uv run python - << 'EOF'
import sys; sys.path.insert(0, ".")
from scripts.generate_sequences import validate_sequence

steps = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "SHIP LOT", "WAFER SORT TEST"]
violations = validate_sequence(steps)
for v in violations:
    print(f"VIOLATION: {v.rule} at position {v.position}")
EOF
```

**Produce Task 3 submission from eval_input_anomaly.csv:**

```python
import csv, sys
sys.path.insert(0, ".")
from scripts.generate_sequences import validate_sequence

with open("data/raw/eval_input_anomaly.csv") as f_in, \
     open("submission_task3.csv", "w") as f_out:
    reader = csv.DictReader(f_in)
    f_out.write("EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE\n")
    for row in reader:
        steps = row["SEQUENCE"].split("|")
        violations = validate_sequence(steps)
        is_valid = 1 if not violations else 0
        score    = 1.0 if not violations else 0.0
        rule     = violations[0].rule if violations else ""
        f_out.write(f"{row['EXAMPLE_ID']},{is_valid},{score},{rule}\n")
```

---

## Eval Metrics Reference

### Task 1 — Next-Step Prediction

| Metric | Formula | Target |
|---|---|---|
| Top-1 Accuracy | exact match at rank 1 | > bigram 68.3% |
| Top-3 Accuracy | correct step in top 3 | > bigram 80.2% |
| Top-5 Accuracy | correct step in top 5 | > bigram 85.4% |
| MRR | mean 1/rank of correct step | > bigram 0.81 |

### Task 2 — Sequence Completion

| Metric | Formula | Target |
|---|---|---|
| Token Accuracy | exact step matches / total steps | > bigram 6.2% |
| Exact Match Rate | % fully correct completions | > 0% |
| Norm. Edit Distance | edit_distance / max_len (lower = better) | < bigram 0.82 |

### Task 3 — Anomaly Detection

| Metric | Formula | Target |
|---|---|---|
| Binary Accuracy | (TP + TN) / total | ~1.0 with rule checker |
| ROC-AUC | area under ROC curve | ~1.0 with rule checker |
| Rule Attribution Accuracy | correct rule ID / detected violations | ~1.0 with rule checker |

---

## GPT-2 Fine-Tuning (Leonardo A100)

Fine-tuning must run on the Leonardo supercomputer. Local CPU is too slow (11 s/prediction point).

```bash
# On Leonardo:
uv run python src/train.py configs/exp/gpt2_finetune_v1.yaml
```

This is the main path to beating bigram on Task 2. Target: Token Accuracy > 50%.

---

## Submission Format

**Task 1** (`submission_task1.csv`):
```
EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5
valid_0001, MEASURE THICKNESS, INSPECT WAFER, CLEAN SURFACE, MEASURE GEOMETRY, HF DIP
```

**Task 2** (`submission_task2.csv`):
```
EXAMPLE_ID, PREDICTED_SEQUENCE
valid_0001, MEASURE THICKNESS|CLEAN SURFACE|DEPOSIT GATE OXIDE|...
```

**Task 3** (`submission_task3.csv`):
```
EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE
valid_0001,  1, 0.95,
forbidden_0042, 0, 0.08, RULE_DEP_NO_CLEAN
```

---

## Team

| Member | Role |
|---|---|
| Lea | Data pipeline, model training, cloud, DevOps |
| Vladimir | Algorithms, system architecture, evaluation framework |
| Kamila | Performance optimisation, scaling experiments |
| Thomas | Scrum master, coordination |

**On-site mentor**: Simeon (Infineon)
