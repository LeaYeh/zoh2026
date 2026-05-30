# ADR 008 — Track 1: Bigram Model as Statistical Baseline

**Status:** Accepted  
**Date:** 2026-05-30

## Context

Track 1 (Infineon) requires a baseline before any GPU training. Three tasks must be scored:
- Task 1: Next-step prediction (Top-1/3/5, MRR)
- Task 2: Sequence completion (Token Accuracy, Edit Distance)
- Task 3: Anomaly detection (ROC-AUC, Rule Attribution Accuracy)

A usable baseline must run without GPU and serve as the lower bound for GPT-2 fine-tuning.

EDA revealed the manufacturing process grammar is near-deterministic:
- Conditional entropy H(next | current) = 0.89–0.95 bits per family
- 38–48% of steps have exactly one valid successor
- All sequences start with the same step (RECEIVE WAFER LOT, 100%)

## Decision

**Bigram model with family conditioning, hard zero for OOV, and BOS token.**

Design choices and alternatives considered:

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Smoothing | Hard zero (OOV → 1e6) | Laplace / Kneser-Ney | Hard zero gives binary anomaly signal; smoothing blurs it |
| Family handling | Three separate transition tables | One global model | Per-family H = 0.78–0.95 bits; combined H = 1.11 bits — 40–46% of bigrams are family-exclusive |
| BOS token | Yes | Global frequency fallback | global_freq top-1 = DEVELOP PHOTORESIST; BOS top-1 = RECEIVE WAFER LOT (100% accuracy at position 0) |
| Task 2 stopping | EOS token + family_median+30 cap | Fixed length | Prevents runaway completion without sequence length prior |
| Rule attribution | PREDICTED_RULE="UNKNOWN" | Attempt rule inference from bigram | Bigram has no semantic rule knowledge; attribution requires explicit rule checker |

## Results

- Task 1 Top-1: 68.3% (MRR: 0.81)
- Task 2 Token Accuracy: 6.2% (error cascading: one wrong step propagates)
- Task 3 ROC-AUC: 0.996 on synthetic swap anomalies

Note: Task 3 result is on synthetic data (random step swaps → OOV bigrams). Performance on real rule violations is expected to be much lower — see ADR-010.

## Implementation

### Sequence format

Raw data is `data/raw/training_data/*_variants.csv` in **long format** — one row per step:

```
SEQUENCE_ID,STEP
seq_0001,RECEIVE WAFER LOT
seq_0001,LOT IDENTIFICATION
seq_0001,INITIAL WAFER INSPECTION
seq_0002,RECEIVE WAFER LOT
...
```

`load_sequences()` groups by `SEQUENCE_ID` and returns `list[list[str]]`.  
Each inner list is raw step names with no special tokens — `fit()` adds `<BOS>` and `<EOS>` internally.

`ProcessStepTokenizer` (integer IDs) is **not used** by the bigram model; step name strings are the keys directly.

### Augmented sequence seen by `fit()`

```
input:      ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "INITIAL WAFER INSPECTION"]
augmented:  ["<BOS>", "RECEIVE WAFER LOT", "LOT IDENTIFICATION", "INITIAL WAFER INSPECTION", "<EOS>"]

bigram pairs counted:
  <BOS>                    → RECEIVE WAFER LOT
  RECEIVE WAFER LOT        → LOT IDENTIFICATION
  LOT IDENTIFICATION       → INITIAL WAFER INSPECTION
  INITIAL WAFER INSPECTION → <EOS>
```

### Training call chain (`scripts/eval_bigram.py`)

```
main()
  load_with_families(["IC","IGBT","MOSFET"])
    └─ load_sequences() per family → list[list[str]] + family labels
  train_val_split(val_ratio=0.1, seed=42)
    └─ global shuffle → 90% train / 10% val  (not stratified per family)
  BigramModel().fit(train_seqs, train_fams)   ← single O(N) scan, no epochs
  eval_task1 / eval_task2 / eval_task3
  wandb.log(all metrics)
```

`fit()` is called exactly once; there is no gradient update, epoch loop, or checkpoint.  
The only callers of `BigramModel` are `scripts/eval_bigram.py` (training + eval) and the three module-level functions (`predict_next_step`, `complete_sequence`, `anomaly_score`) used by `scripts/submit.py`.

### Predict key lookup

| Scenario | Key passed to transition table |
|---|---|
| Position 0 (empty prefix) | `<BOS>` |
| Any mid-sequence position | last step in partial sequence |
| Task 3 anomaly score | every consecutive pair including `<BOS>→first` and `last→<EOS>` |

OOV from-step falls back to global frequency ranking; OOV bigram (known from-step, unseen to-step) returns `1e6` immediately (hard zero).

## Consequences

- Bigram is the safety net: if Leonardo is unavailable, bigram alone can produce a valid Task 1 submission
- Task 2 is the critical gap: bigram's greedy error cascading defines the improvement target for GPT-2 fine-tuning
- Bigram is not a viable Task 3 solution for the real eval — see ADR-010
- Val split is not stratified; family distribution in val set may differ from train — acceptable for a baseline but worth noting if per-family metrics diverge
