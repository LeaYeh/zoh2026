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

## Consequences

- Bigram is the safety net: if Leonardo is unavailable, bigram alone can produce a valid Task 1 submission
- Task 2 is the critical gap: bigram's greedy error cascading defines the improvement target for GPT-2 fine-tuning
- Bigram is not a viable Task 3 solution for the real eval — see ADR-010
