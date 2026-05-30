# ADR 010 — Track 1: Task 3 Anomaly Detection via Deterministic Rule Checker

**Status:** Accepted  
**Date:** 2026-05-30

## Context

Task 3 requires classifying sequences as valid or invalid, and optionally identifying which of the 10 forbidden rules was violated (PREDICTED_RULE field → Rule Attribution Accuracy score).

Two approaches were evaluated:

**Statistical (bigram):** Score sequences by anomaly score; OOV bigrams return 1e6.
- Achieved 99.6% ROC-AUC on synthetic anomalies (random step swaps)
- Synthetic swaps create OOV bigrams → hard zero fires reliably

**Analysis of the real eval:** All 10 forbidden rules in `generation_rules.md` are window-based (N=6–15 steps lookback) or global ordering constraints — none are immediate-successor (bigram) constraints:

| Rule | Type | Bigram detectable? |
|---|---|---|
| RULE_DEP_NO_CLEAN | Window N=12 | No |
| RULE_METAL_ETCH_NO_LITHO | Window N=15 | No |
| RULE_ETCH_NO_MASK | Window N=12 | No |
| RULE_LITHO_LEVEL_SKIP | Global ordering | No |
| RULE_IMPLANT_NO_MASK | Window N=15 | No |
| RULE_CMP_NO_DEP | Window N=6 | No |
| RULE_PAD_OPEN_BEFORE_DEP | Global ordering | No |
| RULE_TEST_BEFORE_PASSIVATION | Global ordering | No |
| RULE_SHIP_BEFORE_TEST | Global ordering | No |
| RULE_BACKSIDE_BEFORE_PASSIVATION | Global ordering | No |

A valid sequence can violate RULE_SHIP_BEFORE_TEST while every individual bigram transition is perfectly valid. Bigram's 99.6% result does not transfer to the real eval.

`generate_sequences.py` already contains a complete `validate_sequence(steps) → list[Violation]` implementation covering all 10 rules, returning the violated rule ID.

## Decision

**Use `validate_sequence()` from `generate_sequences.py` as the Task 3 anomaly detector. Do not use bigram or GPT-2 perplexity for Task 3.**

```python
violations = validate_sequence(steps)
is_valid   = 1 if not violations else 0
pred_rule  = violations[0].rule if violations else ""
score      = 0.0 if violations else 1.0
```

## Rationale

- The organizers used `validate_sequence()` to generate the ground truth eval set — using the same function guarantees alignment
- Rule attribution accuracy is a scored metric; `validate_sequence()` returns exact rule IDs at no extra cost
- No training data, no GPU, no threshold tuning required
- Statistical models (bigram, GPT-2 perplexity) cannot express window or global ordering constraints

## Consequences

- Task 3 is effectively solved before any model training
- Bigram and GPT-2 anomaly scoring code remains in place as a fallback if `validate_sequence()` is found to differ from the held-out eval format
- Effort freed from Task 3 can be directed to Task 2 (sequence completion via GPT-2 fine-tuning)
