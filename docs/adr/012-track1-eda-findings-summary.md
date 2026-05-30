# ADR 012 — Track 1: Consolidated EDA Findings and Architecture Baseline

**Status:** Accepted  
**Date:** 2026-05-30  
**Updated:** 2026-05-30 (re-run on 10k dataset)

## Context

After completing EDA (`notebooks/eda/track1_eda.ipynb`) and establishing the statistical
baseline (ADR-008, ADR-009, ADR-010, ADR-011), a consolidated reference is needed that
captures all findings and their direct implications for GPT-2 fine-tuning on Leonardo.

EDA was originally run on 1k sequences per family (`*_variants.csv`). All statistics below
have been recomputed on the full 10k dataset (`ic_10k.csv`, `igbt_10k.csv`, `mosfet_10k.csv`).
The 1k estimates are noted where they differ.

This ADR supersedes none of the prior ADRs — it is a synthesis record for competition-day
reference.

---

## Corpus Statistics

| Metric | 1k (original) | **10k (current)** |
|---|---|---|
| Total sequences | 3,000 | **30,000** |
| Total step tokens | 388,294 | **3,884,749** |
| Full vocabulary | 198 unique step names | **198 (confirmed stable)** |
| Shared (all 3 families) | 94 steps (47%) | **94 steps (47%)** |
| Family-exclusive steps | 76 (MOSFET 20 / IGBT 27 / IC 29) | **76 (unchanged)** |

**Sequence lengths (10k):**

| Family | Min | p25 | p50 | p75 | Max | Mean |
|---|---|---|---|---|---|---|
| IC | 104 | 114 | 115 | 117 | 124 | 115.2 |
| MOSFET | 117 | 124 | 125 | 127 | 134 | 125.3 |
| IGBT | 138 | 146 | 148 | 150 | 158 | 148.0 |

GPT-2 default context window (1024 tokens) covers the longest possible sequence (~160 tokens)
with 6× headroom. No sliding window or truncation is needed.

---

## Process Grammar

**Conditional entropy H(next | current) — 10k estimates (1k in parentheses):**

| Family | H (10k) | H (1k) | Deterministic steps |
|---|---|---|---|
| IC | 0.8709 bits | (0.89) | 48 / 131 (37%) |
| IGBT | 0.9387 bits | (0.95) | 58 / 148 (39%) |
| MOSFET | 0.7641 bits | (0.78) | 59 / 138 (43%) |
| Combined | **1.0952 bits** | (1.11) | — |

- Entropy estimates converged: 10k values are ≤0.02 bits from 1k estimates. The 1k corpus was statistically sufficient for entropy estimation.
- Every sequence starts with `RECEIVE WAFER LOT` (100% frequency, all families, confirmed on 10k).
- **Optional steps are dominant**: IC 73, IGBT 70, MOSFET 68 steps appear in <100% of sequences. Over 50% of each family's vocabulary is optional — this is the primary source of Task 2 difficulty (bigram greedy completion hits optional branch points and cascades into error).
- High-entropy steps cluster around: litho branch points, via etch variants, measurement steps.

Implication: with 10k sequences, the grammar is fully characterized. The model must capture
*optional-step variation* across 68–73 per-family optional steps, not discover structure from noise.

---

## Structural Differences Between Families

**Shared backbone (~16 anchor steps, 100% presence in all families):**

```
RECEIVE WAFER LOT → LOT IDENTIFICATION → ... → HF DIP
→ ILD block (5 synonym-pair steps)
→ CURE PASSIVATION → CLEAN PAD OPENING → MEASURE PAD OPENING
→ LEAKAGE TEST → SWITCHING TEST → WAFER SORT TEST → YIELD ANALYSIS → SHIP LOT
```

**Critical divergence points:**

| Module | MOSFET | IGBT | IC |
|---|---|---|---|
| Family prep | Epitaxial deposition | Epitaxial wafer check | Backside grind + wet etch |
| Litho levels | **4** | **6** | **4** |
| Implants | WELL / LDD / SOURCE DRAIN | P BODY / N BUFFER / CHANNEL STOP / DRAIN-CATHODE / SOURCE | N-TYPE only |
| Via fill | FILL VIA METAL | FILL VIA METAL | TUNGSTEN SEED + FILL VIA TUNGSTEN |
| Final test | THRESHOLD VOLTAGE TEST | BREAKDOWN VOLTAGE TEST | THRESHOLD VOLTAGE TEST |

IGBT's 6 litho levels vs MOSFET/IC's 4 is the clearest case where the `[FAMILY]` BOS
conditioning token is mandatory — without it, the model cannot know that `ALIGN MASK LEVEL 5`
and `ALIGN MASK LEVEL 6` are legal next steps.

**Family-exclusive bigram transitions (10k vs 1k):**

| Family | Exclusive bigrams (10k) | Total bigrams | % exclusive | 1k estimate |
|---|---|---|---|---|
| IC | 89 | 280 | **31.8%** | (40–46%) |
| IGBT | 81 | 298 | **27.2%** | (40–46%) |
| MOSFET | 56 | 264 | **21.2%** | (40–46%) |

The 1k estimate (40–46%) overstated family exclusivity due to sparse sampling. With 10k
sequences, previously unseen transitions appear in multiple families, reducing exclusive %.
However, the combined H (1.0952) remains higher than any per-family H (0.76–0.94), confirming
that family conditioning still reduces prediction entropy and is worth retaining.

---

## Performance Baseline Ladder

| Method | Task 1 Top-1 | Task 2 Token Acc | Task 3 ROC-AUC | Notes |
|---|---|---|---|---|
| Random | ~0.5% | — | ~0.50 | 1/198 vocab |
| GPT-2 zero-shot | 13.3% | skipped | 0.57 | n=30 samples; CPU only |
| Bigram | **68.3%** | **6.2%** | 0.996* | *synthetic anomalies only |
| GPT-2 fine-tuned | target | **primary gap** | N/A | Rule Checker handles Task 3 |

**Task 2 is the primary improvement target.** Bigram's 6.2% token accuracy is caused by
greedy error cascading: one wrong step invalidates all successors. Fine-tuned GPT-2 with
beam search breaks this cascade.

---

## Task 3 — Deterministic Rule Checker (Closed)

All 10 forbidden rules are window-based (look-back 6–15 steps) or global ordering constraints.
None are bigram-detectable. `validate_sequence()` in `scripts/generate_sequences.py` covers
all 10 rules and returns exact rule IDs.

**Task 3 is solved before any GPU training.** See ADR-010.

---

## Parameter EDA (ADR-011 Summary)

- 353/374 steps (94–95% per family) have at least one parsed numeric parameter.
- 73 steps are shared by all 3 families.
- Top discriminating parameters by PCA PC1 loading:
  `temperature_c` > `time_min` > `thickness_nm` > `pressure_mtorr`
- Cross-family temperature deltas are small (≤50°C), concentrated in thermal steps.
- PCA (2 components, 49% variance) shows partial separability with significant overlap.
- **Parameters are a supplementary signal. Sequence grammar is the primary family discriminator.**

Best parameter-based family discriminators:
- Litho overlay tolerance: IGBT ±120–150 nm vs MOSFET/IC ±80–100 nm
- RTA temperature: IGBT 1050°C vs MOSFET/IC 1000°C
- Implant energy profile: IGBT has 5 distinct implants (30–150 keV); MOSFET/IC have 3

---

## Architecture Decisions

| Decision | Chosen | Rationale |
|---|---|---|
| Model count | One shared model | 47% shared vocab; backbone identical across families |
| Family signal | `[FAMILY]` BOS conditioning token | Mandatory for IGBT litho-level disambiguation |
| Tokenization | One step = one custom token | Preserves step name semantics; no subword splitting (Rule 2) |
| Training format | `[MOSFET] STEP1 \| STEP2 \| ... [EOS]` | Pipe separator; max ~160 tokens per sequence |
| Task 3 solver | Deterministic rule checker | Statistical models cannot express global ordering constraints |
| Compute | Fine-tuning on Leonardo A100 | 11s/prediction on local CPU makes training infeasible locally |

---

## Gate Status at ADR Creation

| Gate | Condition | Status |
|---|---|---|
| Gate 1 | GPT-2 pipeline end-to-end, Top-1 > 0 after 200 steps | Pending Leonardo |
| Gate 2 | Top-1 ≥ 0.40 by step 1,000 | Pending |
| Gate 3 | All 3 submission files valid | Pending |
| Gate 4 | Gradio demo live, files validated | Pending |

## Consequences

- This document is the single reference for competition-day trade-off discussions.
- Task 2 token accuracy improvement (from 6.2% to ≥40%) is the primary GPT-2 fine-tuning goal.
- Parameter features (`temperature_c`, `time_min`, `thickness_nm`) are reserved for a
  potential OOD conditioning extension after Gate 2 is cleared.
- No new architecture experiments should begin until Gate 1 confirms the pipeline is healthy.
