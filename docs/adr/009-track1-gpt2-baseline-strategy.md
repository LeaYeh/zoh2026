# ADR 009 — Track 1: GPT-2 Zero-Shot Baseline Strategy

**Status:** Accepted  
**Date:** 2026-05-30

## Context

GPT-2 (124M, pre-trained on WebText) is the foundation model for Track 1. Before fine-tuning on A100s, a zero-shot eval is needed to:
1. Establish a lower bound so fine-tuning improvement can be measured
2. Validate the inference pipeline before committing training compute

Constraints: local CPU only (no GPU for training); Leonardo supercomputer required for fine-tuning.

## Decision

**Zero-shot GPT-2 with constrained decoding against the 198-step vocabulary, deferred fine-tuning on Leonardo.**

Design choices:

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Decoding strategy | Teacher forcing (one forward pass per candidate) | KV caching across 198 candidates | KV cache retains 198 separate past_key_values tensors → OOM (exit 138 on local machine) |
| Vocabulary constraint | Score all 198 step names by summing BPE token log-probs | Unconstrained generation | Output must always be a valid step name; unconstrained sampling would produce invalid strings |
| Task 3 scoring | Sequence perplexity (single forward pass) | Per-step perplexity aggregate | Single pass is O(1) regardless of sequence length; aggregate requires per-position scoring |
| Context window | Truncate to 1024 tokens | Sliding window | Simplest fix for the IndexError on IGBT sequences (~1036 BPE tokens); sliding window adds complexity for marginal gain |
| Fine-tuning timeline | Deferred to Leonardo | Local fine-tuning | 11s per prediction point on CPU; full Task 1 eval (300 samples) takes ~55 min; training is infeasible locally |

## Results (zero-shot, n=30 Task 1 samples)

- Task 1 Top-1: 13.3% (lower bound — GPT-2 has no domain knowledge)
- Task 2: Skipped (198 forward passes × sequence length steps = too slow for zero-shot)
- Task 3 ROC-AUC: 0.57 (near random — WebText perplexity cannot distinguish process rule violations)

Baseline ladder: Random (0.5%) → GPT-2 zero-shot (13.3%) → Bigram (68.3%) → GPT-2 fine-tuned (target)

## Consequences

- Fine-tuning config (`configs/exp/gpt2_finetune_v1.yaml`) must be designed for Leonardo execution
- Local work is limited to: bigram eval, GPT-2 zero-shot scoring, config authoring, code review
- Task 2 is where fine-tuned GPT-2 creates the most value (bigram error cascading = 6% token accuracy)
- Task 3 will not be solved by GPT-2 perplexity — see ADR-010
