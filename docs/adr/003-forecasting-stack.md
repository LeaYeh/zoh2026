# ADR 003 — Forecasting Stack: Chronos + PatchTST

**Status:** Accepted  
**Date:** 2026-05-21

## Context

Track 03 requires a model that predicts asset price direction to feed the agent's decision. Options evaluated:

- Amazon Chronos (zero-shot foundation model)
- PatchTST (fine-tuned transformer, patch-based)
- iTransformer
- Pure LLM reasoning without a dedicated forecasting model
- Chronos zero-shot as baseline + PatchTST fine-tune as primary

## Decision

**Two-stage pipeline: Chronos zero-shot baseline → PatchTST fine-tuned on A100.**

## Rationale

- Chronos runs zero-shot with no training, giving a working baseline in < 30 minutes on Day 1
- PatchTST's patch-based architecture is well-matched to financial time series (local pattern extraction)
- Two-stage lets the agent loop run before fine-tuning completes — de-risks the 36-hour timeline
- Both models output probabilistic forecasts (median + confidence interval) that the risk node can reason about
- Pure LLM reasoning without a forecasting model sacrifices the A100 advantage

## Consequences

- `src/forecasting/chronos_baseline.py` handles zero-shot inference
- `src/forecasting/patchtst_finetune.py` handles A100 fine-tuning via PyTorch Lightning
- Training uses `bf16-mixed` precision (mandatory on A100) and `torch.set_float32_matmul_precision("high")`
- Forecast outputs are `{median, q10, q90}` — both models must conform to this interface
- Ensemble weights models by backtest Sharpe ratio
