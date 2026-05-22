# ADR 006 — Validation Strategy: Walk-Forward Backtest (No K-Fold)

**Status:** Accepted  
**Date:** 2026-05-21

## Context

Traditional Kaggle competitions use K-fold cross-validation with OOF predictions to estimate leaderboard score. For Track 03, the evaluation metric is Sharpe ratio from the agent's trading decisions, not a direct model prediction score.

## Decision

**Walk-forward backtesting replaces K-fold CV as the primary validation mechanism.**

For PatchTST fine-tuning, a single 80/20 temporal train/val split (not K-fold) is used to detect overfitting.

## Rationale

- Random K-fold on time series causes look-ahead bias — the model sees future data during training
- Walk-forward backtest is temporal CV by definition: the agent is tested on data strictly after its training window
- The primary metric is Sharpe ratio (agent output), not MAE/MSE (forecast output) — a model with lower MAE can have worse Sharpe if it mispredicts at the wrong moments
- K-fold requires 5× training time; the 36-hour budget is better spent on model diversity and stress-testing
- 5-fold OOF ensemble logic (from tabular competitions) is not applicable here: forecasting models are blended by Sharpe-weighted averaging, not stacking

## Consequences

- `data/folds.pkl` is not used; removed from Rule 3 in CLAUDE.md
- Rule 3 updated to: "all backtests must use the same time windows" (defined in `configs/competition.yaml`)
- `src/evaluation/backtesting.py` is the primary validation tool, not `sklearn.TimeSeriesSplit`
- 1000+ parallel scenarios (different assets, time windows, market regimes) replace fold variance as the robustness check
