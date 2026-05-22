# ADR 004 — Experiment Tracking: WandB Only

**Status:** Accepted  
**Date:** 2026-05-21

## Context

Need to track all training runs and backtest results. Options:

- WandB only
- `experiments.csv` only (git-trackable, offline)
- WandB + `experiments.csv` (dual tracking)

## Decision

**WandB only. No `experiments.csv`.**

## Rationale

- WandB is the sole source of truth: real-time metrics, GPU stats, config → Sharpe mapping, model artifact storage
- `experiments.csv` adds maintenance overhead (append logic, merge conflicts) without proportional benefit
- WandB's run comparison UI is faster to navigate than a CSV under hackathon time pressure
- Offline fallback (network issues at venue): WandB has offline mode that syncs when connectivity is restored

## Consequences

- Every training run and backtest must call `wandb.init()` — no exceptions (Rule 2 in CLAUDE.md)
- No local CSV fallback; if WandB is down, log manually and backfill after
- WandB run ID is included in every experiment review report
