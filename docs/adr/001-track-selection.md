# ADR 001 — Track Selection: Track 03 (Sybilion)

**Status:** Accepted  
**Date:** 2026-05-21

## Context

Zero One Hack Vienna 2026 offers three tracks:
- Track 01 (Infineon): sequence models for manufacturing process data
- Track 02 (UNIQA): conversational UI + prompt orchestration for insurance
- Track 03 (Sybilion): agent that decides when to buy (trading decisions)

Team composition: Lea (AI agents, DevOps), Vladimir (algorithms, physics background), Kamila (performance optimization), Thomas (Scrum).

## Decision

**Track 03.**

## Rationale

- Lea's "AI agents" skillset maps directly to the agent design requirement
- Vladimir's algorithms/quantitative background fits backtesting framework design
- Kamila's performance optimization shines on parallel backtest-at-scale (1000+ scenarios on A100)
- Track 02 wastes A100 compute (API calls only); Track 01 requires domain expertise in industrial processes that the team lacks
- Highest prize ceiling: hardest to replicate for teams without agent + systems engineering depth

## Consequences

- All infrastructure targets time-series + agent pipelines; CV/image code removed
- PatchTST and Chronos are the forecasting backbone (not ViT/EfficientNet)
- Evaluation metric is Sharpe ratio via backtesting, not a static ML metric
