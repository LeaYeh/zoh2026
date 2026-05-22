# Skill 06 — Experiment Review Report

Trigger: after EVERY training run or backtest. No exceptions.

## Required output format

```markdown
## Experiment: [run_name]

### Change
- Relative to previous run, only changed: [one variable — if more than one, flag it]

### Results
- Metric: [value]  (e.g. Sharpe: 1.42, MAE: 0.031)
- Fold breakdown (if CV): Fold 1: x, Fold 2: x, ..., mean ± std

### Comparison
- Previous best: [value + run_name]
- Delta: +/- [value]
- WandB run: [link or run ID]

### Anomaly Checks
- [ ] Fold variance too high? (CV std > 20% of mean = flag)
- [ ] Any fold with NaN / Inf loss?
- [ ] Backtest Sharpe < 0? (agent losing money — stop)
- [ ] Max drawdown > 30%? (risk too high — investigate)
- [ ] CV improving but backtest degrading? (overfit to val — stop)

### Recommendation
- Keep this run: [Yes / No + one-line reason]
- Next experiment: [specific change to try]

### Human Decision Required
- [List any open questions explicitly, e.g.:
  "Fold 3 Sharpe is 0.2 vs 1.8 average — suspected regime anomaly in 2020-03, check manually"]
```

## Rules
- Never omit the anomaly checks section
- Never omit the "Human Decision Required" section (write "None" if clean)
- If delta is negative: explicitly state whether to keep or discard
- WandB run link must be included so humans can verify
