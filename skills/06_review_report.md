# Skill 06 — Experiment Review Report

Trigger: after EVERY training run. No exceptions.

## Required output format

```markdown
## Experiment: [run_name]

### Change
- Relative to previous run, only changed: [one variable — if more than one, flag it]

### Results
- Top-1 Accuracy: x.xxxx
- Top-3 Accuracy: x.xxxx
- MRR: x.xxxx
- Edit Distance (val): x.xxxx  (lower = better)
- Anomaly F1 (val): x.xxxx
- WandB run: [run ID or link]

### Comparison
- Previous best Top-1: x.xxxx ([run_name])
- Delta: +/- x.xxxx

### Anomaly Checks
- [ ] Top-1 < 0.10 after 500 steps? (model not learning — check attention_mask)
- [ ] Edit Distance > 0.80? (completions almost random — check EOS handling)
- [ ] F1 < 0.52? (anomaly barely above random — check threshold calibration)
- [ ] Top-1 improving but F1 degrading? (needs GRPO)
- [ ] Any NaN / Inf in predictions?
- [ ] Val Top-1 plateauing while train loss still dropping? (overfitting)

### Recommendation
- Keep: [Yes / No + one-line reason]
- Next step: [specific next experiment]

### Human Decision Required
- [List explicitly, e.g. "Top-1=0.38 — just below 0.40 gate threshold, evaluate if more steps help"]
```

## Rules
- Never omit the anomaly checks section
- Never omit the "Human Decision Required" section (write "None" if clean)
- If delta is negative: explicitly state whether to keep or discard
- WandB run link must be included so humans can verify
- If multiple variables changed: flag it and treat results as uninterpretable
