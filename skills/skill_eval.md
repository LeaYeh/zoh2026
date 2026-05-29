# Skill: Evaluation — Metrics and CV Logic

Trigger: "evaluate", "metrics", "CV setup", any post-training review.

## Primary metric: CRPS (Continuous Ranked Probability Score)
Lower is better. Measures probabilistic forecast quality.
`src/evaluator.py` computes it automatically after every run.

Secondary metrics in order of importance:
1. **CRPS** — probabilistic accuracy (primary)
2. **MASE** — point forecast vs naive baseline (> 1.0 = worse than naive, unacceptable)
3. **Coverage 80%** — should be 75–85%. Far outside = calibration failure.
4. **MAE / RMSE** — point accuracy (less important than CRPS for probabilistic models)

## Using evaluator.py
```python
from src.evaluator import evaluate, naive_scale_from_series

# predictions: (n_series, n_samples, pred_len)
# ground_truth: (n_series, pred_len)
# naive_scale: (n_series,) — mean |y_t - y_{t-1}| on training window

metrics = evaluate(predictions, ground_truth, naive_scale)
print(metrics.to_dict())
```

## Time-series CV: rolling origin (no random splits, ever)
```python
from src.evaluator import rolling_origin_splits

splits = rolling_origin_splits(n=len(series), pred_len=24, n_windows=5)
for train_idx, test_idx in splits:
    train = series[train_idx]
    test = series[test_idx]
    # train model on `train`, predict `test`
```

**Why not K-fold:** random splits let the model see future data during training → results are invalid. Every paper that uses random CV on time series is wrong.

## Anomaly checklist (run after every experiment)
- [ ] MASE > 1.0 → model is worse than naive. Do not submit. Investigate.
- [ ] coverage_80 < 50% → predictions systematically too narrow. Check scaling.
- [ ] coverage_80 > 95% → predictions too wide. Model learned uncertainty > signal.
- [ ] CRPS improving but MAE degrading → probabilistic calibration traded off against point accuracy. Usually fine; check if task cares about point or interval.
- [ ] Any NaN in predictions → data has inf/nan. Check `data/raw/` for bad values.

## MASE denominator (naive scale)
```python
# For non-seasonal data (financial):
scale = np.mean(np.abs(np.diff(train_series)))

# For seasonal data (electricity, freq=24 for daily):
scale = np.mean(np.abs(train_series[24:] - train_series[:-24]))
```
Set `freq` in `naive_scale_from_series(series, freq=<seasonality>)`.

## Pinball loss interpretation
- pinball_10 = loss when you bet "actual will be above q10"
- pinball_50 ≈ MAE/2 (for symmetric distributions)
- pinball_90 = loss when you bet "actual will be below q90"
Lower = better calibrated quantile forecasts.

## Comparing two runs
Use this template (from Skill 06):
```
run_A CRPS: 2.58  coverage_80: 46.9%
run_B CRPS: 2.51  coverage_80: 48.9%
Delta CRPS: -0.07 (-2.7%) ✓ keep run_B
```
A delta of < 0.5% CRPS improvement = noise, not signal.
