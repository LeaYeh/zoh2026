# Skill 02 — Validation Framework (Time Series)

Trigger: "set up CV", "fold", "validation framework".

## Critical rule: never use random K-Fold on time series
Random splits cause look-ahead bias — the model sees future data during training.
Always use walk-forward (expanding window) or blocked time splits.

## Walk-forward split (recommended)

```
|--train--|--val--|
          |--train--|--val--|
                    |--train--|--val--|
```

- `n_folds = 5` (set in `configs/competition.yaml`)
- Each fold's val set is strictly after the train set
- Gap between train end and val start: 1 prediction_length (avoids leakage from autocorrelation)

## Implementation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=prediction_length)
for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
    ...
```

Save fold indices to `data/folds.pkl` once — all models must use the same folds (Rule 3).

## Sanity checks before proceeding
- [ ] Val set never overlaps with train set for any fold
- [ ] Fold sizes are roughly equal
- [ ] Log fold date ranges to WandB config at run start
- [ ] CV metric variance across folds < 20% of mean (high variance = unstable model)

## Gate 1 (Hour 2)
Report CV metric for the Chronos baseline.
Wait for human to confirm CV direction matches LB before continuing.
