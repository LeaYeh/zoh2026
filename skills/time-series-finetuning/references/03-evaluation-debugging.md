# 03 · Evaluation & Debugging
**When to read**: diagnosing prediction quality, analysing failure modes, designing evaluation pipelines

---

## Failure Mode Classification

### Type 1: Systematic Bias Across All Series
- Symptom: all predictions are uniformly too high or too low
- Diagnosis: base model's distributional assumption doesn't match target data
- Quick fix: IKE few-shot correction (see below, no fine-tuning needed)

```python
def detect_systematic_bias(predictions, actuals):
    errors = predictions - actuals
    mean_error = errors.mean()
    if abs(mean_error) > actuals.std() * 0.1:
        direction = "over-predicting" if mean_error > 0 else "under-predicting"
        print(f"Systematic bias: {direction} by {mean_error:.3f} — recommend IKE correction")
```

### Type 2: Specific Series Types Failing
- Symptom: sequences with strong seasonality or high volatility perform poorly
- Diagnosis: fine-tuning data lacks coverage of these patterns
- Quick fix: use zero-shot to filter MaybeKnown, supplement these pattern types (see 01-data-strategy.md)

### Type 3: Unstable Predictions
- Symptom: same series produces very different predictions across runs
- Diagnosis: inference has inherent stochasticity; single-run signal-to-noise is low
- Quick fix: Majority Vote (see below)

### Type 4: Benchmark Regression
- Symptom: target WQL improves, but ETTh1 MASE degrades > 10%
- Diagnosis: catastrophic forgetting
- Quick fix: stop training immediately, add replay data and rerun (see 02-training-execution.md)

---

## Evaluation Metric Design

```python
def full_evaluation(model, target_val, benchmark_val, baseline):
    metrics = {}

    # Primary: target task
    metrics['target_wql']  = compute_wql(model, target_val)
    metrics['target_crps'] = compute_crps(model, target_val)

    # Anti-forgetting: public benchmark
    metrics['benchmark_mase']   = compute_mase(model, benchmark_val)
    metrics['forgetting_ratio'] = metrics['benchmark_mase'] / baseline['mase']

    # Stability: std across multiple samples
    preds = [model.predict(target_val, seed=i) for i in range(5)]
    metrics['prediction_std'] = np.std(preds, axis=0).mean()

    return metrics
```

---

## Majority Vote (standard method for stability improvement)

```python
def majority_vote_forecast(model, context, n_samples=10):
    predictions = [
        model.predict(context, seed=i) for i in range(n_samples)
    ]
    # Median is more robust to outliers than mean
    return np.median(predictions, axis=0)

# Single inference → Majority Vote is almost always more accurate
# Cost: n× inference time
```

## Best-of-N + Verifier (requires a verifier)

```python
def best_of_n_forecast(model, verifier, context, n=10):
    candidates = [model.predict(context, seed=i) for i in range(n)]
    # pinball loss as verifier: lower is better
    scores = [compute_pinball_loss(context, pred) for pred in candidates]
    return candidates[np.argmin(scores)]
```

---

## IKE Systematic Bias Correction (no parameter changes)

```python
def ike_bias_correction_prompt(query, bias_direction, bias_magnitude):
    return f"""
You are a time-series forecasting expert. Adjust your prediction based on the following correction:

[Known Bias]
This series type tends to {bias_direction} by {bias_magnitude:.1%}

[Correction Example]
Input:    [100, 110, 105, 115]
Raw pred: 125 (over-predicted)
Corrected: 119 (corrected by {bias_magnitude:.1%})

[When to apply]
This correction applies to peak periods of periodic series

[When NOT to apply]
Trend-dominant series do not need this correction (keep original prediction)

Now forecast: {query}
"""
```

---

## Evaluation Process Checklist

```
□ Step 1: record zero-shot baseline (WQL + MASE)
□ Step 2: analyse failure modes (systematic? specific type? unstable?)
□ Step 3: choose the right correction strategy (IKE / add data / Majority Vote)
□ Step 4: run full evaluation (target + benchmark) at every checkpoint
□ Step 5: run Majority Vote (10 seeds) before final submission
```

---

## Supplement: Implementation Code and Advanced Settings

---

## 1. Diagnosis Workflow

**Always check training loss first, then validation loss:**
```
training loss high               → underfitting: model too small / lr wrong / not enough training
training loss low, val loss high → overfitting: regularisation / more data / dropout
val low, test high               → data leakage or CV design error
both low but metric poor         → loss and metric are misaligned (change loss function)
```

**Quick diagnosis checklist:**
```
□ Loss NaN from the start?      → lr too large, or inputs not normalised
□ Loss completely flat?          → lr too small, or trainable params = 0 (check LoRA applied)
□ Val loss oscillating wildly?   → batch too small, lr too large, or data shuffle issue
□ GPU utilisation < 50%?        → num_workers too low, or batch_size too small
□ OOM?                           → see 02-training-execution.md §9
□ CRPS worse than baseline?      → verify CV design first, then tune model
```

---

## 2. Probabilistic Forecast Metrics (essential for Track 3)

### CRPS (Continuous Ranked Probability Score)
```python
import numpy as np
from scipy import special

def crps_gaussian(y_true, mu, sigma):
    """When model outputs Gaussian distribution parameters"""
    z = (y_true - mu) / (sigma + 1e-8)
    crps = sigma * (
        z * (2 * special.ndtr(z) - 1) +
        2 * special.ndtr(z) -
        2 * np.exp(-z**2 / 2) / np.sqrt(2 * np.pi) -
        1 / np.sqrt(np.pi)
    )
    return float(np.mean(crps))

def crps_quantile_approx(y_true, quantile_levels, quantile_preds):
    """When model outputs quantiles (Chronos default)"""
    total = 0
    for q, pred in zip(quantile_levels, quantile_preds):
        error = y_true - pred
        total += np.mean(np.where(error >= 0, q * error, (q - 1) * error))
    return 2 * total / len(quantile_levels)

# Chronos usage example
quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
forecast_quantiles = pipeline.predict_quantiles(context, quantile_levels)
crps = crps_quantile_approx(y_true, quantile_levels, forecast_quantiles)
```

### WQL (Weighted Quantile Loss)
```python
def weighted_quantile_loss(y_true, quantile_forecasts, quantile_levels):
    """quantile_forecasts: shape (n_samples, n_quantiles)"""
    losses = []
    for q, forecasts in zip(quantile_levels, quantile_forecasts.T):
        errors = y_true - forecasts
        loss = np.where(errors >= 0, q * errors, (q - 1) * errors)
        losses.append(np.mean(loss))
    return float(np.mean(losses))
```

### Coverage (confidence interval calibration)
```python
def coverage_at_level(y_true, lower, upper):
    """Goal: nominal level and actual coverage should match"""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

def calibration_report(y_true, quantile_preds, quantile_levels):
    print("Level  | Expected | Actual")
    print("-------|----------|-------")
    for i, (q_lo, q_hi) in enumerate(zip(quantile_levels, reversed(quantile_levels))):
        if q_lo >= q_hi:
            break
        nominal = q_hi - q_lo
        actual = coverage_at_level(y_true, quantile_preds[i], quantile_preds[-(i+1)])
        flag = "✅" if abs(nominal - actual) < 0.05 else "⚠️"
        print(f"{nominal:.0%}    | {nominal:.3f}    | {actual:.3f} {flag}")
```

### Point Forecast Metrics
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_point(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    naive = np.mean(np.abs(np.diff(y_true)))   # lag-1 diff MAE
    mase  = mae / (naive + 1e-8)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "MASE": mase}
```

---

## 3. Catastrophic Forgetting: Diagnosis and Defence

**Symptom**: after fine-tuning, performance on original benchmarks drops significantly.

**Diagnosis:**
```python
def forgetting_check(model_before, model_after, benchmark_data):
    score_before = evaluate(model_before, benchmark_data)
    score_after  = evaluate(model_after,  benchmark_data)
    delta = score_after - score_before
    if abs(delta) > 0.05:
        print(f"⚠️  Forgetting detected: {delta:+.3f}")
    return delta
```

**Defence tools:**
```python
# 1. LoRA (most fundamental: change only a small fraction of parameters)

# 2. Small lr + warmup
training_args = TrainingArguments(
    learning_rate=2e-5,      # fine-tuning: 1e-5 ~ 5e-5, not 3e-4
    warmup_ratio=0.1,
    ...
)

# 3. Elastic Weight Consolidation (EWC)
class EWCLoss:
    """Penalise deviations from important parameters"""
    def __init__(self, model, fisher_matrix, lam=5000):
        self.params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = fisher_matrix
        self.lam = lam

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.lam * loss

# 4. Replay (keep small fraction of old data in training)
def create_replay_dataset(new_data, old_data, replay_ratio=0.1):
    n_replay = int(len(new_data) * replay_ratio)
    replay_samples = old_data.select(range(n_replay))
    from datasets import concatenate_datasets
    return concatenate_datasets([new_data, replay_samples]).shuffle(seed=42)
```

---

## 4. CV / LB Score Divergence Diagnosis

```
CV high → LB low (good locally, bad in competition):
    → train/test time ranges differ → run adversarial validation
    → CV gap design is wrong → set gap = prediction_horizon

LB high → CV low (bad locally, good in competition):
    → overfitting CV (k too small) → increase n_splits
    → CV test set too small → increase test_size

Both poor:
    → feature / preprocessing issue
    → check metric calculation with a naive baseline
```

---

## 5. Experiment Tracking (lightweight competition version)

```python
import pandas as pd
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    def __init__(self, log_file="experiments.csv"):
        self.log_file = Path(log_file)
        self.df = pd.read_csv(log_file) if self.log_file.exists() else pd.DataFrame()

    def log(self, config: dict, metrics: dict, note: str = ""):
        row = {
            "timestamp": datetime.now().strftime("%H:%M"),
            "note": note,
            **config,
            **metrics,
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.log_file, index=False)
        print(f"[{row['timestamp']}] {note} | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    def best(self, metric="val_crps", n=3, ascending=True):
        return self.df.sort_values(metric, ascending=ascending).head(n)[
            ["timestamp", "note", metric] + list(self.df.columns[:5])
        ]

# Usage
logger = ExperimentLogger("experiments.csv")
logger.log(
    config={"model": "chronos-small", "lr": 2e-5, "lora_r": 16, "epochs": 3},
    metrics={"val_crps": 0.234, "val_coverage_80": 0.812, "train_loss": 0.412},
    note="LoRA r=16 baseline"
)
```

---

## 6. Gate Checkpoint Report Template

```python
def gate_report(results: list[dict]) -> str:
    lines = ["=" * 55, "  GATE CHECKPOINT REPORT", "=" * 55]
    for i, exp in enumerate(results):
        lines += [
            f"\n[EXP {i+1}] {exp.get('note', exp.get('model', '?'))}",
            f"  Config : lr={exp.get('lr')}, lora_r={exp.get('lora_r')}, epochs={exp.get('epochs')}",
            f"  CRPS   : {exp.get('val_crps', 'N/A'):.4f}",
            f"  Cov@80 : {exp.get('val_coverage_80', 'N/A'):.3f}",
            f"  Time   : {exp.get('train_minutes', '?')} min",
        ]
    best = min(results, key=lambda x: x.get("val_crps", float("inf")))
    lines += ["", f"✅ RECOMMEND: {best.get('note', best.get('model'))} (CRPS={best['val_crps']:.4f})", "=" * 55]
    return "\n".join(lines)
```
