# 01 · Data Strategy
**When to read**: designing fine-tuning datasets, cleaning data, filtering training samples

---

## Core Law

### Alignment cannot teach the model brand-new knowledge
Three data types ranked by effectiveness:

```
MaybeKnown (almost right, needs a different framing) → most effective ✓
Known (already gets it)                               → effective but low marginal gain
Unknown (has no idea)                                 → worst, may destroy capability ✗
```

**Pitfall**: fine-tuning on Unknown data — loss goes down but overall capability regresses.

**How to filter MaybeKnown**:
```python
results = chronos_zero_shot(dataset)

# WQL too low = Known; too high = Unknown (completely lost)
maybe_known = [
    s for s in results
    if 0.3 < wql_score(s) < 0.85  # middle band: almost right
]
```

---

## Training Data Diversity Checklist (verify before every run)

```
□ context_lengths:   [32, 64, 128, 256, 512]     varying history window lengths
□ frequencies:       [hourly, daily, weekly]      different sampling rates
□ horizons:          [1, 7, 14, 30]               different forecast horizons
□ missing_rates:     [0%, 5%, 15%]                with and without missing values
□ normalization:     [standard, minmax, none]     different normalisation schemes
□ domains:           at least 2 different source domains   prevents overfitting
```

Missing coverage on any dimension means the fine-tuned model may break
when given different input formats during demo.

---

## Data Quality Filtering Methods

### Method 1: Model Scoring (Alpagasus approach)
```python
scores = evaluator_model.score(training_samples)
high_quality = [s for s, sc in zip(samples, scores) if sc > 0.7]
# Curated subset typically outperforms training on the full dataset
```

### Method 2: Length Filtering (fast baseline)
Simply select samples with the longest targets — length is a rough proxy for quality:
```python
sorted_by_length = sorted(samples, key=lambda x: len(x['target']), reverse=True)
training_set = sorted_by_length[:top_k]
```

### Method 3: Knowledge Distillation
Use a stronger model's outputs as training targets — no human labelling needed:
```python
# Chronos-Large as teacher, training Chronos-Small
teacher_preds = chronos_large.predict(inputs, num_samples=100)
student.finetune(inputs, targets=teacher_preds)
# Often outperforms training with human-annotated data
```

---

## When to Give Up Fine-tuning and Use Prompt Engineering Instead

```
Is zero-shot performance acceptable?
  └─ Yes → add few-shot examples → good enough?
            ├─ Yes → use prompt engineering, skip fine-tuning
            └─ No  → analyse failure modes
                      ├─ Systematic bias → Model Editing (see 05-advanced-techniques.md)
                      └─ Capability gap  → Fine-tuning (see 02-training-execution.md)
```

---

## Fallback Post-processing Rules (no parameter changes)

```python
def postprocess_forecast(predictions, history):
    # Rule 1: clip predictions beyond 3σ of historical range
    mu, sigma = history.mean(), history.std()
    predictions = predictions.clip(mu - 3*sigma, mu + 3*sigma)

    # Rule 2: calibrate prediction intervals
    #         (if base model is systematically over/under-confident)
    predictions = calibrate_intervals(predictions, factor=1.1)

    # Rule 3: restore seasonality washed out by normalisation
    seasonal = extract_seasonality(history, period=7)
    predictions = predictions + seasonal[:len(predictions)]

    return predictions
```

---

## Supplement: Implementation Code and Advanced Settings

---

## 1. Time-Series CV Design (highest priority)

**Never use random split — always use TimeSeriesSplit:**

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def time_series_cv(X, y, n_splits=5, gap=0, test_size=None):
    """
    gap:       time gap between train and val sets to prevent future leakage
    test_size: fix each fold's validation set size (recommended)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        score = evaluate(model.predict(X_val), y_val)
        scores.append(score)
        print(f"Fold {fold}: {score:.4f}")
    print(f"CV: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores
```

**Walk-forward validation (closest to real deployment):**
```python
def walk_forward_validation(data, model_fn, train_size, step=1):
    """Rolling forecast: train only on past data at each step"""
    predictions = []
    for i in range(train_size, len(data), step):
        model = model_fn(data[:i])
        predictions.extend(model.predict(data[i:i+step]))
    return predictions
```

**CV design checklist:**
```
□ Gap present?             → set gap=H for horizon H to prevent future leakage
□ Fixed test_size?         → fixed is more stable than rolling, recommended
□ Enough splits?           → minimum 3, recommend 5
□ Preprocessing inside fold? → scaler.fit only on train_idx, never on full data
```

---

## 2. Data Formats: Chronos / Moirai / Lag-Llama

### Chronos Input Format
```python
import torch

# Each time series = one 1D tensor
context = torch.tensor([1.2, 1.5, 1.3, 1.8, 2.1, ...])  # shape: (T,)

# Batch input = list of tensors (lengths may differ)
batch = [
    torch.tensor([...]),   # series 1, length T1
    torch.tensor([...]),   # series 2, length T2
]

# Fine-tuning data preparation: (context, target) pairs
def make_chronos_dataset(series_list, context_len=512, horizon=24):
    samples = []
    for series in series_list:
        for i in range(len(series) - context_len - horizon):
            ctx = series[i : i + context_len]
            tgt = series[i + context_len : i + context_len + horizon]
            samples.append({"context": ctx, "target": tgt})
    return samples
```

### Moirai Input Format (multivariate)
```python
from gluonts.dataset.pandas import PandasDataset
import pandas as pd

df = pd.DataFrame({
    "item_id": ["series_1"] * T,
    "timestamp": pd.date_range("2020-01-01", periods=T, freq="H"),
    "target": values,
    "feat_dynamic_real": covariates,    # optional
})
dataset = PandasDataset(df, target="target", freq="H")
```

### HuggingFace Dataset (generic)
```python
from datasets import Dataset

def format_for_hf(context_list, target_list):
    return Dataset.from_list([
        {"context": c.tolist(), "target": t.tolist()}
        for c, t in zip(context_list, target_list)
    ])

dataset = format_for_hf(contexts, targets)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
```

---

## 3. Data Cleaning and Pre-processing

```python
import numpy as np
import pandas as pd

def clean_time_series(series: np.ndarray) -> np.ndarray:
    """Standard cleaning pipeline"""
    s = pd.Series(series)

    # 1. Fill missing values (interpolate, not mean-fill, for time series)
    s = s.interpolate(method="time")
    s = s.fillna(method="bfill").fillna(method="ffill")

    # 2. Remove outliers via IQR — clip, don't delete
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    s = s.clip(Q1 - 3 * IQR, Q3 + 3 * IQR)

    return s.values

def normalize_series(train, val=None, test=None):
    """Z-score using train mean/std; val/test follow the same transform"""
    mean, std = train.mean(), train.std() + 1e-8
    train_n = (train - mean) / std
    results = [train_n]
    if val is not None:
        results.append((val - mean) / std)
    if test is not None:
        results.append((test - mean) / std)
    return results, mean, std   # keep mean/std for inverse transform
```

---

## 4. Data Volume Decision Tree

```
Volume:
    < 500 sequences      → zero-shot (no fine-tuning)
    500 ~ 5,000          → LoRA fine-tuning (r=8, few epochs)
    5,000 ~ 50,000       → LoRA fine-tuning (r=16~32)
    > 50,000             → full fine-tuning viable

Horizon:
    short  (≤ 24 steps)  → Chronos-small or Lag-Llama
    medium (24~168)      → Chronos-base / Moirai-base
    long   (> 168)       → Moirai-large / TimesFM

Multivariate:
    has covariates       → Moirai (native multivariate)
    purely univariate    → Chronos (faster, less memory)
```

---

## 5. Adversarial Validation (diagnose train/test distribution shift)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def adversarial_validation(X_train, X_test):
    """
    AUC ≈ 0.5 → distributions are similar, CV is trustworthy
    AUC > 0.8 → large distribution gap, CV scores are unreliable
    """
    X = np.vstack([X_train, X_test])
    y = np.array([0] * len(X_train) + [1] * len(X_test))
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
    print(f"Adversarial AUC: {auc:.3f}")
    if auc > 0.8:
        print("⚠️  Train/test distribution mismatch!")
    return auc
```
