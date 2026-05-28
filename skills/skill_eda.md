# Skill: EDA — First Hour with Competition Data

Trigger: competition data arrives, any "explore data" / "EDA" request.

## Goal
Produce a 1-page findings doc before touching any model. Every minute spent here saves 30 minutes of debugging later.

## Step 1: Shape and coverage (5 min)
```python
import pandas as pd
df = pd.read_csv("data/raw/<file>")   # or parquet
print(df.shape, df.dtypes)
print(df.isnull().sum())
print(df.index.min(), df.index.max())
```
- [ ] How many series? How many timesteps each?
- [ ] Is the index a datetime? If not, convert immediately.
- [ ] Are there gaps (missing dates)? Count them.
- [ ] Are series all the same length, or ragged?

## Step 2: Distribution (10 min)
```python
target = df["target"]   # or whatever the column is
print(target.describe())
target.hist(bins=50)    # look for bimodality, heavy tails
```
- [ ] Any zeros that shouldn't be zero?
- [ ] Any negative values?
- [ ] Log-transform needed? (right-skewed → yes)
- [ ] Any obvious outliers (> 5 std from mean)?

## Step 3: Temporal structure (10 min)
```python
# Plot a few representative series
for i in [0, 1, -1]:
    series = get_series(df, i)
    plt.plot(series); plt.show()
```
- [ ] Strong trend? (linear, exponential, or breaking)
- [ ] Seasonality visible? (daily/weekly/yearly)
- [ ] Regime change / structural break? (look for sudden level shifts)
- [ ] Stationarity: run ADF test on log-returns → expect stationary

## Step 4: Classify the data type
Based on what you see, map to known benchmark data:

| Characteristics | Likely similar to | Best context_length |
|---|---|---|
| Hourly, strong daily cycle | Electricity, Traffic | 168 (1 week) |
| Daily, weekly seasonality | M5, Retail | 30–90 |
| Monthly, irregular | M4 Monthly | 24–48 |
| High-frequency, noisy | Financial | 512–1024 |
| Multiple related series | M5, Traffic | depends |

**Write this classification down** — it drives context_length and patch_size decisions.

## Step 5: Quick stationarity check
```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(series.dropna())
print(f"ADF p-value: {result[1]:.4f}")
# p < 0.05 → stationary (good for forecasting)
# p > 0.05 → non-stationary, consider differencing or log-transform
```

## Output (mandatory)
Write 3–5 bullet findings into CLAUDE.md → "Known issues / open questions":
- Data type classification + benchmark analogue
- Key preprocessing needed (log, diff, clip)
- Seasonality period (for MASE denominator)
- Any anomaly or data quality issue that needs human review

Update CLAUDE.md current phase to "Phase 1 EDA complete".
