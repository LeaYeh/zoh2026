# Skill 01 — EDA (Time-Series Financial Data)

Trigger: any "explore data", "analyze data", "EDA" request.

## Steps (do not skip)

### 1. Shape and coverage
- [ ] Print shape, date range, symbols covered
- [ ] Check for gaps in the time index (missing trading days)
- [ ] Check `null` counts per column

### 2. Price distribution
- [ ] Plot close price over full history — identify regime changes (structural breaks)
- [ ] Plot log returns distribution — check for fat tails, skewness
- [ ] Compute rolling mean and std (20d, 60d) — flag periods of high volatility

### 3. Stationarity
- [ ] Run ADF test on raw close prices (expect non-stationary)
- [ ] Run ADF test on log returns (expect stationary)
- [ ] If non-stationary returns: flag as anomaly, investigate splits/dividends

### 4. Autocorrelation
- [ ] Plot ACF/PACF of log returns (lags 1-30)
- [ ] Check if any lag is significant — informs context window choice for PatchTST

### 5. Seasonality and calendar effects
- [ ] Day-of-week effect: average return by weekday
- [ ] Month effect: average return by month
- [ ] Note: financial time series often have weak seasonality — don't over-engineer

### 6. Multi-asset correlation (if multiple symbols)
- [ ] Correlation matrix of returns
- [ ] Flag pairs with |r| > 0.8 — potential redundancy in ensemble

### 7. Output
- Save EDA summary notebook to `notebooks/00_eda.ipynb`
- Write 3-5 bullet findings to the "Known issues / open questions" field in CLAUDE.md
