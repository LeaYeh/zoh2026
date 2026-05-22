# Skill 05 — Backtesting & Evaluation

Trigger: "backtest", "evaluate agent", "ensemble", Hour 20+.

## Primary metric: Sharpe ratio
Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
- > 1.0: acceptable
- > 2.0: strong
- < 0: agent is losing money — do not submit

## Walk-forward backtest (no look-ahead, ever)

The agent only sees data up to time `t` when deciding at time `t`.
Implementation in `src/evaluation/backtesting.py`.

```python
result = run_backtest(df, decision_fn=agent_fn, symbol="AAPL")
log_backtest(result, run_name="agent_v0_backtest")
```

## Stress-testing at scale (A100 advantage)

Run backtest across N scenarios in parallel:
- Different symbols (10-20 assets)
- Different time windows (rolling 1-year windows across history)
- Different market regimes (bull / bear / sideways — label manually from EDA)

Target: 1000+ scenario runs before submitting.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=32) as ex:
    futures = [ex.submit(run_backtest, df_slice, agent_fn, symbol) for ...]
results = [f.result() for f in futures]
```

## Gate 3 (Hour 24)
Before ensemble/fine-tuning phase:
- [ ] Sharpe > 1.0 on at least 60% of scenarios
- [ ] Max drawdown < 20% on average
- [ ] At least 5 OOF files in `data/oof/`

Report to human. Wait for Gate 3 approval.

## Ensemble of forecasting models

Stack Chronos + PatchTST predictions:
- Simple average: `(pred_chronos + pred_patchtst) / 2`
- Weighted by validation Sharpe: `w_i = sharpe_i / sum(sharpe_all)`
- Hill climbing: greedily add models that improve ensemble Sharpe

## After backtest: invoke Skill 06 (review report)
