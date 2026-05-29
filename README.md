# ZOH 2026 — Track 03: Trading Decision Agent

**Competition**: Zero One Hack · Vienna · May 29–31, 2026 · 36 hours · 64× NVIDIA A100  
**Track**: 03 — Sybilion · *An agent that decides when to buy*  
**Goal**: Real trained model + live Gradio demo. A slide deck without a running model will not clear judging.

---

## Competition Day Pipeline

This is the authoritative sequence. Do not skip steps or proceed past a Gate without human sign-off.

### Hour 0–2 · Data Arrival → Gate 1

```bash
# 1. Verify environment is healthy
uv run python scripts/verify_setup.py

# 2. Fill in competition data format in the config
#    Edit configs/exp/chronos_zeroshot_v0.yaml:
#      dataset.path           → path to competition CSV
#      dataset.price_col      → column name for prices
#      dataset.prediction_length → horizon (from briefing)

# 3. Run zero-shot baseline immediately
uv run python src/train.py configs/exp/chronos_zeroshot_v0.yaml

# 4. Launch demo to confirm Gate 1
uv run python app/demo.py
```

**Gate 1 pass criteria** (human must confirm before continuing):
- Zero-shot Chronos runs end-to-end without error
- Agent produces BUY / SELL / HOLD output
- Gradio demo loads at http://localhost:7860

---

### Hour 2–12 · Fine-tuning → Gate 2

```bash
# 5. Update LoRA config with competition data format
#    Edit configs/exp/chronos_lora_v1.yaml (same fields as above)

# 6. Run LoRA fine-tune (uses experience replay automatically)
uv run python src/train.py configs/exp/chronos_lora_v1.yaml

# 7. Compare results in WandB — check eval/crps improved
#    Also run backtest to verify Sharpe improvement
bash scripts/run_backtest.sh <symbol> chronos_lora_v1
```

**Gate 2 pass criteria**: LoRA CRPS < zero-shot CRPS on held-out window.  
**Fail** → check `skills/time-series-finetuning/references/03-evaluation-debugging.md` for diagnosis.

---

### Hour 12–24 · Scale + Stress Test → Gate 3

```bash
# 8. T5-base ablation (quick win — swap checkpoint)
#    Edit configs/exp/chronos_lora_v1.yaml:
#      model.checkpoint: amazon/chronos-t5-base

uv run python src/train.py configs/exp/chronos_lora_v1.yaml

# 9. Ensemble ZS + LoRA (2-line win)
#    samples = np.concatenate([samples_zs, samples_ft], axis=0)

# 10. Run full backtest across all available symbols/windows
bash scripts/run_backtest.sh ALL chronos_lora_v1
```

**Gate 3 pass criteria**: Sharpe > 1.0 on ≥ 60% of backtest scenarios.

---

### Hour 24–30 · Demo Polish → Gate 4

```bash
# 11. Confirm demo auto-loads best WandB checkpoint
uv run python app/demo.py
# Should print: [demo] Loaded checkpoint: <run_name>  (CRPS=x.xxxx)

# 12. Final smoke test
uv run python scripts/verify_setup.py
```

**Gate 4 pass criteria**: live Gradio demo runs, best model loaded, backtest results visible.

---

### Fallback (pipeline broken / < 4 hours remaining)

```bash
# Zero-shot + post-processing rules — always works, always demonstrable
uv run python app/demo.py   # auto-falls back to zero-shot if no checkpoint
```

---

## Architecture

```
Data (yfinance / competition CSV)
  └─ Experience Replay (31 assets, 2015–2025)
       └─ Chronos T5-small (zero-shot CRPS=0.054 Sharpe=1.25)
            └─ LoRA Fine-tune (r=8, cosine LR, grad clip → CRPS=0.053 Sharpe=1.50)
                 └─ LangGraph Agent
                      ├─ fetch_context  → yfinance 3yr daily prices
                      ├─ forecast_node  → Chronos predict() → median/q10/q90
                      ├─ risk_node      → expected return + vol regime + confidence
                      └─ llm_decision   → Claude sonnet-4-6 → BUY/SELL/HOLD
                           └─ Gradio Demo (port 7860, WandB best-run auto-load)
```

See `docs/architecture.html` for the full interactive diagram.

---

## Key Commands

```bash
# Environment check
uv run python scripts/verify_setup.py

# Download models (pre-competition)
uv run python scripts/download_models.py

# Zero-shot baseline
uv run python src/train.py configs/exp/chronos_zeroshot_v0.yaml

# LoRA fine-tune
uv run python src/train.py configs/exp/chronos_lora_v1.yaml

# Backtest
bash scripts/run_backtest.sh AAPL chronos_lora_v1

# Benchmark vs financial leaderboard
uv run python scripts/benchmark_financial.py
uv run python scripts/benchmark_kaggle_gresearch.py

# Launch demo
uv run python app/demo.py
```

---

## Config Checklist (fill before competition starts)

After the briefing, update these fields in **both** config files:

| Field | File | What to fill |
|-------|------|-------------|
| `dataset.path` | `configs/exp/chronos_*.yaml` | path to competition CSV |
| `dataset.price_col` | same | column name containing prices |
| `dataset.timestamp_col` | same | timestamp column (if present) |
| `dataset.series_col` | same | symbol/id column for multi-series (null if single) |
| `dataset.prediction_length` | same | forecast horizon from briefing |
| `model.prediction_length` | same | same value |

---

## Mandatory Rules

| Rule | Detail |
|------|--------|
| **Every run → WandB** | No silent runs. `wandb.init()` is called automatically. |
| **Every run → data/oof/** | `[run_name]_preds.npy` saved automatically by `src/train.py`. |
| **Walk-forward CV only** | No random K-fold. Windows defined in `configs/competition.yaml`. |
| **One variable at a time** | Ablations must differ by exactly one variable. |
| **data/raw/ is immutable** | Raw data is never modified. Processing goes to `data/processed/`. |
| **Gates need human sign-off** | AI must not skip gates autonomously. |

---

## Benchmark Results (pre-competition, yfinance data)

| Model | CRPS | Sharpe | Direction Acc | Return R² |
|-------|------|--------|---------------|-----------|
| Seasonal Naive | ~0.098 | ~0.0 | ~50% | ~0.000 |
| Chronos ZS | 0.054 | +1.25 | 61% | 0.48 |
| Chronos LoRA (200 steps) | 0.053 | +1.50 | 59% | 0.45 |
| M4 competition winner (WQL ref) | 0.056 | — | — | — |

---

## Team

| Member | Role |
|--------|------|
| Lea | Data pipeline, AI agents, cloud, DevOps |
| Vladimir | Algorithms, system architecture, backtesting |
| Kamila | Performance optimisation, parallel backtesting |
| Thomas | Scrum master, coordination |
