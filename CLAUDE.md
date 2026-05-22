# CLAUDE.md — Zero One Hack Vienna 2026
> This file is auto-loaded by Claude Code. All AI development behavior must follow these rules.

---

## Project Context

**Competition:** Zero One Hack · Vienna (May 29–31, 2026) — 36-hour hackathon, 64× NVIDIA A100s.
**Track:** 03 — Sybilion · *An agent that decides when to buy.*
**Goal:** Real trained model + live Gradio demo. A slide deck without a running model will not clear judging.
**Portfolio target:** AI Engineer job applications (full-stack: idea → trained model → deployed demo).

**Team:**
- Lea — data pipeline, AI agents, cloud, DevOps
- Vladimir — algorithms, system architecture, backtesting framework
- Kamila — performance optimization, parallel backtesting at scale
- Thomas — Scrum master, coordination

**Fill in after competition briefing:**
- **Data source**: [TBD — Sybilion will provide]
- **Eval metric**: [Sharpe ratio / TBD]
- **Current phase**: [Phase 1 EDA / Phase 2 Baseline / Phase 3 Agent / Phase 4 Evaluation]
- **Best backtest Sharpe**: [score + run name]
- **Gates completed**: [1 / 2 / 3 / 4]
- **Next milestone**: [specific task]
- **Known issues / open questions**: [list]

---

## Repo Structure

```
.
├── app/demo.py                        # Gradio live demo
├── configs/
│   ├── competition.yaml               # global: metric, seed, prediction_length
│   └── exp/                           # one yaml per experiment
├── data/
│   ├── raw/                           # immutable — never modify
│   ├── processed/                     # feature-engineered data
│   └── oof/                           # forecast predictions (for backtesting)
├── notebooks/
│   ├── 00_eda.ipynb
│   └── 01_chronos_baseline.ipynb
├── scripts/
│   ├── verify_setup.py
│   ├── train_patchtst.sh
│   └── run_backtest.sh
├── skills/                            # Claude Code must read before each task
├── src/
│   ├── agent/                         # LangGraph state machine + Claude tool calling
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   ├── state.py
│   │   └── tools.py
│   ├── data/loader.py                 # yfinance / competition data ingestion
│   ├── evaluation/backtesting.py      # walk-forward backtest, Sharpe, drawdown
│   ├── forecasting/
│   │   ├── chronos_baseline.py        # zero-shot baseline
│   │   └── patchtst_finetune.py       # A100 fine-tune
│   └── utils/wandb_utils.py
└── pyproject.toml                     # uv managed deps
```

---

## Tech Stack

- Python (`uv` for package management)
- PyTorch + CUDA 12.4 + BF16 (`torch.set_float32_matmul_precision("high")` on A100)
- **Forecasting**: `chronos-forecasting` (zero-shot baseline) + `PatchTST` via HuggingFace (fine-tuned on A100)
- **Agent**: `LangGraph` (state machine) + `anthropic` SDK (Claude tool calling)
- **Training loop**: PyTorch Lightning (`precision="bf16-mixed"` on A100)
- `wandb` for experiment tracking (sole source of truth — no exceptions)
- **Demo**: Gradio (`app/demo.py`)
- Data: `yfinance` for pre-hackathon practice; competition data from Sybilion

---

## Run Commands

```bash
# Verify environment
uv run python scripts/verify_setup.py

# Chronos zero-shot baseline (run this first)
uv run python -c "
from src.data.loader import fetch_and_save
from src.forecasting.chronos_baseline import load_pipeline, run_zero_shot_eval
import pandas as pd
fetch_and_save('AAPL', '2020-01-01', '2024-12-31')
df = pd.read_parquet('data/raw/AAPL.parquet')
pipeline = load_pipeline()
run_zero_shot_eval(pipeline, df, context_length=512, prediction_length=30, run_name='chronos-zeroshot-v0')
"

# Fine-tune PatchTST on A100
bash scripts/train_patchtst.sh patchtst_v0

# Run backtest
bash scripts/run_backtest.sh AAPL chronos

# Launch Gradio demo
uv run python app/demo.py
```

---

## Mandatory Rules (non-negotiable)

### Rule 1: Every forecasting run must save predictions
Every model training must output to `data/oof/`:
- `data/oof/[name]_preds.npy` — predictions on the held-out test window

These are required inputs for the backtesting framework and ensemble.

### Rule 2: Every experiment must be logged to wandb
No silent runs. No record = never happened.
wandb captures real-time metrics, GPU stats, and config → Sharpe mapping.

### Rule 3: All backtests must use the same time windows
Backtest scenarios are defined once in `configs/competition.yaml` and never changed.
Comparing models on different windows = invalid comparison.

### Rule 4: Never touch `data/raw/`
Raw data is immutable. All processing goes to `data/processed/`.

### Rule 5: Change one variable at a time
Ablations must differ by exactly one variable. Otherwise attribution is impossible.

---

## Review Gates (human must approve before proceeding)

```
Gate 1 (Hour 2):  Is the baseline pipeline working end-to-end?
  → Human confirms: Chronos zero-shot runs, agent produces BUY/SELL/HOLD, Gradio demo loads
  → Fail → fix pipeline, do not proceed to fine-tuning

Gate 2 (Hour 12): Is PatchTST fine-tune improving over Chronos baseline?
  → Human confirms: backtest Sharpe of PatchTST > Chronos on held-out window
  → Fail → investigate overfitting, try smaller model or more data

Gate 3 (Hour 24): Is the agent producing consistent decisions across scenarios?
  → Human confirms: Sharpe > 1.0 on ≥ 60% of backtest scenarios
  → Pass → lock agent logic, shift to stress-testing at scale

Gate 4 (Hour 30): Is the final demo ready?
  → Human confirms: Gradio demo runs live, best model loaded, backtest results visible
  → Pass → enter buffer time
```

Each gate requires explicit human approval. AI must not skip gates autonomously.

---

## Skills Index

Claude Code must `Read` the relevant skill file before executing any task in that category.

| Skill | Trigger | File |
|-------|---------|------|
| EDA | "analyze data", "explore data", "EDA" | `skills/01_eda.md` |
| Validation setup | "validation", "time split", "backtest setup" | `skills/02_validation.md` |
| Model training | "train", "baseline", "fine-tune", "Chronos", "PatchTST" | `skills/03_training.md` |
| Agent design | "agent", "LangGraph", "tool calling", "decision loop" | `skills/04_agent.md` |
| Evaluation | "backtest", "Sharpe", "evaluate", "ensemble", Hour 20+ | `skills/05_evaluation.md` |
| Experiment review | after every training run or backtest | `skills/06_review_report.md` |

---

## Experiment Report Format

After every training run or backtest, AI must produce this report (no exceptions):

```markdown
## Experiment: [run_name]

### Change
- Relative to previous run, only changed: [one variable]

### Results
- Sharpe ratio: x.xx
- Total return: x.xx%
- Max drawdown: x.xx%
- Win rate: x.xx%
- WandB run: [run ID or link]

### Comparison
- Previous best Sharpe: x.xx ([run_name])
- Delta: +/- x.xx

### Anomaly Checks
- [ ] Sharpe < 0? (agent losing money — do not proceed)
- [ ] Max drawdown > 30%? (risk too high — investigate)
- [ ] Backtest Sharpe improving but val loss degrading? (overfit — stop)
- [ ] Any NaN / Inf in predictions?

### Recommendation
- Keep: [Yes / No + reason]
- Next step: [specific next experiment]

### Human Decision Required
- [List explicitly, e.g. "Sharpe drops to 0.3 in 2020-03 window — COVID regime, check if model handles volatility spikes"]
```
