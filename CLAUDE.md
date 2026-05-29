# CLAUDE.md — Zero One Hack Vienna 2026
> This file is auto-loaded by Claude Code. All AI development behavior must follow these rules.

---

## Project Context

**Competition:** Zero One Hack · Vienna (May 29–31, 2026) — 36-hour hackathon, 64× NVIDIA A100s.
**Track:** #3 — Forecasting (Sybilion) · *An agent that decides when to buy.*
**Technical focus:** Agent design, multi-step decisions, and stress-tested evaluation at scale.
**Goal:** Real trained model + live Gradio demo. A slide deck without a running model will not clear judging.
**Portfolio target:** AI Engineer job applications (full-stack: idea → trained model → deployed demo).

**Other tracks (for context):**
- #1 Industrial (Infineon) — sequence models (LLMs, transformers, hybrids) for process modeling
- #2 Insurance (Uniqa) — prompt orchestration + conversational UI for customer guidance

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
├── README.md                          # competition day pipeline (start here)
├── app/demo.py                        # Gradio live demo
├── configs/
│   ├── competition.yaml               # global: metric, seed, prediction_length
│   └── exp/                           # one yaml per experiment
│       ├── chronos_zeroshot_v0.yaml   # zero-shot baseline
│       └── chronos_lora_v1.yaml       # LoRA fine-tune with experience replay
├── data/
│   ├── raw/                           # immutable — never modify
│   ├── processed/                     # feature-engineered data
│   └── oof/                           # forecast predictions + LoRA checkpoints
├── docs/
│   └── architecture.html              # interactive pipeline diagram
├── scripts/
│   ├── verify_setup.py                # pre-competition environment check
│   ├── download_models.py             # cache Chronos T5-small/base
│   ├── benchmark_financial.py         # CRPS+Sharpe+DirAcc vs yfinance baseline
│   ├── benchmark_kaggle_gresearch.py  # G-Research Crypto Kaggle benchmark
│   └── run_backtest.sh
├── skills/                            # Claude Code must read before each task
├── src/
│   ├── train.py                       # unified entry point (zero-shot + LoRA)
│   ├── evaluator.py                   # CRPS · MASE · coverage_80
│   ├── agent/                         # LangGraph state machine + Claude tool calling
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   ├── state.py
│   │   └── tools.py
│   ├── data/loader.py                 # yfinance / competition data ingestion
│   ├── evaluation/backtesting.py      # walk-forward backtest, Sharpe, drawdown
│   ├── forecasting/
│   │   └── chronos_baseline.py        # zero-shot inference helper
│   └── utils/wandb_utils.py           # best-run query + checkpoint loading
└── pyproject.toml                     # uv managed deps
```

---

## Tech Stack

- Python (`uv` for package management)
- PyTorch + CUDA 12.4 + BF16 (`torch.set_float32_matmul_precision("high")` on A100)
- **Forecasting**: `chronos-forecasting` — zero-shot baseline + LoRA fine-tune via `peft`
- **Fine-tuning**: LoRA r=8 [q,v], cosine LR scheduler, gradient clipping, experience replay (31 assets, 2015–2025)
- **Agent**: `LangGraph` (state machine) + `anthropic` SDK (Claude tool calling)
- `wandb` for experiment tracking (sole source of truth — no exceptions)
- **Demo**: Gradio (`app/demo.py`) — auto-loads best WandB checkpoint at startup
- Data: `yfinance` for pre-hackathon practice; competition data from Sybilion

---

## Run Commands

```bash
# Verify environment
uv run python scripts/verify_setup.py

# Run a single experiment (zero-shot or fine-tune, config-driven)
uv run python src/train.py configs/exp/<name>.yaml

# Run backtest
bash scripts/run_backtest.sh AAPL chronos

# Launch Gradio demo
uv run python app/demo.py
```

### Example experiment sequence
```bash
# 1. Zero-shot baseline (run first, always)
uv run python src/train.py configs/exp/chronos_zeroshot_v0.yaml

# 2. LoRA fine-tune (after Gate 1 approval)
uv run python src/train.py configs/exp/chronos_lora_v1.yaml

# 3. Compare in wandb → keep if CRPS improves
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

Gate 2 (Hour 12): Is LoRA fine-tune improving over Chronos baseline?
  → Human confirms: backtest Sharpe of LoRA > zero-shot on held-out window
  → Fail → investigate forgetting, adjust replay ratio or reduce LoRA rank

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
| EDA | "explore data", "EDA", data arrives | `skills/skill_eda.md` |
| Zero-shot baseline | "run baseline", "zero-shot", Gate 1 setup | `skills/skill_baseline.md` |
| LoRA fine-tune | "fine-tune", "LoRA", Gate 1 passed | `skills/skill_finetune.md` |
| Evaluation & CV | "evaluate", "metrics", "CV", post-training | `skills/skill_eval.md` |
| Decision agent | "agent", "LangGraph", Gate 2 passed | `skills/skill_decision_agent.md` |
| Demo & deploy | "demo", "deploy", Gate 4 setup | `skills/skill_demo.md` |
| Experiment review | after every training run or backtest | `skills/06_review_report.md` |
| **Time-series fine-tuning** | "fine-tune", "LoRA", "CRPS", "WQL", "Chronos", "catastrophic forgetting", "model merging", "SLERP", "TIES", "DARE", "agent design", "inference pipeline" | `skills/time-series-finetuning/` |

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
