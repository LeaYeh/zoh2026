# Skill: Demo — Deployment and Live Demo

Trigger: "demo", "deploy", Gate 4 setup, Hour 28+.

## Goal
`uv run python app/demo.py` → Gradio UI running on port 7860, ready in < 2 minutes.
Evaluator can type a symbol, see BUY/SELL/HOLD + reasoning live.

## Pre-demo checklist (do this at Hour 28, not Hour 30)
- [ ] Best model checkpoint is in `data/oof/<best_run>/`
- [ ] `.env` has valid `ANTHROPIC_API_KEY` and `WANDB_API_KEY`
- [ ] `uv run python scripts/verify_setup.py` passes all checks
- [ ] `uv run python app/demo.py` starts without error
- [ ] Test with at least 3 symbols manually (AAPL, TSLA, MSFT or competition assets)
- [ ] Agent produces a decision (not an error) for each

## Loading the best checkpoint into the demo

Edit `app/demo.py` to load your fine-tuned model:
```python
# At top of demo.py, after imports:
from chronos import ChronosPipeline
from peft import PeftModel
import torch

BASE = "amazon/chronos-t5-small"
CKPT = "data/oof/<best_run_name>"

pipeline = ChronosPipeline.from_pretrained(BASE, dtype=torch.float32, device_map="cpu")
pipeline.model.model = PeftModel.from_pretrained(pipeline.model.model, CKPT)
pipeline.model.model.eval()
```
Then pass `pipeline` to your `forecast_node` binding.

## Gradio UI tips for live demo
- Add a "Random Example" button → pre-fill a symbol so evaluator can click one button
- Show the forecast chart (matplotlib inline) next to the decision
- Print the agent reasoning below the BUY/SELL/HOLD badge

```python
with gr.Row():
    decision_badge = gr.Label(label="Decision")
    chart = gr.Plot(label="Forecast")
reasoning_box = gr.Textbox(label="Agent Reasoning", lines=4)
```

## If the demo crashes during judging
- Have fallback mode ready: `run(symbol, price, llm_client=None)` uses rule-based decisions
- Keep a Jupyter notebook `notebooks/demo_backup.ipynb` with the same flow (manual run)
- Always have the WandB dashboard open: https://wandb.ai/lea-yeh-ml-42-vienna/zoh2026

## Gate 4 (Hour 30) checklist
- [ ] Demo starts in < 2 minutes
- [ ] Agent produces a decision for any ticker input
- [ ] WandB shows best run metrics
- [ ] Backup notebook works
- [ ] CRPS / Sharpe of best model is recorded in CLAUDE.md

After Gate 4 approval: enter buffer time. No new experiments. Only demo polish.
