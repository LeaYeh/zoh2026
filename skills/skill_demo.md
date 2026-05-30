# Skill: Demo & Submission

Trigger: "demo", "submit", "nextstep.csv", "completion.csv", "anomaly.csv", Gate 4 setup.

## Goal
1. `uv run python app/demo.py` → Gradio UI at port 7860, model loaded, all 3 actions working
2. `uv run python scripts/submit.py ...` → 3 valid submission CSVs generated

## Pre-demo checklist
- [ ] Best model checkpoint at `data/oof/<best_run>/model.pt`
- [ ] `data/raw/eval_input_valid.csv` exists (from organizers)
- [ ] `data/raw/eval_input_anomaly.csv` exists (from organizers)
- [ ] `.env` has valid `WANDB_API_KEY`
- [ ] Demo loads without error: `uv run python app/demo.py`
- [ ] Test all 3 Gradio actions with a sample sequence

## Generate submission files
```bash
uv run python scripts/submit.py \
  --eval-valid   data/raw/eval_input_valid.csv \
  --eval-anomaly data/raw/eval_input_anomaly.csv \
  --output-dir   submission/
```

Output: `submission/nextstep.csv`, `submission/completion.csv`, `submission/anomaly.csv`

## Validate before submitting
```bash
# Check format — first few lines of each file
head -3 submission/nextstep.csv
head -3 submission/completion.csv
head -3 submission/anomaly.csv

# Validate against organizer's eval_metrics.py (if ground truth available)
python data/raw/training_data/../eval_metrics.py \
  --task anomaly --predictions submission/anomaly.csv
```

## Demo fallback (if model fails to load)
```bash
# Re-run quick training (200 steps, ~2 min)
WANDB_MODE=disabled uv run python src/train.py configs/exp/gpt2_dummy.yaml
uv run python app/demo.py
```

## Gate 4 checklist
- [ ] Demo starts in < 2 minutes
- [ ] Predict Next Step → bar chart appears
- [ ] Complete Sequence → full sequence shown
- [ ] Check Anomaly → perplexity verdict shown
- [ ] 3 submission CSVs generated and format-checked
- [ ] WandB shows best run metrics

## Submission repo checklist (before Tally form)
- [ ] Repo is PUBLIC
- [ ] `LICENSE` file at root (MIT)
- [ ] `REPORT.md` at root (filled in with actual results)
- [ ] `requirements.txt` present
- [ ] No secrets in repo history
