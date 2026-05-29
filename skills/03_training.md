# Skill 03 — Model Training

Trigger: "train model", "run baseline", "fine-tune".

## Stage 1: Chronos zero-shot baseline (do this first, always)

Goal: get a working number in < 30 minutes.

```python
from src.forecasting.chronos_baseline import load_pipeline, run_zero_shot_eval

pipeline = load_pipeline("amazon/chronos-t5-small")
results = run_zero_shot_eval(pipeline, df, context_length=512, prediction_length=30)
```

- Log to WandB as run `chronos-zeroshot-v0`
- This is your baseline — every subsequent model must beat this
- If it doesn't run in < 5 minutes, something is wrong with the setup

## Stage 2: PatchTST fine-tune (A100 required)

Config template (`configs/exp/patchtst_v0.yaml`):
```yaml
run_name: patchtst_v0
context_length: 512
prediction_length: 30
batch_size: 64           # A100 80GB can handle 128+, start conservative
max_epochs: 20
learning_rate: 1e-4
precision: bf16-mixed    # mandatory on A100
```

- Always start from a pretrained checkpoint, never train from scratch
- Use `precision: bf16-mixed` — halves memory, ~2x speed on A100
- Save checkpoint to `data/oof/[run_name].ckpt` (Rule 1)

## A100-specific settings (set at session start)

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

## WandB logging (mandatory, Rule 2)

Every run must log:
- Config (automatic via `wandb.init(config=...)`)
- `train/loss` and `val/loss` per epoch
- Final CV metric per fold
- Checkpoint artifact

## OOF generation (mandatory, Rule 1)

After training, generate OOF predictions:
```python
np.save(f"data/oof/train_oof_{run_name}.npy", oof_preds)
np.save(f"data/oof/test_preds_{run_name}.npy", test_preds)
```

## After any training run: invoke Skill 06 (review report)
