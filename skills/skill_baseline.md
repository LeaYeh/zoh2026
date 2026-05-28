# Skill: Baseline — Zero-Shot Chronos Standard Flow

Trigger: "run baseline", "zero-shot", Gate 1 setup.

## Goal
Get a real number in wandb within 30 minutes of data arriving.
This is the number every subsequent experiment must beat.

## Step 1: Create baseline config (2 min)
Copy `configs/exp/chronos_zeroshot_dummy.yaml` → `configs/exp/chronos_zeroshot_v0.yaml`.

Fill in based on EDA output:
```yaml
run_name: chronos_zeroshot_v0
dataset:
  name: <actual dataset path or gluonts name>
model:
  name: chronos
  checkpoint: amazon/chronos-t5-small   # start small, upgrade if time
  context_length: <from EDA classification>
  prediction_length: <competition spec>
  num_samples: 20
  finetune:
    method: none
training:
  max_steps: 0
```

## Step 2: Run (< 5 min on CPU, < 1 min on A100)
```bash
uv run python src/train.py configs/exp/chronos_zeroshot_v0.yaml
```

## Step 3: Record results
Copy the wandb run ID + all metrics into CLAUDE.md:
```
Best baseline Sharpe / CRPS: <value>  (run: chronos_zeroshot_v0)
```

## Step 4: Gate 1 check (Hour 2)
Present to human:
- Baseline CRPS: X.XX
- Baseline MAE: X.XX
- Does coverage_80 ≈ 80%? If far off, quantile calibration is broken — flag immediately.

Wait for Gate 1 approval before starting fine-tuning.

## Common issues
| Symptom | Cause | Fix |
|---|---|---|
| CRPS very high (> 10×MAE) | Samples have huge variance | Reduce num_samples to 10, check normalization |
| coverage_80 < 40% | Predictions too narrow | Check prediction_length matches data |
| coverage_80 > 95% | Predictions too wide | Check if target was scaled |
| OOM on CPU | Series too long | Reduce context_length to 128 |

## Upgrade path (if time allows)
- `chronos-t5-small` → `chronos-t5-base` → `chronos-t5-large`
- Each size roughly 2× better CRPS, 4× slower
- On A100: large is feasible; on CPU: stick with small
