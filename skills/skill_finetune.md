# Skill: Fine-tune — LoRA Standard Flow

Trigger: "fine-tune", "LoRA", Gate 1 passed, baseline established.

## Prerequisites
- [ ] Gate 1 approved (baseline CV direction confirmed)
- [ ] `data/oof/chronos_zeroshot_v0_preds.npy` exists
- [ ] CLAUDE.md shows baseline CRPS/MAE

## Step 1: Create LoRA config
Copy `configs/exp/chronos_lora_dummy.yaml` → `configs/exp/chronos_lora_v0.yaml`.

Key settings:
```yaml
run_name: chronos_lora_v0
dataset:
  name: <same as baseline>
model:
  name: chronos
  checkpoint: amazon/chronos-t5-small
  context_length: <same as baseline>
  prediction_length: <same as baseline>
  finetune:
    method: lora
    lora_r: 8          # start conservative
    lora_alpha: 32     # alpha = 4×r is a good default
    lora_dropout: 0.1
    target_modules: [q, v]   # attention layers only
training:
  max_steps: 200      # CPU: 200 steps ~5min; A100: use 1000+
  batch_size: 4       # increase on A100 (try 32-64)
  learning_rate: 5.0e-5  # lower than full FT
```

## Step 2: Run
```bash
uv run python src/train.py configs/exp/chronos_lora_v0.yaml
```

Watch for:
- Loss at step 0 (should match baseline roughly)
- Loss at step 100 (should be clearly lower)
- If loss is *increasing*: lr is too high → halve it, restart

## Step 3: A100-specific settings
Before running on A100, prepend to your script:
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```
In config: `precision: bf16-mixed` (already set in Lightning trainer).
Typical A100 speedup: 8–12× vs CPU.

## Step 4: Checkpoint verification
After run completes, confirm checkpoint loads:
```python
from peft import PeftModel
from transformers import T5ForConditionalGeneration
base = T5ForConditionalGeneration.from_pretrained("amazon/chronos-t5-small")
model = PeftModel.from_pretrained(base, "data/oof/chronos_lora_v0")
print("Checkpoint OK")
```

## Step 5: Compare vs baseline
Run Skill 06 review report. Key check:
- CRPS(LoRA) < CRPS(zero-shot)? → keep, increment v number
- CRPS higher? → do not use, investigate (lr too high? not enough steps?)

## Ablation order (change one at a time, Rule 5)
1. `lora_r`: 4 → 8 → 16 (more params, higher risk of overfit)
2. `learning_rate`: 1e-4 → 5e-5 → 1e-5
3. `max_steps`: 200 → 500 → 1000 (diminishing returns after ~500)
4. `target_modules`: [q, v] → [q, k, v, o] (more layers)
5. Checkpoint size: small → base → large (only after pipeline is stable)

## Gate 2 signal (Hour 12)
If last 3 runs improved CRPS by < 0.01 each → stop FT, move to agent layer.
