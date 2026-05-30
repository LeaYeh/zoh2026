# ZOH 2026 — Track #1 Industrial AI: Process Sequence Modeling

**Competition**: Zero One Hack · Vienna · May 29–31, 2026 · 36 hours · 64× NVIDIA A100  
**Track**: #1 — Industrial AI (Infineon) · *Models that learn how processes unfold*  
**Technical focus**: Train, fine-tune, or build sequence models (LLMs, transformers, hybrids)  
**Goal**: Real trained model + live Gradio demo. A slide deck without a running model will not clear judging.

---


## Architecture

```
Training Data (CSV)
  IC / IGBT / MOSFET  ── 1 000 sequences each
  generate_sequences.py ── unlimited augmentation
       │
       ▼
  ProcessStepTokenizer
  build vocab from all unique step names (~50–100)
  encode: step name → int  /  decode: int → step name
       │
       ▼
  GPT-2  (from scratch, custom vocab)
  n_embd=256 · n_layer=6 · n_head=8 · ~2.5M params
  n_positions=256  ·  causal LM objective
       │
       ▼
  Training Loop
  AdamW + cosine LR (warmup=300, decay to 1%)
  grad clip=1.0  ·  labels=-100 on PAD  ·  BF16 on A100
       │
       ├─ WandB  (loss / top-1 / top-3 / mrr, every step)
       │
       ▼
  Checkpoint  →  data/oof/<run_name>/model.pt + tokenizer.json
       │
       ├─ Task 1: Next-Step Prediction   → Top-1/3/5, MRR
       ├─ Task 2: Sequence Completion    → Exact Match, Edit Dist, Token Acc
       └─ Task 3: Anomaly Detection      → Perplexity → F1, ROC-AUC
            └─ Gradio Demo (port 7860, auto-loads best WandB checkpoint)
```

See `docs/architecture.html` for the full interactive diagram.

---

## Running the Baseline

### 1. Smoke test — verify the pipeline locally (no WandB, ~1 min)

```bash
WANDB_MODE=disabled uv run python src/train.py configs/exp/gpt2_dummy.yaml
```

Uses a synthetic dummy dataset (300 sequences, 20 vocab). Eval runs every 50 steps.  
Expected terminal output:

```
[train] 270 train / 30 val sequences
[train] vocab size: 20
[train] parameters: 804,884
[eval] step=50   top1=0.xxxx  top3=0.xxxx  mrr=0.xxxx
[eval] step=100  top1=0.xxxx  top3=0.xxxx  mrr=0.xxxx
...
[done] best_top1=0.xxxx
```

### 2. Baseline training on competition data (logs to WandB)

```bash
uv run python src/train.py configs/exp/gpt2_finetune_v1.yaml
```

Trains on `data/raw/training_data/` (IC / IGBT / MOSFET CSVs). Eval runs every 300 steps.  
Metrics printed to terminal **and** logged to [wandb.ai](https://wandb.ai) project `zoh2026`:

| Metric | Logged key | When |
|--------|------------|------|
| Training loss | `train/loss` | every 10 steps |
| Learning rate | `train/lr` | every 10 steps |
| Top-1 Accuracy | `eval/top1` | every 300 steps |
| Top-3 Accuracy | `eval/top3` | every 300 steps |
| Top-5 Accuracy | `eval/top5` | every 300 steps |
| MRR | `eval/mrr` | every 300 steps |

Best checkpoint (by Top-1) is saved to `data/oof/<run_name>/model.pt`.

### 3. Full evaluation (after organiser eval CSVs arrive)

```bash
uv run python -c "
from src.evaluator import run_full_eval
from src.utils.wandb_utils import load_best_model
model, tok, _, _ = load_best_model()
results = run_full_eval(model, tok, 'data/raw/eval_input_valid.csv', 'data/raw/eval_input_anomaly.csv')
import json; print(json.dumps(results, indent=2))
"
```

### 4. Launch Gradio demo

```bash
uv run python app/demo.py
```

---

## Config Checklist (fill after data arrives)

| Field | File | What to fill |
|-------|------|-------------|
| `dataset.train_path` | `configs/exp/gpt2_finetune_v1.yaml` | path to training CSV folder |
| `dataset.sequence_col` | same | column containing sequence ID (default: SEQUENCE_ID) |
| `dataset.step_col` | same | column containing step name (default: STEP) |
| `dataset.product_families` | same | list of family names matching file names |

---

## Mandatory Rules

| Rule | Detail |
|------|--------|
| **Every run → WandB** | No silent runs. `wandb.init()` called in `src/train.py`. |
| **Every run → data/oof/** | Checkpoint saved on best Top-1. |
| **One variable at a time** | Ablations must differ by exactly one variable. |
| **data/raw/ is immutable** | Raw data never modified. |
| **Gates need human sign-off** | AI must not skip gates autonomously. |

---

## Submission Tasks Summary

| # | Task | Metric(s) | Eval File |
|---|------|-----------|-----------|
| 1 | Next-Step Prediction | Top-1/3/5, MRR | `eval_input_valid.csv` |
| 2 | Sequence Completion | Exact Match, Norm. Edit Dist, Token Acc | `eval_input_valid.csv` |
| 3 | Anomaly Detection | Binary Acc, Precision, Recall, F1, ROC-AUC | `eval_input_anomaly.csv` |
| 4 | OOD Generalization | ID→OOD drop (organiser-only, post-submission) | hidden |

---

## Team

| Member | Role |
|--------|------|
| Lea | Data pipeline, model training, cloud, DevOps |
| Vladimir | Algorithms, system architecture, evaluation framework |
| Kamila | Performance optimisation, scaling experiments |
| Thomas | Scrum master, coordination |

**On-site mentor**: Simeon (Infineon)
