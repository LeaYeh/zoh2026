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

## Key Commands

```bash
# Environment check
uv run python scripts/verify_setup.py

# Smoke-test training (200 steps, no WandB)
WANDB_MODE=disabled uv run python src/train.py configs/exp/gpt2_dummy.yaml

# Full training on competition data
uv run python src/train.py configs/exp/gpt2_finetune_v1.yaml

# Full evaluation (all 3 tasks)
uv run python -c "
from src.evaluator import run_full_eval
from src.utils.wandb_utils import load_best_model
model, tok, _, _ = load_best_model()
results = run_full_eval(model, tok, 'data/raw/eval_input_valid.csv', 'data/raw/eval_input_anomaly.csv')
import json; print(json.dumps(results, indent=2))
"

# Launch demo
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
