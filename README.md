# ZOH 2026 — Track #1 Industrial AI: Process Sequence Modeling

**Competition**: Zero One Hack · Vienna · May 29–31, 2026 · 36 hours · 64× NVIDIA A100  
**Track**: #1 — Industrial AI (Infineon) · *Models that learn how processes unfold*  
**Technical focus**: Train, fine-tune, or build sequence models (LLMs, transformers, hybrids)  
**Goal**: Real trained model + live Gradio demo. A slide deck without a running model will not clear judging.

---

## Competition Day Pipeline

This is the authoritative sequence. Do not skip steps or proceed past a Gate without human sign-off.

### Hour 0–2 · Data Arrival → Gate 1

```bash
# 1. Verify environment
uv run python scripts/verify_setup.py

# 2. Inspect data — confirm column names and sequence format
#    Expected: CSV with SEQUENCE_ID, STEP columns (long format)
#    Data lives in: data/raw/training_data/

# 3. Quick smoke-test: verify tokenizer builds correctly
uv run python -c "
from src.data.process_loader import load_sequences, ProcessStepTokenizer
seqs = load_sequences('data/raw/training_data/')
tok = ProcessStepTokenizer.build(seqs)
print(f'vocab: {tok.vocab_size}, sequences: {len(seqs)}')
"

# 4. Run dummy training to confirm pipeline end-to-end
WANDB_MODE=disabled uv run python src/train.py configs/exp/gpt2_dummy.yaml

# 5. Launch demo
uv run python app/demo.py
```

**Gate 1 pass criteria** (human must confirm):
- Tokenizer builds from competition CSV without error
- Training loop runs end-to-end (loss decreasing)
- Gradio demo loads at http://localhost:7860

---

### Hour 2–12 · Full Training → Gate 2

```bash
# 6. Update gpt2_finetune_v1.yaml if needed:
#    dataset.train_path        → data/raw/training_data/
#    dataset.sequence_col      → actual column name (default: SEQUENCE_ID)
#    dataset.step_col          → actual column name (default: STEP)
#    dataset.product_families  → [IC, IGBT, MOSFET] or adjust to match

# 7. Launch full training (3 000 steps, A100)
uv run python src/train.py configs/exp/gpt2_finetune_v1.yaml

# 8. Monitor in WandB — watch eval/top1, eval/top3, eval/mrr
#    Target: Top-1 > 0.5 by step 1000
```

**Gate 2 pass criteria**: Top-1 accuracy improving; checkpoint saved at `data/oof/gpt2_finetune_v1/`.

---

### Hour 12–24 · Eval + Scaling → Gate 3

```bash
# 9. Run full evaluation on competition eval set
uv run python -c "
from src.evaluator import run_full_eval
from src.utils.wandb_utils import load_best_model
model, tok, run, score = load_best_model()
results = run_full_eval(model, tok,
    'data/raw/eval_input_valid.csv',
    'data/raw/eval_input_anomaly.csv')
import json; print(json.dumps(results, indent=2))
"

# 10. Scaling ablation — larger model
#     In gpt2_finetune_v1.yaml: n_embd=512, n_layer=8  → run_name: gpt2_large_v1
uv run python src/train.py configs/exp/gpt2_finetune_v1.yaml

# 11. Compare baseline vs trained vs large in WandB
```

**Gate 3 pass criteria**: Task 1 Top-1 ≥ 0.5, Task 3 F1 ≥ 0.6 on eval set.

---

### Hour 24–30 · Demo Polish → Gate 4

```bash
# 12. Confirm demo auto-loads best checkpoint
uv run python app/demo.py
# Should print: [wandb_utils] Loaded model: <run_name>  (eval/top1=x.xxxx)

# 13. Test all three Gradio actions manually:
#     - "Predict Next Step" → top-5 bar chart shown
#     - "Complete Sequence" → generated completion shown
#     - "Check Anomaly" → perplexity score + verdict shown

# 14. Final smoke test
uv run python scripts/verify_setup.py
```

**Gate 4 pass criteria**: live demo runs, all three tasks working, best model loaded.

---

### Fallback (pipeline broken / < 4 hours remaining)

```bash
# Demo still works with a partially-trained or dummy checkpoint
# If no WandB checkpoint: re-run gpt2_dummy.yaml (200 steps, ~2 min)
WANDB_MODE=disabled uv run python src/train.py configs/exp/gpt2_dummy.yaml
uv run python app/demo.py
```

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
