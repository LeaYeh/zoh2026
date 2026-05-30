# ZOH 2026 — Track #1 Industrial AI: Process Sequence Modeling

> **Team:** Lea · Vladimir · Kamila · Thomas  
> **Track:** #1 Industrial AI (Infineon) · Zero One Hack Vienna · May 29–31, 2026

---

## TL;DR

<!-- 2–3 sentences: what you built, who it's for, what it achieved -->
TODO

---

## Problem

<!-- What specifically you decided to solve and why -->
Semiconductor fabrication processes consist of 100+ ordered steps whose validity depends on complex domain-specific rules. We trained a transformer model to learn these rules from sequence data — enabling next-step prediction, sequence completion, and anomaly detection without explicit rule engineering.

---

## Approach

<!-- 3–5 bullets on architecture and key technical decisions -->
- **Model**: GPT-2 style transformer trained from scratch on custom vocabulary (~120 unique process step names). No pre-trained weights — the model learns purely from process sequences.
- **Training**: Causal language modelling (next-token prediction) on 3,000 pre-generated sequences across IC, IGBT, MOSFET families. Cosine LR + gradient clipping. PAD tokens masked from loss.
- **Task 1 (Next-Step)**: Greedy top-K logits at the last position → RANK_1…RANK_5.
- **Task 2 (Completion)**: Autoregressive greedy decoding from the cut point to EOS.
- **Task 3 (Anomaly)**: Per-sequence perplexity; median threshold separates valid from anomalous.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt
# or: uv sync

# 2. Train
uv run python src/train.py configs/exp/gpt2_finetune_v1.yaml

# 3. Generate submission files
uv run python scripts/submit.py \
  --eval-valid   data/raw/eval_input_valid.csv \
  --eval-anomaly data/raw/eval_input_anomaly.csv \
  --output-dir   submission/

# 4. Run demo
uv run python app/demo.py   # opens at http://localhost:7860
```

---

## Results

<!-- Headline metrics, baseline comparison, evidence -->
| Task | Metric | Baseline (random) | Our model |
|------|--------|-------------------|-----------|
| 1 — Next-Step | Top-1 Accuracy | ~1/120 ≈ 0.8% | TODO |
| 1 — Next-Step | MRR | — | TODO |
| 2 — Completion | Exact Match Rate | ~0% | TODO |
| 2 — Completion | Norm. Edit Distance | ~1.0 | TODO |
| 3 — Anomaly | F1 | ~0.50 | TODO |
| 3 — Anomaly | ROC-AUC | 0.50 | TODO |

WandB run: TODO

---

## What Worked / What Didn't

<!-- Honest engineering reporting -->
**Worked:**
- TODO

**Didn't work:**
- TODO

---

## What We'd Do With Another 36 Hours

- GRPO fine-tuning using `generation_rules.md` violations as negative reward signal
- Scaling ablation: n_embd=512, n_layer=8 vs. current 256/6
- Family-conditioned anomaly thresholds (per-family perplexity calibration)
- Beam search for Task 2 completion

---

## Credits & Dependencies

- **Model**: GPT-2 architecture via HuggingFace `transformers`
- **Training data**: Provided by Infineon / Lumos Data (see `data/raw/training_data/`)
- **Experiment tracking**: Weights & Biases
- **Demo**: Gradio
- **AI coding assistant**: Claude Code (Anthropic)
- Python packages: see `requirements.txt`
