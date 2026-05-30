# CLAUDE.md — Zero One Hack Vienna 2026
> This file is auto-loaded by Claude Code. All AI development behavior must follow these rules.

---

## Project Context

**Competition:** Zero One Hack · Vienna (May 29–31, 2026) — 36-hour hackathon, 64× NVIDIA A100s.
**Track:** #1 — Industrial AI (Infineon) · *Models that learn how processes unfold.*
**Technical focus:** Train, fine-tune, or build sequence models (LLMs, transformers, hybrids).
**Goal:** Real trained model + live Gradio demo. A slide deck without a running model will not clear judging.
**Portfolio target:** AI Engineer job applications (full-stack: idea → trained model → deployed demo).

**Other tracks (for context):**
- #2 Insurance (Uniqa) — prompt orchestration + conversational UI for customer guidance
- #3 Forecasting (Sybilion) — agent design, multi-step decisions, stress-tested evaluation

**Team:**
- Lea — data pipeline, model training, cloud, DevOps
- Vladimir — algorithms, system architecture, evaluation framework
- Kamila — performance optimisation, parallel/scaling experiments
- Thomas — Scrum master, coordination

**On-site mentor:** Simeon (Infineon)

**Competition state:**
- **Data**: `data/raw/training_data/` — IC / IGBT / MOSFET variant CSVs (1K seqs each)
- **Eval metric**: Top-1 Accuracy (Task 1 primary); F1 (Task 3)
- **Current phase**: Phase 1 — Baseline training
- **Best Top-1**: [fill after first run]
- **Gates completed**: [0 / 4]
- **Next milestone**: Gate 1 — GPT-2 baseline Top-1 ≥ 0.40
- **Known issues**: eval_input_valid.csv and eval_input_anomaly.csv not yet distributed by organizers

---

## Repo Structure

```
.
├── README.md                          # project overview and key commands
├── REPORT.md                          # competition submission report (fill results before submit)
├── LICENSE                            # MIT — required for submission
├── requirements.txt                   # pip-installable deps (from uv lockfile)
├── app/demo.py                        # Gradio live demo (3 task buttons)
├── configs/
│   ├── competition.yaml               # global: metric, seed, families
│   └── exp/
│       ├── gpt2_dummy.yaml            # 200-step smoke test (no WandB)
│       └── gpt2_finetune_v1.yaml      # full 3000-step training config
├── data/
│   ├── raw/training_data/             # competition CSVs — immutable, never modify
│   └── oof/                           # model checkpoints (model.pt + tokenizer.json)
├── docs/
│   ├── architecture.html              # interactive pipeline diagram
│   └── track_briefing.md              # official Track 1 briefing (EN)
├── scripts/
│   ├── generate_sequences.py          # organizer augmentation + validation tool
│   └── submit.py                      # generates nextstep.csv / completion.csv / anomaly.csv
├── skills/                            # Claude Code must read before each task
│   └── process-sequence-modeling/     # full knowledge base (李宏毅 ML 2025 + Track 1)
├── src/
│   ├── train.py                       # unified training entry point
│   ├── evaluator.py                   # internal val-set evaluation (Top-1, MRR, F1)
│   ├── data/process_loader.py         # ProcessStepTokenizer + CSV loader
│   ├── models/process_lm.py           # GPT-2 wrapper: predict/complete/anomaly_score
│   ├── evaluation/process_metrics.py  # Top-K, MRR, edit distance, F1, ROC-AUC
│   └── utils/wandb_utils.py           # load_best_model() + WandB logging
└── pyproject.toml                     # uv managed deps
```

---

## Tech Stack

- Python (`uv` for package management)
- PyTorch + CUDA 12.4 + BF16 (`torch.set_float32_matmul_precision("high")` on A100)
- **Model**: GPT-2 style transformer trained from scratch — custom vocabulary of process step names (~120 unique steps)
- **Training**: Causal LM (next-token prediction), cosine LR + warmup, gradient clipping, attention mask on PAD
- **Evaluation**: Top-1/3/5 Accuracy, MRR (Task 1); Edit Distance, Token Accuracy (Task 2); F1, ROC-AUC (Task 3)
- `wandb` for experiment tracking (sole source of truth — no exceptions)
- **Demo**: Gradio (`app/demo.py`) — predict next step / complete sequence / anomaly check
- Data: `data/raw/training_data/` (IC/IGBT/MOSFET variant CSVs from Infineon)

---

## Run Commands

```bash
# Smoke test (no WandB, fast)
WANDB_MODE=disabled uv run python src/train.py configs/exp/gpt2_dummy.yaml

# Full training
uv run python src/train.py configs/exp/gpt2_finetune_v1.yaml

# Generate augmented data
uv run python scripts/generate_sequences.py \
  --family mosfet --count 2000 \
  --output data/raw/training_data/MOSFET_extra.csv

# Validate generated sequences against process rules
uv run python scripts/generate_sequences.py \
  --validate data/raw/training_data/MOSFET_extra.csv --family mosfet

# Generate submission files (after eval CSVs arrive from organizers)
uv run python scripts/submit.py \
  --eval-valid   data/raw/eval_input_valid.csv \
  --eval-anomaly data/raw/eval_input_anomaly.csv \
  --output-dir   submission/

# Launch Gradio demo
uv run python app/demo.py
```

---

## Mandatory Rules (non-negotiable)

### Rule 1: Every run must be logged to WandB
No silent runs. `wandb.init()` is called in `src/train.py` automatically.
`WANDB_MODE=disabled` is allowed only for smoke tests.

### Rule 2: One token = one complete step name
Never split "DEPOSIT GATE OXIDE" into sub-words. The entire string is one vocabulary item.
`ProcessStepTokenizer` handles this — do not replace it with BPE or character-level tokenization.

### Rule 3: PAD positions must always be masked
`labels=-100` on padding, `attention_mask=0` on padding.
Violating this makes the model waste capacity and produces misleadingly low loss.

### Rule 4: Never touch `data/raw/`
Raw data is immutable. Augmented data goes to `data/raw/training_data/` with a distinct filename.
Never overwrite the organizer-provided CSVs.

### Rule 5: Change one variable at a time
Ablations must differ by exactly one variable. Otherwise attribution is impossible.

### Rule 6: Validate sequences before submitting
Always run `scripts/generate_sequences.py --validate` on completion outputs before handing to organizers.
Rule violations in submission = zero score for that example.

---

## Review Gates (human must approve before proceeding)

```
Gate 1: Is the baseline pipeline working end-to-end?
  → Human confirms: GPT-2 trains, eval/top1 > 0 after 200 steps, demo loads
  → Fail → fix pipeline, do not proceed

Gate 2: Is Top-1 ≥ 0.40 on val set?
  → Human confirms: gpt2_finetune_v1 reaches Top-1 ≥ 0.40 by step 1000
  → Fail → lower LR, generate more data, or try larger model

Gate 3: Are all 3 submission tasks producing valid outputs?
  → Human confirms: nextstep.csv / completion.csv / anomaly.csv generated and format-correct
  → Fail → fix scripts/submit.py, re-run

Gate 4: Is the final demo ready?
  → Human confirms: Gradio demo runs, all 3 actions work, submission files validated
  → Pass → enter buffer time, no new experiments
```

Each gate requires explicit human approval. AI must not skip gates autonomously.

---

## Skills Index

Claude Code must `Read` the relevant skill file before executing any task in that category.

| Skill | Trigger | File |
|-------|---------|------|
| EDA | "explore data", "EDA", data arrives | `skills/skill_eda.md` |
| Baseline training | "train", "baseline", "GPT-2", Gate 1 setup | `skills/skill_baseline.md` |
| GRPO fine-tuning | "fine-tune", "GRPO", "RL training", "rule-based reward", "forgetting" | `skills/skill_finetune.md` |
| Evaluation & metrics | "evaluate", "Top-1", "MRR", "edit distance", "F1", "ROC-AUC", "anomaly" | `skills/skill_eval.md` |
| Demo & submission | "demo", "submit", "nextstep.csv", "completion.csv", "anomaly.csv" | `skills/skill_demo.md` |
| Experiment review | after every training run | `skills/06_review_report.md` |
| **Process sequence modeling** | "GPT-2", "transformer", "sequence-model", "process-steps", "causal-LM", "tokenizer", "GRPO", "catastrophic-forgetting", "Top-1", "MRR", "perplexity", "IC", "IGBT", "MOSFET", "generation_rules", "submit.py", "industrial", "semiconductor" | `skills/process-sequence-modeling/` |

---

## Experiment Report Format

After every training run, AI must produce this report (no exceptions):

```markdown
## Experiment: [run_name]

### Change
- Relative to previous run, only changed: [one variable]

### Results
- Top-1 Accuracy: x.xxxx
- Top-3 Accuracy: x.xxxx
- MRR: x.xxxx
- Edit Distance (val): x.xxxx
- Anomaly F1 (val): x.xxxx
- WandB run: [run ID or link]

### Comparison
- Previous best Top-1: x.xxxx ([run_name])
- Delta: +/- x.xxxx

### Anomaly Checks
- [ ] Top-1 < 0.10 after 500 steps? (model not learning — check attention_mask / labels)
- [ ] Edit Distance > 0.80? (completions random — check EOS handling)
- [ ] F1 < 0.52? (anomaly barely above random — check perplexity threshold calibration)
- [ ] Top-1 improving but F1 degrading? (needs GRPO)
- [ ] Any NaN / Inf in predictions?
- [ ] Val Top-1 plateauing while train loss dropping? (overfitting)

### Recommendation
- Keep: [Yes / No + reason]
- Next step: [specific next experiment]

### Human Decision Required
- [List explicitly, e.g. "Top-1=0.38 — below gate threshold, try reducing LR or adding data"]
```
