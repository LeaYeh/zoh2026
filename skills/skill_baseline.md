# Skill: GPT-2 Baseline Training

Trigger: "train", "baseline", "GPT-2", "run model", Gate 1 setup.

## Goal
`uv run python src/train.py configs/exp/gpt2_finetune_v1.yaml` completes,
WandB shows decreasing loss and `eval/top1` improving past 0.40 by step 1000.

## Pre-training checklist
- [ ] `data/raw/training_data/IC_variants.csv` (and IGBT, MOSFET) exists
- [ ] `.env` has valid `WANDB_API_KEY`
- [ ] Smoke-test passes: `WANDB_MODE=disabled uv run python src/train.py configs/exp/gpt2_dummy.yaml`
- [ ] GPU visible: `python -c "import torch; print(torch.cuda.is_available())"`

## Training command
```bash
uv run python src/train.py configs/exp/gpt2_finetune_v1.yaml
```

## What to watch in WandB
| Metric | Target |
|---|---|
| `train/loss` | ~4.8 → <2.0 over 3000 steps |
| `eval/top1` | ≥ 0.40 by step 1000 |
| `eval/top3` | Typically 2–3× top1 |
| `eval/mrr` | Should exceed top1 |

## Gate 1 pass criteria (human must confirm)
- [ ] Training runs without error
- [ ] Loss decreasing after 100 steps
- [ ] Checkpoint saved at `data/oof/gpt2_finetune_v1/model.pt`
- [ ] WandB run visible

## Troubleshooting
| Symptom | Fix |
|---|---|
| Loss stuck at ~4.8 | Check `labels=-100` on PAD and `attention_mask` passed |
| `vocab_size mismatch` | Delete checkpoint, rebuild tokenizer |
| OOM on GPU | Reduce `batch_size` from 64 to 32 |
| `No CSV files found` | Check `dataset.train_path` in yaml |
| Top-1 < 0.10 at step 500 | Lower LR to 2e-4, or generate 2000 more sequences per family |

See `process-sequence-modeling/references/01-architecture-training.md` for full guide.
