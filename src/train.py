"""Unified training entry point — Track 1: Industrial Process Sequence Modeling.

Trains a small GPT-2-style transformer from scratch on process step sequences.

Usage:
    uv run python src/train.py configs/exp/gpt2_finetune_v1.yaml
"""
from __future__ import annotations
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import wandb
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv

from src.data.process_loader import ProcessStepTokenizer, load_sequences
from src.models.process_lm import build_model, predict_next_step
from src.evaluation.process_metrics import top_k_accuracy, mrr

load_dotenv()
CKPT_DIR = Path("data/oof")
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ── dataset ───────────────────────────────────────────────────────────────────

class ProcessSequenceDataset(Dataset):
    def __init__(self, encoded: list[list[int]], max_len: int, pad_id: int) -> None:
        self.encoded = encoded
        self.max_len = max_len
        self.pad_id  = pad_id

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.encoded[idx][: self.max_len]
        real_len = len(seq)
        seq = seq + [self.pad_id] * (self.max_len - real_len)
        input_ids   = torch.tensor(seq, dtype=torch.long)
        attn_mask   = torch.zeros(self.max_len, dtype=torch.long)
        attn_mask[:real_len] = 1
        return input_ids, attn_mask


def _load_dataset(cfg: dict) -> tuple[list[list[str]], list[list[str]]]:
    name = cfg["name"]

    if name == "process_csv":
        seqs = load_sequences(
            data_dir=cfg["train_path"],
            product_families=cfg.get("product_families"),
            train_files=cfg.get("train_files"),
            sequence_col=cfg.get("sequence_col", "SEQUENCE_ID"),
            step_col=cfg.get("step_col", "STEP"),
            min_length=cfg.get("min_length", 5),
        )
    elif name == "dummy":
        rng = np.random.default_rng(42)
        vocab = [f"STEP_{i:02d}" for i in range(20)]
        n = cfg.get("n_series", 300)
        seqs = [
            [vocab[rng.integers(0, len(vocab))] for _ in range(rng.integers(10, 40))]
            for _ in range(n)
        ]
    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    val_ratio = cfg.get("val_ratio", 0.1)
    n_val = max(1, int(len(seqs) * val_ratio))
    rng = np.random.default_rng(cfg.get("seed", 42))
    idx = rng.permutation(len(seqs)).tolist()
    return [seqs[i] for i in idx[n_val:]], [seqs[i] for i in idx[:n_val]]


# ── eval ──────────────────────────────────────────────────────────────────────

def _eval_step(
    model: torch.nn.Module,
    tokenizer: ProcessStepTokenizer,
    val_seqs: list[list[str]],
    device: torch.device,
    n_samples: int = 300,
) -> dict:
    model.eval()
    rng = np.random.default_rng(0)
    indices = rng.choice(len(val_seqs), size=min(n_samples, len(val_seqs)), replace=False)
    ranked_preds: list[list[str]] = []
    targets: list[str] = []
    for i in indices:
        seq = val_seqs[i]
        if len(seq) < 2:
            continue
        cut = int(rng.integers(1, len(seq)))
        preds = predict_next_step(model, tokenizer, seq[:cut], top_k=5, device=device)
        ranked_preds.append([p for p, _ in preds])
        targets.append(seq[cut])
    model.train()
    return {
        "eval/top1": top_k_accuracy(ranked_preds, targets, k=1),
        "eval/top3": top_k_accuracy(ranked_preds, targets, k=3),
        "eval/top5": top_k_accuracy(ranked_preds, targets, k=5),
        "eval/mrr":  mrr(ranked_preds, targets),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    run_name  = cfg["run_name"]
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    ds_cfg    = cfg.get("dataset", {})

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "zoh2026"),
        name=run_name,
        config=cfg,
        tags=["track1", model_cfg.get("type", "gpt2_scratch")],
    )

    # ── data ─────────────────────────────────────────────────────────────────
    train_seqs, val_seqs = _load_dataset(ds_cfg)
    print(f"[train] {len(train_seqs)} train / {len(val_seqs)} val sequences")

    tokenizer = ProcessStepTokenizer.build(train_seqs + val_seqs)
    print(f"[train] vocab size: {tokenizer.vocab_size}")
    wandb.config.update({"model/vocab_size": tokenizer.vocab_size}, allow_val_change=True)

    max_len    = model_cfg.get("max_seq_len", 256)
    batch_size = train_cfg.get("batch_size", 32)

    train_ds = ProcessSequenceDataset(
        [tokenizer.encode(s) for s in train_seqs], max_len, tokenizer.pad_id
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # ── model ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"[train] device: {device}")

    model = build_model(
        tokenizer,
        max_seq_len=max_len,
        n_embd=model_cfg.get("n_embd", 256),
        n_layer=model_cfg.get("n_layer", 6),
        n_head=model_cfg.get("n_head", 8),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] parameters: {n_params:,}")
    wandb.config.update({"model/n_params": n_params}, allow_val_change=True)

    # ── optimiser + scheduler ────────────────────────────────────────────────
    lr        = train_cfg.get("lr", 5e-4)
    max_steps = train_cfg.get("steps", 2000)
    warmup    = train_cfg.get("warmup_steps", max(1, max_steps // 10))
    grad_clip = train_cfg.get("grad_clip", 1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── training loop ────────────────────────────────────────────────────────
    eval_every = train_cfg.get("eval_every", 200)
    log_every  = train_cfg.get("log_every", 10)
    step       = 0
    best_top1  = 0.0
    data_iter  = iter(train_loader)

    model.train()
    while step < max_steps:
        try:
            batch_ids, batch_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch_ids, batch_mask = next(data_iter)

        batch_ids  = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        # labels: -100 on PAD positions so they are ignored in loss
        labels = batch_ids.clone()
        labels[batch_mask == 0] = -100
        loss = model(batch_ids, attention_mask=batch_mask, labels=labels).loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % log_every == 0:
            lr_now = scheduler.get_last_lr()[0]
            wandb.log({"train/loss": loss.item(), "train/lr": lr_now}, step=step)
            print(f"  step {step:>5}/{max_steps}  loss={loss.item():.4f}  lr={lr_now:.2e}")

        if step % eval_every == 0 or step == max_steps:
            metrics = _eval_step(model, tokenizer, val_seqs, device)
            wandb.log(metrics, step=step)
            top1 = metrics["eval/top1"]
            print(f"[eval] step={step}  top1={top1:.4f}  top3={metrics['eval/top3']:.4f}"
                  f"  mrr={metrics['eval/mrr']:.4f}")

            if top1 > best_top1:
                best_top1 = top1
                ckpt_path = CKPT_DIR / run_name
                ckpt_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path / "model.pt")
                tokenizer.save(ckpt_path / "tokenizer.json")
                wandb.log({"eval/best_top1": best_top1}, step=step)
                print(f"[ckpt] saved  best_top1={best_top1:.4f}")

    wandb.summary["eval/best_top1"] = best_top1
    print(f"[done] best_top1={best_top1:.4f}")
    wandb.finish()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/train.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
