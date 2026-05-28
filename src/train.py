"""Unified training entry point.

Usage:
    uv run python src/train.py configs/exp/patchtst_electricity_v0.yaml
"""

from __future__ import annotations
import sys
import os
import json
from pathlib import Path

import numpy as np
import yaml
import torch
import wandb
from dotenv import load_dotenv

load_dotenv()

CKPT_DIR = Path("data/oof")
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ── dataset loading ───────────────────────────────────────────────────────────

def load_dataset(dataset_cfg: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns (train_series_list, test_series_list)."""
    name = dataset_cfg["name"]

    if name == "dummy":
        rng = np.random.default_rng(42)
        n = dataset_cfg.get("n_series", 10)
        length = dataset_cfg.get("length", 512)
        series = [rng.standard_normal(length).cumsum() for _ in range(n)]
        pred_len = dataset_cfg.get("prediction_length", 24)
        train = [s[:-pred_len] for s in series]
        test = [s[-pred_len:] for s in series]
        return train, test

    if name in ("electricity", "traffic", "exchange_rate", "solar-energy", "m4_hourly", "m4_daily"):
        from gluonts.dataset.repository import get_dataset
        ds = get_dataset(name, regenerate=False)
        pred_len = ds.metadata.prediction_length
        train_list, test_list = [], []
        for entry in ds.train:
            v = entry["target"].astype(np.float32)
            train_list.append(v)
        for entry in ds.test:
            v = entry["target"].astype(np.float32)
            test_list.append(v[-pred_len:])
        return train_list, test_list

    raise ValueError(f"Unknown dataset: {name}")


# ── model loading + optional LoRA ────────────────────────────────────────────

def load_model(model_cfg: dict):
    name = model_cfg["name"]
    finetune_cfg = model_cfg.get("finetune", {})
    method = finetune_cfg.get("method", "none")

    if name == "chronos":
        from chronos import ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained(
            model_cfg.get("checkpoint", "amazon/chronos-t5-small"),
            device_map="cpu",
            dtype=torch.float32,
        )
        if method == "lora":
            from peft import get_peft_model, LoraConfig, TaskType
            # Apply LoRA to the inner T5 model, not the ChronosModel wrapper
            inner_t5 = pipeline.model.model
            lora_cfg = LoraConfig(
                r=finetune_cfg.get("lora_r", 8),
                lora_alpha=finetune_cfg.get("lora_alpha", 32),
                lora_dropout=finetune_cfg.get("lora_dropout", 0.1),
                target_modules=finetune_cfg.get("target_modules", ["q", "v"]),
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            pipeline.model.model = get_peft_model(inner_t5, lora_cfg)
            pipeline.model.model.print_trainable_parameters()
        return ("chronos", pipeline)

    if name == "patchtst":
        import importlib
        ft = importlib.import_module("src.forecasting.patchtst_finetune")
        return ("patchtst", ft)

    raise ValueError(f"Unknown model: {name}")


# ── predict (unified interface) ───────────────────────────────────────────────

def predict_series(
    model_tuple,
    context: np.ndarray,
    prediction_length: int,
    num_samples: int = 20,
) -> np.ndarray:
    """Returns (n_samples, pred_len) array of sample draws."""
    kind, model = model_tuple

    if kind == "chronos":
        ctx = torch.tensor(context, dtype=torch.float32)
        forecast = model.predict(ctx, prediction_length, num_samples=num_samples)
        return forecast[0].numpy()  # (n_samples, pred_len)

    raise NotImplementedError(f"predict_series not implemented for {kind}")


# ── training loop (chronos LoRA / PatchTST) ──────────────────────────────────

def finetune_chronos_lora(
    pipeline,
    train_series: list[np.ndarray],
    cfg: dict,
) -> None:
    """Simple next-step quantile loss fine-tune for Chronos with LoRA."""
    from chronos import ChronosPipeline

    model = pipeline.model
    context_len = cfg["model"].get("context_length", 512)
    # Use the tokenizer's configured prediction_length for training batches
    pred_len = pipeline.tokenizer.config.prediction_length
    lr = cfg["training"].get("learning_rate", 1e-4)
    max_steps = cfg["training"].get("max_steps", 100)
    batch_size = cfg["training"].get("batch_size", 8)

    optimizer = torch.optim.AdamW(
        [p for p in model.model.model.parameters() if p.requires_grad], lr=lr
    )

    rng = np.random.default_rng(42)
    model.train()
    step = 0

    while step < max_steps:
        # sample a random batch of windows
        batch_ctx, batch_tgt = [], []
        for _ in range(batch_size):
            s = rng.choice(train_series)
            if len(s) < context_len + pred_len:
                continue
            i = rng.integers(0, len(s) - context_len - pred_len)
            batch_ctx.append(s[i : i + context_len])
            batch_tgt.append(s[i + context_len : i + context_len + pred_len])

        if not batch_ctx:
            continue

        ctx_t = torch.tensor(np.array(batch_ctx), dtype=torch.float32)
        tgt_t = torch.tensor(np.array(batch_tgt), dtype=torch.float32)

        # context_input_transform returns (input_ids, attention_mask, scale)
        ctx_input_ids, ctx_attn_mask, scale = pipeline.tokenizer.context_input_transform(ctx_t)
        tgt_input_ids, _ = pipeline.tokenizer.label_input_transform(tgt_t, scale=scale)

        # Call inner T5 directly (LoRA is applied there)
        out = model.model.model(
            input_ids=ctx_input_ids,
            attention_mask=ctx_attn_mask,
            labels=tgt_input_ids,
        )
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            wandb.log({"train/loss": loss.item(), "step": step})
            print(f"  step {step:4d}  loss={loss.item():.4f}")

        step += 1


# ── main ─────────────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    run_name = cfg["run_name"]

    entity = os.getenv("WANDB_ENTITY") or None
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "zoh2026-track03"),
        entity=entity,
        name=run_name,
        config=cfg,
    )

    print(f"\n=== {run_name} ===")

    # 1. load data
    print("Loading dataset...")
    train_series, test_series = load_dataset(cfg["dataset"])
    pred_len = cfg["model"].get("prediction_length", 24)
    print(f"  {len(train_series)} series, pred_len={pred_len}")

    # 2. load model
    print("Loading model...")
    model_tuple = load_model(cfg["model"])
    kind = model_tuple[0]

    # 3. fine-tune (if method != none)
    method = cfg["model"].get("finetune", {}).get("method", "none")
    if method != "none" and kind == "chronos":
        print(f"Fine-tuning ({method})...")
        finetune_chronos_lora(model_tuple[1], train_series, cfg)

        # save LoRA weights
        # Save LoRA adapter weights from the inner T5 model
        model_tuple[1].model.model.save_pretrained(str(CKPT_DIR / run_name))
        print(f"  Checkpoint saved → {CKPT_DIR / run_name}")
    elif method == "none":
        print("Skipping fine-tune (zero-shot).")

    # 4. predict + evaluate
    print("Evaluating...")
    from src.evaluator import evaluate, naive_scale_from_series

    all_preds, all_truth, all_scales = [], [], []
    context_len = cfg["model"].get("context_length", 512)
    num_samples = cfg["model"].get("num_samples", 20)

    for i, (train_s, test_s) in enumerate(zip(train_series, test_series)):
        ctx = train_s[-context_len:]
        if kind == "chronos":
            preds = predict_series(model_tuple, ctx, pred_len, num_samples)
        else:
            continue
        all_preds.append(preds)
        all_truth.append(test_s[:pred_len])
        all_scales.append(naive_scale_from_series(train_s))

    if not all_preds:
        print("No predictions produced — check model/data config.")
        return

    preds_arr = np.stack(all_preds)       # (n_series, n_samples, pred_len)
    truth_arr = np.stack(all_truth)       # (n_series, pred_len)
    scale_arr = np.array(all_scales)

    metrics = evaluate(preds_arr, truth_arr, scale_arr)
    metrics_dict = metrics.to_dict()

    print("\n── Metrics ──────────────────────────────")
    for k, v in metrics_dict.items():
        print(f"  {k:<20} {v:.4f}")

    wandb.log({f"eval/{k}": v for k, v in metrics_dict.items()})

    # 5. save predictions
    np.save(CKPT_DIR / f"{run_name}_preds.npy", preds_arr)
    np.save(CKPT_DIR / f"{run_name}_truth.npy", truth_arr)
    print(f"\nPredictions → data/oof/{run_name}_preds.npy")

    wandb.finish()
    print(f"\n✓ {run_name} done")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/train.py configs/exp/<name>.yaml")
        sys.exit(1)
    main(sys.argv[1])
