"""Unified training entry point.

Usage:
    uv run python src/train.py configs/exp/chronos_lora_v1.yaml
"""

from __future__ import annotations
import sys
import os
import copy
import json
import math
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

    if name == "csv":
        return _load_csv_dataset(dataset_cfg)

    raise ValueError(f"Unknown dataset: {name}")


def _load_csv_dataset(
    dataset_cfg: dict,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generic CSV loader for competition data.

    Accepts any CSV with a price column and optional timestamp column.
    Each unique value in `series_col` becomes one time series (set to None
    for single-series files). The last `prediction_length` rows of each
    series become the test target; the rest becomes train context.

    Config keys:
      path          str   path to CSV file (required)
      price_col     str   column name for price/target values (default: "close")
      timestamp_col str   column for timestamps, used only for sorting (default: None)
      series_col    str   column that identifies separate series, e.g. "symbol"
                          (default: None → treat whole file as one series)
      prediction_length int  number of steps to hold out as test target
      min_length    int   drop series shorter than this (default: 100)
    """
    import pandas as pd

    path          = dataset_cfg["path"]
    price_col     = dataset_cfg.get("price_col", "close")
    timestamp_col = dataset_cfg.get("timestamp_col", None)
    series_col    = dataset_cfg.get("series_col", None)
    pred_len      = dataset_cfg["prediction_length"]
    min_length    = dataset_cfg.get("min_length", 100)

    df = pd.read_csv(path)

    # Normalise column names to lowercase for robustness
    df.columns = [c.lower().strip() for c in df.columns]
    price_col     = price_col.lower().strip()
    if timestamp_col:
        timestamp_col = timestamp_col.lower().strip()
    if series_col:
        series_col = series_col.lower().strip()

    if timestamp_col and timestamp_col in df.columns:
        df = df.sort_values(timestamp_col)

    def _extract(sub: pd.DataFrame) -> np.ndarray:
        return sub[price_col].dropna().values.astype(np.float32)

    if series_col and series_col in df.columns:
        groups = [g for _, g in df.groupby(series_col, sort=False)]
    else:
        groups = [df]

    train_list, test_list = [], []
    for g in groups:
        v = _extract(g)
        if len(v) < min_length + pred_len:
            continue
        train_list.append(v[:-pred_len])
        test_list.append(v[-pred_len:])

    if not train_list:
        raise ValueError(
            f"No series loaded from {path} — check price_col='{price_col}', "
            f"series_col='{series_col}', min_length={min_length}"
        )
    print(f"  [csv loader] {len(train_list)} series from {path}")
    return train_list, test_list


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


def _predict_batched(
    pipeline,
    context_arrays: list[np.ndarray],
    prediction_length: int,
    num_samples: int = 20,
    batch_size: int = 16,
) -> list[np.ndarray]:
    """Run Chronos prediction on a list of context arrays in batches.

    Returns a list of (num_samples, prediction_length) arrays, one per input.
    Batching avoids per-series Python loop overhead and improves GPU utilisation.
    """
    results: list[np.ndarray] = []
    pipeline.model.eval()
    with torch.no_grad():
        for start in range(0, len(context_arrays), batch_size):
            batch = [
                torch.tensor(arr, dtype=torch.float32)
                for arr in context_arrays[start : start + batch_size]
            ]
            # pipeline.predict accepts list[1D Tensor] → (batch, num_samples, pred_len)
            preds = pipeline.predict(
                batch, prediction_length, num_samples=num_samples,
                limit_prediction_length=False,
            )
            for i in range(len(batch)):
                results.append(preds[i].numpy())
    return results


def load_lora_checkpoint(run_name: str, base_checkpoint: str = "amazon/chronos-t5-small"):
    """Reload a saved LoRA adapter into a fresh Chronos pipeline."""
    from chronos import ChronosPipeline
    from peft import PeftModel
    ckpt_path = CKPT_DIR / run_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    pipeline = ChronosPipeline.from_pretrained(base_checkpoint, dtype=torch.float32, device_map="cpu")
    pipeline.model.model = PeftModel.from_pretrained(pipeline.model.model, str(ckpt_path))
    pipeline.model.model.eval()
    return pipeline


# ── financial experience replay ──────────────────────────────────────────────

_REPLAY_SYMBOLS_DEFAULT = [
    # US equity indices
    "^GSPC", "^IXIC", "^DJI", "^FTSE", "^N225",
    # Volatility / bear-market regime anchors
    "^VIX", "SQQQ", "SH",
    # Equity ETFs
    "QQQ", "SPY", "IWM",
    # Tech
    "AAPL", "MSFT", "GOOGL", "NVDA", "BABA", "TSM",
    # Commodities + bonds
    "GC=F", "CL=F", "TLT", "GLD",
    # Crypto — covers 2017 AND 2021 bull markets
    "BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "LINK-USD",
    # Forex
    "EURUSD=X", "JPY=X", "GBPUSD=X",
]

# Extended date range to cover 2017 crypto bull + 2015 China correction
_REPLAY_START_DEFAULT = "2015-01-01"
_REPLAY_END_DEFAULT   = "2025-06-01"


def _load_financial_replay(
    symbols: list[str] | None = None,
    start: str = _REPLAY_START_DEFAULT,
    end: str = _REPLAY_END_DEFAULT,
    min_length: int = 600,
) -> list[np.ndarray]:
    """Download daily close prices for a diverse set of financial assets.

    Returns raw price series (float32) for use as experience replay data.
    Silently skips symbols that fail to download.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  [replay] yfinance not installed — skipping financial replay")
        return []

    symbols = symbols or _REPLAY_SYMBOLS_DEFAULT
    series_list: list[np.ndarray] = []
    for sym in symbols:
        try:
            df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                continue
            col = "Close" if "Close" in df.columns else df.columns[0]
            prices = df[col].dropna().values.astype(np.float32)
            if len(prices) >= min_length:
                series_list.append(prices)
        except Exception:
            pass
    print(f"  [replay] loaded {len(series_list)}/{len(symbols)} financial series")
    return series_list


# ── val split + eval helpers ─────────────────────────────────────────────────

def _split_val(
    train_series: list[np.ndarray],
    pred_len: int,
    context_len: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Split training series into finetune / val sets.

    Layout per series:
      [======== finetune ========|== gap ==|== val target ==]
                                 pred_len     pred_len
    gap prevents any overlap between finetune context and val target.
    """
    finetune, val_ctx, val_tgt = [], [], []
    needed = context_len + 2 * pred_len
    for s in train_series:
        if len(s) < needed:
            continue
        finetune.append(s[: -(2 * pred_len)])
        val_ctx.append(s[-(context_len + pred_len) : -pred_len])
        val_tgt.append(s[-pred_len:])
    return finetune, val_ctx, val_tgt


def _eval_val_crps(
    pipeline,
    val_ctx: list[np.ndarray],
    val_tgt: list[np.ndarray],
    pred_len: int,
    num_samples: int,
) -> float:
    from src.evaluator import evaluate, naive_scale_from_series
    batched_preds = _predict_batched(pipeline, val_ctx, pred_len, num_samples)
    pipeline.model.train()
    all_scales = [naive_scale_from_series(ctx) for ctx in val_ctx]
    metrics = evaluate(
        np.stack(batched_preds),
        np.stack(val_tgt),
        np.array(all_scales),
    )
    return metrics.crps


# ── training loop ─────────────────────────────────────────────────────────────

def finetune_chronos_lora(
    pipeline,
    train_series: list[np.ndarray],
    val_ctx: list[np.ndarray],
    val_tgt: list[np.ndarray],
    cfg: dict,
    replay_series: list[np.ndarray] | None = None,
) -> None:
    """LoRA fine-tune with early stopping on val CRPS. Restores best weights in-place.

    replay_series: optional financial price series mixed into batches at replay_ratio.
    Val CRPS is always evaluated on competition data only (not replay).
    """
    model = pipeline.model
    context_len = cfg["model"].get("context_length", 512)
    pred_len = pipeline.tokenizer.config.prediction_length
    lr = cfg["training"].get("learning_rate", 1e-4)
    max_steps = cfg["training"].get("max_steps", 1000)
    batch_size = cfg["training"].get("batch_size", 8)
    patience = cfg["training"].get("early_stop_patience", 3)
    eval_every = cfg["training"].get("eval_every", 100)
    num_samples = cfg["model"].get("num_samples", 20)
    replay_cfg = cfg["training"].get("experience_replay", {})
    replay_ratio = replay_cfg.get("ratio", 0.0) if replay_series else 0.0

    # A100: enable TF32 + bf16 if requested
    if cfg["training"].get("precision") == "bf16-mixed" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    trainable_params = [p for p in model.model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # Cosine LR with linear warmup: ramp up for warmup_steps, decay to lr*0.01
    warmup_steps = cfg["training"].get("warmup_steps", max(1, max_steps // 10))
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    rng = np.random.default_rng(42)
    model.train()

    best_val_crps = float("inf")
    best_lora_state: dict | None = None
    patience_counter = 0
    step = 0

    while step < max_steps:
        batch_ctx, batch_tgt = [], []
        for _ in range(batch_size):
            # experience replay: sample from financial series with probability replay_ratio
            if replay_series and rng.random() < replay_ratio:
                pool = replay_series
            else:
                pool = train_series
            s = pool[rng.integers(0, len(pool))]
            if len(s) < context_len + pred_len:
                continue
            i = rng.integers(0, len(s) - context_len - pred_len)
            batch_ctx.append(s[i : i + context_len])
            batch_tgt.append(s[i + context_len : i + context_len + pred_len])

        if not batch_ctx:
            continue

        ctx_t = torch.tensor(np.array(batch_ctx), dtype=torch.float32)
        tgt_t = torch.tensor(np.array(batch_tgt), dtype=torch.float32)

        ctx_input_ids, ctx_attn_mask, scale = pipeline.tokenizer.context_input_transform(ctx_t)
        tgt_input_ids, _ = pipeline.tokenizer.label_input_transform(tgt_t, scale=scale)

        out = model.model.model(
            input_ids=ctx_input_ids,
            attention_mask=ctx_attn_mask,
            labels=tgt_input_ids,
        )
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "step": step,
            })
            print(f"  step {step:4d}  loss={loss.item():.4f}")

        # early stopping check
        if step > 0 and step % eval_every == 0 and val_ctx:
            val_crps = _eval_val_crps(pipeline, val_ctx, val_tgt, pred_len, num_samples)
            wandb.log({"val/crps": val_crps, "step": step})
            print(f"           val_crps={val_crps:.4f}  best={best_val_crps:.4f}")
            if val_crps < best_val_crps:
                best_val_crps = val_crps
                patience_counter = 0
                # snapshot LoRA weights in memory
                best_lora_state = copy.deepcopy(
                    {k: v.cpu() for k, v in model.model.model.state_dict().items()}
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at step {step} (patience={patience} exhausted)")
                    break

        step += 1

    # restore best weights
    if best_lora_state is not None:
        model.model.model.load_state_dict(best_lora_state)
        print(f"  Restored best checkpoint (val_crps={best_val_crps:.4f})")


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
        context_len = cfg["model"].get("context_length", 512)
        finetune_series, val_ctx, val_tgt = _split_val(train_series, pred_len, context_len)
        print(f"Fine-tuning ({method})...  finetune={len(finetune_series)} series, val={len(val_ctx)} series")

        # load financial experience replay if enabled
        replay_series: list[np.ndarray] = []
        replay_cfg = cfg["training"].get("experience_replay", {})
        if replay_cfg.get("enabled", False):
            print("Loading financial replay data...")
            replay_series = _load_financial_replay(
                symbols=replay_cfg.get("symbols"),
                start=replay_cfg.get("start", "2018-01-01"),
                end=replay_cfg.get("end", "2025-01-01"),
                min_length=context_len + pred_len,
            )
            wandb.config.update({"replay_n_series": len(replay_series)}, allow_val_change=True)

        finetune_chronos_lora(model_tuple[1], finetune_series, val_ctx, val_tgt, cfg, replay_series or None)
        model_tuple[1].model.model.save_pretrained(str(CKPT_DIR / run_name))
        print(f"  Checkpoint saved → {CKPT_DIR / run_name}")
    elif method == "none":
        print("Skipping fine-tune (zero-shot).")

    # 4. predict + evaluate (batched for speed)
    print("Evaluating...")
    from src.evaluator import evaluate, naive_scale_from_series

    context_len = cfg["model"].get("context_length", 512)
    num_samples = cfg["model"].get("num_samples", 20)
    infer_batch  = cfg.get("inference", {}).get("batch_size", 16)

    if kind == "chronos":
        pipeline = model_tuple[1]
        ctx_arrays  = [s[-context_len:] for s in train_series]
        all_preds   = _predict_batched(pipeline, ctx_arrays, pred_len, num_samples, infer_batch)
        all_truth   = [s[:pred_len] for s in test_series]
        all_scales  = [naive_scale_from_series(s) for s in train_series]
    else:
        all_preds, all_truth, all_scales = [], [], []

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
