from __future__ import annotations
import os
from pathlib import Path

import wandb

CKPT_DIR = Path("data/oof")


def init(run_name: str, config: dict, tags: list[str] | None = None) -> wandb.run:
    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "zoh2026"),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=run_name,
        config=config,
        tags=tags or [],
    )


def log_forecast_metrics(median: list, q10: list, q90: list, actuals: list | None = None) -> None:
    metrics = {
        "forecast/median_mean": sum(median) / len(median),
        "forecast/ci_width_mean": sum(q - p for q, p in zip(q90, q10)) / len(q90),
    }
    if actuals:
        mae = sum(abs(m - a) for m, a in zip(median, actuals)) / len(actuals)
        metrics["forecast/mae"] = mae
    wandb.log(metrics)


def get_best_run_name(
    project: str = "zoh2026",
    metric: str = "eval/top1",
    minimize: bool = False,
) -> tuple[str | None, float | None]:
    """Query WandB for the finished run with the best eval metric.

    Returns (run_name, metric_value) or (None, None) if no runs found.
    run_name matches the key used in data/oof/<run_name>/.
    """
    try:
        api = wandb.Api()
        entity = os.getenv("WANDB_ENTITY") or None
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path, filters={"state": "finished"})
        valid = [r for r in runs if metric in r.summary and r.config.get("run_name")]
        if not valid:
            return None, None
        best = (
            min(valid, key=lambda r: r.summary[metric])
            if minimize
            else max(valid, key=lambda r: r.summary[metric])
        )
        return best.config["run_name"], float(best.summary[metric])
    except Exception as exc:
        print(f"[wandb_utils] get_best_run_name failed: {exc}")
        return None, None


def load_best_model(
    project: str = "zoh2026",
    metric: str = "eval/top1",
):
    """Load the best Track-1 model (GPT-2 scratch) from a local checkpoint.

    Returns (model, tokenizer, run_name, metric_value).
    Returns (None, None, None, None) when no checkpoint is found.
    """
    import torch
    from src.data.process_loader import ProcessStepTokenizer
    from src.models.process_lm import load_model

    run_name, score = get_best_run_name(project=project, metric=metric, minimize=False)
    if run_name is None:
        print("[wandb_utils] No finished runs found.")
        return None, None, None, None

    ckpt_path = CKPT_DIR / run_name
    tok_path  = ckpt_path / "tokenizer.json"
    model_path = ckpt_path / "model.pt"

    if not (ckpt_path.exists() and tok_path.exists() and model_path.exists()):
        print(f"[wandb_utils] Checkpoint for '{run_name}' not found locally.")
        return None, None, run_name, score

    tokenizer = ProcessStepTokenizer.load(tok_path)
    model = load_model(str(ckpt_path), tokenizer)
    model.eval()
    print(f"[wandb_utils] Loaded model: {run_name}  ({metric}={score:.4f})")
    return model, tokenizer, run_name, score


def load_best_pipeline(
    project: str = "zoh2026",
    metric: str = "eval/top1",
    fallback_checkpoint: str = "amazon/chronos-t5-small",
):
    """Load the Chronos pipeline from the best WandB run's local checkpoint.

    Falls back to zero-shot pipeline if no checkpoint found locally.
    Returns (pipeline, run_name, metric_value).
    """
    import torch
    from chronos import ChronosPipeline

    run_name, score = get_best_run_name(project=project, metric=metric)

    if run_name is None:
        print("[wandb_utils] No finished runs found — loading zero-shot pipeline.")
        pipeline = ChronosPipeline.from_pretrained(
            fallback_checkpoint, dtype=torch.float32, device_map="cpu"
        )
        return pipeline, None, None

    ckpt_path = CKPT_DIR / run_name
    if ckpt_path.exists():
        print(f"[wandb_utils] Loading LoRA checkpoint: {ckpt_path}  ({metric}={score:.4f})")
        from src.train import load_lora_checkpoint
        # read base checkpoint from WandB config if available
        try:
            api = wandb.Api()
            entity = os.getenv("WANDB_ENTITY") or None
            path = f"{entity}/{project}" if entity else project
            runs = api.runs(path, filters={"config.run_name": run_name, "state": "finished"})
            base = next(iter(runs)).config.get("model", {}).get("checkpoint", fallback_checkpoint)
        except Exception:
            base = fallback_checkpoint
        pipeline = load_lora_checkpoint(run_name, base_checkpoint=base)
    else:
        print(f"[wandb_utils] Best run '{run_name}' has no local checkpoint — zero-shot fallback.")
        pipeline = ChronosPipeline.from_pretrained(
            fallback_checkpoint, dtype=torch.float32, device_map="cpu"
        )

    return pipeline, run_name, score
