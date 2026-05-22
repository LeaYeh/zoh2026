import os
import wandb


def init(run_name: str, config: dict, tags: list[str] | None = None) -> wandb.run:
    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "zoh2026-track03"),
        entity=os.getenv("WANDB_ENTITY"),
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
