import torch
import numpy as np
import pandas as pd
import wandb
from chronos import ChronosPipeline


def load_pipeline(model_id: str = "amazon/chronos-t5-small", device: str = "auto") -> ChronosPipeline:
    return ChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )


def predict(
    pipeline: ChronosPipeline,
    context: pd.Series,
    prediction_length: int,
    num_samples: int = 20,
) -> dict:
    context_tensor = torch.tensor(context.values, dtype=torch.float32)
    forecast = pipeline.predict(
        context=context_tensor,
        prediction_length=prediction_length,
        num_samples=num_samples,
    )
    samples = forecast[0].numpy()
    return {
        "median": np.median(samples, axis=0),
        "q10": np.quantile(samples, 0.1, axis=0),
        "q90": np.quantile(samples, 0.9, axis=0),
        "samples": samples,
    }


def run_zero_shot_eval(
    pipeline: ChronosPipeline,
    df: pd.DataFrame,
    context_length: int,
    prediction_length: int,
    run_name: str = "chronos-zero-shot",
) -> pd.DataFrame:
    wandb.init(project=None, name=run_name, config={
        "model": "chronos-zero-shot",
        "context_length": context_length,
        "prediction_length": prediction_length,
    })

    results = []
    context = df["close"].iloc[-context_length:]
    preds = predict(pipeline, context, prediction_length)

    wandb.log({"median_forecast_mean": float(preds["median"].mean())})
    wandb.finish()

    return pd.DataFrame(preds)
