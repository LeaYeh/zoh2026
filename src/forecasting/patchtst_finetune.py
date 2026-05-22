from pathlib import Path
import torch
import numpy as np
import pandas as pd
import wandb
import lightning as L
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import PatchTSTForPrediction, PatchTSTConfig


CKPT_DIR = Path("data/oof")


class TimeSeriesDataset(Dataset):
    def __init__(self, values: np.ndarray, context_len: int, pred_len: int):
        self.values = values
        self.context_len = context_len
        self.pred_len = pred_len
        self.indices = range(context_len, len(values) - pred_len)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.values[i - self.context_len : i]
        y = self.values[i : i + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class PatchTSTLightning(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        model_config = PatchTSTConfig(
            num_input_channels=1,
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
        )
        self.model = PatchTSTForPrediction(model_config)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        out = self.model(past_values=x.unsqueeze(-1))
        return out.prediction_outputs.squeeze(-1)

    def training_step(self, batch, _):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["learning_rate"])


def train(
    df: pd.DataFrame,
    config: dict,
    run_name: str,
) -> Path:
    wandb.init(project=None, name=run_name, config=config)

    values = df["close"].values.astype(np.float32)
    split = int(len(values) * 0.8)
    train_ds = TimeSeriesDataset(values[:split], config["context_length"], config["prediction_length"])
    val_ds = TimeSeriesDataset(values[split:], config["context_length"], config["prediction_length"])

    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=4)

    model = PatchTSTLightning(config)

    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        precision=config.get("precision", "bf16-mixed"),
        logger=False,
        enable_checkpointing=True,
        default_root_dir=str(CKPT_DIR),
    )
    trainer.fit(model, train_dl, val_dl)

    ckpt_path = CKPT_DIR / f"{run_name}.ckpt"
    trainer.save_checkpoint(ckpt_path)
    wandb.save(str(ckpt_path))
    wandb.finish()

    return ckpt_path


def predict(ckpt_path: Path, context: np.ndarray, config: dict) -> np.ndarray:
    model = PatchTSTLightning.load_from_checkpoint(ckpt_path)
    model.eval()
    x = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return model(x).squeeze(0).numpy()
