"""Score invalid_sequences.csv with a trained model's anomaly_score (perplexity).

Usage:
    uv run python scripts/score_invalid.py \
        --input  data/raw/training_data/invalid_sequences.csv \
        --ckpt   data/oof/ablation_base_6l256d \
        --output data/oof/ablation_base_6l256d/invalid_scored.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import torch

from src.data.process_loader import ProcessStepTokenizer
from src.models.process_lm import load_model, anomaly_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/training_data/invalid_sequences.csv")
    parser.add_argument("--ckpt",   default="data/oof/ablation_base_6l256d")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    out_path = Path(args.output) if args.output else ckpt / "invalid_scored.csv"

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"[score] device={device}  ckpt={ckpt}")

    tokenizer = ProcessStepTokenizer.load(ckpt / "tokenizer.json")
    model = load_model(str(ckpt), tokenizer, device=device)
    model.eval()
    print(f"[score] vocab_size={tokenizer.vocab_size}")

    df = pd.read_csv(args.input)
    print(f"[score] {len(df)} sequences  columns={list(df.columns)}")

    scores: list[float] = []
    unk_counts: list[int] = []
    for i, row in df.iterrows():
        steps = [s.strip() for s in str(row["SEQUENCE"]).split("|") if s.strip()]
        unk = sum(1 for s in steps if s not in tokenizer.step_to_id)
        unk_counts.append(unk)
        score = anomaly_score(model, tokenizer, steps, device=device)
        scores.append(score)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(df)}  last_perplexity={score:.2f}", flush=True)

    df["perplexity"] = scores
    df["unk_steps"]  = unk_counts
    df.to_csv(out_path, index=False)
    print(f"\n[score] saved → {out_path}")

    print("\n── perplexity by RULE ──────────────────────")
    print(df.groupby("RULE")["perplexity"].describe().round(2).to_string())

    print("\n── overall ─────────────────────────────────")
    print(df["perplexity"].describe().round(2).to_string())
    print(f"unk_steps > 0: {(df['unk_steps'] > 0).sum()} / {len(df)}")


if __name__ == "__main__":
    main()
