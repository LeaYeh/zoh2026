"""Generate all three submission CSV files from a trained model.

Reads the organizer-distributed eval files and produces:
  nextstep.csv   — Task 1: top-5 next-step predictions per example
  completion.csv — Task 2: predicted remaining steps (pipe-separated)
  anomaly.csv    — Task 3: IS_VALID, SCORE, PREDICTED_RULE per example

Usage:
  uv run python scripts/submit.py \\
    --eval-valid   data/raw/eval_input_valid.csv \\
    --eval-anomaly data/raw/eval_input_anomaly.csv \\
    --output-dir   submission/
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def _load_model():
    from src.utils.wandb_utils import load_best_model
    model, tokenizer, run_name, score = load_best_model()
    if model is None:
        raise RuntimeError("No trained model found. Run src/train.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"[submit] loaded: {run_name}  (eval/top1={score:.4f})")
    return model, tokenizer, device


def _parse_pipe(seq_str: str) -> list[str]:
    return [s.strip() for s in seq_str.split("|") if s.strip()]


def task1_nextstep(eval_valid_csv: Path, out_csv: Path) -> None:
    from src.models.process_lm import predict_next_step
    model, tokenizer, device = _load_model()

    rows = []
    with open(eval_valid_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            partial = _parse_pipe(row["PARTIAL_SEQUENCE"])
            preds = predict_next_step(model, tokenizer, partial, top_k=5, device=device)
            top5 = [p for p, _ in preds]
            while len(top5) < 5:
                top5.append("")
            rows.append([row["EXAMPLE_ID"]] + top5[:5])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        w.writerows(rows)
    print(f"[submit] Task 1 → {out_csv}  ({len(rows)} rows)")


def task2_completion(eval_valid_csv: Path, out_csv: Path) -> None:
    from src.models.process_lm import complete_sequence
    model, tokenizer, device = _load_model()

    rows = []
    with open(eval_valid_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            partial = _parse_pipe(row["PARTIAL_SEQUENCE"])
            completed = complete_sequence(model, tokenizer, partial,
                                          max_new_steps=200, device=device)
            rows.append([row["EXAMPLE_ID"], "|".join(completed)])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        w.writerows(rows)
    print(f"[submit] Task 2 → {out_csv}  ({len(rows)} rows)")


def task3_anomaly(eval_anomaly_csv: Path, out_csv: Path) -> None:
    from src.models.process_lm import anomaly_score
    model, tokenizer, device = _load_model()

    # calibrate threshold on a small sample of valid-looking scores
    scores_all: list[tuple[str, float]] = []
    with open(eval_anomaly_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            steps = _parse_pipe(row["SEQUENCE"])
            score = anomaly_score(model, tokenizer, steps, device=device)
            scores_all.append((row["EXAMPLE_ID"], score))

    # use median as threshold: below = valid (IS_VALID=1), above = anomaly (IS_VALID=0)
    import statistics
    threshold = statistics.median(s for _, s in scores_all)
    max_score = max(s for _, s in scores_all)

    rows = []
    for example_id, score in scores_all:
        is_valid = 1 if score < threshold else 0
        # normalise score to [0,1]: prob of being VALID (1 = definitely valid)
        validity_prob = round(1.0 - min(score / (max_score + 1e-9), 1.0), 4)
        rows.append([example_id, is_valid, validity_prob, ""])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        w.writerows(rows)
    print(f"[submit] Task 3 → {out_csv}  ({len(rows)} rows, threshold={threshold:.2f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-valid",   required=True, help="eval_input_valid.csv path")
    p.add_argument("--eval-anomaly", required=True, help="eval_input_anomaly.csv path")
    p.add_argument("--output-dir",   default="submission", help="output directory")
    p.add_argument("--tasks", nargs="+", choices=["1", "2", "3"],
                   default=["1", "2", "3"], help="which tasks to run (default: all)")
    args = p.parse_args()

    out = Path(args.output_dir)
    valid_csv   = Path(args.eval_valid)
    anomaly_csv = Path(args.eval_anomaly)

    if "1" in args.tasks:
        task1_nextstep(valid_csv, out / "nextstep.csv")
    if "2" in args.tasks:
        task2_completion(valid_csv, out / "completion.csv")
    if "3" in args.tasks:
        task3_anomaly(anomaly_csv, out / "anomaly.csv")

    print(f"\n[submit] Done. Files in {out}/")


if __name__ == "__main__":
    main()
