"""Track 1 evaluator — runs all three submission tasks on a trained model.

Usage:
    from src.evaluator import run_full_eval
    results = run_full_eval(model, tokenizer, eval_valid_csv, eval_anomaly_csv)

Tasks
-----
Task 1  Next-Step Prediction  eval_input_valid.csv   (truncated at 60% and 80%)
Task 2  Sequence Completion   eval_input_valid.csv   (complete from truncation point)
Task 3  Anomaly Detection     eval_input_anomaly.csv (labelled valid=0 / anomaly=1)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.process_loader import ProcessStepTokenizer
from src.models.process_lm import (
    GPT2LMHeadModel,
    predict_next_step,
    complete_sequence,
    anomaly_score,
)
from src.evaluation.process_metrics import (
    Task1Metrics, Task2Metrics, Task3Metrics,
    evaluate_next_step, evaluate_completion, evaluate_anomaly,
)


def _load_eval_valid(
    csv_path: str | Path,
    sequence_col: str = "SEQUENCE_ID",
    step_col: str = "STEP",
    truncation_col: str = "TRUNCATION",
) -> list[dict]:
    """Load eval_input_valid.csv.

    Returns list of dicts:
      {sequence_id, partial_steps, true_next_step, true_remaining, truncation}
    """
    df = pd.read_csv(csv_path)
    col_map = {c.upper(): c for c in df.columns}
    seq_col  = col_map.get(sequence_col.upper(),  sequence_col)
    step_col_actual = col_map.get(step_col.upper(), step_col)
    trunc_col = col_map.get(truncation_col.upper(), truncation_col)

    records = []
    for seq_id, group in df.groupby(seq_col, sort=False):
        steps = group[step_col_actual].astype(str).tolist()
        trunc = float(group[trunc_col].iloc[0]) if trunc_col in group.columns else 0.6
        cut = max(1, int(len(steps) * trunc))
        records.append({
            "sequence_id":   seq_id,
            "partial_steps": steps[:cut],
            "true_next":     steps[cut] if cut < len(steps) else None,
            "true_remaining": steps[cut:],
            "truncation":    trunc,
        })
    return records


def _load_eval_anomaly(
    csv_path: str | Path,
    sequence_col: str = "SEQUENCE_ID",
    step_col: str = "STEP",
    label_col: str = "LABEL",
) -> list[dict]:
    """Load eval_input_anomaly.csv.

    Returns list of dicts: {sequence_id, steps, label}
    label: 0=valid, 1=anomaly
    """
    df = pd.read_csv(csv_path)
    col_map = {c.upper(): c for c in df.columns}
    seq_col  = col_map.get(sequence_col.upper(), sequence_col)
    step_col_actual = col_map.get(step_col.upper(), step_col)
    lbl_col  = col_map.get(label_col.upper(), label_col)

    records = []
    for seq_id, group in df.groupby(seq_col, sort=False):
        steps = group[step_col_actual].astype(str).tolist()
        label = int(group[lbl_col].iloc[0]) if lbl_col in group.columns else 0
        records.append({"sequence_id": seq_id, "steps": steps, "label": label})
    return records


def run_full_eval(
    model: GPT2LMHeadModel,
    tokenizer: ProcessStepTokenizer,
    eval_valid_csv: str | Path,
    eval_anomaly_csv: str | Path,
    device: str | torch.device = "cpu",
    top_k: int = 5,
) -> dict:
    """Run all three tasks and return a results dict."""
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.eval()

    results: dict = {}

    # ── Task 1 & 2: valid sequences ──────────────────────────────────────────
    valid_records = _load_eval_valid(eval_valid_csv)
    ranked_preds: list[list[str]] = []
    next_targets: list[str]       = []
    completions:  list[list[str]] = []
    comp_targets: list[list[str]] = []

    for rec in valid_records:
        partial = rec["partial_steps"]
        preds   = predict_next_step(model, tokenizer, partial, top_k=top_k, device=device)
        ranked_preds.append([p for p, _ in preds])
        if rec["true_next"]:
            next_targets.append(rec["true_next"])

        if rec["true_remaining"]:
            completed = complete_sequence(
                model, tokenizer, partial,
                max_new_steps=len(rec["true_remaining"]) + 10,
                device=device,
            )
            completions.append(completed)
            comp_targets.append(rec["true_remaining"])

    t1 = evaluate_next_step(ranked_preds, next_targets)
    t2 = evaluate_completion(completions, comp_targets)
    results["task1"] = t1.to_dict()
    results["task2"] = t2.to_dict()

    # ── Task 3: anomaly detection ─────────────────────────────────────────────
    anomaly_records = _load_eval_anomaly(eval_anomaly_csv)
    scores = [
        anomaly_score(model, tokenizer, rec["steps"], device=device)
        for rec in anomaly_records
    ]
    labels = [rec["label"] for rec in anomaly_records]
    t3 = evaluate_anomaly(scores, labels)
    results["task3"] = t3.to_dict()

    return results
