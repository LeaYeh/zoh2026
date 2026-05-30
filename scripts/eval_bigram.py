"""Bigram baseline evaluation against training data (no generation required).

Usage:
    uv run python scripts/eval_bigram.py
    uv run python scripts/eval_bigram.py --data-dir data/raw/training_data --val-ratio 0.1

Tasks evaluated:
    Task 1  Next-step prediction  (Top-1/3/5, MRR)
    Task 2  Sequence completion   (Exact Match, NED, Token Accuracy)
    Task 3  Anomaly detection     (F1, ROC-AUC) — uses synthetic anomalies from step shuffling
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.process_loader import load_sequences
from src.evaluation.process_metrics import (
    evaluate_next_step,
    evaluate_completion,
    evaluate_anomaly,
)
from src.models.bigram_model import BigramModel, predict_next_step, complete_sequence, anomaly_score

FAMILIES = ["IC", "IGBT", "MOSFET"]
COMPLETION_FRACTIONS = [0.6, 0.8]


def _detect_family(sequences: list[list[str]], name: str) -> str:
    for fam in FAMILIES:
        if fam.upper() in name.upper():
            return fam
    return "UNKNOWN"


def load_with_families(
    data_dir: Path,
    families: list[str],
) -> tuple[list[list[str]], list[str]]:
    """Load sequences and attach family label to each."""
    all_seqs: list[list[str]] = []
    all_fams: list[str] = []
    for fam in families:
        seqs = load_sequences(data_dir, product_families=[fam])
        all_seqs.extend(seqs)
        all_fams.extend([fam] * len(seqs))
    return all_seqs, all_fams


def train_val_split(
    sequences: list[list[str]],
    families: list[str],
    val_ratio: float,
    seed: int = 42,
) -> tuple[list, list, list, list]:
    rng = random.Random(seed)
    pairs = list(zip(sequences, families))
    rng.shuffle(pairs)
    split = max(1, int(len(pairs) * (1 - val_ratio)))
    train = pairs[:split]
    val = pairs[split:]
    train_seqs, train_fams = zip(*train) if train else ([], [])
    val_seqs, val_fams = zip(*val) if val else ([], [])
    return list(train_seqs), list(train_fams), list(val_seqs), list(val_fams)


def eval_task1(model: BigramModel, val_seqs: list[list[str]], val_fams: list[str]) -> dict:
    ranked_preds: list[list[str]] = []
    targets: list[str] = []

    for seq, fam in zip(val_seqs, val_fams):
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            partial = seq[:i + 1]
            top5 = predict_next_step(model, None, partial, top_k=5, family=fam)
            ranked_preds.append([s for s, _ in top5])
            targets.append(seq[i + 1])

    metrics = evaluate_next_step(ranked_preds, targets)
    return {"n_samples": len(targets), **metrics.to_dict()}


def eval_task2(model: BigramModel, val_seqs: list[list[str]], val_fams: list[str]) -> dict:
    results = {}
    for frac in COMPLETION_FRACTIONS:
        completions: list[list[str]] = []
        targets: list[list[str]] = []

        for seq, fam in zip(val_seqs, val_fams):
            if len(seq) < 4:
                continue
            cut = max(1, int(len(seq) * frac))
            partial = seq[:cut]
            remaining = seq[cut:]
            predicted = complete_sequence(model, None, partial, family=fam)
            completions.append(predicted)
            targets.append(remaining)

        metrics = evaluate_completion(completions, targets)
        results[f"completion_{int(frac*100)}pct"] = {
            "n_samples": len(completions),
            **metrics.to_dict(),
        }
    return results


def make_synthetic_anomalies(
    valid_seqs: list[list[str]],
    valid_fams: list[str],
    n: int,
    seed: int = 42,
) -> tuple[list[list[str]], list[str], list[int]]:
    """Create synthetic anomalies by randomly swapping 2 non-adjacent steps."""
    rng = random.Random(seed)
    anomaly_seqs: list[list[str]] = []
    anomaly_fams: list[str] = []

    pool = [(s, f) for s, f in zip(valid_seqs, valid_fams) if len(s) >= 4]
    for _ in range(n):
        seq, fam = rng.choice(pool)
        corrupted = list(seq)
        i, j = rng.sample(range(len(corrupted)), 2)
        corrupted[i], corrupted[j] = corrupted[j], corrupted[i]
        anomaly_seqs.append(corrupted)
        anomaly_fams.append(fam)

    all_seqs = list(valid_seqs) + anomaly_seqs
    all_fams = list(valid_fams) + anomaly_fams
    labels = [0] * len(valid_seqs) + [1] * n
    return all_seqs, all_fams, labels


def eval_task3(
    model: BigramModel,
    val_seqs: list[list[str]],
    val_fams: list[str],
    n_anomalies: int | None = None,
) -> dict:
    n = n_anomalies or len(val_seqs)
    all_seqs, all_fams, labels = make_synthetic_anomalies(val_seqs, val_fams, n)

    scores = [
        anomaly_score(model, None, seq, family=fam)
        for seq, fam in zip(all_seqs, all_fams)
    ]

    # cap inf scores for ROC-AUC computation
    finite_max = max((s for s in scores if s < 1e5), default=100.0)
    capped = [min(s, finite_max * 2) for s in scores]

    metrics = evaluate_anomaly(capped, labels)
    oov_count = sum(1 for s in scores if s >= 1e5)
    return {
        "n_valid": len(val_seqs),
        "n_anomaly": n,
        "oov_flagged": oov_count,
        **metrics.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/training_data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"\n{'='*60}")
    print(f"Bigram Baseline Evaluation")
    print(f"Data dir : {data_dir}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"{'='*60}\n")

    # ── load ──────────────────────────────────────────────────────────────────
    all_seqs, all_fams = load_with_families(data_dir, FAMILIES)
    print(f"Total sequences: {len(all_seqs)}\n")

    train_seqs, train_fams, val_seqs, val_fams = train_val_split(
        all_seqs, all_fams, args.val_ratio, args.seed
    )
    print(f"Train: {len(train_seqs)}  Val: {len(val_seqs)}\n")

    # ── fit ───────────────────────────────────────────────────────────────────
    model = BigramModel().fit(train_seqs, train_fams)
    total_bigrams = sum(
        sum(len(c) for c in by_from.values())
        for by_from in model._trans.values()
    )
    print(f"Bigram transitions learned: {total_bigrams:,}")
    for fam, med in sorted(model._family_medians.items()):
        print(f"  {fam:6s} median length: {med:.0f}")
    print()

    # ── Task 1 ────────────────────────────────────────────────────────────────
    print("── Task 1: Next-Step Prediction ─────────────────────────────")
    t1 = eval_task1(model, val_seqs, val_fams)
    print(f"  Samples : {t1['n_samples']:,}")
    print(f"  Top-1   : {t1['top1']:.4f}  ({t1['top1']*100:.1f}%)")
    print(f"  Top-3   : {t1['top3']:.4f}  ({t1['top3']*100:.1f}%)")
    print(f"  Top-5   : {t1['top5']:.4f}  ({t1['top5']*100:.1f}%)")
    print(f"  MRR     : {t1['mrr']:.4f}")
    print()

    # ── Task 2 ────────────────────────────────────────────────────────────────
    print("── Task 2: Sequence Completion ──────────────────────────────")
    t2 = eval_task2(model, val_seqs, val_fams)
    for key, res in t2.items():
        print(f"  [{key}]  n={res['n_samples']}")
        print(f"    Exact Match     : {res['exact_match_rate']:.4f}")
        print(f"    Norm Edit Dist  : {res['normalized_edit_distance']:.4f}  (lower=better)")
        print(f"    Token Accuracy  : {res['token_accuracy']:.4f}")
    print()

    # ── Task 3 ────────────────────────────────────────────────────────────────
    print("── Task 3: Anomaly Detection (synthetic anomalies) ──────────")
    t3 = eval_task3(model, val_seqs, val_fams)
    print(f"  Valid   : {t3['n_valid']}  Anomaly: {t3['n_anomaly']}")
    print(f"  OOV-flagged bigrams: {t3['oov_flagged']}")
    print(f"  Accuracy: {t3['binary_accuracy']:.4f}")
    print(f"  F1      : {t3['f1']:.4f}")
    print(f"  ROC-AUC : {t3['roc_auc']:.4f}")
    print(f"  Threshold (perplexity): {t3['threshold']:.4f}")
    print()
    print("NOTE: Task 3 uses step-swap synthetic anomalies.")
    print("      Real eval requires organizer-provided eval_input_anomaly.csv.")


if __name__ == "__main__":
    main()
