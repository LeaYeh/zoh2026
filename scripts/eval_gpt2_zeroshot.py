"""GPT-2 zero-shot baseline evaluation (no fine-tuning).

Scores all 198 known step names using constrained decoding:
  P(step | context) = sum of BPE token log-probs under pre-trained gpt2

Usage:
    uv run python scripts/eval_gpt2_zeroshot.py
    uv run python scripts/eval_gpt2_zeroshot.py --n-task1 500 --device cpu

Task 2 (sequence completion) is skipped — greedy generation is too slow
for 198-vocab scoring at each step. Run GPT-2 fine-tuned for Task 2 eval.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.process_loader import load_sequences, ProcessStepTokenizer
from src.evaluation.process_metrics import evaluate_next_step, evaluate_anomaly
from src.models.gpt2_zeroshot import GPT2ZeroShot, predict_next_step, anomaly_score

FAMILIES = ["IC", "IGBT", "MOSFET"]


def load_with_families(data_dir: Path) -> tuple[list[list[str]], list[str]]:
    all_seqs, all_fams = [], []
    for fam in FAMILIES:
        seqs = load_sequences(data_dir, product_families=[fam])
        all_seqs.extend(seqs)
        all_fams.extend([fam] * len(seqs))
    return all_seqs, all_fams


def train_val_split(sequences, families, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    pairs = list(zip(sequences, families))
    rng.shuffle(pairs)
    split = max(1, int(len(pairs) * (1 - val_ratio)))
    train = pairs[:split]
    val = pairs[split:]
    ts, tf = zip(*train) if train else ([], [])
    vs, vf = zip(*val) if val else ([], [])
    return list(ts), list(tf), list(vs), list(vf)


def make_synthetic_anomalies(valid_seqs, valid_fams, n, seed=42):
    rng = random.Random(seed)
    anomaly_seqs, anomaly_fams = [], []
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/training_data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--n-task1", type=int, default=300,
                        help="Max prediction points for Task 1 (scoring is slow)")
    parser.add_argument("--n-task3", type=int, default=100,
                        help="Val sequences for Task 3 anomaly eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"\n{'='*60}")
    print("GPT-2 Zero-Shot Baseline Evaluation")
    print(f"Model  : gpt2 (124M, pre-trained, no fine-tuning)")
    print(f"Device : {args.device}")
    print(f"{'='*60}\n")

    # ── load data ─────────────────────────────────────────────────────────────
    all_seqs, all_fams = load_with_families(data_dir)
    train_seqs, train_fams, val_seqs, val_fams = train_val_split(
        all_seqs, all_fams, args.val_ratio, args.seed
    )
    print(f"Train: {len(train_seqs)}  Val: {len(val_seqs)}\n")

    # build vocab from training sequences (no gradient update)
    tokenizer = ProcessStepTokenizer.build(train_seqs)
    step_vocab = [s for s in tokenizer.id_to_step
                  if s not in ("<PAD>", "<BOS>", "<EOS>", "<UNK>")]
    print(f"Vocab size: {len(step_vocab)} step names\n")

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading gpt2 from HuggingFace...")
    model = GPT2ZeroShot(device=args.device)
    model.set_vocab(step_vocab)
    print("Model loaded.\n")

    # ── Task 1 ────────────────────────────────────────────────────────────────
    print("── Task 1: Next-Step Prediction ─────────────────────────────")
    print(f"  (sampling up to {args.n_task1} prediction points)")

    rng = random.Random(args.seed)
    prediction_points: list[tuple[list[str], str]] = []
    for seq in val_seqs:
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            prediction_points.append((seq[:i + 1], seq[i + 1]))

    if len(prediction_points) > args.n_task1:
        prediction_points = rng.sample(prediction_points, args.n_task1)

    ranked_preds, targets = [], []
    for idx, (partial, target) in enumerate(prediction_points):
        if idx % 50 == 0:
            print(f"  scoring {idx}/{len(prediction_points)}...", end="\r")
        top5 = predict_next_step(model, tokenizer, partial, top_k=5)
        ranked_preds.append([s for s, _ in top5])
        targets.append(target)

    print()
    metrics = evaluate_next_step(ranked_preds, targets)
    print(f"  Samples : {len(targets)}")
    print(f"  Top-1   : {metrics.top1:.4f}  ({metrics.top1*100:.1f}%)")
    print(f"  Top-3   : {metrics.top3:.4f}  ({metrics.top3*100:.1f}%)")
    print(f"  Top-5   : {metrics.top5:.4f}  ({metrics.top5*100:.1f}%)")
    print(f"  MRR     : {metrics.mrr:.4f}")
    print()

    # ── Task 2 ────────────────────────────────────────────────────────────────
    print("── Task 2: Sequence Completion ──────────────────────────────")
    print("  SKIPPED — greedy generation requires 198 forward passes per step.")
    print("  Run eval after fine-tuning for meaningful Task 2 scores.\n")

    # ── Task 3 ────────────────────────────────────────────────────────────────
    print("── Task 3: Anomaly Detection (synthetic anomalies) ──────────")
    n3 = min(args.n_task3, len(val_seqs))
    val3_seqs = val_seqs[:n3]
    val3_fams = val_fams[:n3]

    all3_seqs, all3_fams, labels = make_synthetic_anomalies(val3_seqs, val3_fams, n3, args.seed)

    print(f"  Computing perplexity for {len(all3_seqs)} sequences...")
    scores = []
    for idx, (seq, fam) in enumerate(zip(all3_seqs, all3_fams)):
        if idx % 20 == 0:
            print(f"  {idx}/{len(all3_seqs)}...", end="\r")
        scores.append(anomaly_score(model, tokenizer, seq))

    print()
    t3 = evaluate_anomaly(scores, labels)
    print(f"  Valid: {n3}  Anomaly: {n3}")
    print(f"  Accuracy : {t3.binary_accuracy:.4f}")
    print(f"  F1       : {t3.f1:.4f}")
    print(f"  ROC-AUC  : {t3.roc_auc:.4f}")
    print(f"  Threshold (perplexity): {t3.threshold:.2f}")
    print()

    # ── comparison summary ────────────────────────────────────────────────────
    print("── Comparison vs Bigram Baseline ────────────────────────────")
    print("  Task 1 Top-1  :  Bigram=68.3%  |  GPT-2 zero-shot=?")
    print("  Task 3 ROC-AUC:  Bigram=0.996  |  GPT-2 zero-shot=?")
    print("  (fill in after run completes above)")


if __name__ == "__main__":
    main()
