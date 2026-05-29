"""Track 1 evaluation metrics.

Task 1  Next-Step Prediction  — Top-1/3/5 Accuracy, MRR
Task 2  Sequence Completion   — Exact Match Rate, Normalized Edit Distance, Token Accuracy
Task 3  Anomaly Detection     — Binary Accuracy, Precision, Recall, F1, ROC-AUC
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Task1Metrics:
    top1: float
    top3: float
    top5: float
    mrr: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Task2Metrics:
    exact_match_rate: float
    normalized_edit_distance: float  # lower is better
    token_accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Task3Metrics:
    binary_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    threshold: float  # perplexity threshold used for classification

    def to_dict(self) -> dict:
        return asdict(self)


# ── Task 1 ────────────────────────────────────────────────────────────────────

def top_k_accuracy(
    ranked_preds: list[list[str]],
    targets: list[str],
    k: int,
) -> float:
    """Fraction of samples where the true next step is in the top-k predictions."""
    if not targets:
        return 0.0
    return float(sum(t in p[:k] for p, t in zip(ranked_preds, targets)) / len(targets))


def mrr(ranked_preds: list[list[str]], targets: list[str]) -> float:
    """Mean Reciprocal Rank of the true next step in the ranked predictions."""
    if not targets:
        return 0.0
    rrs = []
    for preds, target in zip(ranked_preds, targets):
        try:
            rrs.append(1.0 / (preds.index(target) + 1))
        except ValueError:
            rrs.append(0.0)
    return float(np.mean(rrs))


def evaluate_next_step(
    ranked_preds: list[list[str]],
    targets: list[str],
) -> Task1Metrics:
    return Task1Metrics(
        top1=top_k_accuracy(ranked_preds, targets, k=1),
        top3=top_k_accuracy(ranked_preds, targets, k=3),
        top5=top_k_accuracy(ranked_preds, targets, k=5),
        mrr=mrr(ranked_preds, targets),
    )


# ── Task 2 ────────────────────────────────────────────────────────────────────

def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein edit distance between two sequences of strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def normalized_edit_distance(
    completions: list[list[str]],
    targets: list[list[str]],
) -> float:
    """Mean Levenshtein distance normalised by max(len_completion, len_target)."""
    if not completions:
        return 0.0
    scores = [
        _edit_distance(c, t) / max(len(c), len(t), 1)
        for c, t in zip(completions, targets)
    ]
    return float(np.mean(scores))


def exact_match_rate(
    completions: list[list[str]],
    targets: list[list[str]],
) -> float:
    if not completions:
        return 0.0
    return float(np.mean([c == t for c, t in zip(completions, targets)]))


def token_accuracy(
    completions: list[list[str]],
    targets: list[list[str]],
) -> float:
    """Per-token accuracy; length mismatch counts as incorrect for extra tokens."""
    total = correct = 0
    for c, t in zip(completions, targets):
        length = min(len(c), len(t))
        correct += sum(c[i] == t[i] for i in range(length))
        total += max(len(c), len(t))
    return float(correct / total) if total else 0.0


def evaluate_completion(
    completions: list[list[str]],
    targets: list[list[str]],
) -> Task2Metrics:
    return Task2Metrics(
        exact_match_rate=exact_match_rate(completions, targets),
        normalized_edit_distance=normalized_edit_distance(completions, targets),
        token_accuracy=token_accuracy(completions, targets),
    )


# ── Task 3 ────────────────────────────────────────────────────────────────────

def evaluate_anomaly(
    scores: list[float],
    labels: list[int],
    threshold: float | None = None,
) -> Task3Metrics:
    """
    scores  — anomaly score per sequence (higher = more anomalous, e.g. perplexity)
    labels  — 0 = valid sequence, 1 = anomaly
    threshold — if None, uses median of scores
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    )
    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)

    thr = float(np.median(scores_arr)) if threshold is None else threshold
    preds = (scores_arr >= thr).astype(int)

    roc = (
        float(roc_auc_score(labels_arr, scores_arr))
        if len(set(labels_arr.tolist())) > 1
        else 0.5
    )

    return Task3Metrics(
        binary_accuracy=float(accuracy_score(labels_arr, preds)),
        precision=float(precision_score(labels_arr, preds, zero_division=0)),
        recall=float(recall_score(labels_arr, preds, zero_division=0)),
        f1=float(f1_score(labels_arr, preds, zero_division=0)),
        roc_auc=roc,
        threshold=thr,
    )
