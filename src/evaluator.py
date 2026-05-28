"""Unified evaluator for probabilistic time-series forecasts.

Usage:
    metrics = evaluate(predictions, ground_truth, naive_scale)
    # predictions: np.ndarray shape (n_series, n_samples, pred_len)  — sample-based
    # ground_truth: np.ndarray shape (n_series, pred_len)
    # naive_scale: np.ndarray shape (n_series,) — mean |diff| of training series (for MASE)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict


QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class Metrics:
    mae: float
    mase: float
    rmse: float
    crps: float
    pinball_10: float
    pinball_50: float
    pinball_90: float
    coverage_80: float  # fraction of actuals inside [q10, q90]
    coverage_90: float  # fraction inside [q05, q95]

    def to_dict(self) -> dict:
        return asdict(self)


# ── point metrics ────────────────────────────────────────────────────────────

def _mae(median: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(median - truth)))


def _rmse(median: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((median - truth) ** 2)))


def _mase(median: np.ndarray, truth: np.ndarray, naive_scale: np.ndarray) -> float:
    """MASE = MAE / naive_scale.  naive_scale = mean |y_t - y_{t-1}| on train."""
    abs_err = np.abs(median - truth).mean(axis=-1)          # (n_series,)
    safe_scale = np.where(naive_scale == 0, 1.0, naive_scale)
    return float(np.mean(abs_err / safe_scale))


# ── probabilistic metrics ────────────────────────────────────────────────────

def _pinball(samples: np.ndarray, truth: np.ndarray, q: float) -> float:
    """Pinball / quantile loss. samples: (n_series, n_samples, pred_len)."""
    forecast_q = np.quantile(samples, q, axis=1)            # (n_series, pred_len)
    err = truth - forecast_q
    loss = np.where(err >= 0, q * err, (q - 1) * err)
    return float(np.mean(loss))


def _crps(samples: np.ndarray, truth: np.ndarray) -> float:
    """Energy-form CRPS: E|X-y| - 0.5*E|X-X'|.  O(n_samples^2) in last term."""
    # (n_series, n_samples, pred_len)
    n_samples = samples.shape[1]
    truth_exp = truth[:, np.newaxis, :]                     # (n_series, 1, pred_len)
    term1 = np.abs(samples - truth_exp).mean(axis=1)        # (n_series, pred_len)

    # pairwise |X - X'| via broadcasting — use subset if n_samples > 100
    s = samples if n_samples <= 100 else samples[:, :100, :]
    diff = np.abs(s[:, :, np.newaxis, :] - s[:, np.newaxis, :, :])  # expensive
    term2 = diff.mean(axis=(1, 2))                          # (n_series, pred_len)

    return float(np.mean(term1 - 0.5 * term2))


def _interval_coverage(samples: np.ndarray, truth: np.ndarray, lo_q: float, hi_q: float) -> float:
    lo = np.quantile(samples, lo_q, axis=1)
    hi = np.quantile(samples, hi_q, axis=1)
    inside = (truth >= lo) & (truth <= hi)
    return float(inside.mean())


# ── public API ───────────────────────────────────────────────────────────────

def evaluate(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    naive_scale: np.ndarray | None = None,
) -> Metrics:
    """
    Args:
        predictions:  (n_series, n_samples, pred_len)  sample draws
        ground_truth: (n_series, pred_len)
        naive_scale:  (n_series,)  mean |diff| on training window (for MASE)
                      If None, MASE is computed with scale=1 (same as MAE).
    """
    if naive_scale is None:
        naive_scale = np.ones(predictions.shape[0])

    median = np.median(predictions, axis=1)                 # (n_series, pred_len)

    return Metrics(
        mae=_mae(median, ground_truth),
        mase=_mase(median, ground_truth, naive_scale),
        rmse=_rmse(median, ground_truth),
        crps=_crps(predictions, ground_truth),
        pinball_10=_pinball(predictions, ground_truth, 0.1),
        pinball_50=_pinball(predictions, ground_truth, 0.5),
        pinball_90=_pinball(predictions, ground_truth, 0.9),
        coverage_80=_interval_coverage(predictions, ground_truth, 0.1, 0.9),
        coverage_90=_interval_coverage(predictions, ground_truth, 0.05, 0.95),
    )


# ── time-series CV split (no leakage) ────────────────────────────────────────

def rolling_origin_splits(
    n: int,
    pred_len: int,
    n_windows: int,
    min_train: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns list of (train_idx, test_idx) tuples.
    Each test window is pred_len steps; test windows do not overlap.
    Train is expanding (all data before the test window).
    """
    if min_train is None:
        min_train = max(pred_len * 2, n // 4)

    total_test = pred_len * n_windows
    if n - min_train < total_test:
        raise ValueError(
            f"Not enough data: need {min_train + total_test} points, got {n}."
        )

    splits = []
    for i in range(n_windows):
        test_end = n - pred_len * (n_windows - 1 - i)
        test_start = test_end - pred_len
        train_idx = np.arange(test_start)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    return splits


def naive_scale_from_series(series: np.ndarray, freq: int = 1) -> float:
    """Mean absolute seasonal difference. Use freq=1 for non-seasonal."""
    diffs = np.abs(np.diff(series[::freq]))
    return float(diffs.mean()) if len(diffs) > 0 else 1.0
