#!/usr/bin/env python3
"""
Benchmark Chronos T5-small on G-Research Crypto Forecasting (Kaggle 2022).

Competition: https://www.kaggle.com/competitions/g-research-crypto-forecasting
Metric: Weighted Pearson Correlation between predicted and actual 15-min
        residualized returns (per coin, weighted by competition weights).

What we do:
  1. Download competition data via Kaggle API
  2. Resample minute OHLCV → daily close prices per asset
  3. Run Chronos zero-shot + quick LoRA fine-tune
  4. Evaluate: CRPS, Direction Acc, Pearson r, simulated Sharpe
  5. Compare to published leaderboard scores

Note on metric gap:
  Competition evaluates on residualized (market-factor-removed) 15-min returns.
  We evaluate on daily absolute prices → our Pearson r is NOT directly comparable
  to the leaderboard. We show it as an indicative measure of signal quality.

Leaderboard reference (final public lb):
  1st place:  ~0.0268  (weighted Pearson on residualized returns)
  Top 10%:    ~0.0140
  Median:     ~0.0050
  Baseline:    0.0000  (random / constant prediction)
"""
import sys, os, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from chronos import ChronosPipeline
from pathlib import Path
from scripts.benchmark_financial import (
    crps_samples, trading_metrics, predict, quick_lora_finetune,
    CONTEXT_LEN, NUM_SAMPLES, LORA_STEPS,
)

# ─── Config ────────────────────────────────────────────────────────────────────
COMPETITION   = "g-research-crypto-forecasting"
DATA_DIR      = Path("data/raw/gresearch")
PRED_LEN      = 30        # 30-day horizon (mirroring benchmark_financial)
TRAIN_END_DAY = "2021-06-01"  # competition training data ends ~mid-2021
TEST_DAYS     = 30

# G-Research coin Asset_IDs and names
ASSET_MAP = {
    1: "Bitcoin", 2: "Litecoin", 6: "Ethereum", 7: "Ethereum Classic",
    8: "XRP", 10: "Stellar", 11: "Dogecoin", 12: "Cardano",
    13: "IOTA", 14: "Maker", 15: "Bitcoin Cash (Binance)", 16: "EOS",
}

# Published leaderboard reference scores (Weighted Pearson Correlation)
LEADERBOARD_REF = {
    "1st place":   0.02682,
    "Top 10%":     0.01400,
    "Top 25%":     0.00830,
    "Median":      0.00500,
    "Random/0":    0.00000,
}


# ─── Kaggle download ───────────────────────────────────────────────────────────
def download_data() -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_csv = DATA_DIR / "train.csv"
    if train_csv.exists():
        print(f"  Data already exists at {DATA_DIR}/")
        return True

    try:
        import kaggle
        print(f"  Downloading {COMPETITION} data (may take a few minutes)...")
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            COMPETITION,
            path=str(DATA_DIR),
            quiet=False,
        )
        # Unzip if needed
        for zf in DATA_DIR.glob("*.zip"):
            print(f"  Extracting {zf.name}...")
            with zipfile.ZipFile(zf) as z:
                z.extractall(DATA_DIR)
            zf.unlink()
        return True
    except Exception as e:
        print(f"  Kaggle download failed: {e}")
        print("  Make sure ~/.config/kaggle/kaggle.json is present and")
        print(f"  you have accepted the competition rules at:")
        print(f"  https://www.kaggle.com/competitions/{COMPETITION}/rules")
        return False


# ─── Data loading ──────────────────────────────────────────────────────────────
def load_daily_series() -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """
    Read train.csv, resample minute-level to daily close, split train/test.
    Returns (train_seqs, test_targets, asset_names).
    """
    csv_path = DATA_DIR / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — run download first")

    print(f"  Reading {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=["timestamp", "Asset_ID", "Close"])

    # Load per-asset weights from asset_details.csv
    details_path = DATA_DIR / "asset_details.csv"
    weight_map = {}
    if details_path.exists():
        details = pd.read_csv(details_path)
        weight_map = dict(zip(details["Asset_ID"], details["Weight"]))

    # timestamp is Unix seconds
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.normalize()
    df = df.dropna(subset=["Close"])

    # Resample to daily last close per asset
    daily = (
        df.groupby(["Asset_ID", "date"])["Close"]
        .last()
        .reset_index()
    )

    split_date = pd.Timestamp(TRAIN_END_DAY)
    train_seqs, test_targets, asset_names = [], [], []

    # Use all available asset IDs from the data + details file
    all_ids = daily["Asset_ID"].unique()
    for asset_id in sorted(all_ids):
        name = ASSET_MAP.get(asset_id) or weight_map.get(asset_id, f"Asset_{asset_id}")
        # get name from asset_details if available
        asset_df = daily[daily["Asset_ID"] == asset_id].sort_values("date")
        if len(asset_df) < CONTEXT_LEN + TEST_DAYS:
            continue

        prices = asset_df["Close"].values.astype(np.float32)
        dates  = asset_df["date"].values

        split_idx = int(np.searchsorted(dates, np.datetime64(TRAIN_END_DAY)))
        if split_idx < CONTEXT_LEN:
            continue
        if len(prices) - split_idx < TEST_DAYS:
            continue

        w = weight_map.get(asset_id, 1.0)
        train_seqs.append(prices[:split_idx])
        test_targets.append(prices[split_idx:split_idx + TEST_DAYS])
        asset_names.append(f"{name}(w={w:.2f})")
        print(f"  OK {name:<22}  train={split_idx}  test={TEST_DAYS}  weight={w:.3f}")

    print(f"\n  Loaded {len(asset_names)} assets.\n")
    return train_seqs, test_targets, asset_names


# ─── Weighted Pearson correlation (G-Research metric proxy) ───────────────────
def weighted_pearson(pred_returns: list[float], actual_returns: list[float],
                     weights: list[float]) -> float:
    """
    Weighted Pearson correlation across assets.
    Mirrors competition evaluation but on daily, not 15-min residualized.
    """
    p = np.array(pred_returns)
    a = np.array(actual_returns)
    w = np.array(weights) / np.sum(weights)

    p_wmean = np.sum(w * p)
    a_wmean = np.sum(w * a)
    p_c = p - p_wmean
    a_c = a - a_wmean

    cov = np.sum(w * p_c * a_c)
    std_p = np.sqrt(np.sum(w * p_c**2))
    std_a = np.sqrt(np.sum(w * a_c**2))
    if std_p < 1e-8 or std_a < 1e-8:
        return 0.0
    return float(cov / (std_p * std_a))


# ─── Evaluate one model ────────────────────────────────────────────────────────
def evaluate_model(pipeline, train_series, test_targets, asset_names,
                   label: str) -> list[dict]:
    print(f"─" * 50)
    print(f"Evaluating: {label}")
    print(f"─" * 50)

    rows = []
    for i, (name, train, test) in enumerate(zip(asset_names, train_series, test_targets)):
        samples = predict(pipeline, train, PRED_LEN)
        c  = crps_samples(samples, test)
        tm = trading_metrics(samples, test, entry_price=float(train[-1]))
        row = {"name": name, "crps": c, **tm}
        rows.append(row)
        print(f"  [{i+1:2d}/{len(asset_names)}] {name:<22}  "
              f"CRPS={c:.4f}  DirAcc={tm['direction_acc']:.0%}  "
              f"r={tm['pearson_r']:+.3f}")

    return rows


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("G-Research Crypto Forecasting — Kaggle Benchmark")
    print("=" * 60)

    # ── Download ───────────────────────────────────────────────────────────────
    print("\n[Step 1] Data download")
    if not download_data():
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("[Step 2] Load and resample to daily")
    train_series, test_targets, asset_names = load_daily_series()
    if not asset_names:
        print("No assets loaded.")
        sys.exit(1)

    # ── Load model ─────────────────────────────────────────────────────────────
    print("[Step 3] Load Chronos T5-small")
    pipeline_zs = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        dtype=torch.float32,
    )

    # ── Zero-shot eval ─────────────────────────────────────────────────────────
    print("\n[Step 4] Zero-shot evaluation")
    zs_rows = evaluate_model(pipeline_zs, train_series, test_targets,
                             asset_names, "Chronos T5-small (zero-shot)")

    # ── LoRA fine-tune ─────────────────────────────────────────────────────────
    print(f"\n[Step 5] LoRA fine-tune ({LORA_STEPS} steps)")
    pipeline_ft = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        dtype=torch.float32,
    )
    pipeline_ft = quick_lora_finetune(pipeline_ft, train_series, steps=LORA_STEPS)
    print()

    ft_rows = evaluate_model(pipeline_ft, train_series, test_targets,
                             asset_names, "Chronos T5-small (LoRA fine-tuned)")

    # ── Weighted Pearson (competition proxy metric) ────────────────────────────
    # Equal weights here since we don't have the original Weight column per-row
    weights = [1.0] * len(asset_names)

    zs_wp = weighted_pearson(
        [r["pearson_r"] for r in zs_rows],
        [1.0] * len(zs_rows),   # actual = 1.0 as reference direction
        weights,
    )
    # Use trade_pnl sign as predicted return direction vs actual return sign
    def _wpc_from_rows(rows):
        pred_rets   = [r["trade_pnl"] for r in rows]
        actual_rets = [r["trade_pnl"] / abs(r["trade_pnl"]) if abs(r["trade_pnl"]) > 1e-8 else 0
                       for r in rows]
        return weighted_pearson(
            [r["pearson_r"] for r in rows],
            [abs(r["pearson_r"]) for r in rows],
            weights,
        )

    # ── Per-asset table ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("PER-ASSET RESULTS")
    print("=" * 70)
    print(f"{'Asset':<22} {'ZS-CRPS':>8} {'FT-CRPS':>8} "
          f"{'ZS-Dir':>7} {'FT-Dir':>7} {'ZS-r':>7} {'FT-r':>7} {'Win':>5}")
    print("-" * 70)
    for zr, fr in zip(zs_rows, ft_rows):
        winner = "FT" if fr["crps"] < zr["crps"] else "ZS"
        print(f"{zr['name']:<22} {zr['crps']:8.4f} {fr['crps']:8.4f} "
              f"{zr['direction_acc']:7.0%} {fr['direction_acc']:7.0%} "
              f"{zr['pearson_r']:+7.3f} {fr['pearson_r']:+7.3f} {winner:>5}")

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    def _agg(rows, key):
        return float(np.nanmean([r[key] for r in rows]))

    def _sharpe(rows):
        pnls = np.array([r["trade_pnl"] for r in rows])
        ann  = np.sqrt(252 / PRED_LEN)
        return float(pnls.mean() / pnls.std() * ann) if pnls.std() > 1e-8 else 0.0

    print()
    print("=" * 70)
    print("AGGREGATE METRICS (our model on G-Research daily data)")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Zero-shot':>12} {'LoRA FT':>12} {'Delta':>10}")
    print("-" * 66)
    for label, key, lower_better in [
        ("CRPS (↓)",           "crps",         True),
        ("Direction Acc (↑)",  "direction_acc", False),
        ("Sharpe (↑)",         None,            False),
        ("Pearson r (↑)",      "pearson_r",     False),
        ("Return R² (↑)",      "return_r2",     False),
    ]:
        if key is None:
            zv, fv = _sharpe(zs_rows), _sharpe(ft_rows)
        else:
            zv, fv = _agg(zs_rows, key), _agg(ft_rows, key)
        d = fv - zv
        ok = "✓" if (d < 0) == lower_better else "✗"
        print(f"  {label:<28} {zv:>12.4f} {fv:>12.4f} {d:>+10.4f}  {ok}")

    # ── Gap vs leaderboard ─────────────────────────────────────────────────────
    our_pearson_r = _agg(zs_rows, "pearson_r")

    print()
    print("=" * 70)
    print("GAP VS G-RESEARCH LEADERBOARD")
    print("=" * 70)
    print()
    print("IMPORTANT: Leaderboard uses 15-min RESIDUALIZED returns.")
    print("Our metric uses daily RAW price Pearson r. Not a direct comparison.")
    print("Use as relative benchmark only.\n")
    print(f"{'Leaderboard tier':<25} {'Score (Pearson r)':>18} {'Our gap':>12}")
    print("-" * 58)
    for tier, score in LEADERBOARD_REF.items():
        gap = our_pearson_r - score
        print(f"  {tier:<23} {score:>18.5f} {gap:>+12.5f}")
    print(f"\n  Our ZS Pearson r (daily):  {our_pearson_r:+.5f}")
    print(f"  Our FT Pearson r (daily):  {_agg(ft_rows,'pearson_r'):+.5f}")

    # ── Final slide table ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("HACKATHON SLIDE TABLE — G-Research Crypto")
    print("=" * 70)
    print()
    print(f"| {'Model':<25} | {'CRPS':>6} | {'DirAcc':>7} | {'Sharpe':>7} | {'Pearson r':>9} |")
    print(f"|{'-'*27}|{'-'*8}|{'-'*9}|{'-'*9}|{'-'*11}|")
    print(f"| {'Chronos ZS (ours)':<25} | {_agg(zs_rows,'crps'):.4f} | "
          f"{_agg(zs_rows,'direction_acc'):7.0%} | "
          f"{_sharpe(zs_rows):+7.2f} | "
          f"{_agg(zs_rows,'pearson_r'):+9.4f} |")
    print(f"| {'Chronos LoRA (ours)':<25} | {_agg(ft_rows,'crps'):.4f} | "
          f"{_agg(ft_rows,'direction_acc'):7.0%} | "
          f"{_sharpe(ft_rows):+7.2f} | "
          f"{_agg(ft_rows,'pearson_r'):+9.4f} |")
    print(f"| {'G-Research 1st place':<25} | {'—':>6} | {'—':>7} | {'—':>7} | {'0.02682':>9} |")
    print(f"| {'G-Research Top 10%':<25} | {'—':>6} | {'—':>7} | {'—':>7} | {'0.01400':>9} |")
    print(f"| {'G-Research Median':<25} | {'—':>6} | {'—':>7} | {'—':>7} | {'0.00500':>9} |")
    print()
    print("* Leaderboard Pearson r measured on 15-min residualized returns.")
    print("  Our Pearson r measured on daily raw prices. Not directly comparable.")


if __name__ == "__main__":
    main()
