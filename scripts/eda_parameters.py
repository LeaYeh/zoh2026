"""EDA on parsed fab-level parameters: cross-family comparison and PCA separability.

Usage:
    uv run python scripts/eda_parameters.py
    uv run python scripts/eda_parameters.py --parsed data/processed/parameters_parsed.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FAMILIES = ["IC", "IGBT", "MOSFET"]
FAMILY_COLORS = {"IC": "#4C72B0", "IGBT": "#DD8452", "MOSFET": "#55A868"}
OUT_DIR = Path("docs/eda_figures")

# Numeric params we care about for PCA (must appear in multiple families)
NUMERIC_KEYS = [
    "temperature_c", "time_min", "time_s", "pressure_mtorr",
    "power_w", "speed_rpm", "thickness_nm", "thickness_um",
    "dose_cm2", "energy_kev", "flow_sccm", "concentration_pct",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _scalar(v) -> float | None:
    """Extract a single float from a param value (point, range, or tolerance)."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        if "value" in v:
            return float(v["value"])
        if "min" in v and "max" in v:
            return (float(v["min"]) + float(v["max"])) / 2
    return None


def build_df(records: list[dict]) -> pd.DataFrame:
    """Flatten parsed records into a long-form DataFrame."""
    rows = []
    for r in records:
        base = {"family": r["family"], "step": r["step"]}
        for k, v in r.get("params", {}).items():
            s = _scalar(v)
            rows.append({**base, "param": k, "value": s,
                         "raw_value": v, "is_numeric": s is not None})
    return pd.DataFrame(rows)


def build_step_matrix(records: list[dict], keys: list[str]) -> pd.DataFrame:
    """Build a (step × param) matrix of midpoint values for PCA."""
    data: dict[tuple, dict] = {}
    for r in records:
        key = (r["family"], r["step"])
        for k in keys:
            v = r.get("params", {}).get(k)
            s = _scalar(v)
            if s is not None:
                data.setdefault(key, {})[k] = s

    rows = []
    for (fam, step), params in data.items():
        row = {"family": fam, "step": step}
        row.update({k: params.get(k) for k in keys})
        rows.append(row)
    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_param_coverage(df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: which steps have which numeric params, per family."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    for ax, fam in zip(axes, FAMILIES):
        sub = df[(df["family"] == fam) & df["is_numeric"]]
        pivot = sub.pivot_table(index="step", columns="param", values="value",
                                aggfunc="count").fillna(0)
        pivot = pivot.loc[:, (pivot > 0).any()]
        sns.heatmap(pivot, ax=ax, cmap="Blues", linewidths=0.3, cbar=False,
                    xticklabels=True, yticklabels=True)
        ax.set_title(f"{fam} — numeric param coverage", fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()
    path = out_dir / "param_coverage_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_cross_family_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    """For each numeric param shared by ≥2 families, compare distributions."""
    numeric = df[df["is_numeric"]].copy()
    # Params present in ≥2 families
    family_counts = numeric.groupby("param")["family"].nunique()
    shared_params = family_counts[family_counts >= 2].index.tolist()
    shared_params = [p for p in NUMERIC_KEYS if p in shared_params][:8]  # top 8

    if not shared_params:
        print("  no shared numeric params found, skipping boxplots")
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for ax, param in zip(axes, shared_params):
        sub = numeric[numeric["param"] == param]
        data_by_fam = [sub[sub["family"] == f]["value"].dropna().values for f in FAMILIES]
        bp = ax.boxplot(data_by_fam, labels=FAMILIES, patch_artist=True, notch=False)
        for patch, fam in zip(bp["boxes"], FAMILIES):
            patch.set_facecolor(FAMILY_COLORS[fam])
            patch.set_alpha(0.7)
        ax.set_title(param, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(shared_params):]:
        ax.set_visible(False)
    plt.suptitle("Cross-family parameter distributions (shared params)", fontsize=12)
    plt.tight_layout()
    path = out_dir / "cross_family_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_pca(matrix: pd.DataFrame, out_dir: Path) -> dict:
    """PCA on numeric params — do the 3 families separate in parameter space?"""
    feature_cols = [c for c in NUMERIC_KEYS if c in matrix.columns]
    X = matrix[feature_cols].copy()

    # Keep only params present in ≥10% of rows
    col_coverage = X.notna().mean()
    feature_cols = [c for c in feature_cols if col_coverage.get(c, 0) >= 0.10]
    X = matrix[feature_cols].copy()

    # Drop rows with too many nulls (>90% missing across retained params)
    row_null_frac = X.isnull().mean(axis=1)
    X = X[row_null_frac <= 0.9]
    labels = matrix.loc[X.index, "family"]
    steps = matrix.loc[X.index, "step"]

    # Impute remaining nulls with column median
    X = X.fillna(X.median())

    if len(X) < 10:
        print("  not enough rows for PCA, skipping")
        return {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(3, len(feature_cols)))
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    for fam in FAMILIES:
        mask = labels == fam
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=fam, color=FAMILY_COLORS[fam],
                   alpha=0.7, s=60, edgecolors="white", linewidths=0.5)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)", fontsize=10)
    ax.set_title("PCA of fab-level parameters — family separability", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Annotate outliers (top-5 most extreme PC1 points per family)
    for fam in FAMILIES:
        mask = np.array(labels == fam)
        if mask.sum() == 0:
            continue
        fam_pcs = X_pca[mask, 0]
        fam_steps = steps[labels == fam].values
        top_idx = np.argsort(np.abs(fam_pcs))[-3:]
        for idx in top_idx:
            ax.annotate(fam_steps[idx], (X_pca[mask][idx, 0], X_pca[mask][idx, 1]),
                        fontsize=5, alpha=0.6)

    plt.tight_layout()
    path = out_dir / "pca_family_separability.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")

    # Feature loadings
    loadings = pd.DataFrame(pca.components_.T, index=feature_cols,
                            columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    print("\n  Top PC1 loadings (most discriminating parameters):")
    print(loadings["PC1"].abs().sort_values(ascending=False).head(6).to_string())

    return {"explained_variance": explained.tolist(), "loadings": loadings.to_dict()}


def print_shared_steps(records: list[dict]) -> None:
    """Print steps shared by all 3 families and their parameter diffs."""
    by_family: dict[str, dict] = defaultdict(dict)
    for r in records:
        by_family[r["family"]][r["step"]] = r.get("params", {})

    shared = set(by_family["IC"]) & set(by_family["IGBT"]) & set(by_family["MOSFET"])
    print(f"\n  Steps shared by all 3 families: {len(shared)}")

    # Find steps where temperature_c differs across families
    diffs = []
    for step in shared:
        temps = {}
        for fam in FAMILIES:
            v = by_family[fam][step].get("temperature_c")
            s = _scalar(v)
            if s is not None:
                temps[fam] = s
        if len(temps) >= 2:
            values = list(temps.values())
            spread = max(values) - min(values)
            if spread > 0:
                diffs.append((spread, step, temps))

    diffs.sort(reverse=True)
    print(f"\n  Steps where temperature_c differs most between families:")
    for spread, step, temps in diffs[:8]:
        temp_str = "  ".join(f"{f}={v:.0f}°C" for f, v in temps.items())
        print(f"    Δ={spread:.0f}°C  {step:<40s}  {temp_str}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed", default="data/processed/parameters_parsed.json")
    parser.add_argument("--out-dir", default="docs/eda_figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.parsed, encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} parsed records")

    df = build_df(records)
    matrix = build_step_matrix(records, NUMERIC_KEYS)
    print(f"Step matrix: {len(matrix)} rows × {len([c for c in NUMERIC_KEYS if c in matrix.columns])} numeric params")

    print("\n── Coverage ─────────────────────────────────────────────────")
    for fam in FAMILIES:
        sub = df[df["family"] == fam]
        n_steps = sub["step"].nunique()
        n_with_numeric = sub[sub["is_numeric"]]["step"].nunique()
        print(f"  {fam:6s}: {n_with_numeric}/{n_steps} steps have numeric params")
        top_params = (sub[sub["is_numeric"]].groupby("param")["step"]
                      .count().sort_values(ascending=False).head(5))
        print(f"    top params: {', '.join(f'{k}({v})' for k, v in top_params.items())}")

    print_shared_steps(records)

    print("\n── Plots ────────────────────────────────────────────────────")
    plot_param_coverage(df, out_dir)
    plot_cross_family_boxplots(df, out_dir)
    pca_results = plot_pca(matrix, out_dir)

    if pca_results:
        ev = pca_results["explained_variance"]
        print(f"\n  PCA explained variance: PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%  total={sum(ev[:2])*100:.1f}%")
        sep = sum(ev[:2])
        if sep >= 0.4:
            print("  → Parameters have discriminative power (≥40% variance in 2 PCs)")
            print("    Proceed to step-level family affinity scoring (ADR-008 family conditioning)")
        else:
            print("  → Weak separation — parameters alone may not reliably identify family")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
