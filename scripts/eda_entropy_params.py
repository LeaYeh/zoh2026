"""Entropy analysis: does conditioning on parameter buckets reduce next-step prediction entropy?

Key question (ADR-013 Level 2 gate): does adding temperature_c / time_min bucket tokens
to training sequences reduce H(next_step | current_step, family)?

Run:
    uv run python scripts/eda_entropy_params.py
    uv run python scripts/eda_entropy_params.py --params data/processed/parameters_parsed.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


FAMILIES = ["IC", "IGBT", "MOSFET"]

NUMERIC_KEYS = [
    "temperature_c", "time_min", "time_s", "pressure_mtorr",
    "power_w", "speed_rpm", "thickness_nm", "thickness_um",
    "dose_cm2", "energy_kev", "flow_sccm",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _scalar(v) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        if "value" in v:
            return float(v["value"])
        if "min" in v and "max" in v:
            return (float(v["min"]) + float(v["max"])) / 2.0
    return None


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total) for c in counts.values() if c > 0
    )


def weighted_conditional_entropy(bigrams: list[tuple[str, str, str]]) -> float:
    """H(next | current, family) averaged over (current, family) contexts."""
    ctx_counts: dict[tuple, Counter] = defaultdict(Counter)
    for cur, nxt, fam in bigrams:
        ctx_counts[(cur, fam)][nxt] += 1
    total = len(bigrams)
    return sum(
        (sum(c.values()) / total) * _entropy(c)
        for c in ctx_counts.values()
    )


def load_sequences_by_family(data_dir: Path) -> dict[str, list[list[str]]]:
    """Load CSV sequences, infer family from filename."""
    import csv
    result: dict[str, list[list[str]]] = defaultdict(list)
    for csv_path in sorted(data_dir.glob("*.csv")):
        stem = csv_path.stem.upper()
        family = next((f for f in FAMILIES if f in stem), None)
        if family is None:
            continue
        # Only load variant files (skip longdescr / parameter files)
        if "LONGDESCR" in stem or "PARAMETER" in stem or "LONGDESCRIPT" in stem:
            continue
        with open(csv_path, encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            col_map = {c.upper(): c for c in (reader.fieldnames or [])}
            seq_col = col_map.get("SEQUENCE_ID")
            step_col = col_map.get("STEP")
            if not step_col:
                continue
            current_seq_id = None
            current_steps: list[str] = []
            for row in reader:
                sid = row.get(seq_col, "") if seq_col else ""
                step = row[step_col].strip()
                if sid != current_seq_id:
                    if current_steps:
                        result[family].append(current_steps)
                    current_seq_id = sid
                    current_steps = [step]
                else:
                    current_steps.append(step)
            if current_steps:
                result[family].append(current_steps)
        print(f"  {csv_path.name}: {len(result[family])} sequences ({family})")
    return dict(result)


def build_bigrams(seqs_by_family: dict[str, list[list[str]]]) -> list[tuple[str, str, str]]:
    bigrams = []
    for fam, seqs in seqs_by_family.items():
        for seq in seqs:
            for cur, nxt in zip(seq, seq[1:]):
                bigrams.append((cur, nxt, fam))
    return bigrams


def build_params_lookup(records: list[dict]) -> dict[tuple[str, str], dict]:
    """(step, family) → params dict with scalar values."""
    lookup: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r["step"], r["family"])
        scalars = {k: _scalar(v) for k, v in r.get("params", {}).items()}
        lookup[key] = {k: v for k, v in scalars.items() if v is not None}
    return lookup


def percentile_bucket(value: float, p33: float, p67: float) -> str:
    if value <= p33:
        return "low"
    if value <= p67:
        return "mid"
    return "high"


def compute_param_entropy(
    bigrams: list[tuple[str, str, str]],
    lookup: dict[tuple[str, str], dict],
    param_key: str,
) -> dict:
    """
    For bigrams where current_step has param_key:
    1. Check if param is deterministic per (step, family) — it should be.
    2. Compute H(next | current, family) and H(next | current, family, param_bucket).
    3. Return stats.
    """
    # Collect values to compute percentile buckets
    values = []
    for cur, _nxt, fam in bigrams:
        v = lookup.get((cur, fam), {}).get(param_key)
        if v is not None:
            values.append(v)

    if not values:
        return {"coverage": 0.0, "n_bigrams": 0}

    values_sorted = sorted(values)
    n = len(values_sorted)
    p33 = values_sorted[n // 3]
    p67 = values_sorted[2 * n // 3]

    # Check determinism: does each (step, family) always map to the same bucket?
    step_fam_buckets: dict[tuple, set] = defaultdict(set)
    covered_bigrams = []
    for cur, nxt, fam in bigrams:
        v = lookup.get((cur, fam), {}).get(param_key)
        if v is not None:
            bucket = percentile_bucket(v, p33, p67)
            step_fam_buckets[(cur, fam)].add(bucket)
            covered_bigrams.append((cur, nxt, fam, bucket))

    non_deterministic = {k: v for k, v in step_fam_buckets.items() if len(v) > 1}

    # Baseline H(next | current, family) on covered bigrams only
    baseline_ctx: dict[tuple, Counter] = defaultdict(Counter)
    for cur, nxt, fam, _b in covered_bigrams:
        baseline_ctx[(cur, fam)][nxt] += 1

    total_covered = len(covered_bigrams)
    baseline_h = sum(
        (sum(c.values()) / total_covered) * _entropy(c)
        for c in baseline_ctx.values()
    )

    # H(next | current, family, param_bucket)
    param_ctx: dict[tuple, Counter] = defaultdict(Counter)
    for cur, nxt, fam, bucket in covered_bigrams:
        param_ctx[(cur, fam, bucket)][nxt] += 1

    param_h = sum(
        (sum(c.values()) / total_covered) * _entropy(c)
        for c in param_ctx.values()
    )

    return {
        "param_key": param_key,
        "n_bigrams": total_covered,
        "coverage_pct": 100.0 * total_covered / len(bigrams),
        "baseline_entropy": baseline_h,
        "param_entropy": param_h,
        "entropy_reduction": baseline_h - param_h,
        "is_deterministic": len(non_deterministic) == 0,
        "non_deterministic_count": len(non_deterministic),
        "p33": p33,
        "p67": p67,
    }


def estimate_level2_length_increase(
    seqs_by_family: dict[str, list[list[str]]],
    lookup: dict[tuple[str, str], dict],
    param_keys: list[str],
) -> None:
    """Estimate how many extra tokens Level 2 would add per sequence."""
    total_steps = 0
    steps_with_any_param = 0
    for fam, seqs in seqs_by_family.items():
        for seq in seqs:
            for step in seq:
                total_steps += 1
                has_param = any(
                    lookup.get((step, fam), {}).get(k) is not None
                    for k in param_keys
                )
                if has_param:
                    steps_with_any_param += 1

    pct = 100.0 * steps_with_any_param / total_steps if total_steps else 0
    print(f"\n── Level 2 sequence length impact ───────────────────────────────")
    print(f"  Total step tokens : {total_steps:,}")
    print(f"  Steps with ≥1 param token : {steps_with_any_param:,} ({pct:.1f}%)")
    print(f"  → Adding top-2 param tokens would increase sequence length by ~{pct:.0f}%")
    print(f"    (each such step gets 1–2 extra bucket tokens appended)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="data/processed/parameters_parsed.json")
    parser.add_argument("--data-dir", default="data/raw/training_data")
    args = parser.parse_args()

    print("── Loading sequences ─────────────────────────────────────────────")
    seqs_by_family = load_sequences_by_family(Path(args.data_dir))
    total_seqs = sum(len(v) for v in seqs_by_family.values())
    print(f"  Total sequences: {total_seqs}")

    bigrams = build_bigrams(seqs_by_family)
    print(f"  Total bigrams  : {len(bigrams):,}")

    print("\n── Baseline entropy H(next | current, family) ───────────────────")
    baseline_all = weighted_conditional_entropy(bigrams)
    print(f"  All families combined (with family conditioning): {baseline_all:.4f} bits")

    for fam in FAMILIES:
        fam_bigrams = [(c, n, f) for c, n, f in bigrams if f == fam]
        h = weighted_conditional_entropy(fam_bigrams)
        print(f"  {fam:6s}: {h:.4f} bits  ({len(fam_bigrams):,} bigrams)")

    print("\n── Loading parameters ────────────────────────────────────────────")
    with open(args.params, encoding="utf-8") as fh:
        records = json.load(fh)
    lookup = build_params_lookup(records)
    print(f"  {len(lookup)} (step, family) entries with numeric params")

    print("\n── Parameter entropy reduction ranking ──────────────────────────")
    print(f"  {'parameter':<20s}  {'n_bigrams':>9s}  {'coverage':>8s}  "
          f"{'baseline_H':>10s}  {'param_H':>8s}  {'reduction':>9s}  {'deterministic':>13s}")
    print("  " + "-" * 90)

    results = []
    for key in NUMERIC_KEYS:
        r = compute_param_entropy(bigrams, lookup, key)
        if r["n_bigrams"] == 0:
            continue
        results.append(r)

    results.sort(key=lambda x: x["entropy_reduction"], reverse=True)

    for r in results:
        det_str = "yes" if r["is_deterministic"] else f"no ({r['non_deterministic_count']} ambiguous)"
        print(
            f"  {r['param_key']:<20s}  {r['n_bigrams']:>9,}  {r['coverage_pct']:>7.1f}%  "
            f"{r['baseline_entropy']:>10.4f}  {r['param_entropy']:>8.4f}  "
            f"{r['entropy_reduction']:>+9.4f}  {det_str:>13s}"
        )

    estimate_level2_length_increase(seqs_by_family, lookup, NUMERIC_KEYS[:2])

    print("\n── Interpretation ────────────────────────────────────────────────")
    all_det = all(r["is_deterministic"] for r in results)
    max_reduction = max((r["entropy_reduction"] for r in results), default=0.0)

    if all_det and abs(max_reduction) < 0.001:
        print("  FINDING: Parameters are DETERMINISTIC per (step, family).")
        print("  Adding param bucket tokens to training is INFORMATION-REDUNDANT for Task 1.")
        print("  H(next | step, family, param_bucket) == H(next | step, family)")
        print()
        print("  RECOMMENDATION: SKIP Level 2 (parameter tokens).")
        print("  → [FAMILY] prefix token (Level 1) is the only useful data format change.")
        print("  → Spend saved time on: more training steps / larger model on Leonardo.")
    elif max_reduction > 0.05:
        print(f"  FINDING: Top parameter reduces entropy by {max_reduction:.4f} bits.")
        print("  Level 2 MAY be worth implementing — verify it is non-deterministic first.")
    else:
        print(f"  FINDING: Max entropy reduction is {max_reduction:.4f} bits (below 0.05 threshold).")
        print("  Level 2 provides marginal benefit — not recommended within 36-hour constraint.")


if __name__ == "__main__":
    main()
