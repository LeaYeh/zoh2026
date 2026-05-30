"""Parse REALISTIC FAB-LEVEL PARAMETERS from all 3 families using Claude API.

Reads the *_longdescription_parameters.csv files, sends batches of rows to
Claude, and saves structured JSON to data/processed/parameters_parsed.json.

Usage:
    uv run python scripts/parse_parameters.py
    uv run python scripts/parse_parameters.py --batch-size 15 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

PARAM_FILES = {
    "IC":    "IC_longdescription_parameters.csv",
    "IGBT":  "IGBT_longdescription_parameters.csv",
    "MOSFET":"MOSFET_longdescription_parameters.csv",
}

SYSTEM_PROMPT = """You are a semiconductor process engineer parsing fab-level process parameters.

For each step, extract ALL numeric and categorical parameters from the raw string.
Return a JSON array — one object per step — with this shape:

{
  "step": "<STEP NAME>",
  "family": "<IC|IGBT|MOSFET>",
  "params": {
    "<param_name_with_unit>": <value>
  }
}

Rules for params values:
- Point value → use the number directly: {"temperature_c": 1050}
- Range (min–max) → {"temperature_c": {"min": 10, "max": 20}}
- Value ± tolerance → {"thickness_um": {"value": 725, "tolerance": 10}}
- Categorical / text → string: {"atmosphere": "dry O2", "method": "RIE"}
- Chemical ratio (e.g. 1:1:5) → string: {"chemistry_ratio": "1:1:5"}
- Dose with exponent (e.g. 5×10¹²) → number: {"dose_cm2": 5e12}

Key naming rules:
- Always snake_case
- Always include the unit in the key name: temperature_c, time_min, time_s,
  thickness_nm, thickness_um, pressure_mtorr, power_w, dose_cm2, energy_kev,
  speed_rpm, concentration_pct, flow_sccm, wavelength_nm
- If the same parameter appears with two units, prefer SI-adjacent (nm over Å, °C over K)

Only extract what is explicitly in the raw string. Do not infer or hallucinate values.
If a string has no numeric parameters (e.g. "MES verification"), return "params": {}.

Return ONLY the JSON array. No markdown, no explanation."""


def load_all_rows(data_dir: Path) -> list[dict]:
    rows = []
    for fam, fname in PARAM_FILES.items():
        path = data_dir / fname
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "family": fam,
                    "step": r["STEP"].strip(),
                    "raw": r["REALISTIC FAB‑LEVEL PARAMETERS"].strip(),
                })
    return rows


def parse_batch(client: anthropic.Anthropic, batch: list[dict], dry_run: bool) -> list[dict]:
    if dry_run:
        return [{"step": r["step"], "family": r["family"], "params": {}} for r in batch]

    user_content = "Parse these process steps:\n\n" + json.dumps(
        [{"step": r["step"], "family": r["family"], "raw": r["raw"]} for r in batch],
        ensure_ascii=False,
        indent=2,
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    text = response.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/training_data")
    parser.add_argument("--out", default="data/processed/parameters_parsed.json")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, write empty params")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_all_rows(data_dir)
    print(f"Loaded {len(rows)} rows across {len(PARAM_FILES)} families")

    if args.dry_run:
        print("DRY RUN — skipping API calls")
        client = None
    else:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    results: list[dict] = []
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]

    for i, batch in enumerate(batches):
        print(f"  batch {i+1}/{len(batches)}  ({len(batch)} rows)...", end=" ", flush=True)
        parsed = parse_batch(client, batch, args.dry_run)
        results.extend(parsed)
        print("ok")
        if not args.dry_run and i < len(batches) - 1:
            time.sleep(0.5)  # avoid rate limit

    # Merge raw string back in for reference
    raw_by_key = {(r["family"], r["step"]): r["raw"] for r in rows}
    for item in results:
        item["raw"] = raw_by_key.get((item["family"], item["step"]), "")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} parsed records → {out_path}")

    # Quick coverage stats
    with_params = sum(1 for r in results if r.get("params"))
    print(f"Coverage: {with_params}/{len(results)} steps have at least one parsed parameter")
    by_family: dict[str, dict] = {}
    for r in results:
        fam = r["family"]
        by_family.setdefault(fam, {"total": 0, "with_params": 0})
        by_family[fam]["total"] += 1
        if r.get("params"):
            by_family[fam]["with_params"] += 1
    for fam, counts in sorted(by_family.items()):
        pct = counts["with_params"] / counts["total"] * 100
        print(f"  {fam:6s}: {counts['with_params']}/{counts['total']} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
