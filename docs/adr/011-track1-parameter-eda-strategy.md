# ADR 011 — Track 1: Fab-Level Parameter Parsing and EDA Strategy

**Status:** Accepted  
**Date:** 2026-05-30

## Context

Each of the three product families (IC, IGBT, MOSFET) has a `*_longdescription_parameters.csv`
file containing a `REALISTIC FAB-LEVEL PARAMETERS` column. These are free-text strings
mixing numeric values, units, ranges, tolerances, chemical ratios, and categorical descriptions:

```
"NH₄OH:H₂O₂:H₂O (1:1:5), 75°C, 10 min, overflow DI rinse"
"200 keV, 5×10¹² cm⁻², tilt 7°"
"KLA optical scan, sensitivity 0.3 µm"
```

Two goals were identified for this data:
1. **Analysis**: Understand cross-family commonalities and what parameters best discriminate
   which process family a given sequence is closest to.
2. **Model input**: Produce structured parameter features to condition the sequence model
   (step-level family affinity scoring as an extension of ADR-008 family conditioning).

## Decisions

### Parsing strategy: LLM batch extraction

Alternatives considered:

| Strategy | Verdict | Reason |
|---|---|---|
| Regex (value, unit) pairs | Rejected | Cannot handle chemical ratios, superscript exponents, or compound descriptions |
| LLM batch parse (chosen) | Accepted | Handles all formats; 374 rows in 19 batches (~2 min); output stored as static JSON |
| Hybrid regex + LLM | Rejected | Extra complexity with no quality benefit given the small dataset |

Model used: `claude-haiku-4-5-20251001` (fast, low cost for structured extraction).

### Output schema: free schema per step

Each step gets only the parameters it actually has — no global fixed schema, no null padding:

```json
{
  "step": "THERMAL OXIDATION",
  "family": "IC",
  "params": {
    "temperature_c": 1000,
    "time_min": 30,
    "atmosphere": "dry O2",
    "target_thickness_nm": 45
  },
  "raw": "1000°C, 30 min, dry O2, target 45 nm gate oxide"
}
```

Fixed schema was rejected: 12 possible numeric key types × 374 rows → 94%+ null rate per row,
which destroys PCA and any distance-based analysis.

### Range handling: preserve structure

Point values, ranges, and tolerances are stored distinctly:

```json
{"temperature_c": 1050}                         // point
{"resistivity_ohm_cm": {"min": 10, "max": 20}}  // range
{"thickness_um": {"value": 725, "tolerance": 10}} // tolerance
```

Midpoint extraction (`(min+max)/2`) is deferred to the consumer (EDA, model feature) —
the raw structure is preserved so range width can itself be an EDA dimension.

### Step alignment: outer join

Parameter files and sequence files have partial misalignment (~4–5 steps per family
appear in one but not the other). All steps are kept; those missing parameters are
stored with `"params": {}` and flagged in coverage reporting.

Fixed schema / inner join was rejected: coverage rate (94–95% per family) is itself
an EDA finding. Silently dropping uncovered steps hides this information.

## EDA Findings

**Coverage:** 353/374 steps (94–95% per family) have at least one parsed numeric parameter.  
**Shared steps:** 73 steps appear in all 3 families.  
**Top discriminating parameters** (by PCA PC1 loading):
`temperature_c` > `time_min` > `thickness_nm` > `pressure_mtorr`

**Cross-family temperature differences** are small (≤50°C) and concentrated in thermal steps:

| Step | IC | IGBT | MOSFET |
|---|---|---|---|
| DRIVE IN DIFFUSION | 1100°C | 1150°C | 1100°C |
| RAPID THERMAL ANNEAL | 1000°C | 1050°C | 1000°C |
| THERMAL OXIDATION | 1000°C | 1000°C | 950°C |

**PCA separability (2 PCs, 49% total variance explained):**  
The three families show partially separable distributions in parameter space, but with
significant overlap. Parameters alone cannot reliably identify the family of an unknown
sequence — sequence grammar (bigram step transitions) remains the primary discriminator.

## Consequences

- `data/processed/parameters_parsed.json` is the single source of truth for structured
  parameters. Re-run `scripts/parse_parameters.py` only if the source CSVs change.
- Parameters are a **supplementary signal**, not a replacement for sequence grammar.
  The best family affinity model combines bigram transitions + parameter features.
- EDA figures are saved to `docs/eda_figures/` for competition day reference.
- Step-level family affinity scoring (next engineering step) should use `temperature_c`,
  `time_min`, `thickness_nm`, and `pressure_mtorr` as the primary feature dimensions,
  as these have the highest PC1 discriminating power.
