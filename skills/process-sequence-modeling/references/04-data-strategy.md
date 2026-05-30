# 04 · Data Strategy
**When to read**: loading data, augmenting training set, debugging loader issues

> 李宏毅 ML 2025 — Lec 6 (data for LLM pretraining), HW4 (data pipeline)

---

## What Data We Have

| File | Content | Format | Rows |
|---|---|---|---|
| `IC_variants.csv` | 1 000 IC sequences | long (SEQUENCE_ID, STEP) | ~115K |
| `IGBT_variants.csv` | 1 000 IGBT sequences | long | ~148K |
| `MOSFET_variants.csv` | 1 000 MOSFET sequences | long | ~125K |
| `*_Longdescr.csv` | Steps + text descriptions | wide (STEP, DESCRIPTION) | ~120 |
| `*_longdescription_parameters.csv` | Steps + fab parameters | wide | ~120 |
| `synthetic_*.csv` | Single canonical reference | long | 107–151 |

**Use `*_variants.csv` for training.** The other files are reference material.

---

## Loading Training Data

```python
from src.data.process_loader import load_sequences, ProcessStepTokenizer

# Load all 3 families (3 000 sequences total)
seqs = load_sequences(
    data_dir="data/raw/training_data/",
    product_families=["IC", "IGBT", "MOSFET"],
    sequence_col="SEQUENCE_ID",
    step_col="STEP",
)
print(len(seqs))        # 3 000
print(seqs[0][:5])      # ['RECEIVE WAFER LOT', 'LOT IDENTIFICATION', ...]

tok = ProcessStepTokenizer.build(seqs)
print(tok.vocab_size)   # ~124
```

**Column name gotcha**: CSV files use `SEQUENCE_ID` and `STEP`. Check first:
```bash
head -1 data/raw/training_data/IC_variants.csv
# SEQUENCE_ID,STEP
```

---

## Generating More Training Data

The combinatoric space is huge (IC ~6B, IGBT ~13T, MOSFET ~51B distinct valid sequences).
More data = better generalisation, especially for the GRPO phase.

```bash
# Generate 2 000 extra IC sequences
uv run python scripts/generate_sequences.py \
  --family ic --count 2000 \
  --output data/raw/training_data/IC_variants_extra.csv \
  --seed 100

# Validate your generated file
uv run python scripts/generate_sequences.py \
  --validate data/raw/training_data/IC_variants_extra.csv \
  --family ic
```

**When to generate more:**
- After Gate 2: if val Top-1 < 0.4 at step 1000, double the training data
- Before GRPO: add 2000+ sequences per family for more diverse prompts
- Scaling experiment: test 1K vs 3K vs 9K to see scaling curve

---

## ProcessStepTokenizer Details

```python
class ProcessStepTokenizer:
    # Special tokens always at IDs 0–3:
    # 0=<PAD>, 1=<BOS>, 2=<EOS>, 3=<UNK>
    # IDs 4+: step names ordered by frequency (most common first)
    
    def encode(self, steps, add_special=True):
        # [BOS_ID] + [step_id, ...] + [EOS_ID]
        
    def decode(self, ids, skip_special=True):
        # returns list of step name strings
        
    def save(self, path):   # saves as JSON list
    def load(cls, path):    # loads from JSON list
```

**Save/load the tokenizer with the model checkpoint:**
```
data/oof/<run_name>/
├── model.pt        ← model weights
└── tokenizer.json  ← vocabulary (JSON list of step names)
```

Never rebuild tokenizer from scratch at inference time — always load the saved one.

---

## Val Split Strategy

From `configs/exp/gpt2_finetune_v1.yaml`:
```yaml
dataset:
  val_ratio: 0.1   # 10% held out = 300 sequences
  seed: 42         # fixed — same split every run for comparability
```

**Warning**: do NOT change `val_ratio` or `seed` between runs. That would create leakage.

The 10% val split is stratified by sequence index (not by family), so each family contributes ~100 val sequences.

---

## Enriched Data (Optional)

The `*_Longdescr.csv` and `*_longdescription_parameters.csv` files have text descriptions and fab parameters per step. These are NOT used in the current pipeline (we only use step names).

Potential use: add description text as a second modality (multi-modal conditioning). Not recommended under 36-hour constraint — complexity too high for uncertain benefit.
