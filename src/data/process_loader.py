"""Process step data loader and tokenizer for Track 1 (Industrial AI).

CSV format (long format): SEQUENCE_ID, STEP — one row per step.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import pandas as pd

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


class ProcessStepTokenizer:
    """Maps process step names ↔ integer IDs."""

    def __init__(self) -> None:
        self.step_to_id: dict[str, int] = {}
        self.id_to_step: list[str] = []

    @classmethod
    def build(cls, sequences: list[list[str]]) -> "ProcessStepTokenizer":
        tok = cls()
        for sp in SPECIAL_TOKENS:
            tok.step_to_id[sp] = len(tok.id_to_step)
            tok.id_to_step.append(sp)
        counts = Counter(step for seq in sequences for step in seq)
        for step, _ in counts.most_common():
            if step not in tok.step_to_id:
                tok.step_to_id[step] = len(tok.id_to_step)
                tok.id_to_step.append(step)
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_step)

    @property
    def pad_id(self) -> int: return self.step_to_id[PAD_TOKEN]
    @property
    def bos_id(self) -> int: return self.step_to_id[BOS_TOKEN]
    @property
    def eos_id(self) -> int: return self.step_to_id[EOS_TOKEN]
    @property
    def unk_id(self) -> int: return self.step_to_id[UNK_TOKEN]

    def encode(self, steps: list[str], add_special: bool = True) -> list[int]:
        ids = [self.step_to_id.get(s, self.unk_id) for s in steps]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> list[str]:
        special = set(SPECIAL_TOKENS) if skip_special else set()
        return [
            self.id_to_step[i]
            for i in ids
            if 0 <= i < len(self.id_to_step) and self.id_to_step[i] not in special
        ]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.id_to_step))

    @classmethod
    def load(cls, path: str | Path) -> "ProcessStepTokenizer":
        tok = cls()
        tok.id_to_step = json.loads(Path(path).read_text())
        tok.step_to_id = {s: i for i, s in enumerate(tok.id_to_step)}
        return tok


def load_sequences(
    data_dir: str | Path,
    product_families: list[str] | None = None,
    train_files: list[str] | None = None,
    sequence_col: str = "SEQUENCE_ID",
    step_col: str = "STEP",
    min_length: int = 5,
) -> list[list[str]]:
    """Load process sequences from CSV files in data_dir.

    Expects long-format CSVs: one row per step with sequence_col and step_col.
    If train_files is given, loads exactly those filenames (no globbing).
    Otherwise, if product_families is given, globs for files containing each family name.
    """
    data_dir = Path(data_dir)
    csv_files: list[Path] = []
    if train_files:
        csv_files = [data_dir / f for f in train_files]
    elif product_families:
        for family in product_families:
            for pattern in (f"*{family}*.csv", f"*{family.lower()}*.csv"):
                csv_files.extend(data_dir.glob(pattern))
    if not csv_files:
        csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    all_sequences: list[list[str]] = []
    for csv_path in sorted(set(csv_files)):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[loader] skipping {csv_path.name}: {exc}")
            continue

        # case-insensitive column lookup
        col_map = {c.upper(): c for c in df.columns}
        step_col_actual = col_map.get(step_col.upper(), step_col)
        seq_col_actual  = col_map.get(sequence_col.upper(), sequence_col)

        if step_col_actual not in df.columns:
            print(f"[loader] {csv_path.name}: column '{step_col}' not found, skipping")
            continue

        if seq_col_actual in df.columns:
            for _, group in df.groupby(seq_col_actual, sort=False):
                steps = group[step_col_actual].astype(str).tolist()
                if len(steps) >= min_length:
                    all_sequences.append(steps)
        else:
            steps = df[step_col_actual].astype(str).tolist()
            if len(steps) >= min_length:
                all_sequences.append(steps)

    print(f"[loader] {len(all_sequences)} sequences from {len(set(csv_files))} file(s)")
    return all_sequences
