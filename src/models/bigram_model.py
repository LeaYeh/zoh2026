"""Bigram baseline model for Track 1 process-step prediction.

Drop-in replacement for process_lm.py: exposes the same module-level functions
(predict_next_step, complete_sequence, anomaly_score) so callers need no changes.

Design decisions:
- Family-conditioned transition matrices (IC / IGBT / MOSFET separate)
- Hard-zero for unseen bigrams; OOV step falls back to global most-common
- Task 2 stopping: <EOS> first, family_median+30 safety cap
- No gradient training — fit() is a single scan of sequences
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median
from typing import Any


EOS = "<EOS>"
BOS = "<BOS>"
_LARGE_SCORE = 1e6  # returned when a bigram is OOV (hard zero → anomaly)


class BigramModel:
    def __init__(self) -> None:
        # family → from_step → Counter({to_step: count})
        self._trans: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        # global fallback (all families merged)
        self._global: dict[str, Counter] = defaultdict(Counter)
        # global step frequencies for OOV next-step fallback
        self._global_freq: Counter = Counter()
        # family → median sequence length (for Task 2 stopping)
        self._family_medians: dict[str, float] = {}

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit(self, sequences: list[list[str]], families: list[str]) -> "BigramModel":
        """Build transition tables from (sequence, family) pairs.

        Each sequence should be the raw step list *without* BOS/EOS —
        fit() adds them internally so boundary transitions are counted.
        """
        lengths_by_family: dict[str, list[int]] = defaultdict(list)

        for seq, fam in zip(sequences, families):
            fam = fam.upper()
            augmented = [BOS] + list(seq) + [EOS]
            lengths_by_family[fam].append(len(seq))
            self._global_freq.update(seq)

            for a, b in zip(augmented, augmented[1:]):
                self._trans[fam][a][b] += 1
                self._global[a][b] += 1

        for fam, lens in lengths_by_family.items():
            self._family_medians[fam] = median(lens)

        return self

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_counter(self, from_step: str, family: str | None) -> Counter | None:
        fam = family.upper() if family else None
        if fam and fam in self._trans and from_step in self._trans[fam]:
            return self._trans[fam][from_step]
        if from_step in self._global:
            return self._global[from_step]
        return None

    def _top_k_steps(self, from_step: str, family: str | None, top_k: int) -> list[tuple[str, float]]:
        """Return [(step, probability), ...] sorted by probability descending."""
        counter = self._get_counter(from_step, family)
        if counter is None:
            # OOV from_step: return global most-common non-special steps
            fallback = [
                (s, c) for s, c in self._global_freq.most_common(top_k + 4)
                if s not in (BOS, EOS)
            ]
            total = sum(c for _, c in fallback) or 1
            return [(s, c / total) for s, c in fallback[:top_k]]

        # exclude BOS/EOS from ranked next-step candidates
        items = [(s, c) for s, c in counter.items() if s not in (BOS, EOS)]
        items.sort(key=lambda x: -x[1])
        total = sum(c for _, c in items) or 1
        return [(s, c / total) for s, c in items[:top_k]]

    def _family_median(self, family: str | None) -> float:
        if family:
            return self._family_medians.get(family.upper(), 150.0)
        if self._family_medians:
            return median(self._family_medians.values())
        return 150.0


# ── module-level API (matches process_lm.py) ─────────────────────────────────

def predict_next_step(
    model: BigramModel,
    tokenizer: Any,  # accepted for interface compatibility, not used
    partial_steps: list[str],
    top_k: int = 5,
    device: Any = "cpu",
    family: str | None = None,
) -> list[tuple[str, float]]:
    """Return [(step_name, probability), ...] for the most likely next steps."""
    last = partial_steps[-1] if partial_steps else BOS
    return model._top_k_steps(last, family, top_k)


def complete_sequence(
    model: BigramModel,
    tokenizer: Any,
    partial_steps: list[str],
    max_new_steps: int = 100,
    device: Any = "cpu",
    family: str | None = None,
) -> list[str]:
    """Greedily complete partial_steps until <EOS> or safety cap.

    Returns only the *new* steps appended (not the input steps).
    """
    safety_cap = int(model._family_median(family)) + 30
    current = list(partial_steps)
    generated: list[str] = []

    while len(generated) < max_new_steps and len(current) < safety_cap:
        candidates = model._top_k_steps(current[-1] if current else BOS, family, top_k=1)
        if not candidates:
            break
        next_step = candidates[0][0]
        if next_step == EOS:
            break
        generated.append(next_step)
        current.append(next_step)

    return generated


def anomaly_score(
    model: BigramModel,
    tokenizer: Any,
    steps: list[str],
    device: Any = "cpu",
    family: str | None = None,
) -> float:
    """Return mean negative log-probability of the bigram sequence.

    Higher score = more anomalous. Returns _LARGE_SCORE if any transition is OOV.
    Includes BOS→first and last→EOS boundary transitions.
    """
    if not steps:
        return 0.0

    augmented = [BOS] + list(steps) + [EOS]
    fam = family.upper() if family else None
    log_prob_sum = 0.0
    n = len(augmented) - 1  # number of bigram transitions

    for a, b in zip(augmented, augmented[1:]):
        counter = model._get_counter(a, fam)
        if counter is None or b not in counter:
            return _LARGE_SCORE  # hard zero — unseen transition

        total = sum(counter.values())
        prob = counter[b] / total
        log_prob_sum += math.log(prob)

    # mean negative log prob ≈ per-step perplexity exponent
    return float(-log_prob_sum / n)
