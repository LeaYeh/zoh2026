"""Description-based embedding initializer for process step tokens (ADR-014).

Builds a 27-dim feature vector per step from:
  - 8-dim process category one-hot (from step name keywords)
  - 3-dim normalized numeric params (temperature, time, litho level)
  - 16-dim TF-IDF SVD of step names

Projects to model n_embd via Linear(27, n_embd) and writes the result into
model.transformer.wte.weight (special tokens keep random init).
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from src.data.process_loader import SPECIAL_TOKENS


# ── process category keywords ─────────────────────────────────────────────────

_CATEGORIES: list[tuple[str, list[str]]] = [
    ("litho",   ["ALIGN", "EXPOSE", "DEVELOP", "COAT PHOTORESIST", "PHOTO", "RETICLE"]),
    ("thermal", ["OXIDATION", "ANNEAL", "RTA", "DIFFUSION", "CURE", "RAPID THERMAL"]),
    ("deposit", ["DEPOSIT", "CVD", "PVD", "EPITAX", "GROW", "FILL"]),
    ("etch",    ["ETCH", "STRIP", "CMP", "POLISH", "HF DIP", "OXIDE STRIP"]),
    ("implant", ["IMPLANT", "ION"]),
    ("measure", ["MEASURE", "INSPECT", "SCAN", "KLA", "CHECK", "VERIFY", "TEST"]),
    ("clean",   ["CLEAN", "RINSE", "DRY", "RCA", "WASH", "DEGAS"]),
    ("other",   []),
]
N_CATEGORIES = len(_CATEGORIES)  # 8


def _category_onehot(step_name: str) -> list[float]:
    name = step_name.upper()
    for i, (_, keywords) in enumerate(_CATEGORIES[:-1]):
        if any(kw in name for kw in sorted(keywords, key=len, reverse=True)):
            vec = [0.0] * N_CATEGORIES
            vec[i] = 1.0
            return vec
    vec = [0.0] * N_CATEGORIES
    vec[-1] = 1.0  # "other"
    return vec


# ── numeric param features ────────────────────────────────────────────────────

def _extract_scalar(v) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        if "value" in v:
            return float(v["value"])
        if "min" in v and "max" in v:
            return (float(v["min"]) + float(v["max"])) / 2.0
    return None


def _litho_level(step_name: str) -> float:
    m = re.search(r"LEVEL\s+(\d+)", step_name.upper())
    return float(m.group(1)) if m else 0.0


def build_param_lookup(desc_path: Path) -> dict[str, dict[str, float]]:
    """Return {step_name: {temperature_c: float, time_min: float}} averaged across families."""
    with open(desc_path, encoding="utf-8") as fh:
        records = json.load(fh)
    accum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        step = rec["step"]
        for key in ("temperature_c", "time_min", "time_s"):
            v = _extract_scalar(rec.get("params", {}).get(key))
            if v is not None:
                canonical = "time_min" if key == "time_s" else key
                if key == "time_s":
                    v = v / 60.0
                accum[step][canonical].append(v)
    return {
        step: {k: sum(vs) / len(vs) for k, vs in params.items()}
        for step, params in accum.items()
    }


def _numeric_features(
    step_name: str,
    param_lookup: dict[str, dict[str, float]],
    temp_max: float,
    time_max: float,
    litho_max: float,
) -> list[float]:
    params = param_lookup.get(step_name, {})
    temp = params.get("temperature_c", 0.0) / max(temp_max, 1.0)
    time = params.get("time_min", 0.0) / max(time_max, 1.0)
    level = _litho_level(step_name) / max(litho_max, 1.0)
    return [temp, time, level]


# ── TF-IDF + SVD on step names ────────────────────────────────────────────────

def _tokenize_name(name: str) -> list[str]:
    return re.findall(r"[A-Z0-9]+", name.upper())


def build_tfidf_svd(step_names: list[str], n_components: int = 16) -> torch.Tensor:
    """Return (len(step_names), n_components) float tensor via TF-IDF + truncated SVD."""
    # build term frequency matrix
    vocab: dict[str, int] = {}
    docs: list[list[str]] = []
    for name in step_names:
        tokens = _tokenize_name(name)
        docs.append(tokens)
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    n_docs = len(docs)
    n_terms = len(vocab)
    tf = torch.zeros(n_docs, n_terms)
    for i, tokens in enumerate(docs):
        for t in tokens:
            tf[i, vocab[t]] += 1.0
        if tokens:
            tf[i] /= len(tokens)

    # IDF
    df = (tf > 0).float().sum(0)
    idf = torch.log((n_docs + 1.0) / (df + 1.0)) + 1.0
    tfidf = tf * idf

    # Truncated SVD via torch.linalg.svd
    k = min(n_components, n_docs - 1, n_terms - 1)
    U, S, _Vh = torch.linalg.svd(tfidf, full_matrices=False)
    result = U[:, :k] * S[:k]

    if k < n_components:
        pad = torch.zeros(n_docs, n_components - k)
        result = torch.cat([result, pad], dim=1)

    # L2-normalize each row
    norms = result.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return result / norms


# ── main entry point ──────────────────────────────────────────────────────────

def build_description_feature_matrix(
    step_names: list[str],
    desc_path: Path,
    n_embd: int,
) -> torch.Tensor:
    """Return (len(step_names), n_embd) tensor to initialize wte.

    Special tokens in step_names are skipped (left as zeros for caller to
    keep as random init).
    """
    param_lookup = build_param_lookup(desc_path)

    # Compute normalisation bounds from all non-special steps
    process_steps = [s for s in step_names if s not in SPECIAL_TOKENS]
    all_temps = [param_lookup.get(s, {}).get("temperature_c", 0.0) for s in process_steps]
    all_times = [param_lookup.get(s, {}).get("time_min", 0.0) for s in process_steps]
    temp_max = max(all_temps) if all_temps else 1.0
    time_max = max(all_times) if all_times else 1.0
    litho_max = max(_litho_level(s) for s in process_steps) or 1.0

    tfidf_vecs = build_tfidf_svd(process_steps, n_components=16)

    # Assemble 27-dim feature rows for process steps only
    rows: list[list[float]] = []
    for name in process_steps:
        cat = _category_onehot(name)          # 8
        num = _numeric_features(name, param_lookup, temp_max, time_max, litho_max)  # 3
        rows.append(cat + num)                # 11 so far; TF-IDF appended below

    features = torch.tensor(rows, dtype=torch.float32)          # [n_steps, 11]
    features = torch.cat([features, tfidf_vecs], dim=1)         # [n_steps, 27]

    # Project 27 → n_embd; caller sets global seed before build_model()
    proj_weight = torch.empty(n_embd, features.shape[1]).normal_(0, 1.0 / math.sqrt(features.shape[1]))
    projected = features @ proj_weight.T                        # [n_steps, n_embd]

    # L2-normalise then scale to match GPT-2 default init std (~0.02)
    norms = projected.norm(dim=1, keepdim=True).clamp(min=1e-8)
    projected = projected / norms * 0.02

    # Place into full vocab matrix; special tokens stay zero (caller keeps random init)
    special_set = set(SPECIAL_TOKENS)
    full = torch.zeros(len(step_names), n_embd)
    proc_idx = 0
    for i, name in enumerate(step_names):
        if name not in special_set:
            full[i] = projected[proc_idx]
            proc_idx += 1

    return full
