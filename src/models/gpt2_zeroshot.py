"""HuggingFace GPT-2 zero-shot baseline for Track 1 process-step prediction.

Drop-in replacement for process_lm.py and bigram_model.py: exposes the same
module-level functions so callers need no changes.

Strategy:
  - Use pre-trained gpt2 (BPE tokenizer, 124M params, no fine-tuning)
  - Steps are joined as space-separated text: "CLEAN OXIDATION DEPOSITION_POLY"
  - Next-step scoring: for each candidate in vocab, compute log P(candidate | context)
    by summing token-level log-softmax over the candidate's BPE tokens
  - Constrained to the known vocabulary — output is always a valid step name
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

_MODEL_NAME = "gpt2"
EOS = "<EOS>"
BOS = "<BOS>"
_LARGE_SCORE = 1e6


class GPT2ZeroShot:
    """Wraps HuggingFace gpt2 with constrained next-step scoring."""

    def __init__(self, model_name: str = _MODEL_NAME, device: str = "cpu") -> None:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.device = torch.device(device)
        self.hf_tok = GPT2TokenizerFast.from_pretrained(model_name)
        self.hf_tok.pad_token = self.hf_tok.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # cache step-name → BPE token ids (with leading space for mid-sequence)
        self._step_ids_cache: dict[str, list[int]] = {}
        # known vocabulary of process step names (set by caller)
        self.vocab: list[str] = []

    def set_vocab(self, step_names: list[str]) -> None:
        """Register the set of valid step names for constrained decoding."""
        self.vocab = list(step_names)
        self._step_ids_cache = {}
        for name in step_names:
            # leading space → GPT-2 treats it as a mid-sentence token
            ids = self.hf_tok.encode(" " + name, add_special_tokens=False)
            self._step_ids_cache[name] = ids

    def _context_ids(self, partial_steps: list[str]) -> list[int]:
        if not partial_steps:
            return []
        text = " ".join(s for s in partial_steps if s not in (BOS, EOS))
        return self.hf_tok.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def score_next_steps(
        self,
        partial_steps: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Score all vocab items and return top_k by log probability.

        For each candidate, runs one forward pass over [context + step_tokens]
        and sums the log-softmax at each step-token position. No KV cache is
        retained between candidates to keep memory bounded.
        """
        if not self.vocab:
            return []

        ctx_ids = self._context_ids(partial_steps)
        scores: list[tuple[str, float]] = []

        for step_name in self.vocab:
            step_ids = self._step_ids_cache.get(step_name)
            if not step_ids:
                continue

            full_ids = ctx_ids + step_ids
            full_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)
            logits = self.model(full_tensor).logits[0]  # (seq_len, vocab_hf)

            lp = 0.0
            for i, tok_id in enumerate(step_ids):
                pos = len(ctx_ids) + i - 1  # position before this token
                lp += float(F.log_softmax(logits[pos], dim=-1)[tok_id])

            scores.append((step_name, lp))
            del full_tensor, logits  # free memory immediately

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    @torch.no_grad()
    def sequence_perplexity(self, steps: list[str]) -> float:
        """Compute GPT-2 perplexity of the full sequence in one forward pass.

        Joins step names as space-separated text, then computes mean negative
        log-probability per BPE token. O(1) forward pass regardless of length.
        """
        if not steps:
            return 0.0
        text = " ".join(s for s in steps if s not in (BOS, EOS))
        ids = self.hf_tok.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            return 0.0
        ids = ids[:1024]  # GPT-2 max context window
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids, labels=input_ids)
        # out.loss is mean cross-entropy = mean negative log prob per token
        return float(out.loss.exp())  # perplexity


# ── module-level API (matches process_lm.py and bigram_model.py) ─────────────

def predict_next_step(
    model: GPT2ZeroShot,
    tokenizer: Any,
    partial_steps: list[str],
    top_k: int = 5,
    device: Any = "cpu",
    family: str | None = None,
) -> list[tuple[str, float]]:
    """Return [(step_name, log_prob), ...] for the most likely next steps."""
    return model.score_next_steps(partial_steps, top_k=top_k)


def complete_sequence(
    model: GPT2ZeroShot,
    tokenizer: Any,
    partial_steps: list[str],
    max_new_steps: int = 100,
    device: Any = "cpu",
    family: str | None = None,
) -> list[str]:
    """Greedily complete partial_steps, stopping at max_new_steps."""
    current = list(partial_steps)
    generated: list[str] = []

    for _ in range(max_new_steps):
        candidates = model.score_next_steps(current, top_k=1)
        if not candidates:
            break
        next_step = candidates[0][0]
        generated.append(next_step)
        current.append(next_step)

    return generated


def anomaly_score(
    model: GPT2ZeroShot,
    tokenizer: Any,
    steps: list[str],
    device: Any = "cpu",
    family: str | None = None,
) -> float:
    """Sequence perplexity under GPT-2 (one forward pass). Higher = more anomalous."""
    return model.sequence_perplexity(steps)
