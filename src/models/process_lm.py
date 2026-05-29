"""GPT-2-style language model for industrial process sequence modeling.

Trained from scratch with a custom (small) vocabulary of process step names.
No pre-trained weights are loaded — we initialise randomly and fine-tune on
the provided process CSV data.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from src.data.process_loader import ProcessStepTokenizer, SPECIAL_TOKENS


def build_model(
    tokenizer: ProcessStepTokenizer,
    max_seq_len: int = 256,
    n_embd: int = 256,
    n_layer: int = 6,
    n_head: int = 8,
) -> GPT2LMHeadModel:
    """Build a small GPT-2 from scratch with the process-step vocabulary."""
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=max_seq_len,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
    )
    return GPT2LMHeadModel(config)


def load_model(
    ckpt_dir: str,
    tokenizer: ProcessStepTokenizer,
    device: str | torch.device = "cpu",
) -> GPT2LMHeadModel:
    """Load a previously saved model checkpoint."""
    import os
    from pathlib import Path
    path = Path(ckpt_dir) / "model.pt"
    state = torch.load(path, map_location=device)
    # infer arch from state dict
    n_embd  = state["transformer.wte.weight"].shape[1]
    n_layer = max(int(k.split(".")[2]) for k in state if k.startswith("transformer.h.")) + 1
    model = build_model(tokenizer, n_embd=n_embd, n_layer=n_layer)
    model.load_state_dict(state)
    return model.to(device)


# ── inference helpers ─────────────────────────────────────────────────────────

def predict_next_step(
    model: GPT2LMHeadModel,
    tokenizer: ProcessStepTokenizer,
    partial_steps: list[str],
    top_k: int = 5,
    device: str | torch.device = "cpu",
) -> list[tuple[str, float]]:
    """Return top-k (step_name, probability) for the next step."""
    model.eval()
    input_ids = torch.tensor(
        [tokenizer.encode(partial_steps)], dtype=torch.long, device=device
    )
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]  # last-position logits
    probs = torch.softmax(logits, dim=-1)
    # take top_k * 3 to have room to filter special tokens
    topk = torch.topk(probs, k=min(top_k * 3, tokenizer.vocab_size))
    results: list[tuple[str, float]] = []
    for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
        step = tokenizer.id_to_step[idx]
        if step not in SPECIAL_TOKENS:
            results.append((step, float(prob)))
        if len(results) == top_k:
            break
    return results


def complete_sequence(
    model: GPT2LMHeadModel,
    tokenizer: ProcessStepTokenizer,
    partial_steps: list[str],
    max_new_steps: int = 100,
    device: str | torch.device = "cpu",
) -> list[str]:
    """Autoregressively generate the rest of the sequence."""
    model.eval()
    input_ids = torch.tensor(
        [tokenizer.encode(partial_steps)], dtype=torch.long, device=device
    )
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_steps,
            eos_token_id=tokenizer.eos_id,
            pad_token_id=tokenizer.pad_id,
            do_sample=False,  # greedy — deterministic for eval
        )
    generated_ids = out[0, input_ids.shape[1]:].tolist()
    return tokenizer.decode(generated_ids)


def anomaly_score(
    model: GPT2LMHeadModel,
    tokenizer: ProcessStepTokenizer,
    steps: list[str],
    device: str | torch.device = "cpu",
) -> float:
    """Return perplexity of the sequence.  Higher perplexity → likely anomaly."""
    model.eval()
    ids = tokenizer.encode(steps)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return math.exp(min(loss.item(), 20.0))  # clip to avoid overflow
