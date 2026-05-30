"""GPT-2-style language model for industrial process sequence modeling.

Trained from scratch with a custom (small) vocabulary of process step names.
No pre-trained weights are loaded — we initialise randomly and fine-tune on
the provided process CSV data.
"""
from __future__ import annotations
import math
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from src.data.process_loader import ProcessStepTokenizer, SPECIAL_TOKENS


class CategoryAwareEmbedding(nn.Module):
    """Augments GPT-2's token embedding with a learnable process-category embedding.

    Replaces model.transformer.wte in-place so forward() and generate() work
    transparently without any changes to the training loop or inference helpers.

    The .weight property proxies the original wte.weight so lm_head weight-tying
    remains intact — the output projection still operates in token-ID space.

    cat_embed is zero-initialized so it starts as a no-op and learns incrementally.
    """

    def __init__(
        self,
        wte: nn.Embedding,
        n_categories: int,
        n_embd: int,
        step_to_cat: list[int],
    ) -> None:
        super().__init__()
        self.wte = wte
        self.cat_embed = nn.Embedding(n_categories, n_embd)
        nn.init.zeros_(self.cat_embed.weight)
        self.register_buffer(
            "step_to_cat_ids",
            torch.tensor(step_to_cat, dtype=torch.long),
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.wte.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        cat_ids = self.step_to_cat_ids[input_ids]
        return self.wte(input_ids) + self.cat_embed(cat_ids)


def _build_step_to_cat(tokenizer: ProcessStepTokenizer) -> list[int]:
    from src.models.desc_embed import _category_onehot, N_CATEGORIES
    result = []
    for step in tokenizer.id_to_step:
        if step in SPECIAL_TOKENS:
            result.append(N_CATEGORIES - 1)  # "other" for special tokens
        else:
            onehot = _category_onehot(step)
            result.append(onehot.index(1.0))
    return result


def build_model(
    tokenizer: ProcessStepTokenizer,
    max_seq_len: int = 256,
    n_embd: int = 256,
    n_layer: int = 6,
    n_head: int = 8,
    embedding_init: str | None = None,
    desc_path: str | Path | None = None,
    category_embed: bool = False,
) -> GPT2LMHeadModel:
    """Build a small GPT-2 from scratch with the process-step vocabulary.

    embedding_init='description' initializes wte from step descriptions (ADR-014).
    Special tokens keep their random init regardless of embedding_init.
    """
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
    model = GPT2LMHeadModel(config)

    if embedding_init == "description" and desc_path is not None:
        from src.models.desc_embed import build_description_feature_matrix
        feat = build_description_feature_matrix(
            step_names=tokenizer.id_to_step,
            desc_path=Path(desc_path),
            n_embd=n_embd,
        )
        # Replace process-step rows; keep random init for special tokens
        special_ids = {tokenizer.step_to_id[t] for t in SPECIAL_TOKENS if t in tokenizer.step_to_id}
        with torch.no_grad():
            for i, vec in enumerate(feat):
                if i not in special_ids and vec.norm() > 0:
                    model.transformer.wte.weight[i] = vec
        print(f"[model] embedding_init=description  desc_path={desc_path}")

    if category_embed:
        from src.models.desc_embed import N_CATEGORIES
        step_to_cat = _build_step_to_cat(tokenizer)
        model.transformer.wte = CategoryAwareEmbedding(
            model.transformer.wte, N_CATEGORIES, n_embd, step_to_cat
        )
        print(f"[model] category_embed=True  n_categories={N_CATEGORIES}")

    return model


def load_model(
    ckpt_dir: str,
    tokenizer: ProcessStepTokenizer,
    device: str | torch.device = "cpu",
) -> GPT2LMHeadModel:
    """Load a previously saved model checkpoint."""
    from pathlib import Path
    path = Path(ckpt_dir) / "model.pt"
    state = torch.load(path, map_location=device, weights_only=True)
    # infer arch from state dict shapes
    n_positions, n_embd = state["transformer.wpe.weight"].shape
    n_layer = max(int(k.split(".")[2]) for k in state if k.startswith("transformer.h.")) + 1
    n_head  = state["transformer.h.0.attn.c_attn.weight"].shape[1] // (3 * n_embd // n_embd)
    # c_attn projects to 3*n_embd; heads = n_embd // head_dim, head_dim inferred from config default
    model = build_model(tokenizer, max_seq_len=n_positions, n_embd=n_embd, n_layer=n_layer)
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
    ids = tokenizer.encode(partial_steps)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask).logits[0, -1]
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
    ids = tokenizer.encode(partial_steps)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_steps,
            eos_token_id=tokenizer.eos_id,
            pad_token_id=tokenizer.pad_id,
            do_sample=False,
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
    attn_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        loss = model(input_ids, attention_mask=attn_mask, labels=input_ids).loss
    return math.exp(min(loss.item(), 20.0))  # clip to avoid overflow
