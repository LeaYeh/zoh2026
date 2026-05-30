# 01 · Architecture & Training
**When to read**: setting up the model, debugging loss, designing the training loop

> 李宏毅 ML 2025 — Lec 4 (Transformer internals), Lec 6 (Causal LM), HW3 (Implement Transformer), HW4 (Training Transformer)

---

## Transformer for Process Sequences

### Why GPT-2 (decoder-only) not BERT (encoder)

Process sequences are **ordered and causal**: step N depends on steps 1…N-1.
- BERT-style masked LM sees future context — invalid for next-step prediction at inference time
- GPT-style causal LM never sees future tokens → directly trains the Task 1 inference setup
- Use `GPT2LMHeadModel` with causal attention mask (lower-triangular)

### Architecture parameters for ~120-token vocabulary

```python
# configs/exp/gpt2_finetune_v1.yaml
model:
  n_embd: 256      # embedding dimension
  n_layer: 6       # transformer blocks
  n_head: 8        # attention heads (n_embd must be divisible by n_head)
  max_seq_len: 256 # max sequence length — covers 150-step sequences + special tokens
```

Rule of thumb (李宏毅 Lec 4): `head_dim = n_embd / n_head = 32`. Attention score:
```
α = softmax(Q·Kᵀ / √head_dim)
```
Scaling by `√head_dim` prevents softmax saturation on long sequences.

### Vocabulary construction (ProcessStepTokenizer)

**One token = one complete step string.** Never use BPE or character-level splitting.

```python
# src/data/process_loader.py
SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")
# vocab_size = len(SPECIAL_TOKENS) + len(unique_step_names) ≈ 4 + 120 = 124
```

Vocab of ~124 tokens means the model's embedding table is tiny (124 × 256 = 32K params).
Most parameters are in the attention and FFN layers.

---

## Causal LM Training

### Loss: cross-entropy on next token

```
L = -Σ log P(sₜ | s₁, ..., s_{t-1})
```

Implementation in `src/train.py`:
```python
labels = batch_ids.clone()
labels[batch_mask == 0] = -100   # PAD positions ignored by PyTorch cross-entropy
loss = model(batch_ids, attention_mask=batch_mask, labels=labels).loss
```

**Common bug**: forgetting `attention_mask`. Without it, the model attends to PAD tokens and wastes capacity. Symptom: loss is lower than expected but Top-1 is poor.

### Learning rate schedule (cosine with warmup)

From 李宏毅 HW4 best practices:
```
warmup (300 steps) → cosine decay to lr × 0.01
```

```python
# src/train.py
def _lr_lambda(step):
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return max(0.01, 0.5 * (1 + cos(π × progress)))
```

Typical LR: 5e-4 for n_embd=256; reduce to 2e-4 if loss is unstable in first 100 steps.

### Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Prevents gradient explosions, especially early in training when the model starts from random weights. Always clip before `optimizer.step()`.

### Expected loss trajectory

| Step | Expected loss (vocab=124) | Notes |
|------|--------------------------|-------|
| 0 | ~ln(124) ≈ 4.82 | Random weights |
| 100 | ~3.5–4.0 | Model learning token frequencies |
| 500 | ~2.5–3.0 | Patterns emerging |
| 1000+ | <2.0 | Structural patterns learned |

If loss is above 4.5 after 200 steps: check that `labels=-100` on PAD is set correctly.

---

## BF16 on A100

```python
# In src/train.py — enabled automatically on CUDA
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    # model = model.to(torch.bfloat16)  # add for A100 speed
```

BF16 has the same exponent range as FP32 (important for gradient stability) with half the memory. On A100, BF16 matrix multiplications run in TF32 hardware — ~8× faster than FP32.

---

## Debugging Checklist

- [ ] `attention_mask` passed to every `model()` call? (train + inference)
- [ ] `labels` has `-100` on all PAD positions?
- [ ] `vocab_size` in model config == `tokenizer.vocab_size`?
- [ ] `n_embd` divisible by `n_head`?
- [ ] Initial loss close to `ln(vocab_size)`?
- [ ] `model.train()` called before training loop, `model.eval()` before inference?
