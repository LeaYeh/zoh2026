# ADR 015 — Track 1: Leonardo Three-Way Ablation Design

**Status:** Accepted  
**Date:** 2026-05-30  
**Related:** ADR-012 (EDA findings), ADR-013 (model selection), ADR-014 (description embedding init)

---

## Context

Three experiments are scheduled to run on Leonardo A100 to determine the optimal
embedding strategy before committing to a final training configuration.  
Each experiment changes exactly one variable vs the previous (Rule 5).

---

## Shared Architecture

All three experiments use the same transformer configuration:

| Component | Value |
|---|---|
| Model type | GPT-2-style decoder-only transformer (trained from scratch) |
| Vocabulary | 205 tokens — 198 step names + `<PAD>/<BOS>/<EOS>/<UNK>` + `[IC]/[IGBT]/[MOSFET]` |
| `n_embd` | 256 |
| `n_layer` | 6 |
| `n_head` | 8 |
| FFN width | 1024 (4× n_embd) |
| `max_seq_len` | 256 (sequences peak at ~160 tokens — 6× headroom) |
| Total params | ~4.86M |
| `lm_head` | weight-tied to `wte` (no extra parameters) |

---

## Shared Training Config

| Hyperparameter | Value | Rationale |
|---|---|---|
| Training data | `ic_10k.csv`, `igbt_10k.csv`, `mosfet_10k.csv` | 30k sequences, vocab-converged at 198 steps |
| `val_ratio` | 0.10 | 3k held-out sequences |
| `family_prefix` | `true` | `[IC]/[IGBT]/[MOSFET]` prepended to every sequence — entropy reduction 0.16–0.33 bits |
| `steps` | 3,000 | |
| `batch_size` | 64 | |
| `lr` | 5e-4 | AdamW |
| `warmup_steps` | 300 | cosine decay after warmup |
| `grad_clip` | 1.0 | |
| `eval_every` | 300 | Top-1/3/5 + MRR on 300 val samples |
| `seed` | 42 | torch + numpy unified |
| PAD masking | `labels = -100` on PAD positions | Rule 3 |

---

## Three-Way Ablation Table

| | **Exp A** | **Exp B** | **Exp C** |
|---|:---:|:---:|:---:|
| Config file | `leonardo_A_family_token.yaml` | `leonardo_B_desc_embed.yaml` | `leonardo_C_desc_categ.yaml` |
| WandB run name | `leonardo_A_family_token` | `leonardo_B_desc_embed` | `leonardo_C_desc_categ` |
| **`wte` init** | Random (σ=0.02) | Description-based | Description-based |
| **`CategoryAwareEmbedding`** | — | — | ✓ |
| Extra trainable params | 0 | 0 | +2,048 (8 × 256) |
| **Changed vs previous** | — (baseline) | init method only | +cat_embed only |

### Exp A — Random Init (baseline)

`wte` initialized with standard GPT-2 Gaussian (σ=0.02, mean=0).  
No prior knowledge injected. Sequence co-occurrence is the sole learning signal.

### Exp B — Description Embedding Init (ADR-014)

`wte` initialized from a 27-dim feature vector projected to 256-dim:

```
[0:8]   category one-hot  — 8 process classes (litho/thermal/deposit/etch/implant/measure/clean/other)
[8:11]  numeric params    — temperature_c, time_min, litho_level  (from parameters_parsed.json)
[11:27] TF-IDF SVD        — step names decomposed to 16-dim

→ Linear(27, 256, no bias)
→ L2-normalize, scale to GPT-2 init norm: 0.02 × √256 ≈ 0.32
```

- 136/198 process steps have `parameters_parsed.json` entries; 62 steps use numeric=0 fallback  
- Special tokens (`<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`, `[IC]`, `[IGBT]`, `[MOSFET]`) keep random init  
- `wte` is **not frozen** — description init provides a better gradient starting point, not a constraint

### Exp C — Description Init + CategoryAwareEmbedding

Wraps `model.transformer.wte` with `CategoryAwareEmbedding`:

```python
embedding(token_id) = wte(token_id) + cat_embed(category_id)
```

- `cat_embed`: `nn.Embedding(8, 256)`, zero-initialized (no-op at step 0, learns incrementally)
- `step_to_cat_ids`: static buffer — maps each vocab token to one of 8 category IDs
- `.weight` property proxies `wte.weight` to preserve `lm_head` weight tying
- Hypothesis: shared per-category residual helps all litho steps converge together

---

## Decision Criteria

Results are evaluated at step 3,000 on `eval/top1`:

| Outcome | Action |
|---|---|
| A best | Use Exp A config for final training; discard B and C |
| B best | Use Exp B config; description init confirmed useful |
| C best | Use Exp C config; category embedding provides additional signal |
| B ≈ C | Use B (simpler); category embedding adds no measurable benefit |
| All < 0.40 | Gate 2 fail — lower LR, generate more data, or scale model |

Winning config becomes the base for any subsequent experiments (one variable at a time).

---

## Run Commands (Leonardo)

```bash
sbatch scripts/slurm/train.slurm configs/exp/leonardo_A_family_token.yaml
sbatch scripts/slurm/train.slurm configs/exp/leonardo_B_desc_embed.yaml
sbatch scripts/slurm/train.slurm configs/exp/leonardo_C_desc_categ.yaml
```

Sync WandB after run:

```bash
wandb sync ~/zoh2026/wandb/offline-run-*/
```

---

## Local Smoke Test Results (reference only)

Run on 2-layer / 128-dim toy model, 200 steps, real data. **Not comparable to Leonardo.**

| Exp | best Top-1 | Trend |
|---|---|---|
| A random | 0.047 | Peaks at step 100, then overfits |
| B desc | 0.057 (+21%) | Still improving at step 200 |
| C desc + cat | 0.057 (= B) | Indistinguishable at 200 steps; needs 3,000 steps on A100 |

---

## Consequences

- Three jobs can be submitted in parallel — no inter-dependency
- Gate 2 evaluation uses the winning experiment's checkpoint
- If all three experiments fall below Top-1 = 0.40, the next lever is data augmentation or LR tuning before trying a larger model
