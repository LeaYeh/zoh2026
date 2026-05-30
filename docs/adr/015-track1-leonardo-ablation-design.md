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

Run on 2-layer / 128-dim toy model, 200 steps, 1k sequences per family, MPS (Apple Silicon).
**Not comparable to Leonardo.** Prior numbers (0.047/0.057) were invalidated by an EOS inference
bug fixed on 2026-05-30 — `predict_next_step` was passing `[BOS…EOS]` as input, causing
`logits[-1]` to predict what follows EOS rather than the next process step.

| Exp | WandB run | step 100 Top-1 | step 200 Top-1 | Top-3 | MRR | Trend |
|---|---|:---:|:---:|:---:|:---:|---|
| A random + fam | `quick_A_rand\|fam` | 0.750 | **0.787** | 0.970 | 0.883 | Best at 200 steps; flat after step 100 |
| B desc + fam | `quick_B_desc\|fam\|desc` | 0.707 | 0.753 | 0.960 | 0.862 | Still rising; −3.4% vs A |
| C desc + cat + fam | `quick_C_categ\|fam\|desc\|cat` | 0.687 | 0.760 | 0.967 | 0.866 | Steepest rise (+7.3pp); −2.7% vs A |

**At 200 steps: A > C > B.** C shows the steepest step-100→200 gain, consistent with
`cat_embed` being zero-initialized (no-op at start, learns incrementally). Final ranking
requires 3,000 steps on the full 6-layer/256-dim model — defer to Leonardo.

Reference point: `gpt2_mac_local` (6L/256d, 10k seqs, no family token) reached Top-1=0.810
at step 200 on MPS, confirming the full-size model substantially outperforms the smoke-test
toy model. Leonardo Exp A (same full model + family token) is expected to exceed 0.810.

---

## Consequences

- Three jobs can be submitted in parallel — no inter-dependency
- Gate 2 evaluation uses the winning experiment's checkpoint
- If all three experiments fall below Top-1 = 0.40, the next lever is data augmentation or LR tuning before trying a larger model
