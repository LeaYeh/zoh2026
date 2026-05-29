---
name: time-series-finetuning
description: |
  Complete knowledge base for fine-tuning time-series foundation models and
  building forecasting agent systems, distilled from 李宏毅 ML 2025 + HW01–07.
  Use for ANY task touching time-series forecasting, model fine-tuning,
  probabilistic evaluation, or decision agent design — especially during
  Zero One Hack Track 3 (Sybilion).
  Trigger on: fine-tune LoRA PEFT QLoRA Chronos TimesFM Moirai Lag-Llama
  CRPS WQL coverage training-loop A100 BF16 agent ReAct tool-use CLAUDE.md
  gate checkpoint model-merging SLERP TIES DPO catastrophic-forgetting
  TimeSeriesSplit probabilistic-forecast quantile overfitting optimizer
  scheduler gradient-accumulation experiment-tracking
metadata:
  source: 李宏毅 ML 2025 (Lectures 1, 5, 6, 7, 8, 10, 11) + HW01–07
  competition: Zero One Hack 2026, Vienna
---

# time-series-finetuning

## Seven Core Rules (violating any one will cause problems)

1. **Fine-tuning is a last resort** — confirm zero-shot is insufficient before touching weights
2. **Quality >> Quantity** — 1,000 curated samples beats 100,000 low-quality ones
3. **Lower target loss = more forgetting** — always monitor benchmark metrics simultaneously
4. **Fine-tuning only amplifies existing capabilities** — base model selection sets the ceiling
5. **Model merging requires the same foundation** — different base architectures cannot be merged
6. **Inference length: less is more** — stop as soon as it's good enough, don't chase a perfect chain
7. **Fallback first to keep the demo alive** — broken pipeline → zero-shot + post-processing rules

---

## 36-Hour Competition SOP

```
Receive data
  │
  ▼
① Run zero-shot Chronos, record baseline WQL
  ├─ Good enough → adjust with post-processing rules, submit
  └─ Not good enough → continue analysis
  │
  ▼
② Diagnose zero-shot failure modes
  ├─ Systematic over/under-prediction → IKE few-shot correction (no param changes, <10 min)
  └─ Specific sequence types failing → filter MaybeKnown training set
  │
  ▼
③ Fine-tuning
  ├─ Prepare Pseudo Replay (5% public benchmark sequences to prevent forgetting)
  ├─ Choose Chronos-T5-Large (fits in A100 VRAM)
  ├─ Evaluate target WQL + ETTh1 MASE every epoch
  └─ ETTh1 MASE degrades > 10% → stop immediately, add more replay and rerun
  │
  ▼
④ Validate and submit
  ├─ Majority Vote (10 different seeds)
  ├─ Verifier selects best using pinball loss
  └─ Demo runs → submit
  │
  ⑤ Fallback (pipeline broken / < 4 hours remaining)
     zero-shot + post-processing → clip outliers + calibrate intervals + restore seasonality
```

---

## Symptom → Reference Routing

| Symptom | Diagnosis | Quick Fix | Detailed Reference |
|---------|-----------|-----------|-------------------|
| Benchmark drops after fine-tune | Catastrophic forgetting | Add 5% Pseudo Replay and rerun | `02-training-execution.md` |
| Fine-tune doesn't improve target task | Data is Unknown | Switch to MaybeKnown, filter with zero-shot first | `01-data-strategy.md` |
| All predictions systematically high/low | Distributional bias | Try IKE few-shot first | `03-evaluation-debugging.md` |
| LoRA fine-tune has little effect | Rank too small | rank 8→16→32 or full fine-tune | `02-training-execution.md` |
| Predictions vary wildly across runs | Inference instability | Majority Vote (10 seeds) | `04-inference-agent.md` |
| Checkpoint merge degrades performance | Task vector parameter conflict | DARE / TIES pre-processing | `05-advanced-techniques.md` |
| Inference too slow | Reasoning chain too long | Chain of Draft / Implicit CoT | `04-inference-agent.md` |
| New domain with no labelled data | Transfer learning problem | Task Vector analogy merge | `05-advanced-techniques.md` |
| Want to fix just one small bias | Fine-tuning not needed | Model Editing (IKE is fastest) | `05-advanced-techniques.md` |

---

## Reference Routing

Read only what's relevant — no need to read everything:

| What I want to do | Read this |
|-------------------|-----------|
| Design or clean a fine-tuning dataset | `01-data-strategy.md` |
| Run fine-tuning, design anti-forgetting strategy | `02-training-execution.md` |
| Diagnose model problems, evaluate quality | `03-evaluation-debugging.md` |
| Design inference pipeline or decision agent | `04-inference-agent.md` |
| Merge checkpoints, precise bias correction, domain transfer | `05-advanced-techniques.md` |
