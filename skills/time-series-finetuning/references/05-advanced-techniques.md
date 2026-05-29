# 05 · Advanced Techniques
**When to read**: precise bias correction, merging checkpoints, domain transfer, zero-data capability transfer

---

## Model Editing — Surgical Precision

### When to use (vs fine-tuning)

| | Model Editing | Fine-tuning |
|--|--|--|
| Scope of change | One bias / one fact | Broad skill |
| Data required | A few samples | Large corpus |
| Speed | Fast (minutes) | Slow (training required) |
| Risk | May affect unrelated capabilities | Catastrophic forgetting |

### Three Quality Criteria (check all after editing)
1. **Reliability**: the target input's output actually changed
2. **Generalization**: rephrasing the question gives the same corrected answer
3. **Locality**: unrelated inputs are unaffected

### IKE — Fastest Correction with No Parameter Changes

```python
def ike_prompt(query, correction_fact, examples):
    return f"""
Answer based on the following updated information:

[Reliability Example]
Q: {examples['reliability']['q']}
A: {examples['reliability']['a']}

[Locality Example]
Q: {examples['locality']['q']}
A: {examples['locality']['a']}  (this type is unaffected)

Updated knowledge: {correction_fact}

Now answer: {query}
"""

# Time-series application: correct Monday over-prediction bias
correction = "Periodic series Monday periods need a -12% systematic bias correction"
```

### Pitfalls
- Gradient descent on a single sample → model degenerates to only answering that one question
- Only verify Reliability, forget Locality → editing changes things it shouldn't

---

## Model Merging — Task Vectors

### Core Formula
```
Task Vector τ = θ_finetuned - θ_base

Merge: θ_merged = θ_A + λ * τ_B
```

**Hard requirement**: both models must come from the **same foundation model** with identical architecture.

---

### Addition: Acquire Dual Capabilities

```python
def merge_add(model_a, model_b, base, lam=1.0):
    tau_b = {k: model_b[k] - base[k] for k in base}
    return {k: model_a[k] + lam * tau_b[k] for k in model_a}

# Time-series application:
# chronos_energy = finetune(base, energy_data)
# chronos_retail = finetune(base, retail_data)
# combined = merge_add(chronos_energy, chronos_retail, base, lam=0.7)
```

**λ search strategy**:
```python
for lam in [0.3, 0.5, 0.7, 1.0]:
    merged = merge_add(model_a, model_b, base, lam)
    score = evaluate(merged, val_set)
    print(f"λ={lam}: WQL={score:.4f}")
# Pick the λ with lowest WQL
```

---

### Subtraction: Forget Bad Capability

```python
def merge_subtract(model, bad_model, base, lam=0.5):
    bad_tau = {k: bad_model[k] - base[k] for k in base}
    return {k: model[k] - lam * bad_tau[k] for k in model}

# Time-series application: checkpoint is overfitted
# overfit = load_checkpoint("epoch_20")
# good    = load_checkpoint("epoch_10")
# fixed   = merge_subtract(good, overfit, base)
```

---

### Analogy: Transfer Without Data

```
Task A : Task B = Task C : Task D
Given A, B, C — derive D (even with no real D data)

τ_D = τ_B + (τ_C - τ_A)
```

```python
# Time-series application: have old domain synthetic/real pair, lack new domain real data
tau_A = finetune(base, synthetic_old) - base  # old domain, synthetic
tau_B = finetune(base, real_old) - base       # old domain, real
tau_C = finetune(base, synthetic_new) - base  # new domain, synthetic

tau_D = tau_B + (tau_C - tau_A)               # derived: new domain, real
new_domain_model = base + tau_D
```

---

## Merging Pitfalls and Solutions

### Problem: Two Models Modify the Same Parameters

```
Symptom: merged model is worse than either model individually
```

**Solution 1: DARE (Drop And REscale)**
```python
def dare(tau, drop_rate=0.9):
    mask = (torch.rand_like(tau) > drop_rate).float()
    return tau * mask / (1 - drop_rate)
# Randomly drop 90% of task vector parameters then rescale
```

**Solution 2: TIES (keep only the large ones)**
```python
def ties(tau, top_k=0.2):
    threshold = torch.quantile(tau.abs(), 1 - top_k)
    return tau * (tau.abs() >= threshold).float()
# Keep only the top 20% by absolute value
```

### Important Fact
- **Larger models produce more stable merges** (research finding)
- When VRAM allows, prefer Chronos-T5-Large over Small

---

## Hackathon Quick Merge Workflow

```python
def quick_merge(ckpt_a, ckpt_b, base, val_set):
    best_score, best_model = float('inf'), None
    for lam in [0.3, 0.5, 0.7, 1.0]:
        # DARE pre-processing
        tau_b = {k: ckpt_b[k] - base[k] for k in base}
        tau_b_dare = dare(tau_b, drop_rate=0.9)
        merged = {k: ckpt_a[k] + lam * tau_b_dare[k] for k in ckpt_a}
        score = evaluate_wql(merged, val_set)
        if score < best_score:
            best_score, best_model = score, merged
    return best_model
```

---

## Supplement: Implementation Code and Advanced Settings

---

## 1. Model Merging Overview

**Why merge?**
- Freely combine capabilities from multiple fine-tuned checkpoints without retraining
- In competitions: merge models trained on different seeds / data splits for stability

**Method selection:**
```
checkpoint A + checkpoint B, same base, slightly different  → SLERP (t=0.5)
multiple fine-tuned models, combine different capabilities  → TIES
fine-tuned model, need to denoise                          → DARE + TIES
```

---

## 2. SLERP (Spherical Linear Interpolation)

```python
import torch

def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation — preserves geometric structure in parameter space"""
    v0f, v1f = v0.double(), v1.double()
    norm0, norm1 = torch.norm(v0f), torch.norm(v1f)

    if norm0 < eps or norm1 < eps:
        return ((1 - t) * v0f + t * v1f).to(v0.dtype)

    dot = torch.clamp(torch.dot(v0f.flatten(), v1f.flatten()) / (norm0 * norm1), -1, 1)
    theta = torch.acos(dot)

    if theta.abs() < eps:
        return ((1 - t) * v0f + t * v1f).to(v0.dtype)

    return ((torch.sin((1-t)*theta)/torch.sin(theta)) * v0f +
            (torch.sin(t*theta)/torch.sin(theta)) * v1f).to(v0.dtype)

def merge_slerp(state_a: dict, state_b: dict, t: float = 0.5) -> dict:
    """t=0 → model_a fully; t=1 → model_b fully; t=0.5 → midpoint"""
    return {k: slerp(t, state_a[k], state_b[k]) for k in state_a}

# Usage
state_a = torch.load("checkpoint_epoch2.pt")
state_b = torch.load("checkpoint_epoch3.pt")
merged  = merge_slerp(state_a, state_b, t=0.5)
model.load_state_dict(merged)
```

---

## 3. TIES (Trim, Elect Sign, Merge)

```python
import torch

def compute_task_vector(ft_state: dict, base_state: dict) -> dict:
    """task vector = fine-tuned - pretrained"""
    return {k: ft_state[k].float() - base_state[k].float() for k in ft_state}

def ties_merge(task_vectors: list[dict], density: float = 0.2, lamb: float = 1.0) -> dict:
    """
    density: fraction of parameters with largest changes to keep (trimming)
    lamb:    merge strength (usually 1.0)
    """
    result = {}
    keys = task_vectors[0].keys()

    for key in keys:
        tvs = torch.stack([tv[key] for tv in task_vectors])  # (n_models, ...)

        # Step 1: Trim — keep only top-density parameters per model
        trimmed = []
        for tv in tvs:
            flat = tv.abs().flatten()
            threshold = torch.quantile(flat, 1 - density)
            trimmed.append(tv * (tv.abs() >= threshold).float())
        trimmed = torch.stack(trimmed)

        # Step 2: Elect Sign — majority vote weighted by absolute value
        sign_votes = trimmed.sum(dim=0)
        elected_sign = torch.sign(sign_votes)
        elected_sign[elected_sign == 0] = 1   # tie → positive

        # Step 3: Disjoint Merge — only merge parameters with consistent sign
        aligned = trimmed * (torch.sign(trimmed) == elected_sign.unsqueeze(0)).float()
        result[key] = lamb * aligned.mean(dim=0)

    return result

def apply_task_vector(base_state: dict, merged_tv: dict) -> dict:
    return {k: base_state[k].float() + merged_tv[k] for k in base_state}
```

---

## 4. DARE (Drop And REscale) — Pre-processing Before TIES

```python
def dare(task_vector: dict, p: float = 0.9) -> dict:
    """
    Randomly drop p fraction of parameters then rescale to compensate.
    Typically applied before TIES to denoise fine-tuning artifacts.
    """
    result = {}
    for key, tv in task_vector.items():
        mask = (torch.rand_like(tv) > p).float()
        result[key] = tv * mask / (1 - p + 1e-8)
    return result

# Standard DARE + TIES workflow
base_state = torch.load("chronos_pretrained.pt")
tv1 = compute_task_vector(torch.load("finetuned_seed42.pt"), base_state)
tv2 = compute_task_vector(torch.load("finetuned_seed7.pt"),  base_state)

tv1_dare = dare(tv1, p=0.9)
tv2_dare = dare(tv2, p=0.9)
merged_tv = ties_merge([tv1_dare, tv2_dare], density=0.2)
final_state = apply_task_vector(base_state, merged_tv)
```

---

## 5. mergekit CLI (fastest approach)

```bash
pip install mergekit

# SLERP: merge two checkpoints
cat > merge.yml << 'EOF'
models:
  - model: ./checkpoint_epoch2
  - model: ./checkpoint_epoch3
merge_method: slerp
base_model: ./checkpoint_epoch2
parameters:
  t: 0.5
dtype: bfloat16
EOF

mergekit-yaml merge.yml ./merged_model --cuda

# TIES: merge multiple fine-tuned models
cat > ties_merge.yml << 'EOF'
models:
  - model: ./finetuned_seed42
    parameters: {density: 0.5, weight: 1.0}
  - model: ./finetuned_seed7
    parameters: {density: 0.5, weight: 1.0}
merge_method: ties
base_model: ./pretrained_base
parameters:
  normalize: true
dtype: bfloat16
EOF

mergekit-yaml ties_merge.yml ./ties_merged --cuda
```

---

## 6. DPO Alignment (align output format)

**Use case**: make model output match the competition's required format (structured JSON, specific quantile format)

```python
from trl import DPOConfig, DPOTrainer

# Data format: (prompt, chosen_format, rejected_format)
dpo_data = [
    {
        "prompt": "Forecast next 24 steps given: [1.2, 1.5, 1.3]",
        "chosen":   '{"p10": [1.1, 1.2, ...], "p50": [1.3, 1.4, ...], "p90": [1.6, 1.7, ...]}',
        "rejected": "The forecast is approximately 1.4 for the next period."
    },
    ...
]

dpo_config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,           # DPO uses very small lr
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    bf16=True,
    loss_type="sigmoid",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,          # SFT model, kept frozen
    args=dpo_config,
    train_dataset=Dataset.from_list(dpo_data),
    tokenizer=tokenizer,
)
trainer.train()
```

**ORPO (no ref_model needed, simpler):**
```python
from trl import ORPOConfig, ORPOTrainer

orpo_config = ORPOConfig(
    learning_rate=8e-6,
    lambda=0.1,                   # odds ratio strength
    num_train_epochs=1,
    bf16=True,
)
trainer = ORPOTrainer(model=model, args=orpo_config, ...)
```

---

## 7. Model Editing (inject knowledge without fine-tuning)

**In-Context Editing (most practical, zero cost):**
```python
def inject_domain_knowledge(model_fn, domain_facts: list[str], query: str) -> str:
    """Inject domain knowledge into context for the model to use"""
    facts = "\n".join(f"- {fact}" for fact in domain_facts)
    prompt = f"""Domain-specific knowledge:
{facts}

Based on the above knowledge, answer the following:
{query}"""
    return model_fn(prompt)

# Usage example
facts = [
    "This dataset represents hourly electricity demand in Austria",
    "Peak hours are 8-10am and 6-8pm on weekdays",
    "December-January typically shows 20% higher demand",
]
answer = inject_domain_knowledge(pipeline, facts, "Forecast tomorrow's peak demand")
```

---

## 8. Quantile Ensemble (improve probabilistic calibration)

```python
def quantile_ensemble(models: list, context: np.ndarray, horizon: int,
                      quantile_levels: list = [0.1, 0.5, 0.9]) -> dict:
    """
    Average quantile predictions across multiple models.
    Important: ensemble the quantiles, NOT the samples
    (ensembling samples destroys probabilistic calibration)
    """
    all_quantiles = {q: [] for q in quantile_levels}

    for model in models:
        result = model.predict_quantiles(context, horizon, quantile_levels)
        for q, pred in zip(quantile_levels, result):
            all_quantiles[q].append(pred)

    return {
        f"p{int(q*100)}": np.mean(preds, axis=0)
        for q, preds in all_quantiles.items()
    }
```

---

## References
- mergekit: `github.com/arcee-ai/mergekit`
- 2025 model merging lecture: `speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/merging.pdf`
- TRL DPO: `huggingface.co/docs/trl/dpo_trainer`
- TIES paper: `arxiv.org/abs/2306.01708`
- DARE paper: `arxiv.org/abs/2311.03099`
