# 02 · Training Execution
**When to read**: running fine-tuning, designing anti-forgetting strategies, tuning LoRA parameters

---

## Catastrophic Forgetting Is a Law, Not a Warning

```
target task loss ↓  ←→  forgetting degree ↑
```

Key facts:
- **Larger models don't forget less** (experimentally confirmed)
- LoRA reduces forgetting because it *learns less*, not because it solves forgetting
- Dropout / weight decay cannot prevent forgetting
- Forgetting affects not just safety, but general formatting ability (e.g. JSON output)

---

## Four Anti-Forgetting Methods

### Method 1: Experience Replay (best results, most recommended)
Mix 5% old-task data into the fine-tuning dataset:
```python
REPLAY_DATASETS = ["ETTh1", "ETTh2", "Weather", "Traffic"]

def build_training_set(target_data, replay_ratio=0.05):
    replay = sample_from_benchmarks(REPLAY_DATASETS)
    n_replay = int(len(target_data) * replay_ratio)
    return target_data + replay[:n_replay]
```

### Method 2: Pseudo Replay (when original data is unavailable)
Run the base model first, save its outputs as replay data:
```python
# Step 1: before fine-tuning, run base Chronos on target sequences and save outputs
base_preds = chronos_base.predict(target_sequences)
pseudo_replay = list(zip(target_sequences, base_preds))

# Step 2: mix into fine-tuning set (5%)
training_set = target_data + pseudo_replay[:n_replay]
```
Concept: let the old version leave "memories" that the new version inherits.

### Method 3: Paraphrase (data augmentation + anti-forgetting)
Describe the same series in multiple formats so the model sees more variation:
```python
def paraphrase_series(series, label):
    return [
        (normalize_standard(series), label),
        (normalize_minmax(series), label),
        (resample_weekly(series), label),
        *[(series[-ctx:], label) for ctx in [64, 128, 256]],
    ]
```

### Method 4: Self-output (strong model teaches weak model)
```python
# Chronos-Large as teacher, producing pseudo-labels for Chronos-Small
teacher_preds = chronos_large.predict(inputs, num_samples=100)
chronos_small.finetune(inputs, targets=teacher_preds)
# LLM-teaches-LLM often outperforms human annotations with less forgetting
```

---

## Training Monitoring SOP (run every epoch)

```python
def monitor_checkpoint(model, target_val, benchmark_val, baseline_mase):
    target_wql = compute_wql(model, target_val)
    current_mase = compute_mase(model, benchmark_val)

    print(f"Target WQL:      {target_wql:.4f}")
    print(f"Benchmark MASE:  {current_mase:.4f}  (baseline: {baseline_mase:.4f})")

    # Stop condition: benchmark degrades > 10%
    if current_mase > baseline_mase * 1.10:
        print("⚠️  Forgetting alarm — add more replay data and rerun")
        return "STOP"
    return "CONTINUE"
```

---

## LoRA Parameter Tuning Guide

```python
# rank too small → learns too little; rank too large → learns too much, forgets severely
# Recommended search strategy:
for rank in [8, 16, 32]:
    model = apply_lora(base_model, rank=rank)
    result = train_and_eval(model, target_val, benchmark_val)
    # Goal: smallest rank where target WQL is good AND MASE doesn't degrade
```

If full fine-tuning is clearly better than LoRA: the task is far from the base model's
distribution — you may need more replay data.

---

## Training Execution Checklist

```
□ Record baseline: zero-shot WQL + ETTh1 MASE
□ Prepare replay data: at least 5% public benchmark sequences
□ Training data diversity: verify coverage across lengths/frequencies/missing values
□ Monitoring setup: evaluate target + benchmark every epoch
□ Early stopping: halt immediately if benchmark MASE degrades > 10%
□ Final evaluation: Majority Vote (10 seeds) + verifier selection
```

---

## Supplement: Implementation Code and Advanced Settings

---

## 1. Core Fine-tuning Mindset

```
Pre-training:  large unlabelled corpus, learns world knowledge (expensive, don't touch)
Post-training: small labelled corpus, aligns to target task (what we do)

Catastrophic Forgetting → fine-tuning too aggressively destroys original capabilities
Solution: low lr (1e-5~5e-5) + LoRA (freezes most weights)

When to use full fine-tune vs LoRA:
  data < 10K samples  → LoRA
  data > 50K samples  → full fine-tune is viable (LoRA still preferred for speed)
```

---

## 2. Standard LoRA Configuration

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,        # or SEQ_2_SEQ_LM (Chronos/T5)
    r=16,                                 # rank: 8 (light) / 16 (standard) / 32 (strong)
    lora_alpha=32,                        # scaling = alpha/r, typically set to 2*r
    target_modules=["q_proj", "v_proj"], # attention Q and V only
    # for time-series models try: ["q_proj", "v_proj", "k_proj", "o_proj"]
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Expected: trainable params ~1-3% of total
```

**LoRA rank selection guide:**
```
r=8   → fast idea validation, smallest memory footprint
r=16  → default competition choice
r=32  → larger datasets, stronger expressiveness
r=64  → near full fine-tune, usually unnecessary
```

---

## 3. QLoRA (when memory is tight)

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                # nf4 recommended
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,           # extra memory saving
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# QLoRA: must apply LoRA after loading
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```

---

## 4. Chronos Fine-tuning (Track 3 primary model)

```python
from chronos import ChronosPipeline
import torch

# Step 1: load pipeline to get model and tokenizer
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",   # small: 20M / base: 200M / large: 710M
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
model = pipeline.model.model  # inner T5 model

# Step 2: tokenize (Chronos has its own mean-scale tokenizer)
def tokenize_chronos(context, target, pipeline, device):
    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    tgt_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

    ctx_ids, scale = pipeline.model.context_input_transform(ctx_tensor.to(device))
    tgt_ids = pipeline.model.label_input_transform(tgt_tensor.to(device), scale)
    return ctx_ids, tgt_ids, scale

# Step 3: standard training loop (see §7 below)
```

---

## 5. HuggingFace Trainer Configuration

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./checkpoints",

    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,    # effective batch = 32

    learning_rate=2e-5,               # small lr for fine-tuning
    warmup_ratio=0.1,                 # 10% steps for warmup
    lr_scheduler_type="cosine",
    weight_decay=0.01,

    # A100 acceleration
    bf16=True,
    tf32=True,                        # A100 only
    dataloader_num_workers=4,

    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    report_to="none",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)
trainer.train()
```

---

## 6. A100 Training Optimisations

```python
import torch

# Enable TF32 (A100 only, near-zero precision loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# BF16 autocast (A100 native, more stable than FP16)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
# Note: BF16 does NOT need GradScaler; FP16 does

# Gradient checkpointing (trade time for memory)
model.gradient_checkpointing_enable()

# Flash Attention 2 (if model supports it)
model = AutoModelForCausalLM.from_pretrained(
    model_id, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
)
```

**Gradient accumulation (simulate large batch):**
```python
ACCUM_STEPS = 4
optimizer.zero_grad()
for step, batch in enumerate(loader):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(**batch).loss / ACCUM_STEPS
    loss.backward()
    if (step + 1) % ACCUM_STEPS == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

---

## 7. Optimizer + Scheduler Selection

```python
# Standard fine-tuning configuration
optimizer = torch.optim.AdamW(
    model.parameters(), lr=2e-5, weight_decay=0.01, eps=1e-8
)

# Warmup + Cosine (recommended)
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
warmup = LinearLR(optimizer, start_factor=0.1, total_iters=500)
cosine = CosineAnnealingLR(optimizer, T_max=total_steps - 500)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[500])
```

---

## 8. Saving After Fine-tuning

```python
# LoRA model: save only the adapter (a few MB, not GB)
model.save_pretrained("./lora_adapter")

# Merge and save full model (for deployment)
merged = model.merge_and_unload()
merged.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

# Load adapter to continue training
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./lora_adapter")
```

---

## 9. OOM Troubleshooting Checklist

```
□ Using BF16?                    → add bf16=True
□ Gradient checkpointing?        → model.gradient_checkpointing_enable()
□ Batch size too large?          → halve it, double gradient_accumulation_steps
□ Using QLoRA?                   → load_in_4bit=True
□ device_map on load?            → device_map="auto" for multi-GPU
□ Clearing cache?                → torch.cuda.empty_cache()
```
