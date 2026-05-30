# skills/SKILL.md — Skills Index

> Claude Code reads CLAUDE.md on startup; CLAUDE.md points here.
> Before starting any task, check whether a skill applies and must be followed.

---

## Skill Map — Track 1: Industrial AI (Infineon)

| Task type | Required skill | Trigger keywords |
|-----------|---------------|-----------------|
| Data exploration | `skill_eda.md` | EDA, explore data, look at sequences, data arrives |
| Model training | `skill_baseline.md` | train, baseline, GPT-2, run model |
| Fine-tuning / GRPO | `skill_finetune.md` | fine-tune, GRPO, RL training, rule-based reward, forgetting |
| Evaluation & metrics | `skill_eval.md` | evaluate, Top-1, MRR, edit distance, F1, ROC-AUC, anomaly |
| Submission | `skill_demo.md` | submit, nextstep.csv, completion.csv, anomaly.csv, demo |
| Write report | `06_review_report.md` | after every training run |
| **Full knowledge base** | `process-sequence-modeling/SKILL.md` | GPT-2, transformer, sequence-model, process-steps, causal-LM, tokenizer, GRPO, catastrophic-forgetting, Top-1, MRR, perplexity, IC, IGBT, MOSFET, generation_rules, submit.py |

---

## Claude Code Protocol

### 1. Skill takes priority over memory
Even if Claude knows how to do something, if a skill exists for it, **read that skill first** and follow its steps.

### 2. Report is mandatory
After any training run, **must** produce a report following `06_review_report.md`.
The "anomaly flags" and "human decision required" fields cannot be omitted.

### 3. Gates require human confirmation
When reaching Gate 1/2/3/4, **stop** and explicitly ask: "Gate X — do you approve?"
Never assume approval.

### 4. Declare scope before acting
At the start of every task, state:
- "I will modify: [file list]"
- "I will not touch: [off-limits list]"

### 5. Stop after 3 consecutive failures
Same operation failing 3 times in a row → stop, report to human.
No infinite retry loops.

---

## Example Dialogue

**Bad:**
```
Human: Run the GPT-2 baseline
AI: Sure, let me write the code [immediately starts coding]
```

**Good:**
```
Human: Run the GPT-2 baseline
AI: Reading skills/process-sequence-modeling/SKILL.md first...
    [done]
    Per skill rules:
    - One token = one step name (never split step strings)
    - PAD positions must be masked with labels=-100
    - Save checkpoint on best eval/top1
    - Produce review report after run
    - Log everything to WandB

    I will modify:
    - configs/exp/gpt2_finetune_v1.yaml (verify settings)
    - data/oof/gpt2_finetune_v1/ (new checkpoint)

    I will NOT touch: data/raw/, data/raw/training_data/

    Proceed?
```
