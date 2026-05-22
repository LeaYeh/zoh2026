# skills/SKILL.md — Skills Index

> Claude Code reads CLAUDE.md on startup; CLAUDE.md points here.
> Before starting any task, check whether a skill applies and must be followed.

---

## Skill Map

| Task type | Required skill | Trigger keywords |
|-----------|---------------|-----------------|
| Data exploration | `01_eda.md` | EDA, explore data, look at data, exploratory |
| CV setup | `02_cv_and_validation.md` | CV, fold, validation |
| Feature engineering | `03_features.md` | feature, FE |
| Model training | `04_training.md` | train, baseline, model |
| Ensemble | `05_ensemble.md` | ensemble, stacking, blending |
| Write report | `06_review_report.md` | after every training / ensemble run |

---

## Claude Code Protocol

### 1. Skill takes priority over memory
Even if Claude knows how to do something, if a skill exists for it, **read that skill first** and follow its steps.

### 2. Report is mandatory
After any training or ensemble run, **must** produce a report following `06_review_report.md`.
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
Human: Run an XGBoost baseline
AI: Sure, let me write the code [immediately starts coding]
```

**Good:**
```
Human: Run an XGBoost baseline
AI: Reading skills/04_training.md first...
    [done]
    Per skill rules:
    - Use train_oof() unified function
    - Load data/folds.pkl
    - Save OOF to data/oof/
    - Produce review report after run
    - Log everything to wandb

    I will modify:
    - configs/exp/xgb_v0.yaml (new)
    - data/oof/train_oof_xgb_v0.npy (new)
    - data/oof/test_preds_xgb_v0.npy (new)
    - reports/xgb_v0.md (new)

    I will NOT touch: data/raw/, data/folds.pkl

    Proceed?
```
