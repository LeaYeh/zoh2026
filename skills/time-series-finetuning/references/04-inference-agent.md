# 04 · Inference Optimisation & Agent Design
**When to read**: designing inference pipelines, decision agents, optimising inference speed

---

## Core Principle: Inference Length — Less Is More

```
Asking the same question multiple times ≠ longer reasoning = more accurate
The best AI does the most with limited resources
```

---

## Inference Method Selection

### Method 1: Single Inference (baseline)
Direct prediction — fastest but least stable.

### Method 2: Majority Vote (try this first)
```python
preds = [model.predict(context, seed=i) for i in range(10)]
final = np.median(preds, axis=0)
# Almost always more accurate than single inference; costs 10× inference time
```

### Method 3: Best-of-N + Verifier
```python
candidates = [model.predict(context, seed=i) for i in range(10)]
scores = [pinball_loss(context, p) for p in candidates]
final = candidates[np.argmin(scores)]
```

### Method 4: Beam Search (strongest but slowest)
Keeps N paths at each step. Suitable for agent tasks with longer reasoning chains.
A 1B model with Beam Search can outperform an 8B model's single inference.

---

## Inference Length Control

### Chain of Draft (fast reasoning)
Each step draft is no more than 5 words:
```
trend→rising, volatility→medium, prediction→2.6
# Sometimes more accurate than verbose CoT, and much faster
```

### RL Length-Aware Reward
```python
def length_aware_reward(prediction, ground_truth, reasoning_tokens):
    accuracy = compute_accuracy(prediction, ground_truth)
    if accuracy == 0:
        return 0
    avg_len = compute_avg_correct_length_baseline()
    length_penalty = max(0, reasoning_tokens - avg_len) / avg_len
    return accuracy - 0.1 * length_penalty
```

### Implicit CoT (train mental arithmetic capability)
Progressively reduce reasoning tokens:
```
Epoch 1: full reasoning (100 tokens)
Epoch 2: remove some steps (75 tokens)
Epoch 4: answer only (5 tokens)
# Final capability approaches the full reasoning model but inference is 10×+ faster
```

---

## Decision Agent Design

### When reasoning is appropriate
```
✓ Choosing a forecasting method (trend-dominant vs seasonality-dominant)
✓ Explaining anomalous predictions to users
✓ Recommending actions based on forecasts
```

### When long reasoning is NOT appropriate
```
✗ Batch forecasting (use Majority Vote instead)
✗ Simple extrapolation (compute directly)
✗ Time-sensitive real-time predictions
```

### Agent Reasoning Length Guide

| Task | Recommended method | Token limit |
|------|--------------------|-------------|
| Forecast explanation | Chain of Draft | 50 |
| Anomaly analysis | Few-shot CoT | 200 |
| Action recommendation | Zero-shot CoT | 500 |
| Method selection | Beam Search | Unlimited (offline) |

---

## Journey Learning (train reasoning data with error correction)

Training data must include "wrong then corrected" processes:

```python
# Bad design: only correct paths
bad_data = [{"input": series, "output": correct}]

# Good design: includes errors and corrections
good_data = [{
    "input": series,
    "reasoning": [
        "Initial forecast: 2.8 (too high)",
        "Re-examine: periodic pattern detected, correction needed",
        "Corrected forecast: 2.4"
    ],
    "output": 2.4
}]
# Data with errors→corrections significantly outperforms data with only correct answers
```

---

## Inference Pipeline Template

```python
class TimeSeriesInferencePipeline:
    def __init__(self, model, use_majority_vote=True, n_samples=10):
        self.model = model
        self.use_mv = use_majority_vote
        self.n_samples = n_samples

    def predict(self, context):
        # 1. Forecast
        if self.use_mv:
            preds = [self.model.predict(context, seed=i)
                     for i in range(self.n_samples)]
            raw_pred = np.median(preds, axis=0)
        else:
            raw_pred = self.model.predict(context)

        # 2. Post-process
        pred = self.postprocess(raw_pred, context)

        # 3. Decide (agent layer)
        action = self.decision_agent(pred, context)

        return {"prediction": pred, "action": action}

    def postprocess(self, pred, history):
        mu, sigma = history.mean(), history.std()
        pred = pred.clip(mu - 3*sigma, mu + 3*sigma)
        return pred

    def decision_agent(self, pred, context):
        # Chain of Draft for action recommendation
        prompt = f"trend→{self.detect_trend(pred)}, recommendation→"
        return self.model.generate(prompt, max_tokens=50)
```

---

## Supplement: Implementation Code and Advanced Settings

---

## 1. Agent Core Architecture

**Agent = Model + Memory + Tools + Action capability**

```
Perception → Planning → Action → Observation
                ↑__________________________|
```

**Three planning modes:**

| Mode | Use case | Track 3 relevance |
|------|----------|------------------|
| ReAct (Reason + Act) | Tool calls, dynamic decisions | ✅ Primary mode |
| Chain-of-Thought | Pure reasoning, no tool calls | Auxiliary |
| Plan-then-Execute | Complex tasks with fixed steps | Gate checkpoint design |

---

## 2. ReAct Agent Prototype (Hackathon version)

```python
import anthropic

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "run_forecast",
        "description": "Run probabilistic time-series forecast. Returns mean and P10/P90 quantiles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "series":  {"type": "array", "items": {"type": "number"}, "description": "Historical values"},
                "horizon": {"type": "integer", "description": "Steps to forecast"}
            },
            "required": ["series", "horizon"]
        }
    },
    {
        "name": "detect_anomaly",
        "description": "Check if a forecast value is anomalous given historical distribution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "forecast": {"type": "number"},
                "history":  {"type": "array", "items": {"type": "number"}},
                "sigma_threshold": {"type": "number", "default": 3.0}
            },
            "required": ["forecast", "history"]
        }
    },
    {
        "name": "generate_alert",
        "description": "Generate a human-readable decision alert with severity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "forecast": {"type": "number"},
                "uncertainty": {"type": "number"},
                "context":  {"type": "string"}
            },
            "required": ["forecast", "uncertainty", "context"]
        }
    }
]

def dispatch_tool(name: str, args: dict) -> dict:
    """Tool dispatcher — connects to real implementations"""
    if name == "run_forecast":
        return run_chronos_forecast(**args)
    elif name == "detect_anomaly":
        return check_anomaly(**args)
    elif name == "generate_alert":
        return create_alert(**args)
    raise ValueError(f"Unknown tool: {name}")

def forecasting_agent(series: list, question: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": f"Time series: {series}\n\nQuestion: {question}"}]

    for _ in range(max_steps):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system="""You are a time-series forecasting agent.
Think step by step:
1. Analyse the series pattern and trend
2. Use run_forecast to get probabilistic predictions
3. Use detect_anomaly if the forecast looks unusual
4. Generate a clear decision recommendation

Always report uncertainty alongside point forecasts.""",
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if hasattr(b, "text"))

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = dispatch_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })
        messages.append({"role": "user", "content": tool_results})

    return "Max steps reached"
```

---

## 3. CLAUDE.md Template (core competition document)

```markdown
# CLAUDE.md — Zero One Hack Track 3

## Current Status
CURRENT_GATE: 1          # 0=understand problem / 1=EDA / 2=training / 3=inference / 4=submit
MODEL: chronos-t5-small  # currently selected model
BEST_CRPS: 0.xxx         # best validation score

## Task Description
Track 3 Sybilion: probabilistic forecasting + decision agent
- Input:  historical time series (context)
- Output: future H-step quantile forecasts (P10, P50, P90)
- Metric: CRPS (lower is better)

## Mandatory Rules (never violate under any circumstances)
- Time-series CV must use TimeSeriesSplit — random split is forbidden
- All experiments logged to experiments.csv
- Test set must not influence any decision or hyperparameter choice
- Every Gate requires human confirmation before proceeding

## Current To-Do (Gate 1)
- [ ] EDA: plot 10 representative series
- [ ] Check series length distribution and missing value rates
- [ ] Run Chronos zero-shot baseline
- [ ] Design TimeSeriesSplit (5 folds, gap=H)

## Forbidden Actions
- Must not modify test_submission.csv format
- Must not import external data prohibited by competition rules
- Must not skip Gate checkpoints and proceed to the next phase
```

---

## 4. Human-in-the-Loop Gate Design

```
Gate 0: understand problem  → human confirms: problem definition, metric, data format
    ↓ AI: EDA + data cleaning + CV design
Gate 1: confirm EDA         → human confirms: feature engineering direction, baseline score
    ↓ AI: LoRA fine-tuning + hyperparameter search
Gate 2: confirm training    → human confirms: best model, whether merging is needed
    ↓ AI: test inference + formatting
Gate 3: pre-submission      → human final review: format, score sanity check
```

```python
def gate_checkpoint(gate_num: int, summary: str, blocking: bool = True):
    """Insert a Gate in code to force human confirmation"""
    print(f"\n{'='*60}")
    print(f"  🚦 GATE {gate_num} CHECKPOINT")
    print(f"{'='*60}")
    print(summary)
    print(f"{'='*60}")
    if blocking:
        response = input("Confirm proceed? (y=continue / n=revise / q=terminate): ").strip().lower()
        if response == "q":
            raise SystemExit("Human terminated at gate checkpoint")
        return response == "y"
    return True
```

---

## 5. Chronos Inference Pipeline

```python
import torch
from chronos import ChronosPipeline
import numpy as np

pipeline = ChronosPipeline.from_pretrained(
    "./final_model",          # or "amazon/chronos-t5-small"
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

def forecast_with_uncertainty(series: np.ndarray, horizon: int, num_samples: int = 100):
    """
    Returns:
        mean:      (horizon,)
        quantiles: dict with keys like "p10", "p50", "p90"
    """
    context = torch.tensor(series, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        forecast = pipeline.predict(
            context=context,
            prediction_length=horizon,
            num_samples=num_samples,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        )
    samples = forecast[0].numpy()  # (num_samples, horizon)

    return {
        "mean":    samples.mean(axis=0),
        "std":     samples.std(axis=0),
        "p10":     np.percentile(samples, 10, axis=0),
        "p50":     np.percentile(samples, 50, axis=0),
        "p90":     np.percentile(samples, 90, axis=0),
        "samples": samples,
    }

# Self-Consistency (multiple runs averaged to reduce variance)
def robust_forecast(series, horizon, n_runs=3):
    all_samples = []
    for _ in range(n_runs):
        result = forecast_with_uncertainty(series, horizon, num_samples=50)
        all_samples.append(result["samples"])
    combined = np.vstack(all_samples)  # (n_runs*50, horizon)
    return {
        "mean": combined.mean(axis=0),
        "p10":  np.percentile(combined, 10, axis=0),
        "p90":  np.percentile(combined, 90, axis=0),
    }
```

---

## 6. Common Agent Failure Modes and Defences

| Failure mode | Symptom | Defence |
|-------------|---------|---------|
| Hallucination | Agent fabricates forecast values | Tool result validation + Gate human confirmation |
| Infinite loop | Repeats the same action | max_steps hard limit |
| Context overflow | Long conversation exceeds context | Compress memory (keep only last N turns) |
| Over-confidence | Uncertainty is ignored | Force interval output in system prompt |
| Tool misuse | Calls wrong tool | Precise tool descriptions + few-shot examples |

```python
def compress_memory(messages: list, keep_last_n: int = 6) -> list:
    """Keep system + last N turns; compress the rest into a summary"""
    if len(messages) <= keep_last_n:
        return messages
    summary = f"[Earlier conversation summary: {len(messages)-keep_last_n} turns compressed]"
    return [{"role": "user", "content": summary}] + messages[-keep_last_n:]
```
