# Skill: Decision Agent — LangGraph Design

Trigger: "agent", "LangGraph", "decision loop", Gate 2 passed.

## Architecture (already scaffolded in src/agent/)

```
fetch_context → forecast → risk → llm_decision → END
                                ↘ fallback_decision (if LLM unavailable)
```

State: `src/agent/state.py` (TypedDict — all fields explicit, no dicts-of-dicts).
Nodes: `src/agent/nodes.py` (one pure function per node).
Graph: `src/agent/graph.py` (wiring only, no logic).
Tools: `src/agent/tools.py` (pure functions, registered in TOOL_REGISTRY).

## Wiring a new forecasting model into the agent

In `forecast_node`, replace the placeholder with the actual model:
```python
def forecast_node(state: AgentState) -> AgentState:
    ctx = state["market_context"]["prices"][-512:]  # last 512 steps
    preds = predict_series(model_tuple, ctx, pred_len=30, num_samples=20)
    state["forecast"] = {
        "median": preds.median(axis=0).tolist(),
        "q10": np.quantile(preds, 0.1, axis=0).tolist(),
        "q90": np.quantile(preds, 0.9, axis=0).tolist(),
    }
    return state
```

## Prompt design rules for llm_decision_node
1. Give Claude exactly: current price, forecast median, CI width, expected return, downside risk
2. Request structured JSON output — never parse free text
3. Include a "reasoning" field in the JSON so decisions are auditable
4. Keep prompt < 500 tokens (cost + latency on A100 matters)

```python
prompt = f"""
Symbol: {state['symbol']}  |  Current: {state['current_price']:.2f}
Forecast (next 30 days): median={median:.2f}, CI=[{q10:.2f}, {q90:.2f}]
Expected return: {expected_return:+.1%}  |  Downside: {downside:+.1%}

Decide: BUY, SELL, or HOLD. Output JSON only:
{{"action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reasoning": "..."}}
"""
```

## Testing without API key (fallback mode)
```bash
uv run python -c "
from src.agent.graph import run
result = run('AAPL', 182.5, llm_client=None)
print(result['decision'])
"
```
Fallback uses `decide_action()` in tools.py — rule-based, no LLM.

## Adding a new tool
1. Write pure function in `src/agent/tools.py`
2. Add to `TOOL_REGISTRY`
3. Add name to `configs/competition.yaml → agent.tools`
4. Call it from the appropriate node

## Agent evaluation
Run `run_backtest()` with `decision_fn = lambda symbol, price: run(symbol, price, llm_client)`.
Log each run to wandb with:
```python
wandb.log({
    "agent/action": result["decision"]["action"],
    "agent/confidence": result["decision"]["confidence"],
    "agent/iterations": result["iteration"],
})
```

## Gate 3 signal (Hour 24)
Run 1000+ backtests across different symbols and time windows.
Gate passes when Sharpe > 1.0 in ≥ 60% of scenarios.
Do NOT lock agent logic until Gate 3 approved.
