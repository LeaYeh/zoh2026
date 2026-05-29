# Skill 04 — LangGraph Agent Design

Trigger: "agent", "LangGraph", "decision loop", "tool calling".

## Architecture

```
fetch_context → forecast → risk → llm_decision → END
                                ↘ fallback_decision (if LLM unavailable)
```

State lives in `src/agent/state.py` (TypedDict — all fields explicit).
Nodes live in `src/agent/nodes.py` (one function per node, pure + testable).
Graph wiring in `src/agent/graph.py`.

## Node design rules
- Each node receives `AgentState`, returns `AgentState`
- Nodes must not call each other — only the graph controls flow
- Any exception in a node → set `state["error"]`, return state (don't raise)
- Fallback node handles graceful degradation when LLM is down

## Claude tool calling protocol

Tools are defined in `src/agent/tools.py`.
When adding a new tool:
1. Define the function with type hints
2. Add to `TOOL_REGISTRY`
3. Add the tool name to `configs/competition.yaml → agent.tools`

Tool functions must be pure (no side effects) — side effects go in nodes.

## Prompt design (llm_decision_node)

Structure every LLM prompt as:
1. Context (symbol, price, market state)
2. Forecast (median, confidence interval)
3. Risk metrics (expected return, downside)
4. Instruction (BUY/SELL/HOLD + JSON output)

Always request structured JSON output — parse it, don't trust free text.

## Testing the agent without A100

Use `run_agent(symbol, price, llm_client=None)` — falls back to rule-based decision.
Lets you test the full pipeline without an LLM API key or compute.

## WandB logging

Log each agent run:
```python
wandb.log({
    "agent/action": result["decision"]["action"],
    "agent/confidence": result["decision"]["confidence"],
    "agent/iterations": result["iteration"],
})
```
