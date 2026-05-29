from typing import Any
import numpy as np
from src.agent.state import AgentState
from src.agent.tools import calculate_risk, decide_action


def fetch_context_node(state: AgentState) -> AgentState:
    """Pull recent price data and attach to state."""
    # replace with bound get_market_context at runtime
    state["market_context"] = {}
    return state


def forecast_node(state: AgentState, pipeline: Any = None) -> AgentState:
    """Run Chronos pipeline and store median/q10/q90 forecast in state."""
    if pipeline is None:
        state["forecast"] = {}
        return state

    prices = state["market_context"].get("prices", [])
    if not prices:
        state["forecast"] = {}
        return state

    import torch
    ctx = torch.tensor(prices[-512:], dtype=torch.float32)
    pred_len = state["market_context"].get("pred_len", 30)
    samples = pipeline.predict(ctx, pred_len, num_samples=20)
    draws = samples[0].numpy()  # (n_samples, pred_len)

    state["forecast"] = {
        "median": np.median(draws, axis=0).tolist(),
        "q10": np.quantile(draws, 0.1, axis=0).tolist(),
        "q90": np.quantile(draws, 0.9, axis=0).tolist(),
    }
    return state


def risk_node(state: AgentState) -> AgentState:
    """Compute risk metrics from forecast."""
    f = state["forecast"]
    risk = calculate_risk(
        forecast_median=f.get("median", []),
        forecast_q10=f.get("q10", []),
        forecast_q90=f.get("q90", []),
        current_price=state["current_price"],
    )
    state["risk"] = risk
    return state


def llm_decision_node(state: AgentState, llm_client: Any) -> AgentState:
    """LLM synthesizes context + forecast + risk into a final decision."""
    prompt = (
        f"Symbol: {state['symbol']}\n"
        f"Current price: {state['current_price']}\n"
        f"Forecast: {state['forecast']}\n"
        f"Risk metrics: {state['risk']}\n"
        f"Market context: {state['market_context']}\n\n"
        "Based on this data, should we BUY, SELL, or HOLD? "
        "Explain your reasoning in 2-3 sentences, then output your decision as JSON: "
        '{"action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reasoning": "..."}'
    )
    response = llm_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    state["decision"] = response.content[0].text
    return state


def fallback_decision_node(state: AgentState) -> AgentState:
    """Rule-based fallback when LLM is unavailable."""
    r = state.get("risk", {})
    result = decide_action(
        expected_return=r.get("expected_return", 0),
        downside_risk=r.get("downside_risk", 0),
        ci_width=r.get("ci_width", 1),
    )
    state["decision"] = result
    return state
