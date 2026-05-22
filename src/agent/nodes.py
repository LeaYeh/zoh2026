from typing import Any
from src.agent.state import AgentState
from src.agent.tools import calculate_risk, decide_action


def fetch_context_node(state: AgentState) -> AgentState:
    """Pull recent price data and attach to state."""
    # replace with bound get_market_context at runtime
    state["market_context"] = {}
    return state


def forecast_node(state: AgentState) -> AgentState:
    """Run forecasting model and store predictions."""
    # replace with bound forecast_tool at runtime
    state["forecast"] = {}
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
