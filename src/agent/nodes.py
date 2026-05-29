from typing import Any
import numpy as np
from src.agent.state import AgentState
from src.agent.tools import calculate_risk, decide_action


def fetch_context_node(state: AgentState) -> AgentState:
    """Fetch recent price history for the symbol using yfinance."""
    import yfinance as yf

    symbol = state.get("symbol", "")
    if not symbol:
        state["market_context"] = {}
        state["error"] = "no symbol provided"
        return state

    try:
        df = yf.download(symbol, period="3y", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            state["market_context"] = {}
            state["error"] = f"yfinance returned no data for {symbol}"
            return state

        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        prices = close.dropna().tolist()

        state["market_context"] = {
            "prices":    prices,
            "pred_len":  30,
            "symbol":    symbol,
            "last_date": str(df.index[-1].date()),
        }
    except Exception as exc:
        state["market_context"] = {}
        state["error"] = f"fetch_context failed: {exc}"

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
    """Compute risk metrics and volatility regime from forecast + price history."""
    f = state["forecast"]
    risk = calculate_risk(
        forecast_median=f.get("median", []),
        forecast_q10=f.get("q10", []),
        forecast_q90=f.get("q90", []),
        current_price=state["current_price"],
        price_history=state["market_context"].get("prices"),
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
        vol_regime=r.get("vol_regime", "unknown"),
    )
    state["decision"] = result
    return state
