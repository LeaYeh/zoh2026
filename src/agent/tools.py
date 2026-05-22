import numpy as np
import pandas as pd


def forecast_tool(
    symbol: str,
    context_series: list[float],
    prediction_length: int,
    model: str = "chronos",
) -> dict:
    """Return price forecast for the next `prediction_length` steps."""
    # injected at runtime with the loaded pipeline
    raise NotImplementedError("Bind a forecasting pipeline before use.")


def get_market_context(symbol: str, window_days: int = 30) -> dict:
    """Return recent price stats and trend direction."""
    raise NotImplementedError("Bind a data loader before use.")


def calculate_risk(
    forecast_median: list[float],
    forecast_q10: list[float],
    forecast_q90: list[float],
    current_price: float,
) -> dict:
    """Return expected return, downside risk, and confidence interval width."""
    expected_return = (np.mean(forecast_median) - current_price) / current_price
    downside = (np.min(forecast_q10) - current_price) / current_price
    ci_width = np.mean(np.array(forecast_q90) - np.array(forecast_q10))
    return {
        "expected_return": float(expected_return),
        "downside_risk": float(downside),
        "ci_width": float(ci_width),
    }


def decide_action(
    expected_return: float,
    downside_risk: float,
    ci_width: float,
    threshold_return: float = 0.02,
    threshold_downside: float = -0.03,
) -> dict:
    """Rule-based fallback; LLM overrides with reasoning."""
    if expected_return >= threshold_return and downside_risk >= threshold_downside:
        action = "BUY"
    elif expected_return < 0 or downside_risk < threshold_downside * 2:
        action = "SELL"
    else:
        action = "HOLD"
    return {"action": action, "confidence": 1.0 - ci_width}


TOOL_REGISTRY = {
    "forecast": forecast_tool,
    "get_market_context": get_market_context,
    "calculate_risk": calculate_risk,
    "decide_action": decide_action,
}
