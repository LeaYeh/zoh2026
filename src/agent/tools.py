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
    price_history: list[float] | None = None,
    vol_window: int = 20,
) -> dict:
    """Return expected return, downside risk, CI width, and volatility regime.

    vol_regime "high"/"normal"/"low": rolling 20-day std vs 252-day baseline.
    In high-vol regime, confidence is halved to reduce position sizing.
    """
    expected_return = (np.mean(forecast_median) - current_price) / current_price
    downside = (np.min(forecast_q10) - current_price) / current_price
    ci_width = float(np.mean(np.array(forecast_q90) - np.array(forecast_q10)))

    vol_regime = "unknown"
    vol_current = None
    if price_history and len(price_history) >= vol_window + 1:
        prices = np.array(price_history, dtype=np.float64)
        log_rets = np.diff(np.log(np.maximum(prices, 1e-9)))
        vol_current = float(np.std(log_rets[-vol_window:]))
        if len(log_rets) >= 252:
            vol_baseline = float(np.std(log_rets[-252:]))
            ratio = vol_current / (vol_baseline + 1e-9)
            vol_regime = "high" if ratio > 1.5 else "low" if ratio < 0.7 else "normal"

    vol_penalty = {"high": 0.5, "normal": 1.0, "low": 1.0, "unknown": 0.8}
    raw_confidence = 1.0 - min(ci_width / (current_price + 1e-9), 1.0)
    confidence = float(np.clip(raw_confidence * vol_penalty[vol_regime], 0.0, 1.0))

    return {
        "expected_return": float(expected_return),
        "downside_risk":   float(downside),
        "ci_width":        ci_width,
        "vol_regime":      vol_regime,
        "vol_current":     vol_current,
        "confidence":      confidence,
    }


def decide_action(
    expected_return: float,
    downside_risk: float,
    ci_width: float,
    vol_regime: str = "unknown",
    threshold_return: float = 0.02,
    threshold_downside: float = -0.03,
) -> dict:
    """Rule-based fallback; LLM overrides with reasoning."""
    # Widen thresholds in high-vol regime to avoid over-trading
    multiplier = 1.5 if vol_regime == "high" else 1.0
    if (expected_return >= threshold_return * multiplier
            and downside_risk >= threshold_downside):
        action = "BUY"
    elif expected_return < 0 or downside_risk < threshold_downside * 2:
        action = "SELL"
    else:
        action = "HOLD"
    confidence = max(0.0, 1.0 - ci_width) * (0.5 if vol_regime == "high" else 1.0)
    return {"action": action, "confidence": float(np.clip(confidence, 0.0, 1.0))}


TOOL_REGISTRY = {
    "forecast": forecast_tool,
    "get_market_context": get_market_context,
    "calculate_risk": calculate_risk,
    "decide_action": decide_action,
}
