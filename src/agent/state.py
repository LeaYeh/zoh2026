from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    symbol: str
    current_price: float
    market_context: dict[str, Any]
    forecast: dict[str, Any]
    risk: dict[str, Any]
    decision: Optional[str]
    error: Optional[str]
    iteration: int
