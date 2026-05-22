from functools import partial
from typing import Any

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    fetch_context_node,
    forecast_node,
    risk_node,
    llm_decision_node,
    fallback_decision_node,
)


def _should_use_llm(state: AgentState) -> str:
    return "llm" if state.get("error") is None else "fallback"


def build_graph(llm_client: Any = None) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("fetch_context", fetch_context_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("risk", risk_node)

    if llm_client:
        graph.add_node("decide", partial(llm_decision_node, llm_client=llm_client))
    else:
        graph.add_node("decide", fallback_decision_node)

    graph.set_entry_point("fetch_context")
    graph.add_edge("fetch_context", "forecast")
    graph.add_edge("forecast", "risk")
    graph.add_edge("risk", "decide")
    graph.add_edge("decide", END)

    return graph.compile()


def run(symbol: str, current_price: float, llm_client: Any = None) -> AgentState:
    app = build_graph(llm_client)
    initial_state: AgentState = {
        "symbol": symbol,
        "current_price": current_price,
        "market_context": {},
        "forecast": {},
        "risk": {},
        "decision": None,
        "error": None,
        "iteration": 0,
    }
    return app.invoke(initial_state)
