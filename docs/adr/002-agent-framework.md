# ADR 002 — Agent Framework: LangGraph + Claude

**Status:** Accepted  
**Date:** 2026-05-21

## Context

The agent layer needs to orchestrate: fetch market context → forecast → compute risk → decide BUY/SELL/HOLD. Options evaluated:

- LangGraph (graph-based state machine)
- Claude tool use via raw Anthropic SDK (no framework)
- LangGraph + Claude combined
- Custom agent loop from scratch

## Decision

**LangGraph for orchestration + Claude as the LLM backbone.**

## Rationale

- LangGraph is the most cited agent framework in AI Engineer JDs (2024–2025); naming it on the portfolio is high signal
- Graph-based state machine makes the decision flow inspectable and modifiable without touching node logic
- Claude tool calling provides structured JSON output from the LLM decision node, avoiding brittle string parsing
- Raw SDK alone lacks the state machine structure needed for multi-step reasoning
- Custom loop has higher engineering cost for a 36-hour hackathon
- Fallback mode (`llm_client=None` → rule-based node) allows full pipeline testing without API keys

## Consequences

- `src/agent/graph.py` owns LangGraph wiring; `src/agent/nodes.py` owns pure node functions
- Agent state is a typed `TypedDict` (`AgentState`) — all fields explicit
- Adding a new decision step = adding a node + an edge, not touching existing code
- `langgraph` and `anthropic` are production dependencies
