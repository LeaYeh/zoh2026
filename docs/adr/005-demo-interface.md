# ADR 005 — Demo Interface: Gradio

**Status:** Accepted  
**Date:** 2026-05-21

## Context

Judging requires a live demo. Options evaluated:

- Streamlit
- Gradio
- FastAPI + React frontend
- Jupyter Notebook live run

The organizer requirement: "a small model, fine-tuned on real data, with a live demo beats a big model that only half-runs."

## Decision

**Gradio.**

## Rationale

- `gr.ChatInterface` is purpose-built for agent demos — step-by-step reasoning renders naturally as a chat stream
- Faster to build than FastAPI + frontend (hours, not days)
- HuggingFace ecosystem native — consistent with Chronos/PatchTST tooling
- Streamlit's `st.chat_message` works but Gradio's UX is more polished for model demos out of the box
- FastAPI remains available as an inference endpoint wrapper if needed; Gradio calls it

## Consequences

- `app/demo.py` is the single deliverable frontend
- Demo must start with `uv run python app/demo.py` — no additional setup steps
- Gradio is a production dependency, not dev-only
- Gate 4 (Hour 30): demo must be confirmed running before entering buffer time
