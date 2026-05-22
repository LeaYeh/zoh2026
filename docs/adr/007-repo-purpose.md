# ADR 007 — Repo Purpose: Hackathon-Specific + AI Engineer Portfolio

**Status:** Accepted  
**Date:** 2026-05-21

## Context

The repo was initially conceived as a general-purpose AI competition template (clone-per-competition model). The scope was reassessed given:

- The track is now confirmed (Track 03), not TBD
- The team will use the repo collaboratively during a 36-hour hackathon
- The repo is public and will be referenced in job applications

## Decision

**This repo is dedicated to ZOH 2026 Track 03 and doubles as an AI Engineer portfolio piece.**

It is not a general template. There is no "clone this for other competitions" workflow.

## Rationale

- A general template adds abstraction layers that slow down hackathon execution
- A single-competition repo is cleaner for GitHub portfolio visibility
- The tech stack (LangGraph, Chronos, PatchTST, Gradio) matches AI Engineer JD requirements for 2025–2026
- Hiring managers read repos: code quality, commit hygiene, and clear README matter

## Consequences

- `main` branch is the competition branch — no separate `template` branch
- `learning/` directory is preserved as practice history but is not part of the deliverable
- README (to be written) must explain the project as a portfolio piece, not a template
- Code quality standard is "production-readable": clear module interfaces, typed state, no dead code
