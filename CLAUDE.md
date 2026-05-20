# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Preparation and prototype work for **Zero One Hack · Vienna** (May 29–31, 2026) — a 36-hour supercompute hackathon with 64× NVIDIA A100s and a €10K+ prize pool. The expectation is a real fine-tuned model, not a prompt-engineering demo.

Three competition tracks (see `docs/zero-one-hack-tracks.html` for the full brief):

| Track | Partner | Domain | Core task |
|-------|---------|--------|-----------|
| 01 | Infineon Technologies | Manufacturing AI | Vision defect detection (ViT/EfficientNet + PatchCore + GradCAM) |
| 02 | UNIQA | Insurance AI | Tabular risk scoring (LightGBM ensemble + SHAP explainability) |
| 03 | Sybilion | Probabilistic Forecasting | Two-layer: fine-tuned PatchTST/Chronos → LLM decision agent |

Current team lean is **Track 03** (highest upside, best use of A100 compute, hardest to replicate).

## Expected stack

- **Python** with `uv` for package management
- ML: PyTorch, HuggingFace (transformers, time-series), LightGBM, anomalib
- Explainability: SHAP, GradCAM
- Agent layer: Claude API (tool calling) or similar LLM API
- Experiment tracking: likely MLflow or W&B

## Repo state

Currently pre-code — only planning docs exist. When prototype work starts, expect:
- `notebooks/` for EDA and model experiments
- `src/` for training scripts and inference pipeline
- `agent/` for the LLM decision layer (Track 03)
