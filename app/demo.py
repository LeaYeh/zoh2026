"""Gradio demo — ZOH 2026 Track 03 Trading Agent.

Tab 1 (Trading Agent): enter a ticker → forecast chart + BUY/SELL/HOLD badge + risk metrics.
Tab 2 (Training Monitor): WandB run history table + one-click training launcher with live log.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import anthropic
import gradio as gr
from dotenv import load_dotenv

from src.agent.graph import run as run_agent
from src.utils.wandb_utils import load_best_pipeline

load_dotenv()

_REPO_ROOT = Path(__file__).parent.parent

# ── model loading ─────────────────────────────────────────────────────────────

_pipeline, _best_run, _best_crps = load_best_pipeline()
if _best_run:
    print(f"[demo] Loaded checkpoint: {_best_run}  (CRPS={_best_crps:.4f})")
else:
    print("[demo] No local checkpoint found — running zero-shot.")

_llm_client = None
if os.getenv("ANTHROPIC_API_KEY"):
    _llm_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_CONFIGS = [
    "chronos_zeroshot_v0.yaml",
    "chronos_lora_v1.yaml",
    "chronos_zeroshot_dummy.yaml",
    "chronos_lora_dummy.yaml",
    "chronos_zeroshot_electricity.yaml",
    "chronos_lora_electricity.yaml",
]

_QUICK_EXAMPLES = ["AAPL", "TSLA", "NVDA", "BTC-USD", "ETH-USD"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _fetch_current_price(symbol: str) -> float | None:
    import yfinance as yf
    df = yf.download(symbol, period="5d", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return None
    close = df["Close"]
    if hasattr(close, "squeeze"):
        close = close.squeeze()
    vals = close.dropna()
    return float(vals.iloc[-1]) if len(vals) else None


def _make_forecast_chart(
    prices: list[float],
    median: list[float],
    q10: list[float],
    q90: list[float],
    symbol: str,
) -> plt.Figure:
    hist = prices[-90:] if len(prices) > 90 else list(prices)
    n_hist = len(hist)

    # Connect forecast from the last historical point
    x_hist = list(range(n_hist))
    x_fore = list(range(n_hist - 1, n_hist - 1 + len(median)))
    fore_med = [hist[-1]] + list(median)[:-1]
    fore_q10 = [hist[-1]] + list(q10)[:-1]
    fore_q90 = [hist[-1]] + list(q90)[:-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    ax.plot(x_hist, hist, color="#22d3ee", linewidth=1.5, label="History")
    ax.plot(x_fore, fore_med, color="#34d399", linewidth=1.5, linestyle="--", label="Forecast (median)")
    ax.fill_between(x_fore, fore_q10, fore_q90, alpha=0.25, color="#34d399", label="80% CI")
    ax.axvline(x=n_hist - 1, color="#94a3b8", linestyle=":", linewidth=1, alpha=0.6)

    ax.set_title(f"{symbol} — 30-Day Price Forecast", color="white", fontsize=12, pad=10)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.set_xlabel("Days", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Price", color="#94a3b8", fontsize=9)
    for key, spine in ax.spines.items():
        if key in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color("#1e293b")
    leg = ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=8, framealpha=0.8)
    leg.get_frame().set_edgecolor("#1e293b")

    plt.tight_layout()
    return fig


def _decision_badge(action: str, confidence: float | None = None) -> str:
    palette = {
        "BUY":  ("#22c55e", "rgba(34,197,94,0.12)"),
        "SELL": ("#ef4444", "rgba(239,68,68,0.12)"),
        "HOLD": ("#eab308", "rgba(234,179,8,0.12)"),
    }
    color, bg = palette.get(action, ("#94a3b8", "rgba(148,163,184,0.08)"))
    conf_str = f"<p style='color:#94a3b8;margin:6px 0 0;font-size:0.9rem'>{confidence:.0%} confidence</p>" if confidence is not None else ""
    return (
        f'<div style="text-align:center;padding:24px 0;border:2px solid {color};'
        f'border-radius:12px;background:{bg};margin:8px 0">'
        f'<span style="font-size:3rem;font-weight:700;color:{color}">{action}</span>'
        f'{conf_str}</div>'
    )


# ── tab 1: trading agent ──────────────────────────────────────────────────────

def analyze_symbol(symbol: str):
    symbol = symbol.strip().upper()
    if not symbol:
        return (
            '<div style="color:#94a3b8;text-align:center;padding:30px">Enter a ticker symbol above</div>',
            None,
            "",
            "",
        )

    current_price = _fetch_current_price(symbol)
    if current_price is None:
        err = f'<div style="color:#ef4444;text-align:center;padding:20px">Could not fetch price for <b>{symbol}</b></div>'
        return err, None, f"yfinance returned no data for {symbol}", ""

    try:
        result = run_agent(
            symbol=symbol,
            current_price=current_price,
            pipeline=_pipeline,
            llm_client=_llm_client,
        )
    except Exception as exc:
        err = f'<div style="color:#ef4444;padding:16px">Error running agent: {exc}</div>'
        return err, None, str(exc), ""

    # --- decision badge ---
    raw_decision = result.get("decision", {})
    if isinstance(raw_decision, str):
        m = re.search(r'\{.*\}', raw_decision, re.DOTALL)
        decision_dict = json.loads(m.group()) if m else {}
        reasoning_text = raw_decision
    else:
        decision_dict = raw_decision or {}
        reasoning_text = json.dumps(decision_dict, indent=2)

    action = decision_dict.get("action", "HOLD")
    confidence = decision_dict.get("confidence")
    reasoning = decision_dict.get("reasoning", "")
    badge = _decision_badge(action, confidence)
    if reasoning:
        badge += f'<p style="color:#cbd5e1;padding:0 16px;font-size:0.9rem">{reasoning}</p>'

    # --- forecast chart ---
    forecast = result.get("forecast", {})
    prices = result.get("market_context", {}).get("prices", [])
    chart = None
    if prices and forecast.get("median"):
        chart = _make_forecast_chart(
            prices,
            forecast["median"],
            forecast.get("q10", forecast["median"]),
            forecast.get("q90", forecast["median"]),
            symbol,
        )

    # --- risk summary ---
    risk = result.get("risk", {})
    risk_text = (
        f"Expected return : {risk.get('expected_return', 0): .2%}\n"
        f"Downside risk   : {risk.get('downside_risk', 0): .2%}\n"
        f"CI width        : {risk.get('ci_width', 0):.4f}\n"
        f"Vol regime      : {risk.get('vol_regime', 'unknown')}\n"
        f"Confidence      : {risk.get('confidence', 0):.2%}"
    )

    return badge, chart, reasoning_text, risk_text


# ── tab 2: training monitor ───────────────────────────────────────────────────

def get_run_history() -> list[list]:
    try:
        import wandb
        api = wandb.Api()
        project = os.getenv("WANDB_PROJECT", "zoh2026")
        runs = list(api.runs(project, filters={"state": "finished"}))[:15]
        rows = []
        for r in runs:
            crps = r.summary.get("eval/crps")
            sharpe = r.summary.get("eval/sharpe")
            rows.append([
                r.name or r.id,
                f"{crps:.4f}" if isinstance(crps, float) else "—",
                f"{sharpe:.3f}" if isinstance(sharpe, float) else "—",
                str(r.config.get("training", {}).get("steps", "—")),
                r.state,
            ])
        return rows or [["No finished runs", "", "", "", ""]]
    except Exception as exc:
        return [[f"WandB error: {exc}", "", "", "", ""]]


def launch_training(config_name: str):
    """Generator: streams training stdout line-by-line to the log textbox."""
    cmd = ["uv", "run", "python", "src/train.py", f"configs/exp/{config_name}"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
        cwd=str(_REPO_ROOT),
    )
    output = f"$ {' '.join(cmd)}\n\n"
    yield output
    for line in proc.stdout:
        output += line
        yield output
    proc.wait()
    yield output + f"\n\n[Process exited with code {proc.returncode}]"


# ── UI layout ─────────────────────────────────────────────────────────────────

_ckpt_label = (
    f"LoRA checkpoint: <b>{_best_run}</b> &nbsp;·&nbsp; CRPS = {_best_crps:.4f}"
    if _best_run
    else "zero-shot (no local checkpoint)"
)

with gr.Blocks(title="ZOH 2026 — Track 03 Trading Agent") as demo:

    gr.HTML(
        '<div style="padding:16px 0 8px">'
        '<h1 style="margin:0;font-size:1.6rem;font-weight:700">ZOH 2026 · Track 03 — Trading Agent</h1>'
        f'<p style="color:#94a3b8;margin:6px 0 0">Model: {_ckpt_label}</p>'
        '</div>'
    )

    with gr.Tabs():

        # ── Tab 1 ──
        with gr.Tab("Trading Agent"):
            with gr.Row():
                symbol_input = gr.Textbox(
                    label="Ticker Symbol",
                    placeholder="e.g. AAPL  /  BTC-USD",
                    scale=5,
                )
                analyze_btn = gr.Button("Analyze", variant="primary", scale=1)

            gr.Examples(
                examples=_QUICK_EXAMPLES,
                inputs=[symbol_input],
                label="Quick examples",
            )

            decision_html = gr.HTML(
                '<div style="color:#94a3b8;text-align:center;padding:40px">'
                'Enter a ticker above and click Analyze</div>'
            )
            forecast_chart = gr.Plot(label="Forecast")

            with gr.Row():
                reasoning_box = gr.Textbox(
                    label="Agent Reasoning / Decision", lines=8, scale=3
                )
                risk_box = gr.Textbox(label="Risk Metrics", lines=8, scale=2)

            _outs = [decision_html, forecast_chart, reasoning_box, risk_box]
            analyze_btn.click(fn=analyze_symbol, inputs=[symbol_input], outputs=_outs)
            symbol_input.submit(fn=analyze_symbol, inputs=[symbol_input], outputs=_outs)

        # ── Tab 2 ──
        with gr.Tab("Training Monitor"):
            with gr.Row():
                refresh_btn = gr.Button("Refresh WandB History", variant="secondary", scale=2)
                config_dropdown = gr.Dropdown(
                    choices=_CONFIGS,
                    value=_CONFIGS[0],
                    label="Config",
                    scale=3,
                )
                train_btn = gr.Button("Start Training", variant="primary", scale=1)

            run_table = gr.Dataframe(
                headers=["Run Name", "CRPS ↓", "Sharpe ↑", "Steps", "State"],
                label="WandB Run History (last 15 finished runs)",
                interactive=False,
                wrap=True,
            )

            train_log = gr.Textbox(
                label="Training Log (live)",
                lines=22,
                max_lines=50,
                interactive=False,
            )

            refresh_btn.click(fn=get_run_history, inputs=[], outputs=[run_table])
            train_btn.click(fn=launch_training, inputs=[config_dropdown], outputs=[train_log])


if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=False,
        theme=gr.themes.Base(primary_hue="cyan", secondary_hue="emerald"),
    )
