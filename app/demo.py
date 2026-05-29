"""Gradio demo — ZOH 2026 Track 1: Industrial Process Sequence Modeling.

Tab 1 (Process Agent): enter a partial process sequence →
  - Top-5 next-step predictions with probability bar chart
  - Full sequence completion
  - Anomaly detection verdict

Tab 2 (Training Monitor): WandB run history + one-click training launcher
  with live log streaming.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv

from src.utils.wandb_utils import load_best_model

load_dotenv()

_REPO_ROOT = Path(__file__).parent.parent

# ── model loading ─────────────────────────────────────────────────────────────

_model, _tokenizer, _best_run, _best_top1 = load_best_model()
if _best_run and _model is not None:
    print(f"[demo] Loaded model: {_best_run}  (top1={_best_top1:.4f})")
else:
    print("[demo] No trained model found — train first via Tab 2.")

_CONFIGS = [
    "gpt2_dummy.yaml",
    "gpt2_finetune_v1.yaml",
]

_QUICK_EXAMPLES = [
    "RECEIVE WAFER LOT\nLOT IDENTIFICATION\nINITIAL WAFER INSPECTION",
    "RECEIVE WAFER LOT\nLOT IDENTIFICATION\nINITIAL WAFER INSPECTION\nMEASURE INITIAL THICKNESS\nTHERMAL OXIDATION",
    "RECEIVE WAFER LOT\nLOT IDENTIFICATION",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_steps(text: str) -> list[str]:
    """Parse newline- or comma-separated step names."""
    sep = "\n" if "\n" in text else ","
    return [s.strip() for s in text.split(sep) if s.strip()]


def _bar_chart(predictions: list[tuple[str, float]], title: str) -> plt.Figure:
    steps = [p for p, _ in predictions]
    probs = [v for _, v in predictions]

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(steps) * 0.5)))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    bars = ax.barh(steps[::-1], probs[::-1], color="#22d3ee", edgecolor="none")
    for bar, prob in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", color="white", fontsize=9)

    ax.set_xlim(0, max(probs) * 1.25 + 0.01)
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Probability", color="#94a3b8", fontsize=9)

    plt.tight_layout()
    return fig


def _no_model_html() -> str:
    return (
        '<div style="color:#eab308;border:1px solid #eab308;border-radius:8px;'
        'padding:16px;margin:8px 0">'
        '<b>No trained model loaded.</b><br>'
        'Go to <b>Training Monitor</b> tab → select <code>gpt2_dummy.yaml</code>'
        ' → Start Training, then refresh the page.</div>'
    )


# ── tab 1 handlers ────────────────────────────────────────────────────────────

def predict_next(sequence_text: str):
    if _model is None or _tokenizer is None:
        return _no_model_html(), None, ""
    steps = _parse_steps(sequence_text)
    if not steps:
        return '<div style="color:#94a3b8;padding:12px">Enter at least one step.</div>', None, ""
    from src.models.process_lm import predict_next_step
    preds = predict_next_step(_model, _tokenizer, steps, top_k=5)
    chart = _bar_chart(preds, f"Top-5 Next Steps  (after {len(steps)} steps)")
    top_step = preds[0][0] if preds else "—"
    badge = (
        f'<div style="border:2px solid #22d3ee;border-radius:8px;padding:14px;'
        f'background:rgba(34,211,238,0.08);text-align:center">'
        f'<span style="color:#94a3b8;font-size:0.8rem">Top-1 prediction</span><br>'
        f'<span style="color:#22d3ee;font-size:1.6rem;font-weight:700">{top_step}</span>'
        f'</div>'
    )
    return badge, chart, ""


def complete_seq(sequence_text: str):
    if _model is None or _tokenizer is None:
        return _no_model_html(), None, ""
    steps = _parse_steps(sequence_text)
    if not steps:
        return '<div style="color:#94a3b8;padding:12px">Enter at least one step.</div>', None, ""
    from src.models.process_lm import complete_sequence
    completed = complete_sequence(_model, _tokenizer, steps, max_new_steps=80)
    full_seq  = steps + completed
    numbered  = "\n".join(f"{i+1:>3}. {s}" for i, s in enumerate(full_seq))
    sep_line  = f"{'─'*4} partial ({len(steps)} steps) above · generated ({len(completed)} steps) below {'─'*4}"
    display   = "\n".join(f"{i+1:>3}. {s}" for i, s in enumerate(steps))
    display  += f"\n{sep_line}\n"
    display  += "\n".join(f"{len(steps)+i+1:>3}. {s}" for i, s in enumerate(completed))
    badge = (
        f'<div style="border:2px solid #34d399;border-radius:8px;padding:14px;'
        f'background:rgba(52,211,153,0.08)">'
        f'<span style="color:#34d399;font-weight:700">Completed</span> '
        f'<span style="color:#94a3b8">— {len(completed)} steps generated  '
        f'({len(full_seq)} total)</span></div>'
    )
    return badge, None, display


def check_anomaly(sequence_text: str):
    if _model is None or _tokenizer is None:
        return _no_model_html(), None, ""
    steps = _parse_steps(sequence_text)
    if len(steps) < 2:
        return '<div style="color:#94a3b8;padding:12px">Enter at least 2 steps.</div>', None, ""
    from src.models.process_lm import anomaly_score
    score = anomaly_score(_model, _tokenizer, steps)
    # heuristic threshold: perplexity > 50 is suspicious (tune after training)
    is_anomaly = score > 50.0
    color  = "#ef4444" if is_anomaly else "#22c55e"
    label  = "ANOMALY DETECTED" if is_anomaly else "VALID SEQUENCE"
    badge  = (
        f'<div style="border:2px solid {color};border-radius:8px;padding:16px;'
        f'background:rgba(0,0,0,0.2);text-align:center">'
        f'<span style="color:{color};font-size:1.5rem;font-weight:700">{label}</span><br>'
        f'<span style="color:#94a3b8;font-size:0.9rem">perplexity = {score:.1f}</span>'
        f'</div>'
    )
    # bar chart showing single score vs threshold
    fig, ax = plt.subplots(figsize=(6, 1.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.barh(["perplexity"], [min(score, 200)], color=color, edgecolor="none")
    ax.axvline(x=50, color="#eab308", linestyle="--", linewidth=1.2, label="threshold=50")
    ax.set_xlim(0, 210)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=8)
    plt.tight_layout()
    return badge, fig, ""


# ── tab 2 handlers ────────────────────────────────────────────────────────────

def get_run_history() -> list[list]:
    try:
        import wandb
        api = wandb.Api()
        project = os.getenv("WANDB_PROJECT", "zoh2026")
        runs = list(api.runs(project, filters={"state": "finished"}))[:15]
        rows = []
        for r in runs:
            top1   = r.summary.get("eval/best_top1") or r.summary.get("eval/top1")
            top3   = r.summary.get("eval/top3")
            steps  = r.config.get("training", {}).get("steps", "—")
            rows.append([
                r.name or r.id,
                f"{top1:.4f}" if isinstance(top1, float) else "—",
                f"{top3:.4f}" if isinstance(top3, float) else "—",
                str(steps),
                r.state,
            ])
        return rows or [["No finished runs", "", "", "", ""]]
    except Exception as exc:
        return [[f"WandB error: {exc}", "", "", "", ""]]


def launch_training(config_name: str):
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
    yield output + f"\n\n[Exited with code {proc.returncode}]"


# ── UI ────────────────────────────────────────────────────────────────────────

_ckpt_label = (
    f"model: <b>{_best_run}</b> &nbsp;·&nbsp; Top-1 = {_best_top1:.4f}"
    if (_best_run and _best_top1 is not None)
    else "no model loaded — train first"
)

with gr.Blocks(title="ZOH 2026 — Track 1: Industrial AI") as demo:

    gr.HTML(
        '<div style="padding:14px 0 6px">'
        '<h1 style="margin:0;font-size:1.5rem;font-weight:700">'
        'ZOH 2026 · Track #1 — Industrial Process Sequence Modeling</h1>'
        f'<p style="color:#94a3b8;margin:6px 0 0">{_ckpt_label}</p>'
        '</div>'
    )

    with gr.Tabs():

        # ── Tab 1: Process Agent ──────────────────────────────────────────────
        with gr.Tab("Process Agent"):
            gr.Markdown(
                "Enter a **partial process sequence** (one step per line). "
                "Then choose an action below."
            )
            seq_input = gr.Textbox(
                label="Partial Sequence",
                placeholder="RECEIVE WAFER LOT\nLOT IDENTIFICATION\n...",
                lines=6,
            )
            gr.Examples(
                examples=_QUICK_EXAMPLES,
                inputs=[seq_input],
                label="Quick examples",
            )

            with gr.Row():
                btn_predict  = gr.Button("Predict Next Step",   variant="primary")
                btn_complete = gr.Button("Complete Sequence",   variant="secondary")
                btn_anomaly  = gr.Button("Check Anomaly",       variant="secondary")

            result_badge = gr.HTML(
                '<div style="color:#94a3b8;text-align:center;padding:24px">'
                'Enter a sequence above and choose an action</div>'
            )
            result_chart = gr.Plot(label="")
            result_text  = gr.Textbox(label="Full Sequence", lines=14, interactive=False)

            _outs = [result_badge, result_chart, result_text]
            btn_predict.click( fn=predict_next,  inputs=[seq_input], outputs=_outs)
            btn_complete.click(fn=complete_seq,  inputs=[seq_input], outputs=_outs)
            btn_anomaly.click( fn=check_anomaly, inputs=[seq_input], outputs=_outs)

        # ── Tab 2: Training Monitor ───────────────────────────────────────────
        with gr.Tab("Training Monitor"):
            with gr.Row():
                refresh_btn = gr.Button("Refresh WandB History", variant="secondary", scale=2)
                config_drop = gr.Dropdown(
                    choices=_CONFIGS,
                    value=_CONFIGS[0],
                    label="Config",
                    scale=3,
                )
                train_btn = gr.Button("Start Training", variant="primary", scale=1)

            run_table = gr.Dataframe(
                headers=["Run Name", "Top-1 ↑", "Top-3 ↑", "Steps", "State"],
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
            train_btn.click(fn=launch_training, inputs=[config_drop], outputs=[train_log])


if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=False,
        theme=gr.themes.Base(primary_hue="cyan", secondary_hue="emerald"),
    )
