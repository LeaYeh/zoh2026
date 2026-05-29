import os
import json
import gradio as gr
import anthropic
from dotenv import load_dotenv

from src.data.loader import fetch_ticker
from src.agent.graph import run as run_agent
from src.utils.wandb_utils import load_best_pipeline

load_dotenv()

# Load best checkpoint from WandB on startup
_pipeline, _best_run, _best_crps = load_best_pipeline()
if _best_run:
    print(f"[demo] Loaded checkpoint: {_best_run}  (CRPS={_best_crps:.4f})")
else:
    print("[demo] No local checkpoint found — running zero-shot.")

_llm_client = None
if os.getenv("ANTHROPIC_API_KEY"):
    _llm_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def analyze_symbol(symbol: str, history: list) -> tuple[str, list]:
    symbol = symbol.strip().upper()
    if not symbol:
        return "Please enter a ticker symbol.", history

    try:
        df = fetch_ticker(symbol, start="2024-01-01", end="2025-01-01")
        current_price = float(df["close"].iloc[-1])
        result = run_agent(
            symbol=symbol,
            current_price=current_price,
            pipeline=_pipeline,
            llm_client=_llm_client,
        )
        decision = result.get("decision", "No decision produced.")
        if isinstance(decision, dict):
            decision = json.dumps(decision, indent=2)
    except Exception as e:
        decision = f"Error: {e}"

    history.append((symbol, decision))
    return decision, history


_ckpt_label = f"best run: **{_best_run}**  (CRPS={_best_crps:.4f})" if _best_run else "zero-shot (no checkpoint)"

with gr.Blocks(title="ZOH 2026 — Track 03 Trading Agent") as demo:
    gr.Markdown(f"## Trading Decision Agent\n{_ckpt_label}")

    with gr.Row():
        symbol_input = gr.Textbox(label="Ticker Symbol", placeholder="e.g. AAPL", scale=3)
        submit_btn = gr.Button("Analyze", variant="primary", scale=1)

    output_box = gr.Textbox(label="Agent Decision", lines=10)
    history_state = gr.State([])
    history_display = gr.Chatbot(label="Session History")

    submit_btn.click(
        fn=analyze_symbol,
        inputs=[symbol_input, history_state],
        outputs=[output_box, history_display],
    )


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
