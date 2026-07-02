"""CREST_demo — agentic flash-flood dashboard (HF Space, Docker SDK).

A chat query drives the full pipeline (parse -> basin -> gauge -> data ->
calibrated params -> run) with a LIVE hydrograph + 2-D streamflow that stream
while CREST runs. Locally the run is mocked and parsing uses a deterministic
fallback; on the Space (OPENAI_API_KEY + the fork binary) both are real.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from hf_data.pipeline import analyze
from hf_data.viz import hydrograph_fig, q2d_fig, empty_fig

USE_MOCK = os.environ.get("CREST_DEMO_MOCK", "1") == "1"   # real ef5 on the Space

INTRO = ("Ask me to analyze a flood — e.g. *“flash flood near Allagash this July”*, "
         "*“Kerr County, TX July 2025”*, a `lat,lon`, or a USGS gauge id like `01011000`.")


def chat_run(message, history):
    history = (history or []) + [{"role": "user", "content": message},
                                 {"role": "assistant", "content": "…"}]
    hydro = empty_fig("hydrograph…")
    q2d = empty_fig("2-D streamflow…")
    rows, last_q, log = [], None, []
    yield history, hydro, q2d
    try:
        for kind, payload in analyze(message, use_mock=USE_MOCK, hours=48):
            if kind == "status":
                log.append(payload)
            elif kind == "hydro":
                rows += payload["rows"]
                hydro = hydrograph_fig(rows)
            elif kind == "q2d":
                last_q = payload["path"]
                q2d = q2d_fig(last_q)
            elif kind == "done":
                peak = max((r["sim_q"] for r in rows if r["sim_q"] is not None), default=0.0)
                log.append(f"✅ **Complete** — {len(rows)} timesteps · peak Q **{peak:.1f} m³/s**")
                if last_q:
                    q2d = q2d_fig(last_q)
                hydro = hydrograph_fig(rows, "Hydrograph — complete") if rows else hydro
            history[-1]["content"] = "\n\n".join(log) or "…"
            yield history, hydro, q2d
    except Exception as e:
        history[-1]["content"] = f"⚠️ {e}"
        yield history, hydro, q2d


with gr.Blocks(title="CREST_demo") as demo:
    gr.Markdown("## 🌊 CREST_demo — agentic flash-flood analysis\n"
                "Natural-language query → CREST simulation with a **live** hydrograph + 2-D streamflow.")
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=380, value=[{"role": "assistant", "content": INTRO}])
            msg = gr.Textbox(placeholder="Describe a flood event…", label="Query", autofocus=True)
            with gr.Row():
                send = gr.Button("Analyze", variant="primary")
                clear = gr.Button("Clear")
            gr.Examples(["flash flood near Allagash this July", "Kerr County, TX July 2025",
                         "Fort Cobb Oklahoma", "01011000"], inputs=msg)
        with gr.Column(scale=2):
            hydro_plot = gr.Plot(label="Hydrograph", value=empty_fig("ask a question to start"))
            q2d_plot = gr.Plot(label="2-D streamflow", value=empty_fig("ask a question to start"))

    outs = [chatbot, hydro_plot, q2d_plot]
    send.click(chat_run, [msg, chatbot], outs).then(lambda: "", None, msg)
    msg.submit(chat_run, [msg, chatbot], outs).then(lambda: "", None, msg)
    clear.click(lambda: ([{"role": "assistant", "content": INTRO}],
                         empty_fig("ask a question to start"), empty_fig("ask a question to start")),
                None, outs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
