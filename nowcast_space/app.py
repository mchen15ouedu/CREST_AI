"""CREST_nowcast — Gradio app: DI-LSTM streamflow nowcasting backend.

API (used by the CREST_demo dashboard via Gradio's REST interface):
  /nowcast   JSON in  {gauge_id, lat, lon, area_km2, t0, precip[72], obs[[iso,q],...]}
             JSON out {ok, times[6], q[6], model:{epoch,val_nse,when}}
UI tabs: Status / Data prep (CPU, resumable) / Train (ZeroGPU bursts) / Try it.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta

import numpy as np
import gradio as gr

try:                                   # ZeroGPU decorator (present on Spaces)
    import spaces
    GPU = spaces.GPU
except Exception:                      # local dev fallback
    def GPU(*a, **k):
        def deco(fn):
            return fn
        return deco if not (a and callable(a[0])) else a[0]

from model import DILSTM, H, L, build_features, itq
import data as D
import train as T

import torch

_lock = threading.Lock()
_model: DILSTM | None = None
_ck: dict | None = None
_catalog = None                        # GAGES-II metadata frame
_dataset = None                        # cached training tensors

DEFAULT_GAUGES = "01011000, 08166200, 08167000, 08144500"
DEFAULT_MONTHS = "2023_01-2024_12"
VAL_MONTHS = ["2025_01", "2025_02", "2025_03", "2025_04", "2025_05", "2025_06"]


def _load_catalog():
    global _catalog
    if _catalog is None:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("vincewin/CREST_data", "gauges/gagesII_9322.parquet",
                            repo_type="dataset")
        df = pd.read_parquet(p)
        df["STAID"] = df["STAID"].astype(str).str.zfill(8)
        _catalog = df.set_index("STAID")
    return _catalog


def _gauge_meta(gid: str) -> dict | None:
    try:
        r = _load_catalog().loc[str(gid).zfill(8)]
        return {"id": str(gid).zfill(8), "lat": float(r["LAT_GAGE"]),
                "lon": float(r["LNG_GAGE"]), "area_km2": float(r["DRAIN_SQKM"])}
    except Exception:
        return None


def _months_range(spec: str) -> list[str]:
    """'2023_01-2024_12' -> ['2023_01', ..., '2024_12']."""
    a, b = [s.strip() for s in spec.split("-")]
    y0, m0 = map(int, a.split("_")); y1, m1 = map(int, b.split("_"))
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}_{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _ensure_model():
    global _model, _ck
    with _lock:
        if _model is None:
            _ck = T.load_ckpt()
            if _ck:
                _model = DILSTM()
                _model.load_state_dict(_ck["state_dict"])
                _model.eval()
    return _model, _ck


# ---- /nowcast API -------------------------------------------------------------
def nowcast(payload: str) -> str:
    try:
        req = json.loads(payload)
        model, ck = _ensure_model()
        if model is None:
            return json.dumps({"ok": False, "reason": "not_trained"})
        t0 = datetime.fromisoformat(req["t0"])
        precip = np.asarray(req["precip"], dtype="float64")[-L:]
        if len(precip) < L:
            precip = np.pad(precip, (L - len(precip), 0))
        # last-known-obs + age channels over the lookback window
        obs = sorted((datetime.fromisoformat(a), float(b)) for a, b in req.get("obs", []))
        obs_ff = np.zeros(L); age = np.full(L, 999.0)
        j = -1
        times = [t0 - timedelta(hours=L - 1 - i) for i in range(L)]
        for i, t in enumerate(times):
            while j + 1 < len(obs) and obs[j + 1][0] <= t:
                j += 1
            if j >= 0:
                obs_ff[i] = obs[j][1]
                age[i] = (t - obs[j][0]).total_seconds() / 3600.0
        feat = build_features(precip, obs_ff, age, float(req["area_km2"]), ck["stats"])
        with torch.no_grad():
            y = model(torch.from_numpy(feat[None]).float()).numpy()[0]
        q = np.maximum(itq(y), 0.0)
        return json.dumps({
            "ok": True,
            "times": [(t0 + timedelta(hours=i + 1)).strftime("%Y-%m-%d %H:%M") for i in range(H)],
            "q": [round(float(v), 3) for v in q],
            "model": {"epoch": ck.get("epoch"), "val_nse": ck.get("val_nse"),
                      "when": ck.get("when"), "experimental": True},
        })
    except Exception as e:
        return json.dumps({"ok": False, "reason": f"{type(e).__name__}: {e}"})


# ---- data prep (background thread on the Space; survives closed clients) ------
_prep_log: list[str] = []
_prep_thread: threading.Thread | None = None


def _prep_worker(gauges, months):
    _prep_log.append(f"prep started: {len(gauges)} gauges × {len(months)} months")
    for ym in months:
        y, m = map(int, ym.split("_"))
        try:
            rep = D.prep_month(gauges, y, m, log=_prep_log.append)
            _prep_log.append(f"{ym}: +{rep['obs_added']} obs, +{rep['mrms_added']} mrms "
                             f"({rep['skipped']} already present)")
        except Exception as e:
            _prep_log.append(f"{ym}: FAILED {type(e).__name__}: {e}")
    _prep_log.append("prep DONE")


def prep_start(gauge_ids: str, months_spec: str):
    """Kick the resumable background prep (idempotent while running)."""
    global _prep_thread
    if not os.environ.get("HF_TOKEN"):
        return "❌ HF_TOKEN secret not set yet — add it in Space settings first"
    if _prep_thread is not None and _prep_thread.is_alive():
        return "already running:\n" + "\n".join(_prep_log[-5:])
    gauges = [g for g in (_gauge_meta(s.strip()) for s in gauge_ids.split(",") if s.strip()) if g]
    if not gauges:
        return "no valid gauge ids (must be in the GAGES-II catalog)"
    months = _months_range(months_spec)
    _prep_log.clear()
    _prep_thread = threading.Thread(target=_prep_worker, args=(gauges, months), daemon=True)
    _prep_thread.start()
    return f"started: {[g['id'] for g in gauges]} × {len(months)} months (watch the log)"


def prep_log():
    running = _prep_thread is not None and _prep_thread.is_alive()
    return f"[{'RUNNING' if running else 'idle'}]\n" + "\n".join(_prep_log[-40:])


# ---- training (ZeroGPU bursts) -------------------------------------------------
@GPU(duration=110)                    # account max is lower than 240+queue margin
def _gpu_train(dataset, ck, seconds):
    # ZeroGPU pickles args into a worker process — closures can't cross, so
    # the burst logs to stdout (visible in the Space logs) only
    return T.train_burst(dataset, ck, seconds=seconds, device="cuda", log=print)


def do_train(gauge_ids: str, months_spec: str, seconds: float):
    global _dataset, _model, _ck
    logs = []

    def log(m):
        logs.append(str(m))

    try:
        gauges = [g for g in (_gauge_meta(s.strip()) for s in gauge_ids.split(",") if s.strip()) if g]
        months = _months_range(months_spec)
        if _dataset is None:
            log("building dataset from prepared months…")
            _dataset = T.build_dataset(gauges, months, VAL_MONTHS, log=log)
        if _dataset is None:
            return "no training data — run Data prep first.\n" + "\n".join(logs)
        log(f"train windows: {len(_dataset[0])}, val windows: {len(_dataset[2])}")
        ck = T.load_ckpt()
        ck = _gpu_train(_dataset, ck, min(float(seconds), 90.0))
        T.save_ckpt(ck)
        with _lock:
            _ck, _model = ck, DILSTM()
            _model.load_state_dict(ck["state_dict"])
            _model.eval()
        log(f"checkpoint saved: epoch {ck['epoch']}, val NSE {ck.get('val_nse')}")
        return "\n".join(logs[-60:])
    except Exception:
        import traceback
        return "TRAIN FAILED\n" + "\n".join(logs) + "\n" + traceback.format_exc()[-1200:]


@GPU(duration=30)
def gpu_check():
    if torch.cuda.is_available():
        return f"✅ ZeroGPU OK: {torch.cuda.get_device_name(0)}"
    return "❌ no CUDA device visible"


def status():
    _, ck = _ensure_model()
    if not ck:
        return "model: NOT TRAINED yet — /nowcast returns not_trained"
    return (f"model: epoch {ck['epoch']}, val NSE {ck.get('val_nse')}, "
            f"trained {ck.get('when')}, {ck.get('n_train')} windows, "
            f"lookback {ck.get('lookback')} h → horizon {ck.get('horizon')} h")


def reload_model():
    """Drop the cached model and refetch the checkpoint from the model repo —
    used after an external (HPC) training run uploads new weights."""
    global _model, _ck
    with _lock:
        _model, _ck = None, None
    # hf_hub_download resolves the latest commit on each call, so _ensure_model
    # (via status) fetches the new checkpoint revision automatically
    return "reloaded → " + status()


with gr.Blocks(title="CREST_nowcast") as demo:
    gr.Markdown("# 🔮 CREST_nowcast — DI-LSTM streamflow nowcasting backend\n"
                "Serves the AI nowcast tail for "
                "[CREST_demo](https://huggingface.co/spaces/vincewin/CREST_demo). "
                "Method: data integration (Feng, Fang & Shen 2020) at hourly "
                "timestep with MRMS radar precipitation. **Experimental.**")
    with gr.Tab("Status"):
        st = gr.Textbox(label="model status", value="press refresh")
        gr.Button("refresh").click(status, outputs=st, api_name="status")
        gr.Button("Reload model (after HPC training)").click(
            reload_model, outputs=st, api_name="reload")
        gpu_out = gr.Textbox(label="GPU check")
        gr.Button("ZeroGPU check").click(gpu_check, outputs=gpu_out, api_name="gpu_check")
    with gr.Tab("Data prep"):
        gids = gr.Textbox(label="gauge ids (GAGES-II)", value=DEFAULT_GAUGES)
        mons = gr.Textbox(label="months (YYYY_MM-YYYY_MM)", value=DEFAULT_MONTHS)
        prep_out = gr.Textbox(label="log", lines=16)
        gr.Button("Start background prep", variant="primary").click(
            prep_start, inputs=[gids, mons], outputs=prep_out, api_name="prep_start")
        gr.Button("Refresh log").click(prep_log, outputs=prep_out, api_name="prep_log")
    with gr.Tab("Train"):
        gids2 = gr.Textbox(label="gauge ids", value=DEFAULT_GAUGES)
        mons2 = gr.Textbox(label="train months", value=DEFAULT_MONTHS)
        secs = gr.Slider(30, 90, value=90, label="GPU seconds this burst")
        train_out = gr.Textbox(label="log", lines=16)
        gr.Button("Train burst (ZeroGPU)", variant="primary").click(
            do_train, inputs=[gids2, mons2, secs], outputs=train_out, api_name="train")
    with gr.Tab("API"):
        inp = gr.Textbox(label="payload JSON", lines=8)
        out = gr.Textbox(label="response", lines=8)
        gr.Button("nowcast").click(nowcast, inputs=inp, outputs=out, api_name="nowcast")

if __name__ == "__main__":            # Spaces runs `python app.py`
    demo.queue().launch()
