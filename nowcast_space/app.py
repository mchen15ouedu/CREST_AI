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


# ---- data prep (CPU, month-at-a-time, resumable) ------------------------------
def prep(gauge_ids: str, months_spec: str, progress=gr.Progress()):
    logs = []

    def log(m):
        logs.append(str(m))

    gauges = [g for g in (_gauge_meta(s.strip()) for s in gauge_ids.split(",") if s.strip()) if g]
    if not gauges:
        return "no valid gauge ids (must be in the GAGES-II catalog)"
    months = _months_range(months_spec)
    log(f"{len(gauges)} gauges × {len(months)} months")
    for i, ym in enumerate(months):
        progress((i, len(months)), desc=ym)
        y, m = map(int, ym.split("_"))
        try:
            rep = D.prep_month(gauges, y, m, log=log)
            log(f"{ym}: +{rep['obs_added']} obs, +{rep['mrms_added']} mrms "
                f"({rep['skipped']} already present)")
        except Exception as e:
            log(f"{ym}: FAILED {type(e).__name__}: {e}")
    return "\n".join(logs[-60:])


# ---- training (ZeroGPU bursts) -------------------------------------------------
@GPU(duration=240)
def _gpu_train(dataset, ck, seconds, log):
    return T.train_burst(dataset, ck, seconds=seconds, device="cuda", log=log)


def do_train(gauge_ids: str, months_spec: str, seconds: float):
    global _dataset, _model, _ck
    logs = []

    def log(m):
        logs.append(str(m))

    gauges = [g for g in (_gauge_meta(s.strip()) for s in gauge_ids.split(",") if s.strip()) if g]
    months = _months_range(months_spec)
    if _dataset is None:
        log("building dataset from prepared months…")
        _dataset = T.build_dataset(gauges, months, VAL_MONTHS, log=log)
    if _dataset is None:
        return "no training data — run Data prep first.\n" + "\n".join(logs)
    log(f"train windows: {len(_dataset[0])}, val windows: {len(_dataset[2])}")
    ck = T.load_ckpt()
    ck = _gpu_train(_dataset, ck, float(seconds), log)
    T.save_ckpt(ck)
    with _lock:
        _ck, _model = ck, DILSTM()
        _model.load_state_dict(ck["state_dict"])
        _model.eval()
    log(f"checkpoint saved: epoch {ck['epoch']}, val NSE {ck.get('val_nse')}")
    return "\n".join(logs[-60:])


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


with gr.Blocks(title="CREST_nowcast") as demo:
    gr.Markdown("# 🔮 CREST_nowcast — DI-LSTM streamflow nowcasting backend\n"
                "Serves the AI nowcast tail for "
                "[CREST_demo](https://huggingface.co/spaces/vincewin/CREST_demo). "
                "Method: data integration (Feng, Fang & Shen 2020) at hourly "
                "timestep with MRMS radar precipitation. **Experimental.**")
    with gr.Tab("Status"):
        st = gr.Textbox(label="model status", value="press refresh")
        gr.Button("refresh").click(status, outputs=st, api_name="status")
        gpu_out = gr.Textbox(label="GPU check")
        gr.Button("ZeroGPU check").click(gpu_check, outputs=gpu_out, api_name="gpu_check")
    with gr.Tab("Data prep"):
        gids = gr.Textbox(label="gauge ids (GAGES-II)", value=DEFAULT_GAUGES)
        mons = gr.Textbox(label="months (YYYY_MM-YYYY_MM)", value=DEFAULT_MONTHS)
        prep_out = gr.Textbox(label="log", lines=16)
        gr.Button("Prepare training data", variant="primary").click(
            prep, inputs=[gids, mons], outputs=prep_out)
    with gr.Tab("Train"):
        gids2 = gr.Textbox(label="gauge ids", value=DEFAULT_GAUGES)
        mons2 = gr.Textbox(label="train months", value=DEFAULT_MONTHS)
        secs = gr.Slider(30, 220, value=180, label="GPU seconds this burst")
        train_out = gr.Textbox(label="log", lines=16)
        gr.Button("Train burst (ZeroGPU)", variant="primary").click(
            do_train, inputs=[gids2, mons2, secs], outputs=train_out)
    with gr.Tab("API"):
        inp = gr.Textbox(label="payload JSON", lines=8)
        out = gr.Textbox(label="response", lines=8)
        gr.Button("nowcast").click(nowcast, inputs=inp, outputs=out, api_name="nowcast")

if __name__ == "__main__":            # Spaces runs `python app.py`
    demo.queue().launch()
