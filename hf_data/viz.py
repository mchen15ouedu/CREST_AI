"""Plotly figure builders for the live dashboard.

hydrograph_fig  - simulated Q + observed Q (lines) + precip (bars, reversed right
                  axis), matching the AQUAH web viewer's hydrograph style.
q2d_fig         - 2-D streamflow raster (q.*.tif) as a heatmap over lon/lat.
"""
from __future__ import annotations

import os
import re
from datetime import datetime

import numpy as np
import plotly.graph_objects as go

_BG = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
           font=dict(color="#cdd9e2", size=12))


def hydrograph_fig(rows: list[dict], title: str = "Live hydrograph") -> go.Figure:
    """rows: list of {time, sim_q, obs_q, precip} (from hf_data.runner)."""
    x = [r["time"] for r in rows]
    sim = [r.get("sim_q") for r in rows]
    obs = [r.get("obs_q") for r in rows]
    pr = [r.get("precip") or 0.0 for r in rows]
    maxp = max([p for p in pr if p is not None] + [0.1])
    fig = go.Figure()
    fig.add_bar(x=x, y=pr, name="Precip", marker_color="#5b9bd5", opacity=0.7, yaxis="y2")
    fig.add_scatter(x=x, y=obs, name="Observed Q", mode="lines",
                    line=dict(color="#f4f4f4", width=1.6))
    fig.add_scatter(x=x, y=sim, name="Simulated Q", mode="lines",
                    line=dict(color="#4cc9a0", width=1.8))
    fig.update_layout(
        title=title, height=340, margin=dict(l=54, r=54, t=40, b=36),
        legend=dict(orientation="h", y=1.14, font=dict(size=10)), bargap=0,
        xaxis=dict(title="Time", gridcolor="rgba(255,255,255,.06)"),
        yaxis=dict(title="Discharge (m³/s)", rangemode="tozero", gridcolor="rgba(255,255,255,.06)"),
        yaxis2=dict(title="Precip (mm/h)", overlaying="y", side="right",
                    range=[maxp * 3.4, 0], showgrid=False),
        **_BG,
    )
    return fig


def _q_time_from_name(path: str) -> str:
    m = re.search(r"q\.(\d{8,12})\.", os.path.basename(path))
    if not m:
        return ""
    s = m.group(1)
    fmt = "%Y%m%d%H%M" if len(s) == 12 else ("%Y%m%d%H" if len(s) == 10 else "%Y%m%d")
    try:
        return datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return s


def q2d_fig(tif_path: str, title: str | None = None) -> go.Figure:
    import rasterio
    with rasterio.open(tif_path) as ds:
        a = ds.read(1).astype("float32")
        b = ds.bounds
        nod = ds.nodata
    if nod is not None:
        a = np.where(a == nod, np.nan, a)
    ny, nx = a.shape
    lon = np.linspace(b.left, b.right, nx)
    lat = np.linspace(b.top, b.bottom, ny)          # row 0 = north
    when = _q_time_from_name(tif_path)
    fig = go.Figure(go.Heatmap(
        z=a, x=lon, y=lat, colorscale="Blues", reversescale=False,
        colorbar=dict(title="Q (m³/s)"), hovertemplate="lon %{x:.3f}<br>lat %{y:.3f}<br>Q %{z:.1f}<extra></extra>"))
    fig.update_layout(
        title=title or (f"2-D streamflow · {when}" if when else "2-D streamflow"),
        height=360, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="lon", constrain="domain"),
        yaxis=dict(title="lat", scaleanchor="x",
                   scaleratio=1.0 / max(0.2, np.cos(np.radians((b.top + b.bottom) / 2)))),
        **_BG,
    )
    return fig


def q2d_png(tif_path: str, vmin: float = 1.0, vmax: float | None = None):
    """Render a streamflow raster the EF5 way: discharge on the stream network
    only, LOG color scale (yellow→orange→red), transparent off-network — for a
    Leaflet imageOverlay. Returns (png_bytes, bounds=[[S,W],[N,E]], vmax).
    """
    import io
    import rasterio
    from matplotlib import cm
    from matplotlib.colors import LogNorm
    from PIL import Image
    with rasterio.open(tif_path) as ds:
        a = ds.read(1).astype("float32")
        b = ds.bounds
        nod = ds.nodata
    a = np.where((a == nod) if nod is not None else ~np.isfinite(a), np.nan, a)
    show = np.isfinite(a) & (a >= vmin)              # only the flowing network
    if vmax is None or vmax <= vmin:
        top = float(np.nanmax(a)) if np.isfinite(np.nanmax(a)) else vmin * 10
        vmax = max(top, vmin * 10)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    clipped = np.clip(np.where(show, a, vmin), vmin, vmax)
    rgba = cm.get_cmap("YlOrRd")(norm(clipped))      # pale-yellow -> dark-red
    rgba[..., 3] = np.where(show, 0.92, 0.0)         # transparent off-network
    img = Image.fromarray((rgba * 255).astype("uint8"), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    bounds = [[b.bottom, b.left], [b.top, b.right]]
    return buf.getvalue(), bounds, vmax


def q2d_frames(tif_paths: list[str], vmin: float = 1.0):
    """Render a whole time series of q.*.tif to PNG frames with ONE fixed vmax
    (so colors are stable during animation). Returns (frames, vmax) where
    frames = [(png_bytes, bounds, time_label), ...] in path order."""
    import rasterio
    vmax = vmin * 10.0
    for p in tif_paths:                              # pass 1: global max
        with rasterio.open(p) as ds:
            a = ds.read(1)
            nod = ds.nodata
        a = a[a != nod] if nod is not None else a[np.isfinite(a)]
        if a.size:
            m = float(np.nanmax(a))
            if np.isfinite(m):
                vmax = max(vmax, m)
    frames = []
    for p in tif_paths:                              # pass 2: render each
        png, bounds, _ = q2d_png(p, vmin=vmin, vmax=vmax)
        frames.append((png, bounds, _q_time_from_name(p)))
    return frames, vmax


# --------------------------------------------------------------------------- #
# frame disk cache — lets a cache-served run still show the 2-D animation
# (frames rendered once per (gauge, model, window); invalidated on param change)
# --------------------------------------------------------------------------- #
_WIN_FMT = "%Y-%m-%d %H:%M"


def frames_cache_dir(gauge: str, model: str) -> str:
    from hf_data.statecache import CACHE_DIR
    return os.path.join(CACHE_DIR, "frames", f"{str(gauge).zfill(8)}_{model.lower()}")


def save_frames_cache(gauge: str, model: str, t0: datetime, t1: datetime,
                      frames: list, vmax: float):
    """Persist rendered animation frames so a later cache-hit run can replay
    the 2-D streamflow without re-running EF5."""
    import json
    import shutil
    d = frames_cache_dir(gauge, model)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    for i, (png, _b, _t) in enumerate(frames):
        with open(os.path.join(d, f"frame_{i:04d}.png"), "wb") as fh:
            fh.write(png)
    idx = {"window": [t0.strftime(_WIN_FMT), t1.strftime(_WIN_FMT)],
           "times": [f[2] for f in frames], "bounds": frames[0][1] if frames else None,
           "vmax": vmax, "n": len(frames)}
    with open(os.path.join(d, "index.json"), "w") as fh:
        json.dump(idx, fh)


def load_frames_cache(gauge: str, model: str, t0: datetime, t1: datetime):
    """(frames, vmax) for an exact-window match, else None. Frames are
    (png_bytes, bounds, time_label) like q2d_frames returns."""
    import json
    d = frames_cache_dir(gauge, model)
    try:
        with open(os.path.join(d, "index.json")) as fh:
            idx = json.load(fh)
        if idx.get("window") != [t0.strftime(_WIN_FMT), t1.strftime(_WIN_FMT)]:
            return None
        frames = []
        for i, label in enumerate(idx["times"]):
            with open(os.path.join(d, f"frame_{i:04d}.png"), "rb") as fh:
                frames.append((fh.read(), idx["bounds"], label))
        return frames, idx.get("vmax", 10.0)
    except Exception:
        return None


def has_frames_cache(gauge: str, model: str, t0: datetime, t1: datetime) -> bool:
    import json
    try:
        with open(os.path.join(frames_cache_dir(gauge, model), "index.json")) as fh:
            return json.load(fh).get("window") == [t0.strftime(_WIN_FMT), t1.strftime(_WIN_FMT)]
    except Exception:
        return False


def empty_fig(msg: str = "waiting…") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#8a98a5"))
    fig.update_layout(height=340, xaxis=dict(visible=False), yaxis=dict(visible=False), **_BG)
    return fig
