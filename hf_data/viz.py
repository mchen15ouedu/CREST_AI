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


# --------------------------------------------------------------------------- #
# 2-D streamflow rendering — BASEFLOW-RELATIVE color scale:
#   grey  ≈ nearly dry (well below baseflow)
#   green ≈ normal baseflow
#   yellow → orange → red = flooding (multiples of baseflow)
# The per-cell baseline is the window minimum of the simulated grids, anchored
# to the USGS observed baseflow at the outlet cell when observations exist.
# --------------------------------------------------------------------------- #
NET_MIN = float(os.environ.get("CREST_Q2D_MIN", "0.1"))   # m³/s — a cell that ever exceeds this is drawn
_BASE_FLOOR = NET_MIN / 5.0                               # avoid absurd ratios on trickle cells
_R_LO, _R_HI = -0.5, 1.7                                  # color scale spans log10(q/baseflow)
_RAMP = [(0.00, "#59626c"), (0.14, "#7e8a95"),            # r≈0.3–0.6  dry — grey
         (0.23, "#3fae5a"), (0.36, "#57c258"),            # r≈1–2      baseflow — green
         (0.48, "#c9dd45"), (0.57, "#ffd23f"),            # r≈3        rising — yellow
         (0.70, "#fd8d3c"), (0.84, "#f03b20"),            # r≈10–20    flood — orange/red
         (1.00, "#bd0026")]                               # r≥50       extreme — dark red
_CMAP = None


def _q_cmap():
    global _CMAP
    if _CMAP is None:
        from matplotlib.colors import LinearSegmentedColormap
        _CMAP = LinearSegmentedColormap.from_list("qflow", _RAMP)
    return _CMAP


def _read_q(tif_path: str):
    import rasterio
    with rasterio.open(tif_path) as ds:
        a = ds.read(1).astype("float32")
        b = ds.bounds
        nod = ds.nodata
    a = np.where((a == nod) if nod is not None else ~np.isfinite(a), np.nan, a)
    return a, [[b.bottom, b.left], [b.top, b.right]]


def _render_ratio(a, base, mask) -> bytes:
    """RGBA PNG of q/baseflow: grey→green→yellow→red, transparent off-network."""
    import io
    from PIL import Image
    r = a / np.maximum(base, _BASE_FLOOR)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (np.log10(np.clip(r, 10 ** _R_LO, 10 ** _R_HI)) - _R_LO) / (_R_HI - _R_LO)
    draw = mask & np.isfinite(x)
    rgba = _q_cmap()(np.where(draw, x, 0.0))
    rgba[..., 3] = np.where(draw, 0.9, 0.0)
    img = Image.fromarray((rgba * 255).astype("uint8"), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def q2d_png(tif_path: str, base=None, mask=None):
    """Render one streamflow raster for a Leaflet imageOverlay.
    base = per-cell baseflow grid (None → this frame is treated as baseflow,
    i.e. everything plots green). Returns (png_bytes, bounds, peak_q)."""
    a, bounds = _read_q(tif_path)
    valid = np.isfinite(a)
    if mask is None:
        mask = valid & (a >= NET_MIN)
    if base is None:
        base = np.where(valid, np.maximum(a, _BASE_FLOOR), np.nan)
    png = _render_ratio(a, base, mask & valid)
    peak = float(np.nanmax(a)) if valid.any() else 0.0
    return png, bounds, peak


def q2d_live(tif_path: str, prev_min=None):
    """Incremental renderer for frames arriving DURING a run: the baseline is
    the running per-cell minimum seen so far (the first frames read as normal
    flow, the flood then heats up against them).
    Returns (png_bytes, bounds, updated_min)."""
    a, bounds = _read_q(tif_path)
    cur_min = a if prev_min is None or prev_min.shape != a.shape else np.fmin(prev_min, a)
    mask = np.isfinite(a) & (np.fmax(np.nan_to_num(a, nan=0.0), 0.0) >= NET_MIN)
    base = np.where(np.isfinite(cur_min), np.maximum(cur_min, _BASE_FLOOR), np.nan)
    png = _render_ratio(a, base, mask)
    return png, bounds, cur_min


def obs_baseflow(rows: list[dict]) -> float | None:
    """Baseflow from the USGS observed discharge in the hydrograph rows:
    the 25th percentile of the observations (a standard low-flow proxy)."""
    obs = sorted(v for v in (r.get("obs_q") for r in rows)
                 if isinstance(v, (int, float)) and np.isfinite(v) and v > 0)
    if len(obs) < 8:
        return None
    return float(obs[int(0.25 * (len(obs) - 1))])


def q2d_frames(tif_paths: list[str], baseflow_cms: float | None = None):
    """Render a whole time series of q.*.tif with ONE fixed baseline (stable
    colors while scrubbing). Per-cell baseline = window minimum, scaled so the
    outlet cell's baseline matches the USGS observed baseflow when given.
    Returns (frames, peak_q) with frames = [(png_bytes, bounds, time_label), ...]."""
    cellmin = cellmax = None
    for p in tif_paths:                              # pass 1: per-cell min/max
        a, _ = _read_q(p)
        if cellmin is None or cellmin.shape != a.shape:
            cellmin, cellmax = a.copy(), a.copy()
        else:
            cellmin = np.fmin(cellmin, a)
            cellmax = np.fmax(cellmax, a)
    if cellmin is None:
        return [], 0.0
    mask = np.isfinite(cellmax) & (cellmax >= NET_MIN)
    base = np.where(np.isfinite(cellmin), np.maximum(cellmin, _BASE_FLOOR), np.nan)
    if baseflow_cms and mask.any():                  # anchor to the gauge's real baseflow
        outlet = np.unravel_index(int(np.nanargmax(np.where(mask, cellmax, -np.inf))),
                                  cellmax.shape)
        sim_base = float(base[outlet])
        if np.isfinite(sim_base) and sim_base > 0:
            base = base * float(np.clip(baseflow_cms / sim_base, 0.05, 20.0))
    peak = float(np.nanmax(np.where(mask, cellmax, np.nan))) if mask.any() else 0.0
    frames = []
    for p in tif_paths:                              # pass 2: render each
        a, bounds = _read_q(p)
        png = _render_ratio(a, base, mask & np.isfinite(a))
        frames.append((png, bounds, _q_time_from_name(p)))
    return frames, peak


# --------------------------------------------------------------------------- #
# frame disk cache — lets a cache-served run still show the 2-D animation
# (frames rendered once per (gauge, model, window); invalidated on param change)
# --------------------------------------------------------------------------- #
_WIN_FMT = "%Y-%m-%d %H:%M"
_FRAMES_VER = 2                    # bump when the render style changes — stale
                                   # caches are rejected and re-rendered


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
    idx = {"ver": _FRAMES_VER,
           "window": [t0.strftime(_WIN_FMT), t1.strftime(_WIN_FMT)],
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
        if (idx.get("ver") != _FRAMES_VER
                or idx.get("window") != [t0.strftime(_WIN_FMT), t1.strftime(_WIN_FMT)]):
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
            idx = json.load(fh)
            return (idx.get("ver") == _FRAMES_VER and
                    idx.get("window") == [t0.strftime(_WIN_FMT), t1.strftime(_WIN_FMT)])
    except Exception:
        return False


def empty_fig(msg: str = "waiting…") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#8a98a5"))
    fig.update_layout(height=340, xaxis=dict(visible=False), yaxis=dict(visible=False), **_BG)
    return fig
