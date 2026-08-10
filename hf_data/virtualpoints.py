"""Virtual points: ungauged CONUS simulation outlets (HydroBASINS pour points).

Level-07 sub-basin outlets fill the gaps between USGS gauges so every part of
CONUS has a hindcast target. They have no observations, so a virtual-point
run always uses the speed scheme: the domain is truncated at upstream USGS
gauges and their observed flow is injected as the boundary condition, and the
parameters are borrowed from the nearest calibrated gauge (regionalization —
calibration at the point itself is impossible).

Catalog: gauges/virtual_points.parquet in vincewin/CREST_data
(built by scripts/prep_virtual_points.py). Ids are "V" + HYRIV_ID.
"""
from __future__ import annotations

import threading

import truststore
truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download

HF_REPO = "vincewin/CREST_data"
VP_PATH = "gauges/virtual_points.parquet"

_lock = threading.Lock()
_df = None            # pandas DataFrame indexed by vp id, or "failed" sentinel


def is_virtual(gauge_id) -> bool:
    return str(gauge_id).upper().startswith("V")


def _load():
    global _df
    with _lock:
        if _df is None:
            try:
                import pandas as pd
                path = hf_hub_download(HF_REPO, VP_PATH, repo_type="dataset")
                _df = pd.read_parquet(path).set_index("vp", drop=False)
            except Exception:
                _df = "failed"          # sentinel: no re-download loop per call
        return None if isinstance(_df, str) else _df


def info(vp_id: str) -> dict | None:
    """Outlet dict for the pipeline — same shape as pipeline.gauge_info()."""
    df = _load()
    if df is None or vp_id not in df.index:
        return None
    r = df.loc[vp_id]
    return {"id": vp_id, "name": f"Ungauged point "
                                 f"({float(r.area_km2):,.0f} km² basin)",
            "lat": float(r.lat), "lon": float(r.lon),
            "area": float(r.area_km2), "virtual": True, "tz": str(r.tz)}


def for_bbox(w: float, s: float, e: float, n: float, limit: int = 200) -> list[dict]:
    """Pin dicts for the viewport (map serving)."""
    df = _load()
    if df is None:
        return []
    m = df[(df.lon >= w) & (df.lon <= e) & (df.lat >= s) & (df.lat <= n)]
    if len(m) > limit:                       # center-first, like the gauge pins
        cx, cy = (w + e) / 2, (s + n) / 2
        m = m.assign(_d=(m.lat - cy) ** 2 + (m.lon - cx) ** 2).nsmallest(limit, "_d")
    return [{"id": str(r.vp), "name": "Ungauged point",
             "lat": float(r.lat), "lon": float(r.lon),
             "area_km2": float(r.area_km2), "tz": str(r.tz), "virtual": True}
            for r in m.itertuples()]


def count() -> int:
    df = _load()
    return 0 if df is None else len(df)
