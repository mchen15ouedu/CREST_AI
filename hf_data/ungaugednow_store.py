"""Precomputed ungauged nowcasts — the ungauged analog of nowcaststore.

READ side serves the ungauged (virtual) points from, in order of preference:
  1. nowcast/v3_virtual_latest.parquet — DI-LSTM v3 (statics + NSE loss,
     trained on all 6,036 gauges) run hourly for all 2,676 points by the
     updater Space (scripts/run_nowcast_all.py::_virtual_v3): q12_1..q12_12
     from the 12-h model become q1..q12 here; `hist` is v3's own rolling
     7-day 1-h-lead analysis series (there are no observations at these
     points). Deployed 2026-08-17 — every point gets a nowcast, headwaters
     included (v3 needs no upstream gauge).
  2. nowcast/ungauged_latest<shard>.parquet — the EF5 routed feed below,
     used only when the v3 file is absent (e.g. v3 checkpoints deleted).
The response's "source" field ("dilstm_v3" | "ef5_routed") tells the
frontend which one it got.

The keep-warm Space (ungauged_space/runner.py) advances the ungauged
HydroBASINS points that have an upstream USGS gauge every hour with
hf_data.routednow.compute and writes the result set to
nowcast/ungauged_latest.parquet in the feedback dataset (hubdata.feedback()):

  vp                str    ungauged point id ("V" + HYRIV_ID)
  lat, lon          f32
  area_km2          f32
  q1..qH            f32    routed forecast at t0+1h .. t0+H (H = HORIZON_H)
  hist              str    JSON [[ "YYYY-MM-DD HH:MM", q ], ...] — the ~7-day
                           routed history (observed upstream inflow, no future)

Schema metadata carries the issue time t0 and generation timestamp. Unlike the
gauged DI-LSTM nowcast (a millisecond forward pass), each ungauged point needs
an EF5 run, so it is NOT computed on demand — the app reads this file and serves
every ungauged nowcast instantly, exactly as nowcaststore serves the gauged one.

WRITE side (keep-warm Space): build_table() + upload_latest().
READ side (dashboard): for_bbox() / by_ids() / series(), TTL-cached.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np

from . import hubdata, nowcastfeed
from .hubdata import Put

DATASET = hubdata.feedback()
# One writer (a shard of the keep-warm fleet) publishes ungauged_latest<tag>.parquet;
# the reader concatenates every shard file. tag="" for a single unsharded writer.
LATEST_PREFIX = "nowcast/ungauged_latest"
V3_LATEST = "nowcast/v3_virtual_latest.parquet"
TTL_S = 240
HORIZON_H = 12
TS_FMT = "%Y-%m-%d %H:%M"
MAX_ROW_AGE_H = 26.0                # = ungauged_space CARRY_TTL_H
META_T0_FMT = nowcastfeed.T0_FMT


# --------------------------------------------------------------------------- #
# WRITE side — called by the keep-warm Space
# --------------------------------------------------------------------------- #
def _qcols(horizon: int = HORIZON_H) -> list[str]:
    return [f"q{k}" for k in range(1, horizon + 1)]


def build_table(records: list[dict], t0: datetime, horizon: int = HORIZON_H):
    """Assemble the pyarrow table from routednow.compute() results.

    Each record: {"gid"/"vp", "lat", "lon", "area_km2", "q": [..], "history":
    [rows]} where a history row has "time" and "sim_q". A record may instead
    carry "hist_json" (the pre-encoded string — used when carrying forward a
    previously published row) and its own "t0" string ("%Y-%m-%d %H:%M") when
    it was issued at an earlier hour than this batch. Points with no forecast
    are skipped. Returns a pyarrow.Table with t0/generated schema metadata."""
    import pyarrow as pa

    qcols = _qcols(horizon)
    default_t0 = t0.strftime(TS_FMT)
    vp, lat, lon, area, t0s = [], [], [], [], []
    qarr = {c: [] for c in qcols}
    hist = []
    for r in records:
        if not r or not r.get("q"):
            continue
        vp.append(str(r.get("gid") or r.get("vp")))
        lat.append(float(r.get("lat", np.nan)))
        lon.append(float(r.get("lon", np.nan)))
        area.append(float(r.get("area_km2") or np.nan))
        t0s.append(str(r.get("t0") or default_t0))
        q = list(r.get("q") or [])
        for k, c in enumerate(qcols):
            qarr[c].append(float(q[k]) if k < len(q) and q[k] is not None else np.nan)
        if r.get("hist_json") is not None:
            hist.append(str(r["hist_json"]))
        else:
            rows = r.get("history") or []
            hist.append(json.dumps([[row["time"], round(float(row["sim_q"]), 3)]
                                    for row in rows
                                    if row.get("sim_q") is not None
                                    and isinstance(row.get("sim_q"), (int, float))],
                                   separators=(",", ":")))

    arrays = {
        "vp": pa.array(vp, pa.string()),
        "lat": pa.array(lat, pa.float32()),
        "lon": pa.array(lon, pa.float32()),
        "area_km2": pa.array(area, pa.float32()),
        "t0": pa.array(t0s, pa.string()),
    }
    for c in qcols:
        arrays[c] = pa.array(qarr[c], pa.float32())
    arrays["hist"] = pa.array(hist, pa.string())

    tbl = pa.table(arrays)
    meta = {
        "t0": t0.strftime(META_T0_FMT),
        "generated": datetime.utcnow().strftime(META_T0_FMT),
        "horizon": str(horizon),
        "n_points": str(len(vp)),
    }
    return tbl.replace_schema_metadata(meta)


def upload_latest(table, tag: str = "") -> bool:
    """Write `table` to nowcast/ungauged_latest<tag>.parquet in the feedback
    dataset. Each keep-warm shard passes its own tag (e.g. "__0of3") so
    shards don't clobber each other; the reader concatenates them. False
    without a token; raises RuntimeError when the commit did not land."""
    import io

    import pyarrow.parquet as pq

    if not DATASET.can_write:
        return False
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    t0 = (table.schema.metadata or {}).get(b"t0", b"").decode()
    res = DATASET.commit([Put(f"{LATEST_PREFIX}{tag}.parquet", buf.getvalue())],
                         f"ungauged nowcast{tag} {t0}")
    if not res:
        raise RuntimeError(f"ungauged nowcast upload: {res.error}")
    return True


# --------------------------------------------------------------------------- #
# READ side — served by the dashboard
# --------------------------------------------------------------------------- #
def _v3_table(local: str):
    """DI-LSTM v3 virtual-point file -> (meta, cols) in the store's layout, or
    None when the file carries no usable forecast (then the EF5 routed shards
    are served)."""
    import pyarrow.parquet as pq
    t = pq.read_table(local)
    names = t.schema.names
    src = nowcastfeed.qcols(names, "q12_") or nowcastfeed.qcols(names, "q")
    if not src or "vp" not in names:     # 12-h model preferred (12-h horizon)
        return None
    cols = {n: t.column(n).to_numpy(zero_copy_only=False)
            for n in ("vp", "lat", "lon", "area_km2") if n in names}
    for k, n in enumerate(src):
        cols[f"q{k + 1}"] = t.column(n).to_numpy(zero_copy_only=False)
    if "hist" in names:
        cols["hist"] = t.column("hist").to_numpy(zero_copy_only=False)
    m = nowcastfeed.schema_meta(t)
    meta = {"t0": m.get("t0", ""), "generated": m.get("generated") or m.get("t0", ""),
            "horizon": str(len(src)), "n_points": str(len(cols["vp"])),
            "source": "dilstm_v3", "model_file": m.get("model12_file") or m.get("model_file", ""),
            "model_epoch": m.get("model12_epoch") or m.get("model_epoch", "")}
    return meta, cols


def _read_routed(ds: hubdata.Dataset) -> hubdata.Read:
    """Every ungauged_latest<shard>.parquet, concatenated, rows from a shard
    that stopped publishing dropped. MISSING when no shard is published."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    listed = ds.list_files(LATEST_PREFIX)
    if not listed.ok:
        return listed
    tables, meta = [], {}
    for f in (f for f in listed.value if f.endswith(".parquet")):
        r = ds.read(f, pq.read_table)
        if r.unreachable:
            return r
        if r.missing:                        # deleted since the listing
            continue
        t = r.value
        tables.append(t)
        m = nowcastfeed.schema_meta(t)
        if not meta or m.get("generated", "") > meta.get("generated", ""):
            meta = m                         # report the freshest shard's issue time
    if not tables:
        return hubdata.Read(hubdata.MISSING)
    if len(tables) == 1:
        big = tables[0]
    else:
        try:                                 # shards may differ in schema (t0 col
            big = pa.concat_tables(tables, promote_options="default")   # added later)
        except TypeError:                    # older pyarrow spelling
            big = pa.concat_tables(tables, promote=True)
    cols = {name: big.column(name).to_numpy(zero_copy_only=False)
            for name in big.schema.names}
    # Drop rows whose own issue time is far behind the freshest shard: a shard
    # that stopped publishing (1of3 since 2026-08-13) was served under the
    # fresh shard's t0 label. Rows with no parseable t0 cannot be dated -> drop.
    newest = nowcastfeed.issue_time(meta)
    if newest is not None and "t0" in cols:
        keep = np.zeros(len(cols["t0"]), bool)
        for i, v in enumerate(cols["t0"]):
            try:
                keep[i] = ((newest - datetime.strptime(str(v), TS_FMT))
                           .total_seconds() / 3600.0 <= MAX_ROW_AGE_H)
            except (ValueError, TypeError):
                pass
        if not keep.all():
            cols = {k: v[keep] for k, v in cols.items()}
    meta["source"] = "ef5_routed"
    return hubdata.Read(hubdata.DATA, (meta, cols))


def _read_feed(ds: hubdata.Dataset) -> hubdata.Read:
    """The v3 file when published and usable, else the EF5 routed shards. An
    unreachable v3 read is an outage, not a reason to switch feeds."""
    v3 = ds.read(V3_LATEST, _v3_table, fresh=True)
    if v3.unreachable or (v3.ok and v3.value is not None):
        return v3
    return _read_routed(ds)


def _fresh():
    """(meta, cols) of the current feed, or (None, None)."""
    return nowcastfeed.table(nowcastfeed.read(DATASET, "ungauged", _read_feed, TTL_S))


def _qkeys(cols) -> list[str]:
    return nowcastfeed.qcols(cols, "q")


def issue_t0() -> datetime | None:
    meta, cols = _fresh()
    return None if cols is None else nowcastfeed.issue_time(meta)


def _pack(cols, i, qk, t0):
    """One ungauged point: id/lat/lon/area, forecast q[] with times, history.
    Forecast timestamps are built from the ROW's own issue time when present
    (rows carried forward from an earlier pass keep their original t0), falling
    back to the file-level t0."""
    row_t0 = t0
    if "t0" in cols:
        try:
            row_t0 = datetime.strptime(str(cols["t0"][i]), TS_FMT)
        except (ValueError, TypeError):
            pass
    q = [round(float(cols[n][i]), 3) for n in qk]
    fcst = ([[(row_t0 + timedelta(hours=k + 1)).strftime(TS_FMT), q[k]]
             for k in range(len(q)) if np.isfinite(q[k])] if row_t0 else [])
    try:
        hist = json.loads(cols["hist"][i]) if "hist" in cols else []
    except Exception:
        hist = []
    return {"id": str(cols["vp"][i]),
            "lat": round(float(cols["lat"][i]), 5),
            "lon": round(float(cols["lon"][i]), 5),
            "area_km2": round(float(cols["area_km2"][i]), 1),
            "t0": row_t0.strftime(TS_FMT) if row_t0 else None,
            "q": q, "forecast": fcst, "history": hist, "virtual": True}


def _select(cols, idx, t0):
    qk = _qkeys(cols)
    return [_pack(cols, int(i), qk, t0) for i in idx]


def for_bbox(w: float, s: float, e: float, n: float, limit: int = 200) -> dict:
    """Every ungauged nowcast inside the bbox (largest basins first)."""
    meta, cols = _fresh()
    if cols is None:
        return {"ok": False, "reason": "no precomputed ungauged nowcast yet"}
    idx = nowcastfeed.in_bbox(cols, w, s, e, n)
    total = int(len(idx))
    idx = idx[:max(1, limit)]
    t0 = issue_t0()
    return {"ok": True, "t0": meta.get("t0"), "generated": meta.get("generated"),
            "source": meta.get("source"), "model_file": meta.get("model_file"),
            "n_in_view": total, "truncated": total > len(idx),
            "points": _select(cols, idx, t0)}


NU_TTL_S = 900
NO_UPSTREAM_PREFIX = "nowcast/ungauged_no_upstream"


def _load_ids(local: str) -> set:
    with open(local, encoding="utf-8") as fp:
        return set(json.load(fp))


def _read_no_upstream(ds: hubdata.Dataset) -> hubdata.Read:
    listed = ds.list_files(NO_UPSTREAM_PREFIX)
    if not listed.ok:
        return listed
    ids: set = set()
    for f in (f for f in listed.value if f.endswith(".json")):
        r = ds.read(f, _load_ids)
        if r.unreachable:
            return r
        if r.ok:
            ids |= r.value
    return hubdata.Read(hubdata.DATA, ids)


def _no_upstream_ids() -> set:
    """Points the keep-warm fleet has confirmed have NO upstream USGS gauge
    (union of the per-shard nowcast/ungauged_no_upstream*.json lists); the
    last good set while the Hub is unreachable."""
    r = nowcastfeed.read(DATASET, NO_UPSTREAM_PREFIX, _read_no_upstream, NU_TTL_S)
    return r.value if r.ok else set()


def by_ids(ids) -> dict:
    """Nowcasts for exactly these ungauged ids (comma string or list).

    Ids not in the store are classified in `missing`: "no_upstream" when the
    keep-warm fleet has confirmed the point has no upstream USGS gauge to route
    from, else "pending" (the hourly precompute simply hasn't produced it yet —
    the frontend must NOT claim a hydrologic cause for those)."""
    meta, cols = _fresh()
    if cols is None:
        return {"ok": False, "reason": "no precomputed ungauged nowcast yet"}
    want = ([s.strip() for s in ids.split(",") if s.strip()]
            if isinstance(ids, str) else [str(i) for i in ids])
    idx = np.nonzero(np.isin(cols["vp"], want))[0]
    t0 = issue_t0()
    pts = _select(cols, idx, t0)
    have = {p["id"] for p in pts}
    # "no_upstream" is only a reason under the EF5 routed feed; v3 needs no
    # upstream gauge, so anything absent there is simply not produced yet
    nu = _no_upstream_ids() if meta.get("source") != "dilstm_v3" else set()
    missing = {w: ("no_upstream" if w in nu else "pending")
               for w in want if w not in have}
    return {"ok": True, "t0": meta.get("t0"), "generated": meta.get("generated"),
            "source": meta.get("source"), "model_file": meta.get("model_file"),
            "points": pts, "missing": missing}


def series(vp: str) -> dict | None:
    """A single ungauged point's nowcast, or None if absent."""
    r = by_ids([str(vp)])
    pts = r.get("points") or []
    return pts[0] if pts else None
