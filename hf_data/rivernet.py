"""Upstream river network per gauge — read side of scripts/prep_hydrorivers.py.

HydroRIVERS NA topology (rivers/hydrorivers_na.parquet, ~1M reaches with
NEXT_DOWN links) + the gauge->reach snap table live in vincewin/CREST_data.
Upstream of a gauge = BFS over the reverse (children) adjacency from its
snapped reach. Big basins are thinned by Strahler order so the payload stays
drawable: the smallest order that keeps <= MAX_LINES reaches is shown.

Everything loads lazily on first request (~1-2 s, ~150 MB RAM) and stays in
process; per-gauge results get a small LRU cache.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict

import numpy as np

REPO = os.environ.get("CREST_FEEDBACK_REPO", "vincewin/CREST_data")
MAX_LINES = 2500                 # payload cap; order threshold adapts to this

_lock = threading.Lock()
_net: dict = {}                  # loaded arrays + adjacency, built once
_lru: OrderedDict = OrderedDict()
_LRU_N = 32


def _load():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    t = pq.read_table(hf_hub_download(REPO, "rivers/hydrorivers_na.parquet",
                                      repo_type="dataset", token=tok))
    ids = t.column("id").to_numpy()
    nxt = t.column("next").to_numpy()
    order = t.column("ord").to_numpy()
    la = t.column("lats").combine_chunks()
    lo = t.column("lons").combine_chunks()
    off = la.offsets.to_numpy()
    flat_lat = la.values.to_numpy()
    flat_lon = lo.values.to_numpy()

    idx_of = {int(i): k for k, i in enumerate(ids)}
    children: dict[int, list] = {}
    for k in range(len(ids)):
        d = int(nxt[k])
        if d:
            children.setdefault(d, []).append(k)

    s = pq.read_table(hf_hub_download(REPO, "rivers/gauge_reach.parquet",
                                      repo_type="dataset", token=tok))
    snap = dict(zip(s.column("gid").to_pylist(),
                    s.column("reach_id").to_pylist()))
    return {"ids": ids, "order": order, "off": off, "flat_lat": flat_lat,
            "flat_lon": flat_lon, "idx_of": idx_of, "children": children,
            "snap": snap}


def _net_ready() -> dict:
    with _lock:
        if not _net:
            _net.update(_load())
        return _net


def upstream(gid: str) -> dict:
    """{ok, n_total, n_shown, min_order, lines:[{o, xy:[[lat,lon],..]}, ..]}"""
    gid = gid.strip().zfill(8)
    with _lock:
        if gid in _lru:
            _lru.move_to_end(gid)
            return _lru[gid]
    try:
        net = _net_ready()
    except Exception:
        return {"ok": False, "reason": "river network unavailable"}
    rid = net["snap"].get(gid)
    root = net["idx_of"].get(int(rid)) if rid is not None else None
    if root is None:
        return {"ok": False, "reason": "gauge not snapped to a reach"}

    children, ids = net["children"], net["ids"]
    stack, seen = [root], [root]
    while stack:
        k = stack.pop()
        for c in children.get(int(ids[k]), ()):
            seen.append(c)
            stack.append(c)
    seen = np.array(seen)
    orders = net["order"][seen]

    # thin big basins: smallest Strahler order that keeps <= MAX_LINES reaches
    min_order = 1
    while (orders >= min_order).sum() > MAX_LINES and min_order < 9:
        min_order += 1
    keep = seen[orders >= min_order]

    off, fla, flo = net["off"], net["flat_lat"], net["flat_lon"]
    lines = []
    for k in keep:
        a, b = int(off[k]), int(off[k + 1])
        xy = np.round(np.column_stack([fla[a:b], flo[a:b]]).astype("float64"),
                      4).tolist()
        lines.append({"o": int(net["order"][k]), "xy": xy})
    out = {"ok": True, "gid": gid, "n_total": int(len(seen)),
           "n_shown": len(lines), "min_order": min_order, "lines": lines}
    with _lock:
        _lru[gid] = out
        while len(_lru) > _LRU_N:
            _lru.popitem(last=False)
    return out
