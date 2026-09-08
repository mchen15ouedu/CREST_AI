"""OpenStreetMap infrastructure values at a probed pixel (risk view).

No rendering, no per-basin prefetch: each probe runs one small Overpass
query on a box around the solver pixel and reports the VALUES —

  * buildings   count intersecting the pixel box (+ tagged types)
  * roads       highway classes crossing the box (top class + names)
  * critical    facilities in a wider context ring (hospital, school,
                fire/police, power, water works, bridges), with distance

Overpass fair-use: one query per probe (map pans, ~1/s worst case), tiny
bboxes (pixel ~25 m + 250 m context ring), 25 s timeout, results cached
in-process keyed by rounded pixel. Endpoint overridable via OVERPASS_URL.
"""
from __future__ import annotations

import math
import os
import threading
import time

# public mirrors, tried in order (the main instance 504s under load);
# OVERPASS_URL prepends a preferred endpoint
MIRRORS = [u for u in (os.environ.get("OVERPASS_URL", ""),
                       "https://overpass-api.de/api/interpreter",
                       "https://lz4.overpass-api.de/api/interpreter",
                       "https://overpass.kumi.systems/api/interpreter",
                       "https://overpass.private.coffee/api/interpreter",
                       "https://maps.mail.ru/osm/tools/overpass/api/interpreter") if u]
OVERPASS = MIRRORS[0]
CONTEXT_M = float(os.environ.get("RISK_INFRA_CONTEXT_M", "250"))
# highway class ranking, minor -> major
ROAD_RANK = ["service", "unclassified", "residential", "tertiary",
             "secondary", "primary", "trunk", "motorway"]
CRIT_TAGS = {  # tag=value -> label shown on the arm
    ("amenity", "hospital"): "hospital", ("amenity", "clinic"): "clinic",
    ("amenity", "school"): "school", ("amenity", "kindergarten"): "school",
    ("amenity", "fire_station"): "fire station",
    ("amenity", "police"): "police",
    ("power", "substation"): "power substation",
    ("power", "plant"): "power plant",
    ("man_made", "water_works"): "water works",
    ("man_made", "wastewater_plant"): "wastewater plant",
    ("man_made", "bridge"): "bridge",
}

_cache: dict = {}
_lock = threading.Lock()
CACHE_TTL_S = 6 * 3600
FAIL_TTL_S = 60.0
CACHE_MAX = 4096
_state: dict = {"mirror": 0}


def _bbox_around(lat: float, lon: float, half_m: float):
    dlat = half_m / 111132.0
    dlon = half_m / (111132.0 * max(0.2, math.cos(math.radians(lat))))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)   # s,w,n,e


def _query(lat: float, lon: float, cell_half_m: float) -> str:
    s, w, n, e = _bbox_around(lat, lon, cell_half_m)
    S, W, N, E = _bbox_around(lat, lon, max(CONTEXT_M, cell_half_m))
    crit = "".join(
        f'nwr["{k}"="{v}"]({S},{W},{N},{E});' for k, v in CRIT_TAGS)
    return (
        "[out:json][timeout:25];"
        f'(way["building"]({s},{w},{n},{e});)->.b;'
        f'(way["highway"]({s},{w},{n},{e});)->.r;'
        f"({crit})->.c;"
        ".b out count;"
        ".b out tags center 8;"
        ".r out tags center 12;"
        ".c out tags center 24;"
    )


RISK_REPO = os.environ.get("CREST_RISKDATA_REPO", "vincewin/CREST_riskdata")
RES_1AS = 1.0 / 3600.0
_tiles: dict = {}
_tiles_missing: set = set()


def _tile_of(lat: float, lon: float) -> str:
    lat0, lon0 = math.floor(lat), math.floor(lon)
    return (f"{'N' if lat0 >= 0 else 'S'}{abs(int(lat0)):02d}"
            f"{'W' if lon0 < 0 else 'E'}{abs(int(lon0)):03d}")


def _load_tile(tile: str):
    if tile in _tiles_missing:
        return None
    if tile in _tiles:
        return _tiles[tile]
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        g = pq.read_table(hf_hub_download(
            RISK_REPO, f"osm/grid/{tile}.parquet", repo_type="dataset",
            token=os.environ.get("HF_TOKEN")))
        idx = {}
        rows = g.column("row").to_numpy(); cols = g.column("col").to_numpy()
        bld = g.column("n_bld").to_numpy(); road = g.column("road").to_numpy()
        for i in range(len(rows)):
            idx[(int(rows[i]), int(cols[i]))] = (int(bld[i]), int(road[i]))
        crit = []
        try:
            import pandas as pd
            cf = hf_hub_download(RISK_REPO, f"osm/critical/{tile}.parquet",
                                 repo_type="dataset",
                                 token=os.environ.get("HF_TOKEN"))
            crit = pd.read_parquet(cf).to_dict("records")
        except Exception:
            pass
        if len(_tiles) > 8:
            _tiles.clear()
        _tiles[tile] = {"idx": idx, "crit": crit}
        return _tiles[tile]
    except Exception:
        _tiles_missing.add(tile)
        return None


def _tile_probe(lat: float, lon: float) -> dict | None:
    """Pre-hosted OSM tiles (CREST_riskdata) — the primary path: the 1-arcsec
    cell's building count + road class, and critical facilities within the
    context ring, all from a few-MB parquet the Space caches locally."""
    tl = _load_tile(_tile_of(lat, lon))
    if tl is None:
        return None
    lat0, lon0 = math.floor(lat), math.floor(lon)
    row = 3599 - int((lat - lat0) / RES_1AS)
    col = int((lon - lon0) / RES_1AS)
    n_bld, road = tl["idx"].get((min(max(row, 0), 3599),
                                 min(max(col, 0), 3599)), (0, 0))
    crit = []
    for c in tl["crit"]:
        d = _dist_m(lat, lon, c["lat"], c["lon"])
        if d <= CONTEXT_M:
            crit.append({"kind": c["kind"], "name": c.get("name") or "",
                         "dist_m": round(d)})
    crit.sort(key=lambda c: c["dist_m"])
    return {"status": "ok", "source": "tiles",
            "buildings": int(n_bld), "building_types": {},
            "roads": ([{"class": ROAD_RANK[road - 1], "name": ""}]
                      if road > 0 else []),
            "road_rank": int(road), "road_rank_max": len(ROAD_RANK),
            "critical": crit[:6], "context_m": int(CONTEXT_M)}


def probe(lat: float, lon: float, cell_half_m: float = 15.0) -> dict:
    """Infrastructure values for the pixel at (lat, lon): pre-hosted tiles
    first, live Overpass as the fallback for tiles not yet uploaded."""
    hit = _tile_probe(lat, lon)
    if hit is not None:
        return hit
    return _overpass_probe(lat, lon, cell_half_m)


def _overpass_probe(lat: float, lon: float, cell_half_m: float = 15.0) -> dict:
    """Live Overpass fallback. Cached."""
    key = (round(lat, 4), round(lon, 4))
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL_S:
            return hit[1]
    import requests
    out, q = None, _query(lat, lon, cell_half_m)
    # sticky mirror: keep starting from the endpoint that last worked
    start = _state.get("mirror", 0)
    order = MIRRORS[start:] + MIRRORS[:start]
    for i, url in enumerate(order):
        try:
            r = requests.post(url, data={"data": q}, timeout=30,
                              headers={"User-Agent": "CREST-AI/1.0"})
            r.raise_for_status()
            out = _digest(r.json(), lat, lon)
            _state["mirror"] = (start + i) % len(MIRRORS)
            break
        except Exception as exc:
            host = url.split("/")[2].split(".")[-2]
            errs = (out or {}).get("_errs", [])
            errs.append(f"{host}:{type(exc).__name__}")
            out = {"status": "unavailable (" + "; ".join(errs) + ")",
                   "_errs": errs}
    if out and "_errs" in out:
        out.pop("_errs", None)
    with _lock:
        if len(_cache) > CACHE_MAX:
            _cache.clear()
        # failures are transient (504s under load): cache them for 60 s,
        # not the 6 h a real result deserves
        ttl_now = now if out.get("status") == "ok"             else now - CACHE_TTL_S + FAIL_TTL_S
        _cache[key] = (ttl_now, out)
    return out


def _dist_m(lat, lon, lat2, lon2):
    return math.hypot((lat2 - lat) * 111132.0,
                      (lon2 - lon) * 111132.0
                      * math.cos(math.radians(lat)))


def _digest(js: dict, lat: float, lon: float) -> dict:
    n_build, b_types, roads, crit = 0, {}, {}, []
    for el in js.get("elements", []):
        if el.get("type") == "count":
            n_build = int(el.get("tags", {}).get("ways",
                          el.get("count", 0)) or 0)
            continue
        tags = el.get("tags") or {}
        ctr = el.get("center") or {k: el.get(k) for k in ("lat", "lon")}
        if "building" in tags and "highway" not in tags:
            t = tags["building"]
            if t and t != "yes":
                b_types[t] = b_types.get(t, 0) + 1
            continue
        if "highway" in tags:
            cls = tags["highway"]
            base = cls[:-5] if cls.endswith("_link") else cls
            if base in ROAD_RANK:
                nm = tags.get("name") or tags.get("ref") or ""
                cur = roads.get(base)
                if cur is None or (nm and not cur):
                    roads[base] = nm
            continue
        for (k, v), label in CRIT_TAGS.items():
            if tags.get(k) == v:
                d = (_dist_m(lat, lon, ctr["lat"], ctr["lon"])
                     if ctr.get("lat") is not None else None)
                crit.append({"kind": label,
                             "name": tags.get("name") or "",
                             "dist_m": round(d) if d is not None else None})
                break
    crit.sort(key=lambda c: c.get("dist_m") or 1e9)
    top = max((ROAD_RANK.index(c) for c in roads), default=-1)
    return {
        "status": "ok",
        "buildings": n_build,
        "building_types": dict(sorted(b_types.items(),
                                      key=lambda kv: -kv[1])[:4]),
        "roads": [{"class": c, "name": roads[c]}
                  for c in sorted(roads, key=ROAD_RANK.index,
                                  reverse=True)][:4],
        "road_rank": top + 1,               # 0 none .. 8 motorway
        "road_rank_max": len(ROAD_RANK),
        "critical": crit[:6],
        "context_m": int(CONTEXT_M),
    }
