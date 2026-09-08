"""Local prep: OpenStreetMap infrastructure -> 1x1-degree parquet tiles.

Extracts the three infrastructure values the risk view shows per pixel from
Geofabrik state extracts (canonical OSM downloads), aggregates them onto
the events' native 1-arcsecond grid, and uploads parquet tiles to
vincewin/CREST_riskdata — so the demo Space probes tiles instead of hitting
Overpass live (504s under load, measured 2026-08-25):

  buildings  count of building centroids per 1" cell    (sum-merged)
  roads      max highway class crossing each 1" cell    (max-merged;
             line vertices densified to <1 cell spacing before binning)
  critical   facilities as exact points with kind/name  (deduped)

Layout on HF:
  osm/grid/N{lat}W{lon}.parquet     row,col (uint16, 3600x3600 within the
                                    tile), n_bld (uint16), road (uint8)
  osm/critical/N{lat}W{lon}.parquet lat, lon, kind, name

Runs state by state (resumable) into a LOCAL accumulation store first
(border tiles receive several states), then uploads everything in batched
commits:  python scripts/prep_osm_tiles.py --states ohio west-virginia
          python scripts/prep_osm_tiles.py --all-conus     (the long haul)
          python scripts/prep_osm_tiles.py --upload-only
"""
from __future__ import annotations

import argparse
import io as _io
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import truststore
truststore.inject_into_ssl()

from forcing_update_common import hf_token   # noqa: E402

REPO = "vincewin/CREST_riskdata"
SRC = os.environ.get("OSM_SRC", r"E:\hydroZone\riskdata_src\osm")
ACC = os.environ.get("OSM_ACC", r"E:\hydroZone\riskdata_src\osm_tiles")
GEOFABRIK = "https://download.geofabrik.de/north-america/us/{state}-latest.osm.pbf"
RES = 1.0 / 3600.0                      # 1 arc-second (the events' DEM grid)
N = 3600                                # cells per tile side
ROAD_RANK = ["service", "unclassified", "residential", "tertiary",
             "secondary", "primary", "trunk", "motorway"]
CRIT = {("amenity", "hospital"): "hospital", ("amenity", "clinic"): "clinic",
        ("amenity", "school"): "school",
        ("amenity", "kindergarten"): "school",
        ("amenity", "fire_station"): "fire station",
        ("amenity", "police"): "police",
        ("power", "substation"): "power substation",
        ("power", "plant"): "power plant",
        ("man_made", "water_works"): "water works",
        ("man_made", "wastewater_plant"): "wastewater plant"}
CONUS_STATES = [
    "alabama", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "district-of-columbia", "florida", "georgia",
    "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new-hampshire", "new-jersey", "new-mexico", "new-york",
    "north-carolina", "north-dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode-island", "south-carolina", "south-dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west-virginia", "wisconsin", "wyoming",
]


def tile_name(lat0: int, lon0: int) -> str:
    return (f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}"
            f"{'W' if lon0 < 0 else 'E'}{abs(lon0):03d}")


def download_state(state: str) -> str:
    import requests
    os.makedirs(SRC, exist_ok=True)
    local = os.path.join(SRC, f"{state}-latest.osm.pbf")
    if os.path.exists(local) and os.path.getsize(local) > 1 << 20:
        return local
    url = GEOFABRIK.format(state=state)
    print(f"downloading {state} ...", flush=True)
    t0 = time.time()
    r = requests.get(url, stream=True, timeout=3600,
                     headers={"User-Agent": "CREST-AI/1.0"})
    r.raise_for_status()
    tmp = local + ".part"
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(1 << 22):
            fh.write(chunk)
    os.replace(tmp, local)
    print(f"  {os.path.getsize(local) / 1e6:.0f} MB in "
          f"{time.time() - t0:.0f} s", flush=True)
    return local


class _Grids:
    """Per-tile sparse accumulators held as dense uint16/uint8 arrays for
    the tiles the current state touches (a state spans ~30-90 tiles)."""

    def __init__(self):
        self.bld: dict = {}
        self.road: dict = {}
        self.crit: list = []

    def _key(self, lat: float, lon: float):
        lat0, lon0 = math.floor(lat), math.floor(lon)
        r = int((lat - lat0) / RES)
        c = int((lon - lon0) / RES)
        if r >= N: r = N - 1
        if c >= N: c = N - 1
        return (int(lat0), int(lon0)), N - 1 - r, c   # row 0 = north edge

    def add_building(self, lat, lon):
        k, r, c = self._key(lat, lon)
        g = self.bld.get(k)
        if g is None:
            g = self.bld[k] = np.zeros((N, N), np.uint16)
        if g[r, c] < 65535:
            g[r, c] += 1

    def add_road(self, lat, lon, rank):
        k, r, c = self._key(lat, lon)
        g = self.road.get(k)
        if g is None:
            g = self.road[k] = np.zeros((N, N), np.uint8)
        if rank > g[r, c]:
            g[r, c] = rank


import osmium as _osmium


class Handler(_osmium.SimpleHandler):
    def __init__(self, grids):
        super().__init__()
        self.g = grids
        self.n_ways = 0

    def way(self, w):
        tags = w.tags
        self.n_ways += 1
        if self.n_ways % 2_000_000 == 0:
            print(f"  ...{self.n_ways / 1e6:.0f} M ways", flush=True)
        hwy = tags.get("highway")
        if hwy:
            base = hwy[:-5] if hwy.endswith("_link") else hwy
            if base in ROAD_RANK:
                rank = ROAD_RANK.index(base) + 1
                try:
                    nodes = [(nd.lat, nd.lon) for nd in w.nodes
                             if nd.location.valid()]
                except Exception:
                    return
                for i in range(len(nodes) - 1):
                    (la1, lo1), (la2, lo2) = nodes[i], nodes[i + 1]
                    seg = max(abs(la2 - la1), abs(lo2 - lo1))
                    k = max(1, int(seg / RES) + 1)   # densify < 1 cell
                    for j in range(k + 1):
                        f = j / k
                        self.g.add_road(la1 + (la2 - la1) * f,
                                        lo1 + (lo2 - lo1) * f, rank)
            return
        if "building" in tags:
            try:
                lats = [nd.lat for nd in w.nodes if nd.location.valid()]
                lons = [nd.lon for nd in w.nodes if nd.location.valid()]
            except Exception:
                return
            if lats:
                self.g.add_building(sum(lats) / len(lats),
                                    sum(lons) / len(lons))
            return
        self._crit_tags(tags, w)

    def node(self, n):
        self._crit_tags(n.tags, n, n.location)

    def _crit_tags(self, tags, obj, loc=None):
        for (k, v), label in CRIT.items():
            if tags.get(k) == v:
                if loc is not None and loc.valid():
                    la, lo = loc.lat, loc.lon
                else:
                    try:
                        pts = [(nd.lat, nd.lon) for nd in obj.nodes
                               if nd.location.valid()]
                    except Exception:
                        return
                    if not pts:
                        return
                    la = sum(p[0] for p in pts) / len(pts)
                    lo = sum(p[1] for p in pts) / len(pts)
                self.g.crit.append((la, lo, label,
                                    (tags.get("name") or "")[:80]))
                return


def process_state(state: str):
    """Stream one pbf; merge its grids into the local accumulation store."""
    pbf = download_state(state)
    marker = os.path.join(ACC, f"_done_{state}")
    if os.path.exists(marker):
        print(f"{state}: already accumulated")
        return
    grids = _Grids()
    h = Handler(grids)
    print(f"{state}: streaming ways/nodes ...", flush=True)
    t0 = time.time()
    h.apply_file(pbf, locations=True, idx="sparse_mem_array")
    print(f"{state}: {h.n_ways / 1e6:.1f} M ways in "
          f"{(time.time() - t0) / 60:.1f} min; merging "
          f"{len(set(grids.bld) | set(grids.road))} tiles", flush=True)
    os.makedirs(ACC, exist_ok=True)
    keys = set(grids.bld) | set(grids.road)
    for k in keys:
        lat0, lon0 = k
        f = os.path.join(ACC, f"{tile_name(lat0, lon0)}.npz")
        bld = grids.bld.get(k)
        road = grids.road.get(k)
        if os.path.exists(f):
            # context-manage the npz: np.load keeps the file handle open
            # (lazy NpzFile) and Windows then refuses the os.replace below
            with np.load(f) as z:
                old_b, old_r = z["bld"], z["road"]
            bld = old_b if bld is None else (old_b.astype(np.uint32)
                                             + bld).clip(0, 65535
                                                         ).astype(np.uint16)
            road = old_r if road is None else np.maximum(old_r, road)
        np.savez_compressed(f + ".tmp.npz",
                            bld=bld if bld is not None
                            else np.zeros((N, N), np.uint16),
                            road=road if road is not None
                            else np.zeros((N, N), np.uint8))
        os.replace(f + ".tmp.npz", f)
    if grids.crit:
        import pandas as pd
        cf = os.path.join(ACC, f"crit_{state}.parquet")
        pd.DataFrame(grids.crit,
                     columns=["lat", "lon", "kind", "name"]
                     ).to_parquet(cf, index=False)
    open(marker, "w").close()
    print(f"{state}: accumulated", flush=True)


def upload():
    """Local accumulation store -> HF parquet tiles (batched commits)."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=hf_token())
    ops, n_up = [], 0
    for fn in sorted(os.listdir(ACC)):
        if not fn.endswith(".npz"):
            continue
        tile = fn[:-4]
        z = np.load(os.path.join(ACC, fn))
        bld, road = z["bld"], z["road"]
        rows, cols = np.nonzero((bld > 0) | (road > 0))
        if len(rows) == 0:
            continue
        t = pa.table({"row": rows.astype(np.uint16),
                      "col": cols.astype(np.uint16),
                      "n_bld": bld[rows, cols],
                      "road": road[rows, cols]})
        buf = _io.BytesIO()
        pq.write_table(t, buf, compression="zstd")
        ops.append(CommitOperationAdd(f"osm/grid/{tile}.parquet",
                                      _io.BytesIO(buf.getvalue())))
        n_up += 1
        if len(ops) >= 40:
            api.create_commit(repo_id=REPO, repo_type="dataset",
                              operations=ops,
                              commit_message=f"osm grid: +{len(ops)} tiles")
            print(f"committed ({n_up} so far)", flush=True)
            ops = []
    # critical facilities: dedupe across states, one file per tile
    crits = [f for f in os.listdir(ACC) if f.startswith("crit_")]
    if crits:
        df = pd.concat([pd.read_parquet(os.path.join(ACC, f))
                        for f in crits]).drop_duplicates()
        df["tile"] = [tile_name(math.floor(a), math.floor(o))
                      for a, o in zip(df.lat, df.lon)]
        for tile, sub in df.groupby("tile"):
            buf = _io.BytesIO()
            sub.drop(columns="tile").to_parquet(buf, index=False)
            ops.append(CommitOperationAdd(f"osm/critical/{tile}.parquet",
                                          _io.BytesIO(buf.getvalue())))
            if len(ops) >= 40:
                api.create_commit(repo_id=REPO, repo_type="dataset",
                                  operations=ops,
                                  commit_message="osm critical tiles")
                ops = []
    if ops:
        api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                          commit_message="osm tiles (final)")
    print(f"upload done: {n_up} grid tiles")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="*", default=[])
    ap.add_argument("--all-conus", action="store_true")
    ap.add_argument("--upload-only", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    if not args.upload_only:
        states = CONUS_STATES if args.all_conus else args.states
        for st in states:
            process_state(st)
    if not args.no_upload:
        upload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
