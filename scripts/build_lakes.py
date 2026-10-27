"""Offline build of hf_data/data/lakes.parquet: the lakes, reservoirs and ponds
EF5's lake module simulates (PI 2026-10-27: "identify the reservoirs and lakes
for the simulations and turn on the lake module, start with the west").

Each water body becomes ONE lake cell on EF5's HydroSHEDS 3-arc-second grid,
at its outlet (the module's design: lakes.csv name,lat,lon,th_volume,area,
klake). Inputs (local copies, E:/hydrosheds):
  * USACE National Inventory of Dams (nid/nation.csv): every US dam, with
    max / normal storage, surface area, drainage area and purpose;
  * HydroLAKES v1.0 pour points (hydrolakes/...points_v10.shp): lakes and
    reservoirs >= 10 ha with area, Dis_avg and the watershed area at the
    pour point (HydroSHEDS, the family of our terrain).
A HydroLAKES reservoir within MATCH_KM of a NID dam of similar storage is one
water body (NID volumes, HydroLAKES pour point and area).

Volume V (th_volume) and retention K (klake), PI-approved scheme 2026-10-27.
EF5's law: below V the lake releases Q = S/K*(S/V)^b, so a full lake releases
V/K; above V the excess spills at once.
  * dams: V = NID Max Storage (else NID Storage); runs start at Normal
    Storage (S0), written as a lake state by the run (EF5 cold-starts full);
  * natural lakes: V = area x NATURAL_DEPTH_M of active storage above the
    outlet - not Vol_total, whose water below the outlet never leaves; S0 = V;
  * K = V / mean inflow; mean inflow = the cell's drainage area x the
    observation-based mean annual runoff (GAGES-II RUNAVE7100) of a gauge
    basin containing it of about the same size, else HydroLAKES Dis_avg,
    else the runoff of the smallest gauge basin around it. A full lake then
    passes its mean inflow;
  * small stock, fire and fish ponds without outlet works: K = POND_K_H (no
    release - they spill when full and lose water to evaporation).
NID data guards: a max pool > MAX_NORMAL_RATIO x the normal pool outside
flood control is a data error (V = normal pool); flood-control and debris
dams start empty when no normal pool is listed and release within FLOOD_K_H
(K = V / mean inflow would hold a desert flood pool for decades); a surface
area implying a mean depth > MAX_DEPTH_M is re-estimated at FILL_DEPTH_M.
A water body whose K is under MIN_K_H is dropped: its storage is negligible
next to its flow, and K below a few time steps breaks the module's mass
balance (outflow x dt can exceed storage).

    python scripts/build_lakes.py --sites <basin-extent sites.parquet>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "hf_data", "data", "lakes.parquet")
NID_CSV = "E:/hydrosheds/nid/nation.csv"
HL_SHP = "E:/hydrosheds/hydrolakes/extracted/HydroLAKES_points_v10_shp/HydroLAKES_points_v10.shp"
ACC_COG = "E:/hydrosheds/cog/na_acc_3s.tif"
RUNOFF_TXT = "D:/gageII_char/spreadsheets-in-csv-format/conterm_hydro.txt"

WEST_LON = -104.0            # lakes for western gauges (outlets < -105) + 1 deg margin
MATCH_KM = 3.0               # HydroLAKES reservoir <-> NID dam
MATCH_RATIO = 5.0            # ... storages within this factor
SNAP_KM = 2.0                # outlet snap radius
SNAP_TOL = 0.30              # cell drainage area vs the inventory's, +-30%
AREA_MATCH = 2.0             # gauge basin vs lake drainage area, for its runoff
POND_MAX_KM3 = 0.001         # ponds are small (1 Mm3, ~810 acre-ft)
NATURAL_DEPTH_M = 1.0        # active storage above a natural lake's outlet
POND_K_H = 1.0e7             # "no release" for outlet-less ponds (~1,100 years)
MIN_K_H = 3.0
AF_M3 = 1233.48
ACRE_KM2 = 0.00404686
SQMI_KM2 = 2.58999
POND_PURPOSES = ("Fire Protection, Stock, Or Small Fish Pond", "Fish and Wildlife Pond")
FLOOD_PURPOSES = ("Flood Risk Reduction", "Debris Control")
FLOOD_K_H = 14 * 24.0        # flood pools are evacuated within weeks, not years
MAX_NORMAL_RATIO = 5.0
MAX_DEPTH_M = 100.0          # V / area above this: the area is wrong
FILL_DEPTH_M = 10.0          # ... and is re-estimated at this mean depth
DRY_AREA_KM2 = 0.01          # surface of a dry flood pool with no listed area


def load_nid(path: str, west_lon: float) -> pd.DataFrame:
    n = pd.read_csv(path, skiprows=1, low_memory=False)
    num = lambda c: pd.to_numeric(n[c], errors="coerce")
    d = pd.DataFrame({
        "nid_id": n["NID ID"].astype(str), "name": n["Dam Name"].astype(str),
        "lat": num("Latitude"), "lon": num("Longitude"),
        "max_af": num("Max Storage (Acre-Ft)"), "nid_af": num("NID Storage (Acre-Ft)"),
        "normal_af": num("Normal Storage (Acre-Ft)"),
        "area_km2": num("Surface Area (Acres)") * ACRE_KM2,
        "da_km2": num("Drainage Area (Sq Miles)") * SQMI_KM2,
        "purpose": n["Primary Purpose"].fillna("").astype(str),
        "purposes": n["Purposes"].fillna("").astype(str),
        "gate": n["Outlet Gate Type"].fillna("").astype(str),
        "assoc": n["Is Associated Structure?"].fillna("").astype(str)})
    keep = (d.lat.between(24, 50) & d.lon.between(-125, west_lon)
            & (d.assoc.str.lower() != "yes")             # dikes / saddle dams of one pool
            & (d.purpose != "Tailings"))                 # mine impoundments, off-stream
    d = d[keep].copy()
    d["flood"] = d.purpose.isin(FLOOD_PURPOSES)
    d["v_af"] = d.max_af.where(d.max_af > 0, d.nid_af)
    # a max pool > MAX_NORMAL_RATIO x the normal pool is a data error outside
    # flood control (Lemon Dam CO: 487,660 vs 40,146 acre-ft; ~40,000 real)
    bad = ~d.flood & (d.normal_af > 0) & (d.v_af > MAX_NORMAL_RATIO * d.normal_af)
    d["max_suspect"] = bad
    d.loc[bad, "v_af"] = d.normal_af[bad]
    d = d[d.v_af > 0]
    # S0: Normal Storage; missing -> the median normal/max ratio of its purpose,
    # and empty for flood-control / debris dams (normally dry)
    r = (d.normal_af / d.v_af).where((d.normal_af > 0) & (d.normal_af <= d.v_af))
    med = r.groupby(d.purpose).transform("median").fillna(r.median())
    d["s0_frac"] = r.fillna(med.where(~d.flood, 0.0)).clip(0.0, 1.0)
    # outlet-less pond: a pond purpose (NID lists purposes alphabetically, so
    # "Fish and Wildlife Pond" heads big Reclamation reservoirs - McPhee,
    # Vallecito - and only counts when it is the ONLY purpose), small, and no
    # outlet works recorded
    d["pond"] = (((d.purpose == POND_PURPOSES[0])
                  | (d.purposes.str.strip().str.rstrip(";") == POND_PURPOSES[1]))
                 & (d.v_af * AF_M3 / 1e9 <= POND_MAX_KM3) & (d.gate.str.strip() == ""))
    return d.reset_index(drop=True)


def load_hydrolakes(path: str, west_lon: float) -> pd.DataFrame:
    import pyogrio
    h = pyogrio.read_dataframe(path, read_geometry=False, columns=[
        "Hylak_id", "Lake_name", "Country", "Lake_type", "Grand_id", "Lake_area",
        "Vol_total", "Dis_avg", "Wshd_area", "Pour_long", "Pour_lat"])
    h = h[(h.Country == "United States of America") & h.Pour_lat.between(24, 50)
          & h.Pour_long.between(-125, west_lon)]
    return h.rename(columns={"Pour_lat": "lat", "Pour_long": "lon"}).reset_index(drop=True)


def _km(lat1, lon1, lat2, lon2):
    return np.hypot((lat2 - lat1) * 110.574,
                    (lon2 - lon1) * 111.32 * np.cos(np.radians(lat1)))


def match(h: pd.DataFrame, d: pd.DataFrame) -> dict:
    """HydroLAKES row index -> NID row index for reservoirs / lake-controls
    that are the same water body: the largest-storage dam within MATCH_KM of
    the pour point whose storage is within MATCH_RATIO of Vol_total."""
    from scipy.spatial import cKDTree
    xy = lambda la, lo: np.c_[lo * 111.32 * np.cos(np.radians(40.0)), la * 110.574]
    tree = cKDTree(xy(d.lat.values, d.lon.values))
    out, used = {}, set()
    cand = h[(h.Lake_type != 1) | (h.Grand_id > 0)]
    for i, r in cand.sort_values("Vol_total", ascending=False).iterrows():
        idx = tree.query_ball_point(xy(np.array([r.lat]), np.array([r.lon]))[0], MATCH_KM * 1.3)
        best = None
        for j in idx:
            if j in used or _km(r.lat, r.lon, d.lat[j], d.lon[j]) > MATCH_KM:
                continue
            v_mcm = d.v_af[j] * AF_M3 / 1e6
            ratio = v_mcm / max(float(r.Vol_total), 1e-6)
            if 1.0 / MATCH_RATIO <= ratio <= MATCH_RATIO and (best is None or d.v_af[j] > d.v_af[best]):
                best = j
        if best is not None:
            out[i] = best
            used.add(best)
    return out


class Snapper:
    """Outlet cells on the HydroSHEDS 3-arc-second FAM (windowed reads)."""

    def __init__(self, acc_cog: str):
        import rasterio
        self.ds = rasterio.open(acc_cog)
        self.tr = self.ds.transform
        self.nod = float(self.ds.nodata)

    def __call__(self, lat: float, lon: float, ref_km2: float | None):
        """(row, col, fam_km2, rule) or None. With a reference drainage area:
        basinextent.snap_cell (nearest cell within SNAP_KM whose area matches
        +-SNAP_TOL). Without one (NID dams with no drainage area): the
        largest-FAM cell of the 3x3 around the dam point."""
        from rasterio.windows import Window
        from hf_data.basinextent import cell_km2, snap_cell
        tr = self.tr
        hr = int(SNAP_KM / (tr.a * 110.574)) + 2
        hc = int(SNAP_KM / (tr.a * 111.32 * math.cos(math.radians(lat)))) + 2
        r = int((tr.f - lat) / -tr.e)
        c = int((lon - tr.c) / tr.a)
        r0, c0 = max(0, r - hr), max(0, c - hc)
        r1, c1 = min(self.ds.height, r + hr + 1), min(self.ds.width, c + hc + 1)
        if r1 <= r0 or c1 <= c0:
            return None
        win = Window(c0, r0, c1 - c0, r1 - r0)
        acc = self.ds.read(1, window=win).astype("float64")
        acc[(acc == self.nod) | (acc < 0)] = np.nan
        if not np.isfinite(acc).any():
            return None
        wtr = self.ds.window_transform(win)
        if ref_km2 and ref_km2 > 0:
            rc = snap_cell(acc, wtr, lat, lon, ref_km2, radius_km=SNAP_KM, tol=SNAP_TOL)
            rule = "area"
        else:
            rr, cc = r - r0, c - c0
            sub = acc[max(rr - 1, 0):rr + 2, max(cc - 1, 0):cc + 2]
            if not np.isfinite(sub).any():
                return None
            k = int(np.nanargmax(sub))
            rc = (max(rr - 1, 0) + k // sub.shape[1], max(cc - 1, 0) + k % sub.shape[1])
            rule = "max3x3"
        if rc is None:
            return None
        lat_c = wtr.f + (rc[0] + 0.5) * wtr.e
        return (rc[0] + r0, rc[1] + c0,
                float(acc[rc]) * float(cell_km2(lat_c, wtr.a)), rule)


def runoff_lookup(sites_path: str, runoff_txt: str):
    """f(lat, lon, fam_km2) -> (matched, regional) RUNAVE7100 (mm/yr).
    matched: a GAGES-II basin containing the point whose area is within
    AREA_MATCH of the lake's drainage area - the runoff of (about) the lake's
    own catchment; NaN when none. regional: the smallest basin containing the
    point, else the nearest gauge - only for small water bodies, whose own
    catchment no gauge measures (a desert side canyon's runoff applied to the
    Colorado at Glen Canyon was 6x too low)."""
    import geopandas as gpd
    import shapely
    from scipy.spatial import cKDTree
    s = pd.read_parquet(sites_path, columns=["id", "kind", "lat", "lon", "outline_km2", "poly_wkb"])
    s = s[(s.kind == "g") & s.poly_wkb.notna()]
    ro = pd.read_csv(runoff_txt, dtype={"STAID": str}, usecols=["STAID", "RUNAVE7100"])
    ro["STAID"] = ro.STAID.str.zfill(8)
    s = s.merge(ro, left_on="id", right_on="STAID").dropna(subset=["RUNAVE7100"])
    polys = gpd.GeoDataFrame(s[["id", "outline_km2", "RUNAVE7100"]],
                             geometry=shapely.from_wkb(s.poly_wkb.values), crs=4326)
    tree = cKDTree(np.c_[s.lon.values * 0.77, s.lat.values])

    def f(lats, lons, fam):
        n = len(lats)
        pts = gpd.GeoDataFrame({"fam": np.asarray(fam, float)},
                               geometry=gpd.points_from_xy(lons, lats), crs=4326)
        j = gpd.sjoin(pts, polys, predicate="within", how="inner")
        regional = (j.sort_values("outline_km2").groupby(level=0).RUNAVE7100.first()
                    .reindex(range(n)).values.astype(float))
        j["err"] = np.abs(np.log(j.outline_km2 / j.fam.clip(lower=1e-6)))
        jm = j[j.err <= math.log(AREA_MATCH)]
        matched = (jm.sort_values("err").groupby(level=0).RUNAVE7100.first()
                   .reindex(range(n)).values.astype(float))
        miss = ~np.isfinite(regional)
        if miss.any():
            _, k = tree.query(np.c_[np.asarray(lons)[miss] * 0.77, np.asarray(lats)[miss]])
            regional[miss] = s.RUNAVE7100.values[k]
        return matched, regional
    return f


def _git_sha() -> str:
    if os.environ.get("BUILDER_SHA"):
        return os.environ["BUILDER_SHA"]
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sites", required=True,
                    help="sites.parquet of scripts/build_basin_extent.py (GAGES-II polygons)")
    ap.add_argument("--west-lon", type=float, default=WEST_LON)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    d = load_nid(NID_CSV, args.west_lon)
    h = load_hydrolakes(HL_SHP, args.west_lon)
    m = match(h, d)
    print(f"NID {len(d)} dams, HydroLAKES {len(h)} lakes, {len(m)} matched "
          f"({time.time() - t0:.0f}s)", flush=True)

    rows = []
    matched_nid = set(m.values())
    for i, r in h.iterrows():                        # HydroLAKES (alone or matched)
        j = m.get(i)
        if j is not None:
            n = d.loc[j]
            rows.append({"id": f"NID{n.nid_id}", "name": n["name"], "kind": "pond" if n.pond else "dam",
                         "source": "both", "lat": r.lat, "lon": r.lon, "ref_km2": r.Wshd_area,
                         "v_km3": n.v_af * AF_M3 / 1e9, "s0_frac": n.s0_frac,
                         "area_km2": n.area_km2 if n.area_km2 > 0 else r.Lake_area,
                         "dis_avg": r.Dis_avg, "purpose": n.purpose, "gate": n.gate,
                         "hylak_id": int(r.Hylak_id), "grand_id": int(r.Grand_id), "nid_id": n.nid_id,
                         "flood": bool(n.flood)})
        else:
            rows.append({"id": f"HL{int(r.Hylak_id)}", "name": r.Lake_name or "", "kind": "natural",
                         "source": "hydrolakes", "lat": r.lat, "lon": r.lon, "ref_km2": r.Wshd_area,
                         "v_km3": r.Lake_area * NATURAL_DEPTH_M * 1e-3, "s0_frac": 1.0,
                         "area_km2": r.Lake_area, "dis_avg": r.Dis_avg, "purpose": "", "gate": "",
                         "hylak_id": int(r.Hylak_id), "grand_id": int(r.Grand_id), "nid_id": "",
                         "flood": False})
    for j, n in d.iterrows():                        # NID dams HydroLAKES lacks
        if j in matched_nid:
            continue
        rows.append({"id": f"NID{n.nid_id}", "name": n["name"], "kind": "pond" if n.pond else "dam",
                     "source": "nid", "lat": n.lat, "lon": n.lon, "ref_km2": n.da_km2,
                     "v_km3": n.v_af * AF_M3 / 1e9, "s0_frac": n.s0_frac, "area_km2": n.area_km2,
                     "dis_avg": np.nan, "purpose": n.purpose, "gate": n.gate,
                     "hylak_id": 0, "grand_id": 0, "nid_id": n.nid_id, "flood": bool(n.flood)})
    L = pd.DataFrame(rows)
    # NID ids repeat for multi-structure dams: one water body each
    L = L.sort_values("v_km3", ascending=False).drop_duplicates("id").reset_index(drop=True)

    snap = Snapper(ACC_COG)
    got = [snap(r.lat, r.lon, r.ref_km2 if r.ref_km2 == r.ref_km2 else None) for r in L.itertuples()]
    L["snapped"] = [g is not None for g in got]
    for k, col in enumerate(("snap_row", "snap_col", "fam_km2", "snap_rule")):
        L[col] = [g[k] if g else None for g in got]
    tr = snap.tr
    L["snap_lat"] = [tr.f + (g[0] + 0.5) * tr.e if g else np.nan for g in got]
    L["snap_lon"] = [tr.c + (g[1] + 0.5) * tr.a if g else np.nan for g in got]
    print(f"snapped {int(L.snapped.sum())}/{len(L)} ({time.time() - t0:.0f}s)", flush=True)
    L = L[L.snapped].copy()

    # mean inflow: drainage area at the outlet cell x the observed runoff of a
    # gauge basin of about that size; else HydroLAKES Dis_avg; else the
    # regional runoff around the (small) water body
    ro_m, ro_r = runoff_lookup(args.sites, RUNOFF_TXT)(
        L.snap_lat.values, L.snap_lon.values, L.fam_km2.values)
    a_q = L.fam_km2.values * 1e6 / 1000.0 / (365.25 * 86400.0)     # m3/s per mm/yr
    dis = L.dis_avg.fillna(0.0).values
    L["q_source"] = np.where(ro_m > 0, "runave7100_basin",
                             np.where(dis > 0, "dis_avg", "runave7100_regional"))
    L["runoff_mm"] = np.where(ro_m > 0, ro_m, np.where(dis > 0, dis / a_q, ro_r))
    L["q_mean_m3s"] = L.runoff_mm * a_q
    # natural lakes and reservoirs without a surface area: area from V at the
    # median depth of the dams that have both
    dep = (L.v_km3 * 1e9 / (L.area_km2 * 1e6)).where((L.area_km2 > 0) & (L.kind != "natural"))
    # a normally-dry flood pool has (almost) no lake surface: re-estimating
    # its area from V would rain onto desert the land cells already route
    dry = L.flood & (L.s0_frac <= 0)
    L["area_km2"] = L.area_km2.where(L.area_km2 > 0, L.v_km3 * 1e3 / float(dep.median()))
    deep = (L.v_km3 * 1e3 / L.area_km2 > MAX_DEPTH_M) & ~dry   # NID placeholder areas (1 acre)
    L.loc[deep, "area_km2"] = L.v_km3[deep] * 1e3 / FILL_DEPTH_M
    L.loc[dry, "area_km2"] = L.area_km2[dry].where(
        L.area_km2[dry] < L.v_km3[dry] * 1e3 / MAX_DEPTH_M, DRY_AREA_KM2)
    k = L.v_km3 * 1e9 / L.q_mean_m3s / 3600.0
    L["k_h"] = np.where(L.kind == "pond", POND_K_H,
                        np.where(L.flood, np.minimum(k, FLOOD_K_H), k))
    drop = ~(L.k_h >= MIN_K_H) | ~(L.v_km3 > 0)
    print(f"dropped {int(drop.sum())} with K < {MIN_K_H} h or no volume", flush=True)
    L = L[~drop].copy()
    L["s0_km3"] = L.v_km3 * L.s0_frac
    L["snap_row"] = L.snap_row.astype("int64")
    L["snap_col"] = L.snap_col.astype("int64")
    keep = ["id", "name", "kind", "source", "lat", "lon", "snap_lat", "snap_lon", "snap_row",
            "snap_col", "fam_km2", "ref_km2", "snap_rule", "v_km3", "s0_km3", "area_km2",
            "q_mean_m3s", "q_source", "runoff_mm", "k_h", "flood", "purpose", "gate", "hylak_id",
            "grand_id", "nid_id"]
    L = L[keep].sort_values(["snap_row", "snap_col"]).reset_index(drop=True)
    # one lake per cell: the module keys a lake by its cell; keep the largest
    L = L.sort_values("v_km3", ascending=False).drop_duplicates(["snap_row", "snap_col"])
    L = L.sort_values(["snap_row", "snap_col"]).reset_index(drop=True)

    import pyarrow as pa
    import pyarrow.parquet as pq
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    body = L.to_csv(index=False).encode()
    meta = {"table_version": f"{built}-{hashlib.sha1(body).hexdigest()[:10]}", "built": built,
            "builder_sha": _git_sha(), "west_lon": args.west_lon,
            "natural_depth_m": NATURAL_DEPTH_M, "pond_k_h": POND_K_H, "min_k_h": MIN_K_H,
            "flood_k_h": FLOOD_K_H, "max_normal_ratio": MAX_NORMAL_RATIO,
            "snap_tol": SNAP_TOL, "snap_km": SNAP_KM, "lattice": [tr.c, tr.f, tr.a],
            "inputs": [os.path.basename(p) for p in (NID_CSV, HL_SHP, ACC_COG, RUNOFF_TXT)]}
    tbl = pa.Table.from_pandas(L, preserve_index=False)
    tbl = tbl.replace_schema_metadata({**(tbl.schema.metadata or {}),
                                       b"lakes": json.dumps(meta).encode()})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pq.write_table(tbl, args.out, compression="zstd")
    print(f"wrote {args.out}: {len(L)} lakes, version {meta['table_version']} "
          f"({time.time() - t0:.0f}s)")
    print(L.groupby(["kind", "source"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
