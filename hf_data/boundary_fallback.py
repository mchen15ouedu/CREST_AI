"""Versioned, mass-accounted discharge boundaries for the 30 m event solver.

Only nonnested, topologically connected upstream gauges may replace their
catchments. An observation at the event outlet is never an inflow. A branch
without complete observed coverage remains simulated by distributed runoff.
"""
from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import json
import math
import os
import re

import numpy as np

VERSION = 1
POLICY = "usgs-fallback-v1"
FILENAME = "boundary_forcing.json"
_D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
       16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}


def hourly_times(start, end):
    """All forcing switches, including the terminal coverage check."""
    if start.minute or start.second or end.minute or end.second or end <= start:
        raise ValueError("boundary window must have ordered whole-hour UTC endpoints")
    return [start + dt.timedelta(hours=i)
            for i in range(int((end - start).total_seconds() // 3600) + 1)]


def hourly_values(series, times, max_gap_s=3600):
    """Interpolate only within observed brackets <= 1 hour; never extrapolate.

    Return None unless every required hour is supported. This intentionally
    exceeds the historical 95% gate: a missing river is not a valid cut.
    Padded cache rows can bracket endpoints but cannot inflate coverage.
    """
    clean = {}
    for when, q in series:
        try:
            q = float(q)
        except (ValueError, TypeError):
            continue
        if math.isfinite(q) and q >= 0:
            if when.tzinfo is not None:
                when = when.astimezone(dt.timezone.utc).replace(tzinfo=None)
            clean[when] = q
    ts = sorted(clean)
    out = []
    for when in times:
        k = bisect.bisect_left(ts, when)
        if k < len(ts) and ts[k] == when:
            out.append(clean[when])
        elif k == 0 or k == len(ts):
            return None
        else:
            a, b = ts[k - 1], ts[k]
            span = (b - a).total_seconds()
            if span > max_gap_s:
                return None
            w = (when - a).total_seconds() / span
            out.append((1 - w) * clean[a] + w * clean[b])
    # Exact hourly points alone must not conceal a multi-hour data gap.
    for a, b in zip(ts, ts[1:]):
        if a < times[-1] and b > times[0] and (b - a).total_seconds() > max_gap_s:
            return None
    return out


def select_source(sim, observed, min_ratio=0.3):
    """Severe same-gauge disagreement in matched hourly peak OR mean Q.

    .3 preserves the existing severe-underprediction threshold; its reciprocal
    supplies the symmetric overprediction check. No runoff bias scaling.
    """
    if not 0 < min_ratio < 1:
        raise ValueError("boundary minimum ratio must lie strictly between 0 and 1")
    if not observed or any(not math.isfinite(float(q)) or q < 0 for q in observed):
        raise ValueError("observations must be finite nonnegative matched values")
    if (sim is None or len(sim) != len(observed)
            or any(not math.isfinite(float(q)) or q < 0 for q in sim)):
        return "usgs", {"reason": "modeled_q_unavailable", "sim_obs_ratio": None}
    def ratio(a, b):
        return a / b if b > 0 else (1.0 if a == 0 else None)
    sp, op = max(sim), max(observed)
    sm, om = sum(sim) / len(sim), sum(observed) / len(observed)
    rp, rm = ratio(sp, op), ratio(sm, om)
    bad = any(r is None or r < min_ratio or r > 1 / min_ratio for r in (rp, rm))
    return ("usgs" if bad else "crest"), {
        "reason": "severe_q_disagreement" if bad else "modeled_q_within_threshold",
        "sim_peak_m3s": sp, "obs_peak_m3s": op, "sim_obs_ratio": rp,
        "sim_mean_m3s": sm, "obs_mean_m3s": om, "sim_obs_mean_ratio": rm}


def upstream_mask(fdir, cell):
    """Exact ESRI-D8 upstream catchment, including its pour cell."""
    mask = np.zeros(fdir.shape, dtype=bool)
    mask[cell] = True
    todo = [cell]
    nr, nc = fdir.shape
    while todo:
        r, c = todo.pop()
        for code, (dr, dc) in _D8.items():
            rr, cc = r - dr, c - dc
            if (0 <= rr < nr and 0 <= cc < nc and not mask[rr, cc]
                    and fdir[rr, cc] == code):
                mask[rr, cc] = True
                todo.append((rr, cc))
    return mask


def nonnested_cuts(candidates, outlet_id, outlet_rc, fdir):
    """Established nonnested cut set, with one discharge source per cell."""
    from .domain import cut_set
    eligible = [(g, rc) for g, rc in candidates
                if str(g["id"]) != str(outlet_id)
                and max(abs(rc[0] - outlet_rc[0]), abs(rc[1] - outlet_rc[1])) > 1]
    # The legacy helper considers distinct gauge IDs at an identical pinned
    # cell independent. They represent the same water: retain one record,
    # preferring the largest drainage area and then the stable gauge ID.
    eligible.sort(key=lambda item: (-(item[0].get("snap_area") or
                                     item[0].get("area") or 0.0), str(item[0]["id"])))
    unique = {}
    for g, rc in eligible:
        unique.setdefault(tuple(rc), (g, rc))
    return cut_set(outlet_rc, list(unique.values()), fdir)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _write_policy(output_dir, policy, mask_digest):
    topology = {k: policy[k] for k in ("version", "policy", "mode", "dem_res",
                                      "allow_simulated_inlets", "min_sim_obs_ratio")}
    topology["domain_mask_sha256"] = mask_digest
    topology["q_grid"] = policy.get("q_grid")
    topology["subsurface_routing"] = policy.get("subsurface_routing")
    topology["cuts"] = [{k: inlet[k] for k in
                         ("gauge_id", "source", "source_row", "source_col")}
                        for inlet in policy["inlets"]]
    policy["fingerprint"] = _digest(topology)
    policy["domain_mask_sha256"] = mask_digest
    policy.pop("content_sha256", None)
    policy["content_sha256"] = _digest(policy)
    path = os.path.join(output_dir, FILENAME)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(policy, fp, allow_nan=False)
    return policy


def _cut_domain(path, excluded, fdir_transform, fdir_crs, inlets):
    import rasterio
    from rasterio.warp import reproject, Resampling
    with rasterio.open(path) as ds:
        labels, profile = ds.read(1), ds.profile.copy()
    native = 1 / 3600
    if not (abs(profile["transform"].a - native) < 1e-10
            and abs(profile["transform"].e + native) < 1e-10):
        raise ValueError("boundary domain must stay on the native 1 arc-second (~30 m) grid")
    ex = np.zeros(labels.shape, dtype="uint8")
    reproject(excluded.astype("uint8"), ex, src_transform=fdir_transform,
              src_crs=fdir_crs, dst_transform=profile["transform"],
              dst_crs=profile["crs"], resampling=Resampling.nearest)
    labels[ex > 0] = -1
    # The source cell stays as a boundary port, but its lateral generation
    # is suppressed by exclude_lateral_source_cells in the solver interface.
    if not (labels >= 0).any():
        raise ValueError("boundary cuts removed the entire 2-D basin")
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(labels, 1)
    return labels, profile


def prepare(gauge, output_dir, domain, start, end, *, dem_res="1",
            min_ratio=0.3, model=None, log=print):
    """Create a closed forcing bundle; return (policy, updated domain).

    Complete observation coverage is required before cutting a branch.
    Ungauged/uncovered branches keep their runoff-generating upstream area.
    Network/data lookup failures safely select runoff-only, while failures
    after a cut begins abort preparation instead of publishing partial data.
    """
    if str(dem_res) != "1":
        raise ValueError("event boundary policy requires nominal 30 m DEM_RES=1")
    if not 0 < min_ratio < 1:
        raise ValueError("boundary minimum ratio must lie strictly between 0 and 1")
    import rasterio
    from . import basic, basinextent, neighbors, obs
    times = hourly_times(start, end)
    policy = {"version": VERSION, "policy": POLICY, "mode": "runoff-only",
              "dem_res": "1", "allow_simulated_inlets": False,
              "min_sim_obs_ratio": min_ratio, "max_sim_obs_ratio": 1 / min_ratio,
              "max_observation_gap_s": 3600, "required_hour_coverage": 1.0,
              "subsurface_routing": {"mode": "kw", "leaki": 0.1, "under": 1.0,
                                     "channel_km2": 1.0},
              "inlets": [], "excluded_gauges": [], "exclude_lateral_source_cells": []}
    domain_path = os.path.join(output_dir, "domain_huc.tif")
    with rasterio.open(domain_path) as ds:
        initial_labels = ds.read(1)
        native = 1 / 3600
        if not (ds.crs and ds.crs.is_geographic and
                abs(ds.transform.a - native) < 1e-10 and
                abs(ds.transform.e + native) < 1e-10):
            raise ValueError("boundary domain must remain native 1 arc-second (~30 m)")
    # Q lattice metadata belongs in the sidecar even for zero inlets; the
    # solver can validate coordinates instead of assuming matching rasters.
    rx = re.compile(r"q\.(\d{8}_?\d{4})\.([^.]+)\.tif$")
    qpaths = []
    for filename in sorted(os.listdir(output_dir)):
        m = rx.fullmatch(filename)
        if m and (model is None or m[2].lower() == model.lower()):
            when = dt.datetime.strptime(m[1].replace("_", ""), "%Y%m%d%H%M")
            qpaths.append((when, os.path.join(output_dir, filename)))
    candidates = []
    try:
        ext = basinextent.ef5_extent(gauge)
        basic_dir = basic.store_dir(ext["bounds"])
        gs = {**gauge, "lat": ext["snap"]["lat"], "lon": ext["snap"]["lon"],
              "area": ext["snap"]["area"]}
        with rasterio.open(os.path.join(basic_dir, "fdir_clip.tif")) as ds:
            fdir, fdtr, fdcrs = ds.read(1), ds.transform, ds.crs
            if ds.nodata is not None:
                fdir = np.where(fdir == ds.nodata, 0, fdir)
        outlet_rc = basinextent.cell_of(fdtr, gs["lat"], gs["lon"])
        upstream = neighbors.upstream_gauges(gs, ext["bounds"], basic_dir,
                                              max_gauges=4096)
        if not qpaths:
            raise ValueError("no routed Q grid lattice for upstream-gauge mapping")
        with rasterio.open(qpaths[0][1]) as ds:
            qtr, qcrs, qshape = ds.transform, ds.crs, ds.shape
        policy["q_grid"] = {"transform": list(qtr)[:6], "crs": str(qcrs),
                            "shape": list(qshape)}
        if qcrs != fdcrs:
            raise ValueError("Q and D8 lattice CRSs differ")
        if abs(qtr.a - fdtr.a) > 1e-10 or abs(qtr.e - fdtr.e) > 1e-10:
            raise ValueError("Q and D8 lattice cell sizes differ")
        for offset in ((qtr.c - fdtr.c) / fdtr.a, (qtr.f - fdtr.f) / fdtr.e):
            if abs(offset - round(offset)) > 1e-6:
                raise ValueError("Q and D8 lattice origins are not aligned")
        for g in upstream:
            if str(g["id"]) == str(gauge["id"]):
                continue
            lat, lon = g["snap_lat"], g["snap_lon"]
            rc = basinextent.cell_of(fdtr, lat, lon)
            qr, qc = basinextent.cell_of(qtr, lat, lon)
            if not (0 <= qr < qshape[0] and 0 <= qc < qshape[1]):
                policy["excluded_gauges"].append({"gauge_id": g["id"], "reason": "outside_q_grid"})
                continue
            # A qualifying cut must actually lie inside this worker basin.
            with rasterio.open(domain_path) as ds:
                dr, dc = ds.index(lon, lat)
            if not (0 <= dr < initial_labels.shape[0] and 0 <= dc < initial_labels.shape[1]
                    and initial_labels[dr, dc] >= 0):
                policy["excluded_gauges"].append({"gauge_id": g["id"], "reason": "outside_active_domain"})
                continue
            observed = hourly_values(obs.get_series(g["id"], start, end), times)
            if observed is None:
                policy["excluded_gauges"].append({"gauge_id": g["id"],
                    "reason": "incomplete_observations_keep_full_branch",
                    "note": "includes uncovered future hours; no extrapolation"})
                continue
            candidates.append(({**g, "_observed": observed, "_qrc": (qr, qc)}, rc))
    except Exception as exc:
        log(f"boundary observation/network lookup unavailable ({type(exc).__name__}: {exc}); runoff-only")
        policy["lookup_error"] = f"{type(exc).__name__}: {exc}"
        candidates = []

    if candidates:
        cuts = nonnested_cuts(candidates, gauge["id"], outlet_rc, fdir)
        selected = {g["id"] for g, _ in cuts}
        policy["excluded_gauges"].extend(
            {"gauge_id": g["id"], "reason": "nested_or_same_cell_upstream_gauge"}
            for g, _ in candidates if g["id"] not in selected)
        excluded = np.zeros(fdir.shape, dtype=bool)
        for g, rc in cuts:
            qr, qc = g["_qrc"]
            sim_series = []
            for when, path in qpaths:
                with rasterio.open(path) as ds:
                    if ds.transform != qtr or ds.crs != qcrs or ds.shape != qshape:
                        raise ValueError("routed Q grid lattice changed during event")
                    value = float(ds.read(1, window=((qr, qr + 1), (qc, qc + 1)))[0, 0])
                    if ds.nodata is not None and value == ds.nodata:
                        continue
                sim_series.append((when, value))
            simulated = hourly_values(sim_series, times)
            source, diagnostics = select_source(simulated, g["_observed"], min_ratio)
            values = g["_observed"] if source == "usgs" else simulated
            inlet = {"gauge_id": g["id"], "source": source,
                     "lat": g["snap_lat"], "lon": g["snap_lon"],
                     "source_row": qr, "source_col": qc,
                     "times": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times],
                     "q_m3s": values, **diagnostics}
            policy["inlets"].append(inlet)
            policy["exclude_lateral_source_cells"].append([qr, qc])
            branch = upstream_mask(fdir, rc)
            branch[rc] = False
            excluded |= branch
            log(f"boundary {g['id']}: {source} discharge; {diagnostics['reason']}; upstream runoff removed")
        if policy["inlets"]:
            policy["mode"] = "gauged-incremental"
            labels, profile = _cut_domain(domain_path, excluded, fdtr, fdcrs,
                                           policy["inlets"])
            from rasterio.features import shapes
            from shapely.geometry import shape, mapping
            from shapely.ops import unary_union
            geom = unary_union([shape(s) for s, v in shapes(
                (labels >= 0).astype("uint8"), transform=profile["transform"]) if v == 1])
            geo_path = os.path.join(output_dir, "domain.geojson")
            with open(geo_path, encoding="utf-8") as fp:
                geo = json.load(fp)
            geo["union"] = mapping(geom)
            geo["boundary_cuts"] = [{k: i[k] for k in ("gauge_id", "source", "lat", "lon")}
                                    for i in policy["inlets"]]
            with open(geo_path, "w", encoding="utf-8") as fp:
                json.dump(geo, fp)
            lat = profile["transform"].f + profile["transform"].e * labels.shape[0] / 2
            area = ((labels >= 0).sum() * (native * 111.132) ** 2
                    * math.cos(math.radians(lat)))
            domain = dict(domain, n_active=int((labels >= 0).sum()),
                          parent_area_km2=domain.get("area_km2"), area_km2=round(float(area), 1),
                          boundary_mode="gauged-incremental",
                          boundary_cut_ids=[i["gauge_id"] for i in policy["inlets"]],
                          area_basis="incremental mask; HUC12 area describes parent basin")
            # Preserve the parent HUC basin ring for context; domain.geojson
            # and domain_huc.tif carry the actual incremental geometry.
            initial_labels = labels
    mask_digest = hashlib.sha256(initial_labels.tobytes()).hexdigest()
    policy = _write_policy(output_dir, policy, mask_digest)
    domain = dict(domain, boundary_mode=policy["mode"],
                  boundary_fingerprint=policy["fingerprint"])
    log(f"2-D boundary policy: {policy['mode']}, {len(policy['inlets'])} gauge cut(s); nominal 30 m")
    return policy, domain
