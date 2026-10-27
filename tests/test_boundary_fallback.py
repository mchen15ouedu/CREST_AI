import datetime as dt
import json
import sys
import tarfile
import types

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from hf_data import boundary_fallback as bf


T0 = dt.datetime(2026, 10, 23)


def test_matched_hours_missing_future_gap_and_padding():
    hours = bf.hourly_times(T0, T0 + dt.timedelta(hours=3))
    complete = [(t, 10) for t in hours]
    assert bf.hourly_values(complete, hours) == [10] * 4
    assert bf.hourly_values(complete[:-1], hours) is None
    assert bf.hourly_values([complete[0], complete[-1]], hours) is None
    padding = [(T0 - dt.timedelta(days=1), 999), *complete[:-1]]
    assert bf.hourly_values(padding, hours) is None
    assert bf.hourly_values([(t, float("nan")) for t in hours], hours) is None


@pytest.mark.parametrize("sim,observed,source", [
    ([10, 10], [100, 100], "usgs"),
    ([400, 400], [100, 100], "usgs"),
    ([50, 50], [100, 100], "crest"),
    ([0, 0], [0, 0], "crest"),
    ([1, 1], [0, 0], "usgs"),
    ([0, 0], [1, 1], "usgs"),
    (None, [10, 10], "usgs"),
    ([float("nan"), 10], [10, 10], "usgs"),
])
def test_severe_under_and_over_prediction_and_zero(sim, observed, source):
    assert bf.select_source(sim, observed)[0] == source


def network():
    fd = np.full((9, 7), 4, dtype="int16")
    fd[:, :3] = 1
    fd[:, 4:] = 16
    return fd


def test_nonnested_cuts_never_inject_outlet():
    fd = network()
    cuts = bf.nonnested_cuts([({"id": "upper"}, (1, 3)),
                             ({"id": "lower"}, (4, 3)),
                             ({"id": "outlet"}, (8, 3))], "outlet", (8, 3), fd)
    assert [g["id"] for g, _ in cuts] == ["lower"]
    mask = bf.upstream_mask(fd, (4, 3))
    assert mask[:5].all()
    assert not mask[5:].any()


def test_two_gauges_at_one_source_cell_never_double_inject():
    cuts = bf.nonnested_cuts([({"id": "small", "area": 1}, (4, 3)),
                             ({"id": "large", "area": 2}, (4, 3))],
                            "outlet", (8, 3), network())
    assert [(g["id"], rc) for g, rc in cuts] == [("large", (4, 3))]


def test_fingerprint_stable_across_hourly_data(tmp_path):
    p = {"version": 1, "policy": bf.POLICY, "mode": "gauged-incremental",
         "dem_res": "1", "allow_simulated_inlets": False, "min_sim_obs_ratio": .3,
         "q_grid": {"transform": [1 / 1200, 0, -107, 0, -1 / 1200, 38],
                    "shape": [9, 7], "crs": "EPSG:4326"},
         "inlets": [{"gauge_id": "lower", "source": "usgs", "source_row": 4,
                     "source_col": 3, "times": ["old"], "q_m3s": [100]}]}
    first = bf._write_policy(tmp_path, p, "same-mask")
    fp, content = first["fingerprint"], first["content_sha256"]
    first.pop("content_sha256")
    first["inlets"][0]["times"] = ["new"]
    first["inlets"][0]["q_m3s"] = [200]
    second = bf._write_policy(tmp_path, first, "same-mask")
    assert second["fingerprint"] == fp
    assert second["content_sha256"] != content
    second["q_grid"]["transform"][2] = -108
    moved = bf._write_policy(tmp_path, second, "same-mask")
    assert moved["fingerprint"] != fp


def fake_data(tmp_path, monkeypatch, incomplete=False):
    basic_dir = tmp_path / "basic"
    basic_dir.mkdir()
    out_dir = tmp_path / "forcing"
    out_dir.mkdir()
    tr = from_origin(-107, 38, 1 / 1200, 1 / 1200)
    def point(rc):
        lon, lat = tr * (rc[1] + .5, rc[0] + .5)
        return lat, lon
    def write(path, data, transform, nodata=-9999):
        with rasterio.open(path, "w", driver="GTiff", height=data.shape[0],
                           width=data.shape[1], count=1, dtype=str(data.dtype),
                           crs="EPSG:4326", transform=transform, nodata=nodata) as ds:
            ds.write(data, 1)
    write(basic_dir / "fdir_clip.tif", network(), tr)
    for i in range(3):
        when = T0 + dt.timedelta(hours=i)
        write(out_dir / f"q.{when:%Y%m%d_%H%M}.crest.tif",
              np.full((9, 7), 10, dtype="float32"), tr)
    dtr = from_origin(-107, 38, 1 / 3600, 1 / 3600)
    write(out_dir / "domain_huc.tif", np.zeros((27, 21), dtype="int16"), dtr, -1)
    (out_dir / "domain.geojson").write_text(json.dumps({"features": [], "union": {}}))
    lat, lon = point((8, 3))
    gauge = {"id": "outlet", "lat": lat, "lon": lon, "area": 5}
    ext = {"bounds": (-107, 37, -106, 38), "snap": {"lat": lat, "lon": lon, "area": 5}}
    upstream = []
    for gid, rc in [("upper", (1, 3)), ("lower", (4, 3)), ("outlet", (8, 3))]:
        lat, lon = point(rc)
        upstream.append({"id": gid, "snap_lat": lat, "snap_lon": lon})
    basic = types.SimpleNamespace(store_dir=lambda bounds: str(basic_dir))
    extent = types.SimpleNamespace(ef5_extent=lambda gauge: ext,
        cell_of=lambda transform, lat, lon: (int(np.floor((~transform * (lon, lat))[1])),
                                              int(np.floor((~transform * (lon, lat))[0]))))
    neighbors = types.SimpleNamespace(upstream_gauges=lambda *a, **kw: upstream, _D8=bf._D8)
    def series(*args):
        return [(T0 + dt.timedelta(hours=i), 100) for i in range(2 if incomplete else 3)]
    obs = types.SimpleNamespace(get_series=series)
    for name, module in [("basic", basic), ("basinextent", extent),
                         ("neighbors", neighbors), ("obs", obs)]:
        monkeypatch.setitem(sys.modules, f"hf_data.{name}", module)
        monkeypatch.setattr(sys.modules["hf_data"], name, module, raising=False)
    # domain.py imports its neighbor module once; update it for this synthetic topology.
    from hf_data import domain as domain_module
    monkeypatch.setattr(domain_module, "neighbors", neighbors)
    return gauge, out_dir


def test_prepare_cuts_only_connected_nonnested_branch_and_keeps_30m(tmp_path, monkeypatch):
    gauge, out = fake_data(tmp_path, monkeypatch)
    policy, domain = bf.prepare(gauge, str(out), {"area_km2": 5}, T0,
                                T0 + dt.timedelta(hours=2), model="crest", log=lambda s: None)
    assert policy["mode"] == "gauged-incremental"
    assert [(i["gauge_id"], i["source"]) for i in policy["inlets"]] == [("lower", "usgs")]
    assert policy["exclude_lateral_source_cells"] == [[4, 3]]
    assert policy["allow_simulated_inlets"] is False
    with rasterio.open(out / "domain_huc.tif") as ds:
        labels = ds.read(1)
        assert ds.transform.a == pytest.approx(1 / 3600)
        assert (labels[:12] == -1).all()
        assert (labels[12:15, 9:12] == 0).all()  # Q port stays; lateral forcing excluded
        assert (labels[15:] == 0).all()
    assert domain["n_active"] == 12 * 21 + 9


def test_prepare_incomplete_future_keeps_full_basin_runoff_only(tmp_path, monkeypatch):
    gauge, out = fake_data(tmp_path, monkeypatch, incomplete=True)
    policy, _ = bf.prepare(gauge, str(out), {}, T0, T0 + dt.timedelta(hours=2),
                           model="crest", log=lambda s: None)
    assert policy["mode"] == "runoff-only"
    assert policy["inlets"] == []
    assert all(g["reason"] == "incomplete_observations_keep_full_branch"
               for g in policy["excluded_gauges"])
    with rasterio.open(out / "domain_huc.tif") as ds:
        assert (ds.read(1) == 0).all()


def test_non_30m_request_rejected():
    with pytest.raises(ValueError, match="30 m"):
        bf.prepare({}, "unused", {}, T0, T0 + dt.timedelta(hours=1), dem_res="13")


def test_changed_q_lattice_aborts_before_cutting(tmp_path, monkeypatch):
    gauge, out = fake_data(tmp_path, monkeypatch)
    path = out / "q.20261023_0100.crest.tif"
    with rasterio.open(path, "r+") as ds:
        ds.transform = from_origin(-107.01, 38, 1 / 1200, 1 / 1200)
    with pytest.raises(ValueError, match="lattice changed"):
        bf.prepare(gauge, str(out), {}, T0, T0 + dt.timedelta(hours=2),
                   model="crest", log=lambda s: None)
    with rasterio.open(out / "domain_huc.tif") as ds:
        assert (ds.read(1) == 0).all()
    assert not (out / bf.FILENAME).exists()


def test_outlet_disagreement_still_queues_and_preserves_prior_results(tmp_path, monkeypatch):
    from hf_data import eventsim
    def set_module(name, module):
        monkeypatch.setitem(sys.modules, f"hf_data.{name}", module)
        monkeypatch.setattr(sys.modules["hf_data"], name, module, raising=False)
    monkeypatch.setitem(sys.modules, "crestimap", types.SimpleNamespace(EventConfig=object, run_event=object))
    def run_gauge(*args, **kwargs):
        (tmp_path / "CREST_output").mkdir()
        yield "meta", {"model": "crest"}
        yield "hydro", {"rows": [{"time": "2026-10-23 00:00", "sim_q": 1, "obs_q": 100}]}
    set_module("pipeline", types.SimpleNamespace(gauge_info=lambda gid: {"id": gid, "lat": 38, "lon": -107},
                                                 run_gauge=run_gauge))
    set_module("nowcaststore", types.SimpleNamespace())
    set_module("basinextent", types.SimpleNamespace(ef5_extent=lambda g: {
        "source": "test", "carved_km2": 10, "cells": 9}))
    domain = {"n_hucs": 1, "hucs": [{"huc12": "123"}], "basin": [[38, -107]],
              "bbox": [-107, 37, -106, 38], "n_bbox": 9}
    set_module("hucdomain", types.SimpleNamespace(build_domain=lambda *a, **kw: domain, DomainError=RuntimeError))
    def forbid_delete(*args, **kw):
        pytest.fail("outlet Q advisory must never delete or disown a result")
    set_module("eventstore", types.SimpleNamespace(drop_events=forbid_delete, note_missed=forbid_delete))
    captured = {}
    def enqueue(ev, spec, output_dir, **kw):
        captured.update(spec)
        return True
    set_module("eventqueue", types.SimpleNamespace(mode=lambda: "on", enqueue=enqueue))
    policy = {"version": 1, "policy": bf.POLICY, "mode": "runoff-only", "fingerprint": "topology",
              "domain_mask_sha256": "mask", "content_sha256": "content",
              "subsurface_routing": {"mode": "kw"},
              "inlets": [], "excluded_gauges": []}
    monkeypatch.setattr(bf, "prepare", lambda *a, **kw: (policy, domain))
    monkeypatch.setattr(eventsim.tempfile, "mkdtemp", lambda **kw: str(tmp_path))
    result = eventsim.run_one("outlet", T0, defer=True)
    assert result["deferred"] is True
    assert captured["worker_protocol"] == 2
    assert captured["required_engine_prefix"] == "jetstream2:"
    assert captured["trigger"]["outlet_q_advisory"]["action"] == "continue_runoff_driven_2d"
    assert captured["interflow"] == "kw"


def test_queue_bundles_only_explicit_sidecars(tmp_path, monkeypatch):
    from hf_data import eventqueue, eventstore
    for name in ("runoff.20261023_0000.crest.tif", "domain_huc.tif", "domain.geojson",
                 bf.FILENAME, "credentials.txt", "unrelated.json"):
        (tmp_path / name).write_text("test")
    (tmp_path / bf.FILENAME).write_text(json.dumps(
        {"version": 1, "fingerprint": "topology", "content_sha256": "content"}))
    captured = []
    def create_commit(**kw):
        for op in kw["operations"]:
            if op.path_in_repo.endswith(".forcing.tar.gz"):
                with tarfile.open(op.path_or_fileobj) as tf:
                    captured.extend(tf.getnames())
    api = types.SimpleNamespace(list_repo_files=lambda *a, **kw: [], create_commit=create_commit)
    monkeypatch.setattr(eventstore, "_api", lambda: api)
    monkeypatch.setattr(eventqueue, "_wake_workers", lambda log: None)
    class Add:
        def __init__(self, path, value):
            self.path_in_repo, self.path_or_fileobj = path, value
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        CommitOperationAdd=Add, CommitOperationDelete=object))
    spec = {"worker_protocol": 2, "dem_res": "1", "forcing_fingerprint": "topology",
            "boundary_content_sha256": "content"}
    assert eventqueue.enqueue("2026102300_test", spec, str(tmp_path), log=lambda s: None)
    assert set(captured) == {"runoff.20261023_0000.crest.tif", "domain_huc.tif", "domain.geojson", bf.FILENAME}
    spec["forcing_fingerprint"] = "different-topology"
    assert not eventqueue.enqueue("2026102300_test", spec, str(tmp_path), log=lambda s: None)
