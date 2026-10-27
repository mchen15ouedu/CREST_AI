import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from conftest import RES, write_grid

WEST, NORTH = -110.0, 38.0
T0 = datetime(2026, 10, 20, 16)


def _tr():
    from rasterio.transform import from_origin
    return from_origin(WEST, NORTH, RES, RES)


def _centre(r, c):
    return NORTH - (r + 0.5) * RES, WEST + (c + 0.5) * RES


@pytest.fixture
def table(monkeypatch):
    """Three water bodies: one on a node, one on the raster but off the
    basin, one off the raster."""
    from hf_data import lakes
    rows = []
    for i, (r, c) in enumerate([(1, 2), (3, 3), (40, 40)]):
        la, lo = _centre(r, c)
        rows.append({"id": f"NID{i}", "name": f"L{i}", "kind": "dam", "snap_lat": la,
                     "snap_lon": lo, "v_km3": 0.004, "s0_km3": 0.003, "area_km2": 1.1,
                     "k_h": 1000.0})
    monkeypatch.setattr(lakes, "_tbl", (pd.DataFrame(rows), {"table_version": "tv1"}))
    return lakes


def _nodes():
    m = np.zeros((5, 5), dtype=bool)
    m[1:3, 1:4] = True
    return m


def test_region(monkeypatch):
    from hf_data import lakes
    monkeypatch.delenv("CREST_LAKES", raising=False)
    assert not lakes.enabled(-111.0)                  # off until the host's EF5 has the fix
    monkeypatch.setenv("CREST_LAKES", "west")
    assert lakes.enabled(-111.0) and not lakes.enabled(-97.0)
    monkeypatch.setenv("CREST_LAKES", "all")
    assert lakes.enabled(-97.0)
    monkeypatch.setenv("CREST_LAKES", "bogus")
    assert not lakes.enabled(-111.0)


def test_only_lakes_on_the_runs_nodes(table):
    got = table.for_run(_nodes(), _tr(), (2.0, 0.5))
    assert list(got.id) == ["NID0"]
    assert (got.r[0], got.c[0]) == (1, 2)
    assert got.th_volume_mm3[0] == pytest.approx(8.0)  # 0.004 km3 = 4 Mm3, x2
    assert got.s0_m3[0] == pytest.approx(6.0e6)        # S0 keeps its fraction of V
    assert got.klake_h[0] == pytest.approx(500.0)


def test_csv_volume_in_million_m3(table, tmp_path):
    got = table.for_run(_nodes(), _tr())
    p = table.write_csv(str(tmp_path / "lakes.csv"), got)
    lines = open(p).read().splitlines()
    assert lines[0] == "name,lat,lon,th_volume,area,klake"
    name, lat, lon, vol, area, k = lines[1].split(",")
    assert name == "NID0" and float(vol) == pytest.approx(4.0) and float(k) == 1000.0


def test_multipliers():
    from hf_data import lakes
    assert lakes.multipliers(None, None) == (1.0, 1.0)
    assert lakes.multipliers({"lake_v": 2}, {"lake_v": 3, "lake_k": 4}) == (2.0, 4.0)


def _dem(tmp_path):
    p = str(tmp_path / "dem_clip.tif")
    write_grid(p, np.zeros((5, 5)), WEST, NORTH)
    return p


def _read(p):
    import rasterio
    with rasterio.open(p) as ds:
        return ds.read(1)


def test_cold_start_at_normal_pool(table, tmp_path):
    sd = str(tmp_path / "states")
    got = table.for_run(_nodes(), _tr())
    info = table.prepare_states(sd, T0, got, _dem(tmp_path))
    assert info == {"kept": 0, "initial": 1}
    st = _read(os.path.join(sd, "lake_storage_20261020_1600.tif"))
    assert st[1, 2] == pytest.approx(3.0e6) and st.sum() == pytest.approx(3.0e6)
    assert _read(os.path.join(sd, "lake_outflow_20261020_1600.tif")).sum() == 0
    sig = json.load(open(os.path.join(sd, "lake_set_20261020_1600.json")))
    assert sig["table"] == "tv1" and sig["ids"] == ["NID0"]


def test_saved_storage_kept_only_for_the_same_lake_set(table, tmp_path):
    sd = str(tmp_path / "states")
    os.makedirs(sd)
    got = table.for_run(_nodes(), _tr())
    s = np.zeros((5, 5))
    s[1, 2] = 1.5e6                              # what EF5 saved for NID0
    write_grid(os.path.join(sd, "lake_storage_20261020_1600.tif"), s, WEST, NORTH)
    write_grid(os.path.join(sd, "lake_outflow_20261020_1600.tif"), np.zeros((5, 5)), WEST, NORTH)
    assert table.tag_states(sd, got) == 1
    assert table.prepare_states(sd, T0, got, _dem(tmp_path))["kept"] == 1
    assert _read(os.path.join(sd, "lake_storage_20261020_1600.tif"))[1, 2] == pytest.approx(1.5e6)
    # other multipliers: a different lake set -> back to the normal pool
    got2 = table.for_run(_nodes(), _tr(), (2.0, 1.0))
    assert table.prepare_states(sd, T0, got2, _dem(tmp_path), (2.0, 1.0))["kept"] == 0
    assert _read(os.path.join(sd, "lake_storage_20261020_1600.tif"))[1, 2] == pytest.approx(6.0e6)


def test_staging_ignores_lake_grids_for_coverage(tmp_path):
    from hf_data import runstates
    sh, rd = str(tmp_path / "shared"), str(tmp_path / "run")
    os.makedirs(sh)
    write_grid(os.path.join(sh, "crest_SM_20261020_1600.tif"), np.ones((5, 5)), WEST, NORTH)
    lk = np.full((5, 5), -9999.0)
    lk[1, 2] = 5.0                                     # a lake grid: one valid cell
    write_grid(os.path.join(sh, "lake_storage_20261020_1600.tif"), lk, WEST, NORTH)
    with open(os.path.join(sh, "lake_set_20261020_1600.json"), "w") as fh:
        fh.write("{}")
    with open(os.path.join(sh, "lake_relationships_20261020_1600.txt"), "w") as fh:
        fh.write("stale")
    ok, cov = runstates.stage(sh, rd, T0, _nodes(), _tr())
    assert ok and cov == 1.0
    staged = sorted(os.listdir(rd))
    assert "lake_set_20261020_1600.json" in staged and "lake_storage_20261020_1600.tif" in staged
    assert "lake_relationships_20261020_1600.txt" not in staged     # EF5 rebuilds it


def test_control_switches_the_module_on(tmp_path):
    from hf_data.control import ControlSpec, Gauge, build_control
    spec = ControlSpec(control_path=str(tmp_path / "control.txt"),
                       time_begin=T0, time_end=datetime(2026, 10, 22), timestep="1h",
                       basic_dir=str(tmp_path), precip_dir=str(tmp_path), pet_dir=str(tmp_path),
                       output_dir=str(tmp_path / "out"), gauges=[Gauge("09337500", -111.5, 37.8, None)],
                       state_dir=str(tmp_path / "st"), warmup_start=datetime(2026, 9, 20),
                       lakes_csv=str(tmp_path / "lakes.csv"))
    build_control(spec)
    main = open(spec.control_path).read()
    warm = open(tmp_path / "control_warmup.txt").read()
    for txt in (main, warm):
        assert f"LAKES_CSV={os.path.abspath(spec.lakes_csv)}" in txt
        assert "LAKEMODULE=true" in txt
    spec.lakes_csv = None
    build_control(spec)
    assert "LAKE" not in open(spec.control_path).read()
