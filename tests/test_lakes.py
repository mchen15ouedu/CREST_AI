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
    assert lines[0] == "name,lat,lon,th_volume,area,klake,exact"
    name, lat, lon, vol, area, k, exact = lines[1].split(",")
    assert name == "NID0" and float(vol) == pytest.approx(4.0) and float(k) == 1000.0
    assert exact == "1"                          # EF5 keeps the lake on this cell
    assert (float(lat), float(lon)) == pytest.approx(_centre(1, 2))


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


def test_bad_saved_storage_starts_at_the_normal_pool(table, tmp_path):
    sd = str(tmp_path / "states")
    os.makedirs(sd)
    got = table.for_run(_nodes(), _tr())
    for bad in (-9999.0, np.nan, -1.0):
        s = np.zeros((5, 5))
        s[1, 2] = bad                            # EF5's nodata / a NaN / a negative volume
        write_grid(os.path.join(sd, "lake_storage_20261020_1600.tif"), s, WEST, NORTH,
                   nodata=None)
        write_grid(os.path.join(sd, "lake_outflow_20261020_1600.tif"), np.zeros((5, 5)),
                   WEST, NORTH)
        table.tag_states(sd, got)
        assert table.prepare_states(sd, T0, got, _dem(tmp_path)) == {"kept": 0, "initial": 1}
        assert _read(os.path.join(sd, "lake_storage_20261020_1600.tif"))[1, 2] == pytest.approx(3.0e6)


def test_lake_set_sidecar_never_written_through_a_hard_link(table, tmp_path):
    sh, rd = tmp_path / "shared", tmp_path / "run"
    sh.mkdir()
    rd.mkdir()
    shared = sh / "lake_set_20261020_1600.json"
    shared.write_text(json.dumps({"table": "other", "mult": [1.0, 1.0], "ids": ["X"]}))
    os.link(shared, rd / "lake_set_20261020_1600.json")      # as runstates.stage links it
    got = table.for_run(_nodes(), _tr())
    table.prepare_states(str(rd), T0, got, _dem(tmp_path))
    assert json.loads(shared.read_text())["ids"] == ["X"]     # the shared file is untouched
    assert json.loads((rd / "lake_set_20261020_1600.json").read_text())["ids"] == ["NID0"]
    os.link(shared, rd / "lake_set_20261020_1700.json")
    write_grid(str(rd / "lake_storage_20261020_1700.tif"), np.zeros((5, 5)), WEST, NORTH)
    table.tag_states(str(rd), got)
    assert json.loads(shared.read_text())["ids"] == ["X"]
    assert not [f for f in os.listdir(rd) if f.endswith(".tmp")]


def test_commit_moves_a_lake_set_only_with_its_grid(tmp_path):
    from hf_data import runstates
    sh, rd = tmp_path / "shared", tmp_path / "run"
    sh.mkdir()
    rd.mkdir()
    # 16:00: the grid is a staged hard link (skipped) - its sidecar must stay
    write_grid(str(sh / "lake_storage_20261020_1600.tif"), np.zeros((5, 5)), WEST, NORTH)
    os.link(sh / "lake_storage_20261020_1600.tif", rd / "lake_storage_20261020_1600.tif")
    (sh / "lake_set_20261020_1600.json").write_text("old")
    (rd / "lake_set_20261020_1600.json").write_text("new")
    # 17:00: new grid + sidecar move together; 18:00: a sidecar without a grid stays
    write_grid(str(rd / "lake_storage_20261020_1700.tif"), np.ones((5, 5)), WEST, NORTH)
    (rd / "lake_set_20261020_1700.json").write_text("new17")
    (rd / "lake_set_20261020_1800.json").write_text("orphan")
    # 19:00: no clobber keeps the shared grid - and so its sidecar
    write_grid(str(sh / "lake_storage_20261020_1900.tif"), np.zeros((5, 5)), WEST, NORTH)
    (sh / "lake_set_20261020_1900.json").write_text("old19")
    write_grid(str(rd / "lake_storage_20261020_1900.tif"), np.ones((5, 5)), WEST, NORTH)
    (rd / "lake_set_20261020_1900.json").write_text("new19")
    assert runstates.commit(str(rd), str(sh), clobber=False) == 2
    assert (sh / "lake_set_20261020_1600.json").read_text() == "old"
    assert (sh / "lake_set_20261020_1700.json").read_text() == "new17"
    assert not (sh / "lake_set_20261020_1800.json").exists()
    assert (sh / "lake_set_20261020_1900.json").read_text() == "old19"


def test_bundle_keeps_lake_storage_and_its_sidecar(tmp_path):
    import rasterio
    from hf_data import statebundle
    sd, out = tmp_path / "s", tmp_path / "u"
    sd.mkdir()
    s = np.zeros((5, 5))
    s[1, 2] = 1.234567e10                        # m3: float16 tops out at 65504
    write_grid(str(sd / "lake_storage_20261020_1600.tif"), s, WEST, NORTH)
    write_grid(str(sd / "lake_outflow_20261020_1600.tif"), np.full((5, 5), 0.3), WEST, NORTH)
    write_grid(str(sd / "crest_SM_20261020_1600.tif"), np.full((5, 5), 0.3), WEST, NORTH)
    (sd / "lake_set_20261020_1600.json").write_text('{"ids": ["NID0"]}')
    statebundle.unpack(statebundle.pack_dir(str(sd)), str(out))
    with rasterio.open(out / "lake_storage_20261020_1600.tif") as ds:
        assert ds.read(1)[1, 2] == np.float32(1.234567e10)
    with rasterio.open(out / "lake_outflow_20261020_1600.tif") as ds:
        assert ds.read(1)[0, 0] == np.float32(0.3)              # lossless
    with rasterio.open(out / "crest_SM_20261020_1600.tif") as ds:
        assert ds.read(1)[0, 0] == np.float32(np.float16(0.3))  # the others stay float16
    assert (out / "lake_set_20261020_1600.json").read_text() == '{"ids": ["NID0"]}'
    # a local lake grid wins - and the bundle's sidecar must not vouch for it
    out2 = tmp_path / "u2"
    out2.mkdir()
    write_grid(str(out2 / "lake_storage_20261020_1600.tif"), np.zeros((5, 5)), WEST, NORTH)
    statebundle.unpack(statebundle.pack_dir(str(sd)), str(out2))
    assert not (out2 / "lake_set_20261020_1600.json").exists()


def test_mosaic_never_pastes_lake_grids(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    from hf_data import conusstate, statebundle
    sd = tmp_path / "s"
    sd.mkdir()
    write_grid(str(sd / "crest_SM_20261020_1600.tif"), np.full((5, 5), 0.5), WEST, NORTH)
    write_grid(str(sd / "kwr_IR_20261020_1600.tif"), np.full((5, 5), 0.1), WEST, NORTH)
    write_grid(str(sd / "lake_storage_20261020_1600.tif"), np.full((5, 5), 7.0), WEST, NORTH)
    (sd / "lake_set_20261020_1600.json").write_text("{}")
    bpath = str(tmp_path / "b.pqf")
    statebundle.pack_dir(str(sd), out_path=bpath)

    def _open(key):
        pf = pq.ParquetFile(bpath)
        md = pf.schema_arrow.metadata
        return pf, {"nrows": int(md[b"nrows"]), "ncols": int(md[b"ncols"]),
                    "xll": float(md[b"xllcorner"]), "yll": float(md[b"yllcorner"]),
                    "cell": float(md[b"cellsize"]), "nodata": float(md[b"nodata"])}

    monkeypatch.setattr(conusstate, "_open", _open)
    monkeypatch.setattr(conusstate, "CELL", RES)
    dst = {"xll": WEST, "yll": NORTH - 5 * RES, "nrows": 5, "ncols": 5}
    res = conusstate.stitch(dst, ["09337500_crest"], T0, workers=1)
    assert sorted(res["grids"]) == ["crest_SM", "kwr_IR"]
    res["grids"]["lake_storage"] = np.full((5, 5), 7.0, dtype="float32")    # even if handed one
    out = tmp_path / "o"
    conusstate.write_states(res, dst, str(out))
    assert sorted(os.listdir(out)) == ["crest_SM_20261020_1600.tif", "kwr_IR_20261020_1600.tif"]


def test_cache_key_separates_lake_runs(table):
    from hf_data import pipeline
    none = table.for_run(np.zeros((5, 5), dtype=bool), _tr())
    some = table.for_run(_nodes(), _tr())
    assert table.cache_sig(None, (1.0, 1.0)) is None and table.cache_sig(none, (1.0, 1.0)) is None
    # a basin without lakes keeps the plain (fleet) key
    assert pipeline.config_tag(None, None, "auto", lake=table.cache_sig(none, (1.0, 1.0))) == ""
    on = pipeline.config_tag(None, None, "auto", lake=table.cache_sig(some, (1.0, 1.0)))
    assert on and on != ""
    # other multipliers or another table version: another key
    assert pipeline.config_tag(None, None, "auto", lake=table.cache_sig(some, (2.0, 1.0))) != on
    table._tbl = (table._tbl[0], {"table_version": "tv2"})
    assert pipeline.config_tag(None, None, "auto", lake=table.cache_sig(some, (1.0, 1.0))) != on
    # with other parts too
    ov = {"wm": 2.0}
    assert pipeline.config_tag(None, ov, "auto", lake=table.cache_sig(some, (1.0, 1.0))) \
        != pipeline.config_tag(None, ov, "auto")


def test_empty_warm_up_leaves_lakes_at_the_normal_pool(table, tmp_path):
    sd = str(tmp_path / "states")
    got = table.for_run(_nodes(), _tr())
    # the warm-up saved nothing at the Simu start: S0 states are written
    assert table.ensure_states(sd, T0, got, _dem(tmp_path)) == {"kept": 0, "initial": 1}
    assert _read(os.path.join(sd, "lake_storage_20261020_1600.tif"))[1, 2] == pytest.approx(3.0e6)
    # the warm-up's own lake state is never replaced
    s = np.zeros((5, 5))
    s[1, 2] = 1.0e6
    write_grid(os.path.join(sd, "lake_storage_20261020_1600.tif"), s, WEST, NORTH)
    assert table.ensure_states(sd, T0, got, _dem(tmp_path)) is None
    assert _read(os.path.join(sd, "lake_storage_20261020_1600.tif"))[1, 2] == pytest.approx(1.0e6)


def test_dam_on_a_natural_lake_follows_the_natural_rule():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "scripts"))
    from build_lakes import NATURAL_EQ_FRAC as f, ef5_param_b, retention_h
    L = pd.DataFrame({"kind": ["dam", "dam", "natural"], "v_km3": [0.9, 0.9, 0.9],
                      "q_mean_m3s": [10.0, 10.0, 10.0], "flood": [False, False, False],
                      "s0_frac": [0.9, f, f], "lake_ctl": [False, True, False]})
    k = retention_h(L)
    full = 0.9e9 / 10.0 / 3600.0
    b = float(ef5_param_b(np.array([0.9]))[0])
    # a reservoir passes its mean inflow when full; a lake (or a dam on one)
    # at its equilibrium f x V
    assert k[0] == pytest.approx(full)
    assert k[1] == pytest.approx(f ** (1.0 + b) * full) == pytest.approx(k[2])
