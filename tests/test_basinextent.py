import os

import numpy as np
import pytest

from conftest import RES, make_table, write_grid


def test_ok_rows(table):
    e = table.ef5_extent("05467000")
    assert e["bounds"] == (-90.975, 41.024, -90.157, 41.263)
    assert e["snap"] == {"lat": 41.2, "lon": -90.5, "area": 433.6}
    assert e["source"] == "gages2" and e["table"] == "test-1"
    assert table.ef5_extent({"id": "V70520944"})["source"] == "hybas"
    assert table.ef5_extent("5467000")["cells"] == 281834      # USGS ids zero-filled


@pytest.mark.parametrize("gid,reason", [("15019990", "no_terrain"),
                                        ("V99999999", "over_cap"),
                                        ("01234567", "no_row")])
def test_no_basin_raises(table, gid, reason):
    with pytest.raises(table.BasinExtentError) as ei:
        table.ef5_extent(gid)
    assert ei.value.reason == reason


def test_cap_env_can_only_lower(table, monkeypatch):
    monkeypatch.setattr(table, "_CAP_ENV", "1000000")
    assert table.ef5_extent("05467000")["cells"] == 281834
    with pytest.raises(table.BasinExtentError) as ei:
        table.ef5_extent("08263500")                 # 6.7M cells > 1M
    assert ei.value.reason == "over_cap"
    monkeypatch.setattr(table, "_CAP_ENV", "1e12")    # cannot raise the table's cap
    assert table.cap_cells() == 36_000_000


def test_lfs_pointer_is_table_missing(tmp_path, monkeypatch):
    from hf_data import basinextent
    p = tmp_path / "basin_extent.parquet"
    p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 1\n")
    monkeypatch.setattr(basinextent, "TABLE_PATH", str(p))
    monkeypatch.setattr(basinextent, "_tbl", None)
    with pytest.raises(basinextent.BasinExtentError) as ei:
        basinextent.ef5_extent("05467000")
    assert ei.value.reason == "table_missing" and "LFS" in str(ei.value)


def test_partial_qa_table_refused(tmp_path, monkeypatch):
    from hf_data import basinextent
    p = make_table(str(tmp_path / "t.parquet"), meta={"partial": True})
    monkeypatch.setattr(basinextent, "TABLE_PATH", p)
    monkeypatch.setattr(basinextent, "_tbl", None)
    with pytest.raises(basinextent.BasinExtentError) as ei:
        basinextent.ef5_extent("05467000")
    assert ei.value.reason == "table_missing"


def _river_acc():
    """50x50 hillslope (acc 1) with a river down column 25 (acc 1000 + 10*row)
    and a big river down column 10 (acc 1e6) - the case where neighbors._snap's
    window-scaled distance penalty pinned gauges to their hillslope cell."""
    acc = np.ones((50, 50))
    acc[:, 25] = 1000 + 10 * np.arange(50)
    acc[:, 10] = 1e6
    return acc


def test_snap_cell_picks_nearest_matching_stream_cell():
    from rasterio.transform import from_origin
    from hf_data.basinextent import cell_km2, cell_of, snap_cell
    tr = from_origin(-90.0, 41.0, RES, RES)
    acc = _river_acc()
    lat, lon = 41.0 - 30.5 * RES, -90.0 + 20.5 * RES          # cell (30, 20), on the hillslope
    area = acc[30, 25] * float(cell_km2(lat, RES))
    assert cell_of(tr, lat, lon) == (30, 20)
    assert snap_cell(acc, tr, lat, lon, area) == (30, 25)
    # an area nothing nearby matches: no outlet (never a hillslope cell)
    assert snap_cell(acc, tr, lat, lon, area * 10) is None
    # the matching river exists but lies beyond the search radius
    assert snap_cell(acc, tr, lat, lon, area, radius_km=0.2) is None
    # a cell centre with its own FAM area snaps to itself (idempotent)
    s_lat, s_lon = 41.0 - 30.5 * RES, -90.0 + 25.5 * RES
    assert snap_cell(acc, tr, s_lat, s_lon, area) == (30, 25)


def _reference_upstream(fdir, r0, c0):
    """Brute force: follow every cell downstream; upstream iff it hits (r0,c0)."""
    from hf_data.basinextent import _D8
    H, W = fdir.shape
    out = np.zeros_like(fdir, dtype=bool)
    for r in range(H):
        for c in range(W):
            rr, cc = r, c
            for _ in range(H * W + 1):
                if (rr, cc) == (r0, c0):
                    out[r, c] = True
                    break
                d = _D8.get(int(fdir[rr, cc]))
                if d is None:
                    break
                rr, cc = rr + d[0], cc + d[1]
                if not (0 <= rr < H and 0 <= cc < W):
                    break
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_upstream_mask_matches_brute_force(seed):
    from hf_data.basinextent import upstream_mask
    rng = np.random.default_rng(seed)
    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128, -9999], dtype="float32")
    fdir = rng.choice(codes, size=(17, 13))
    r0, c0 = 8, 6
    assert (upstream_mask(fdir, r0, c0) == _reference_upstream(fdir, r0, c0)).all()


def _bowl(extra_line=False):
    """7x7 nodata grid with a 3x3 bowl draining to (3,3); optionally a line of
    cells from the top border feeding it (a basin the raster clips)."""
    f = np.full((7, 7), -9999.0)
    f[2, 2], f[2, 3], f[2, 4] = 2, 4, 8
    f[3, 2], f[3, 3], f[3, 4] = 1, 4, 16
    f[4, 2], f[4, 3], f[4, 4] = 128, 64, 32
    if extra_line:
        f[0, 3] = f[1, 3] = 4
    return f


def test_carve_and_edge_guard(tmp_path):
    from hf_data import basinextent
    west, north = -90.0, 41.0
    ext = {"snap": {"lat": north - 3.5 * RES, "lon": west + 3.5 * RES, "area": 1.0}}
    ok = tmp_path / "ok"
    ok.mkdir()
    write_grid(str(ok / "fdir_clip.tif"), _bowl(), west, north)
    cv = basinextent.carve(str(ok), ext, "t")
    assert cv["cells"] == 9 and cv["mask"][3, 3]
    assert os.path.exists(ok / "carve_3_3.npy")                # cached per outlet
    assert basinextent.carve(str(ok), ext, "t")["cells"] == 9

    bad = tmp_path / "bad"
    bad.mkdir()
    write_grid(str(bad / "fdir_clip.tif"), _bowl(extra_line=True), west, north)
    with pytest.raises(basinextent.BasinExtentError) as ei:
        basinextent.carve(str(bad), ext, "t")
    assert ei.value.reason == "carve_touches_edge"


def test_no_box_left_in_model_code():
    """PI 2026-10-23: no EF5 extent may be sized from the outlet point and the
    drainage area. (The DI-LSTM rainfall square in scripts/, nowcast_space/
    and nowcast_eval/ belongs to the separate basin-rainfall plan.)"""
    import re
    from conftest import ROOT
    pat = re.compile(r"def basin_bbox|pipeline\.basin_bbox|import basin_bbox|"
                     r"1\.2\s*\*\s*math\.sqrt|domain_bbox")
    hits = []
    for sub in ("hf_data", "fleet", "autocal", "fleet_space", "ungauged_space",
                "autocal_space", "gpu_worker_space"):
        for dp, _, fs in os.walk(os.path.join(ROOT, sub)):
            for f in fs:
                if f.endswith(".py"):
                    p = os.path.join(dp, f)
                    for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                        if pat.search(line):
                            hits.append(f"{os.path.relpath(p, ROOT)}:{i}: {line.strip()}")
    assert not hits, "\n".join(hits)
