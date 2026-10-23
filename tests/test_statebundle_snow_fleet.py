from datetime import datetime

import numpy as np

from conftest import RES, write_grid

WEST, NORTH = -90.0, 41.0


# --- statebundle: the newest state time sets the bundle geometry -------------
def test_pack_dir_uses_newest_geometry(tmp_path):
    import rasterio
    from hf_data import statebundle
    sd, out = tmp_path / "s", tmp_path / "u"
    sd.mkdir()
    # same SHAPE, origin one cell apart: the old raster must be skipped, not
    # packed under the new origin
    write_grid(str(sd / "SM_20210701_0000.tif"), np.full((3, 3), 1.0), WEST - RES, NORTH)
    write_grid(str(sd / "SM_20210711_0000.tif"), np.full((3, 3), 2.0), WEST, NORTH)
    write_grid(str(sd / "Q_20210711_0000.tif"), np.full((3, 3), 3.0), WEST, NORTH)
    blob = statebundle.pack_dir(str(sd))
    n = statebundle.unpack(blob, str(out))
    names = sorted(p.name for p in out.iterdir())
    assert names == ["Q_20210711_0000.tif", "SM_20210711_0000.tif"] and n == 2
    with rasterio.open(out / "SM_20210711_0000.tif") as ds:
        assert abs(ds.transform.c - WEST) < 1e-9 and ds.read(1).max() == 2.0


# --- snow (M2): decided over the basin, not the rectangle ---------------------
def test_temp_pixels():
    from hf_data.snow import _temp_pixels
    m = _temp_pixels(np.array([-89.95, -89.85]), np.array([40.95, 40.95]),
                     -90.0, 40.0, 0.1, (10, 10))
    assert m.sum() == 2 and m[0, 0] and m[0, 1]


def test_detect_snow_uses_basin_cells(tmp_path):
    from rasterio.transform import from_origin
    from hf_data import snow
    dem = np.full((10, 10), 200.0)
    dem[:3, :3] = 2400.0                       # a high basin in a low rectangle
    write_grid(str(tmp_path / "dem.tif"), dem, -105.0, 39.0)
    mask = np.zeros((10, 10), dtype=bool)
    mask[:3, :3] = True
    bbox = (-105.0, 39.0 - 10 * RES, -105.0 + 10 * RES, 39.0)
    t0, t1 = datetime(2025, 7, 1), datetime(2025, 7, 3)        # warm season, lat < 40
    rect = snow.detect_snow(bbox, t0, t1, dem_path=str(tmp_path / "dem.tif"),
                            use_temp=False)
    assert rect["snow"] is False                               # rectangle mean ~620 m
    basin = snow.detect_snow(bbox, t0, t1, dem_path=str(tmp_path / "dem.tif"),
                             use_temp=False, basin=(mask, from_origin(-105.0, 39.0, RES, RES)))
    assert basin["snow"] is True and "2400" in basin["reason"]


# --- fleet: size split + prefixes ---------------------------------------------
IDS = ["05467000", "08263500", "15019990", "V70520944", "V99999999"]


def test_select_outlets_size_split(table):
    import fleet_run
    small, skipped = fleet_run.select_outlets(IDS, "small")
    assert small == ["05467000"]
    assert dict(skipped) == {"no_terrain": 1, "over_cap": 1}
    big, _ = fleet_run.select_outlets(IDS, "big")
    assert big == ["08263500", "V70520944"]
    allc, _ = fleet_run.select_outlets(IDS, "all")
    assert allc == ["05467000", "08263500", "V70520944"]


def test_done_keys_prefixes(monkeypatch):
    import huggingface_hub
    import fleet_run
    files = ["results/05467000_crestphys.json", "states/05467000_crestphys.pqf",
             # the ungauged nowcast's rolling checkpoint: NOT a finished hindcast
             "results/V70520944_crestphys-spd.json", "states/V70520944_crestphys-spd.pqf",
             "vp/results/V1_crestphys-spd.json", "vp/states/V1_crestphys-spd.pqf",
             "results/08263500_crestphys.json"]                 # no states: not done

    class FakeApi:
        def __init__(self, *a, **k):
            pass

        def list_repo_files(self, *a, **k):
            return files

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setenv("HF_TOKEN", "x")
    assert fleet_run._done_keys() == {"05467000_", "V1_"}


def test_store_keys():
    import fleet_run
    from hf_data import basic
    a = (-90.975, 41.024166666, -90.1575, 41.2633333)
    b = (a[0] + RES, a[1], a[2], a[3])                       # one lattice cell apart
    assert basic.store_key(a) != basic.store_key(b)
    kb, kf = fleet_run._keys(a)
    assert kb == basic.store_key(a)
    assert kf == "_".join(f"{v:.3f}" for v in a).replace("-", "m").replace(".", "p")
