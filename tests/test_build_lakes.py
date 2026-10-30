import numpy as np
import pandas as pd
import pytest

from scripts import build_lakes as bl


def test_param_b_is_ef5s():
    # LakeInfo::CalculateParamB thresholds, in its if/else order
    v = [3.0, 1.5, 1.2, 1.0, 0.5, 0.1, 0.07, 0.05, 0.01]
    assert list(bl.ef5_param_b(v)) == [2.0, 1.8, 1.8, 1.5, 1.5, 1.5, 1.2, 1.2, 1.0]


def _rows(**cols):
    n = len(next(iter(cols.values())))
    base = {"kind": ["dam"] * n, "flood": [False] * n, "s0_frac": [0.8] * n,
            "v_km3": [0.2] * n, "q_mean_m3s": [10.0] * n}
    return pd.DataFrame({**base, **cols})


def test_natural_lake_passes_its_mean_inflow_at_equilibrium():
    L = _rows(kind=["natural", "natural"], v_km3=[0.02, 2.0], q_mean_m3s=[1.0, 50.0])
    k = bl.retention_h(L)
    f = bl.NATURAL_EQ_FRAC
    for i in range(2):
        v = L.v_km3[i] * 1e9
        b = bl.ef5_param_b(L.v_km3[i])
        s = f * v                                   # EF5: Q = S/K (S/V)^b, K in hours
        assert s / (k[i] * 3600.0) * (s / v) ** b == pytest.approx(L.q_mean_m3s[i])


def test_flood_cap_only_for_dry_pools():
    L = _rows(kind=["dam", "dam", "dam", "pond"], flood=[True, True, False, False],
              s0_frac=[0.0, 0.6, 0.6, 1.0], v_km3=[0.5, 0.5, 0.5, 0.0005])
    k = bl.retention_h(L)
    full = 0.5e9 / 10.0 / 3600.0                    # V / mean inflow, h
    assert k[0] == bl.FLOOD_K_H                     # normally dry: evacuated within weeks
    assert k[1] == pytest.approx(full)              # flood dam WITH a normal pool
    assert k[2] == pytest.approx(full)
    assert k[3] == bl.POND_K_H


def test_dis_avg_first_outside_the_us():
    L = pd.DataFrame({"country": [bl.US, "Canada", "Canada", bl.US],
                      "dis_avg": [5.0, 5.0, np.nan, np.nan]})
    src, mm = bl.mean_runoff(L, np.array([300.0, 300.0, 300.0, np.nan]),
                             np.array([100.0] * 4), np.full(4, 0.01))
    assert list(src) == ["runave7100_basin", "dis_avg", "runave7100_basin",
                         "runave7100_regional"]
    assert list(mm) == pytest.approx([300.0, 500.0, 300.0, 100.0])


def _lk(id_, source, r, c, fam, v, area, hylak=0, dis=np.nan, kind="dam"):
    return {"id": id_, "name": id_, "kind": kind, "source": source, "lat": 39.0, "lon": -120.0,
            "snap_lat": 39.0, "snap_lon": -120.0, "snap_row": r, "snap_col": c,
            "fam_km2": fam, "ref_km2": fam, "snap_rule": "area", "v_km3": v, "s0_frac": 0.9,
            "area_km2": area, "dis_avg": dis, "purpose": "Irrigation" if source != "hydrolakes" else "",
            "gate": "", "hylak_id": hylak, "grand_id": 0, "nid_id": id_[3:] if source != "hydrolakes" else "",
            "flood": False, "country": bl.US}


def test_one_water_body_listed_twice_is_merged():
    L = pd.DataFrame([
        _lk("NIDCA10162", "nid", 25005, 35827, 1295.46, 0.903, 485.6),          # Lake Tahoe Dam
        _lk("HL792", "hydrolakes", 25006, 35829, 1295.40, 0.4956, 495.6, hylak=792, dis=22.0,
            kind="natural"),                                                     # 'Tahoe'
        _lk("NIDX1", "nid", 25007, 35828, 12.0, 0.001, 0.1),                    # side creek
        _lk("NIDX2", "nid", 25100, 35900, 1300.0, 0.002, 0.2),                  # far away
    ])
    out = bl.merge_duplicates(L).set_index("id")
    assert sorted(out.index) == ["NIDCA10162", "NIDX1", "NIDX2"]
    t = out.loc["NIDCA10162"]
    assert t.v_km3 == pytest.approx(0.903) and t.kind == "dam"   # the NID volume and purpose
    assert t.area_km2 == pytest.approx(495.6)                    # the larger surface, never the sum
    assert (t.snap_row, t.snap_col) == (25005, 35827)            # the downstream (larger FAM) cell
    assert t.source == "both" and t.hylak_id == 792 and t.dis_avg == pytest.approx(22.0)
    assert t.merged_ids == "HL792"
    assert out.loc["NIDX1"].merged_ids == "" and out.loc["NIDX2"].merged_ids == ""


def test_lake_control_matches_its_dam_without_a_storage_test():
    d = pd.DataFrame({"lat": [39.0919, 39.0950, 39.30, 38.6990, 38.7075, 38.7138],
                      "lon": [-120.1569, -120.1600, -120.30, -121.1530, -121.1567, -121.1741],
                      "v_af": [732000.0, 900000.0, 5000.0, 33.0, 976955.0, 200.0],
                      "da_km2": [1311.0, 60.0, np.nan, 0.05, 2320.6, np.nan]})
    h = pd.DataFrame({"lat": [39.0925, 39.30, 39.31, 38.7104], "lon": [-120.1575, -120.30, -120.31, -121.1562],
                      "Lake_type": [3, 1, 2, 2], "Grand_id": [0, 0, 0, 182],
                      "Vol_total": [150000.0, 10.0, 5.0, 1102.7],
                      "Wshd_area": [1295.0, 50.0, 40.0, 4821.9]})
    m = bl.match(h, d)
    # Tahoe: 150 km3 vs 0.9 km3 regulated, still the same water body (the
    # drainage areas agree); the bigger dam nearby drains another stream
    # (60 km2) and is skipped. A natural lake (type 1, not in GRanD) never
    # matches; a dam without a drainage area needs a similar storage
    assert m[0] == 0 and 1 not in m and m[2] == 2
    # Folsom Lake: NID's drainage area is wrong (896 of 1,875 sq mi) but the
    # storage agrees; the 200 acre-ft dam 1.6 km away (no drainage area) and
    # the 33 acre-ft one are not Folsom Lake
    assert m[3] == 4


def test_distance_alone_never_matches():
    d = pd.DataFrame({"lat": [38.7138], "lon": [-121.1741], "v_af": [200.0], "da_km2": [np.nan]})
    h = pd.DataFrame({"lat": [38.7104], "lon": [-121.1562], "Lake_type": [2], "Grand_id": [182],
                      "Vol_total": [1102.7], "Wshd_area": [4821.9]})
    assert bl.match(h, d) == {}


def test_hydrolakes_region(monkeypatch):
    import pyogrio
    h = pd.DataFrame({"Hylak_id": [1, 2, 3, 4, 5], "Lake_name": [""] * 5,
                      "Country": [bl.US, bl.US, "Canada", "Canada", "Mexico"],
                      "Lake_type": [1] * 5, "Grand_id": [0] * 5, "Lake_area": [1.0] * 5,
                      "Vol_total": [1.0] * 5, "Dis_avg": [1.0] * 5, "Wshd_area": [1.0] * 5,
                      "Pour_long": [-120.0, -135.0, -117.0, -100.0, -105.5],
                      "Pour_lat": [45.0, 57.0, 52.0, 52.0, 27.6]})
    monkeypatch.setattr(pyogrio, "read_dataframe", lambda *a, **k: h)
    got = bl.load_hydrolakes("x.shp", bl.WEST_LON)
    # CONUS; not Alaska; Canada and Mexico west of WEST_LON
    assert list(got.Hylak_id) == [1, 3, 5]
