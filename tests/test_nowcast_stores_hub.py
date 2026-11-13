"""The gauged and ungauged nowcast stores over the in-memory dataset adapter:
what they serve when the feed is there, is a real 404, or the Hub is
unreachable (the last good copy, never "nothing"). Offline."""
from __future__ import annotations

import io
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hf_data import nowcaststore as ns
from hf_data import ungaugednow_store as us
from hf_data.hubdata import Dataset, MemoryHub


def _parquet(cols: dict, meta: dict | None = None) -> bytes:
    t = pa.table(cols)
    if meta:
        t = t.replace_schema_metadata(meta)
    buf = io.BytesIO()
    pq.write_table(t, buf)
    return buf.getvalue()


def _latest(q1=50.0, t0="2026-11-13 12:00 UTC"):
    cols = {"gid": ["01000001", "01000002"], "lat": [35.0, 36.0],
            "lon": [-97.0, -96.0], "area_km2": [100.0, 2000.0],
            "obs_age_h": [1.0, 1.0], "obs_last_q": [40.0, 1.0],
            "obs_last_time": ["2026-11-13 11:00", "2026-11-13 11:00"]}
    for k in range(1, 7):
        cols[f"q{k}"] = [q1, 1.0]
    return _parquet(cols, {"t0": t0, "generated": "2026-11-13 12:20 UTC"})


THRESHOLDS = _parquet({"gid": ["01000001", "01000002"], "qbase_cms": [1.0, 1.0],
                       "q2_cms": [20.0, 20.0], "q5_cms": [40.0, 40.0],
                       "q10_cms": [80.0, 80.0]})


@pytest.fixture
def gauged(monkeypatch):
    h = MemoryHub({"nowcast/latest.parquet": _latest(),
                   "nowcast/flood_thresholds.parquet": THRESHOLDS})
    monkeypatch.setattr(ns, "DATASET", Dataset("me/feedback", adapter=h))
    monkeypatch.setattr(ns, "TTL_S", 0)            # every call goes to the Hub
    monkeypatch.setattr(ns, "THR_TTL_S", 0)
    return h


def test_gauged_risk_from_the_feed(gauged):
    r = ns.all_risk()
    assert r["ok"] and r["t0"] == "2026-11-13 12:00 UTC"
    assert r["tiers"] == {"01000001": 3, "01000002": 0}
    assert ns.issue_t0().hour == 12


def test_gauged_outage_serves_the_last_good_feed(gauged):
    assert ns.for_bbox(-98, 34, -95, 37)["ok"]
    gauged.outage = True
    r = ns.for_bbox(-98, 34, -95, 37)
    assert r["ok"] and [g["id"] for g in r["gauges"]] == ["01000002", "01000001"]
    assert r["gauges"][1]["q2"] == 20.0                     # thresholds stale too
    assert ns.all_risk()["tiers"]["01000001"] == 3


def test_gauged_404_is_no_nowcast_not_a_stale_one(gauged):
    assert ns.all_risk()["ok"]
    del gauged.files["nowcast/latest.parquet"]
    assert ns.all_risk() == {"ok": False,
                             "reason": "no precomputed nowcast available yet"}
    assert ns.issue_t0() is None and ns.nowcast_series("01000001") == []


def test_gauged_outage_with_nothing_cached_is_no_nowcast(gauged):
    gauged.outage = True
    assert not ns.all_risk()["ok"]


# --------------------------------------------------------------------------- #
# ungauged nowcast store
# --------------------------------------------------------------------------- #
T0 = datetime(2026, 11, 13, 12)


def _shard(vp, t0=T0, row_t0=None, area=100.0):
    rec = {"vp": vp, "lat": 35.0, "lon": -97.0, "area_km2": area,
           "q": [1.0] * us.HORIZON_H, "history": [], "t0": row_t0}
    buf = io.BytesIO()
    pq.write_table(us.build_table([rec], t0), buf)
    return buf.getvalue()


def _v3(vp="V1"):
    cols = {"vp": [vp], "lat": [35.0], "lon": [-97.0], "area_km2": [10.0]}
    for k in range(1, 13):
        cols[f"q12_{k}"] = [2.0]
    return _parquet(cols, {"t0": "2026-11-13 12:00 UTC",
                           "generated": "2026-11-13 12:10 UTC"})


@pytest.fixture
def ungauged(monkeypatch):
    h = MemoryHub()
    monkeypatch.setattr(us, "DATASET", Dataset("me/feedback", adapter=h))
    monkeypatch.setattr(us, "TTL_S", 0)
    monkeypatch.setattr(us, "NU_TTL_S", 0)
    return h


def test_ungauged_prefers_v3_then_routed_shards(ungauged):
    assert us.for_bbox(-98, 34, -96, 36) == {
        "ok": False, "reason": "no precomputed ungauged nowcast yet"}
    ungauged.files[us.LATEST_PREFIX + "__0of2.parquet"] = _shard("V7")
    ungauged.files[us.LATEST_PREFIX + "__1of2.parquet"] = _shard(
        "V8", row_t0="2026-11-10 12:00")                   # shard stopped: row dropped
    r = us.for_bbox(-98, 34, -96, 36)
    assert r["source"] == "ef5_routed" and [p["id"] for p in r["points"]] == ["V7"]
    assert r["points"][0]["forecast"][0] == ["2026-11-13 13:00", 1.0]
    ungauged.files[us.V3_LATEST] = _v3()
    r = us.for_bbox(-98, 34, -96, 36)
    assert r["source"] == "dilstm_v3" and r["points"][0]["q"][0] == 2.0
    assert us.issue_t0() == T0


def test_ungauged_outage_serves_the_last_good_feed(ungauged):
    ungauged.files[us.LATEST_PREFIX + ".parquet"] = _shard("V7")
    ungauged.files["nowcast/ungauged_no_upstream__0of1.json"] = b'["V9"]'
    r = us.by_ids("V7,V9,V5")
    assert r["missing"] == {"V9": "no_upstream", "V5": "pending"}
    ungauged.outage = True
    r = us.by_ids("V7,V9,V5")
    assert r["ok"] and r["source"] == "ef5_routed" and r["points"][0]["id"] == "V7"
    assert r["missing"] == {"V9": "no_upstream", "V5": "pending"}
    assert us.series("V7")["id"] == "V7"


def test_ungauged_v3_outage_does_not_flip_to_the_routed_feed(ungauged):
    ungauged.files[us.V3_LATEST] = _v3("V1")
    ungauged.files[us.LATEST_PREFIX + ".parquet"] = _shard("V7")
    assert us.for_bbox(-98, 34, -96, 36)["source"] == "dilstm_v3"
    ungauged.outage = True
    assert us.for_bbox(-98, 34, -96, 36)["source"] == "dilstm_v3"


def test_upload_latest_commits_the_shard_through_the_dataset(ungauged):
    tbl = us.build_table([{"vp": "V7", "lat": 35, "lon": -97, "area_km2": 1,
                           "q": [1.0], "history": []}], T0)
    assert us.upload_latest(tbl, tag="__0of1")
    assert us.LATEST_PREFIX + "__0of1.parquet" in ungauged.files
    assert us.for_bbox(-98, 34, -96, 36)["points"][0]["id"] == "V7"
    ungauged.outage = True
    with pytest.raises(RuntimeError):                      # the caller logs a failure
        us.upload_latest(tbl, tag="__0of1")
    ungauged.writable = False
    assert not us.upload_latest(tbl, tag="__0of1")         # no token: quietly skipped
