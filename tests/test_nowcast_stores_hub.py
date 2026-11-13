"""The gauged and ungauged nowcast stores over the in-memory dataset adapter:
what they serve when the feed is there, is a real 404, or the Hub is
unreachable (the last good copy, never "nothing"). Offline."""
from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hf_data import nowcaststore as ns
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
