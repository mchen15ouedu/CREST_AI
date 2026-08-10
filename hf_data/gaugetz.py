"""gid -> IANA time zone (gauges/gauge_tz.parquet, from prep_gauge_tz.py).

The dashboard shows hydrograph times in the GAUGE'S local zone; the frontend
does the actual conversion with Intl (browser tz database handles DST), this
module just serves the zone name alongside pins/nowcasts. Loaded lazily once.
"""
from __future__ import annotations

import os
import threading

REPO = os.environ.get("CREST_FEEDBACK_REPO", "vincewin/CREST_data")

_lock = threading.Lock()
_map: dict = {}


def tz_of(gid: str) -> str | None:
    with _lock:
        if not _map:
            try:
                import pyarrow.parquet as pq
                from huggingface_hub import hf_hub_download
                t = pq.read_table(hf_hub_download(
                    REPO, "gauges/gauge_tz.parquet", repo_type="dataset",
                    token=os.environ.get("HF_TOKEN")))
                _map.update(zip(t.column("gid").to_pylist(),
                                t.column("tz").to_pylist()))
            except Exception:
                _map["__failed__"] = "UTC"     # don't re-download every call
        return _map.get(gid)
