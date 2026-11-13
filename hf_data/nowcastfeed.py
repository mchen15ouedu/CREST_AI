"""What the gauged (nowcaststore) and ungauged (ungaugednow_store) nowcast
stores share: the feed's freshness, its issue time, its column layout and
the bbox selection.

Freshness is hubdata's one outage policy: a feed is re-read after its TTL,
the last good copy is served while the Hub is unreachable, and a real 404 is
"no nowcast" rather than a stale one."""
from __future__ import annotations

import threading
from datetime import datetime

import numpy as np

from . import hubdata

T0_FMT = "%Y-%m-%d %H:%M UTC"        # schema metadata "t0" / "generated"
RETRY_S = 60.0                        # after an unreachable read, wait this long

_lock = threading.Lock()
_feeds: dict = {}


def read(ds: hubdata.Dataset, name: str, fetch, ttl_s: float) -> hubdata.Read:
    """The cached answer of `fetch(ds) -> Read` (see hubdata.cached), one
    cache per (dataset, name): a store pointed at another dataset starts cold."""
    key = (id(ds), name, ttl_s)
    with _lock:
        hit = _feeds.get(key)
        if hit is None or hit[0] is not ds:
            hit = (ds, hubdata.cached(lambda: fetch(ds), ttl_s,
                                      retry_s=min(RETRY_S, ttl_s)))
            _feeds[key] = hit
    return hit[1].get()


def table(read_result: hubdata.Read):
    """(meta, cols) of a feed read, or (None, None) when there is none."""
    if read_result.ok and read_result.value is not None:
        return read_result.value
    return None, None


def load_parquet(local: str):
    """A feed parquet -> (meta {str: str}, cols {name: ndarray})."""
    import pyarrow.parquet as pq
    t = pq.read_table(local)
    return schema_meta(t), {n: t.column(n).to_numpy(zero_copy_only=False)
                            for n in t.schema.names}


def schema_meta(t) -> dict:
    return {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()
            if not k.startswith(b"ARROW")}


def qcols(cols, prefix: str = "q") -> list[str]:
    """Ordered forecast columns <prefix>1..<prefix>N."""
    n = len(prefix)
    return sorted((c for c in cols if c and c.startswith(prefix) and c[n:].isdigit()),
                  key=lambda c: int(c[n:]))


def issue_time(meta) -> datetime | None:
    """The feed's issue time t0 (UTC), or None."""
    try:
        return datetime.strptime((meta or {}).get("t0", ""), T0_FMT)
    except (ValueError, TypeError):
        return None


def in_bbox(cols, w: float, s: float, e: float, n: float):
    """Row indices inside the bbox, largest basins first, and their count."""
    m = ((cols["lon"] >= w) & (cols["lon"] <= e)
         & (cols["lat"] >= s) & (cols["lat"] <= n))
    return largest_first(cols, np.nonzero(m)[0])


def largest_first(cols, idx):
    return idx[np.argsort(-cols["area_km2"][idx])]
