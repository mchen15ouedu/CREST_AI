"""Persistent per-basin parameter store (task: auto-calibration).

Keeps the BEST-known multiplier set per (gauge, model), replacing it only when
a later run — AI calibration, manual tweaking, or any completed simulation —
achieves a higher NSE. Stored under CACHE_DIR/params/ next to the result/state
caches (set CREST_CACHE_DIR to a persistent volume on the Space to keep them
across restarts).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from hf_data import statecache
from hf_data.statecache import CACHE_DIR


def _path(gauge: str, model: str) -> str:
    d = os.path.join(CACHE_DIR, "params")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{str(gauge).zfill(8)}_{model.lower()}.json")


def get(gauge: str, model: str) -> dict | None:
    """Best stored record: {wb, kw, nse, source, when, window} or None."""
    try:
        with open(_path(gauge, model), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def maybe_save(gauge: str, model: str, wb: dict, kw: dict, nse: float | None,
               source: str, window: list[str] | None = None) -> bool:
    """Persist (wb, kw) if this NSE beats the stored one. Returns True if saved."""
    if nse is None:
        return False
    cur = get(gauge, model)
    if cur is not None and cur.get("nse") is not None and float(cur["nse"]) >= float(nse):
        return False
    rec = {"wb": wb, "kw": kw, "nse": round(float(nse), 4), "source": source,
           "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "window": window}
    with open(_path(gauge, model), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    # cached hydrograph rows were produced with the OLD params — drop them
    try:
        rp = statecache.results_path(gauge, model)
        if os.path.exists(rp):
            os.remove(rp)
    except Exception:
        pass
    return True
