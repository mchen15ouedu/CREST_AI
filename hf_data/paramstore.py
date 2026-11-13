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

from hf_data import hubdata, statecache
from hf_data.hubdata import Put
from hf_data.statecache import CACHE_DIR

# the private state dataset persist.py mirrors params/ to; winners written on
# another Space (CREST_autocal) land there and are picked up here on read
DATASET = hubdata.state()
REMOTE_TTL_S = float(os.environ.get("CREST_PARAMS_REMOTE_TTL_S", "600"))
_remote_seen: dict[str, float] = {}


def _name(gauge: str, model: str) -> str:
    return f"{str(gauge).zfill(8)}_{model.lower()}.json"


def _path(gauge: str, model: str) -> str:
    d = os.path.join(CACHE_DIR, "params")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, _name(gauge, model))


def _local(gauge: str, model: str) -> dict | None:
    try:
        with open(_path(gauge, model), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _remote(gauge: str, model: str) -> hubdata.Read:
    """The record in the state dataset (a private repo: without a token
    there is no answer, UNREACHABLE)."""
    if not hubdata.token():
        return hubdata.Read(hubdata.UNREACHABLE, error="no HF token")
    return DATASET.read_json(f"params/{_name(gauge, model)}", kind=dict)


def get(gauge: str, model: str) -> dict | None:
    """Best-known record {wb, kw, nse, source, when, window} or None.

    Reads the local store, and at most every REMOTE_TTL_S per (gauge, model)
    also the CREST_state copy: the NEWER `when` wins (a calibration finished
    on the autocal Space must reach the runner/demo/fleet without a restart;
    `when` is '%Y-%m-%d %H:%M UTC', so it compares as a string). A newer
    remote record is written locally so persist mirrors it like any other.
    An unreachable remote is not remembered: the next call looks again."""
    loc = _local(gauge, model)
    key = _name(gauge, model)
    now = datetime.now(timezone.utc).timestamp()
    if now - _remote_seen.get(key, 0.0) >= REMOTE_TTL_S:
        r = _remote(gauge, model)
        if not r.unreachable:
            _remote_seen[key] = now
        rem = r.value if r.ok else None
        if rem and (loc is None or str(rem.get("when", "")) > str(loc.get("when", ""))):
            try:
                with open(_path(gauge, model), "w", encoding="utf-8") as fh:
                    json.dump(rem, fh, indent=1)
            except OSError:
                pass
            return rem
    return loc


def push_remote(gauge: str, model: str) -> bool:
    """Upload the local record to the state dataset's params/ (for Spaces that
    run no persist thread, e.g. CREST_autocal). True on success."""
    p = _path(gauge, model)
    if not os.path.exists(p):
        return False
    return bool(DATASET.commit([Put(f"params/{_name(gauge, model)}", p)],
                               f"params {gauge} {model}: calibration winner"))


def maybe_save(gauge: str, model: str, wb: dict, kw: dict, nse: float | None,
               source: str, window: list[str] | None = None,
               baseline_nse: float | None = None) -> bool:
    """Persist (wb, kw) if this NSE beats the stored one. Returns True if saved.
    baseline_nse: compare against THIS value (the stored parameters' NSE on
    the same window) instead of the stored record's NSE, which may come from
    a different period — an auto-calibration triggered by a missed flood
    must be judged on the window it was run on."""
    if nse is None:
        return False
    cur = get(gauge, model)
    if baseline_nse is not None:
        if float(nse) <= float(baseline_nse):
            return False
    elif cur is not None and cur.get("nse") is not None and float(cur["nse"]) >= float(nse):
        return False
    rec = {"wb": wb, "kw": kw, "nse": round(float(nse), 4), "source": source,
           "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "window": window}
    with open(_path(gauge, model), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    # NOTE (UX isolation, 2026-09-28): caches are NOT purged any more. The
    # pipeline keys rows and frames by a tag that includes this record's
    # `when`, so the next run simply lands under a new key while anyone's run
    # in flight keeps its own — deleting shared caches under a live job broke
    # other users' animations.
    return True
