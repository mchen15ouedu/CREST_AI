"""Archive-forcing freshness for /api/health — MRMS Pass2, PET, TEMP.

The health check watched only the near-real-time Pass1 radar feed, which the
updater Space refreshes itself every hour (UPDATER_AUTO_FEEDS=mrms_recent).
The feeds a simulation actually integrates — gauge-corrected MRMS Pass2, PET
and TEMP month tars — refresh ONLY when the weekly routine POSTs the updater
Space explicitly, so a routine that stops running is invisible until a run
hits a forcing gap. Seen live 2026-08-20: PET last archived 2026-07-20, a
month stale, with nothing in the pipeline reporting it.

Two signals per feed, both from ONE repo-tree listing (no tar downloads — the
temp/mrms month tars are hundreds of MB):

  updated_age_d   how long since the newest month tar was last written; this
                  is updater liveness, the thing that actually breaks
  months_behind   current month minus the newest month tar's month; this is
                  archive COVERAGE, i.e. can a run for "now" be forced at all

Both are coarse by construction — the exact newest timestep lives inside the
tar. scripts/check_forcing_freshness.py still opens the tars for that; this
module is the cheap always-on version the dashboard can serve every request.
"""
from __future__ import annotations

import datetime
import os
import re
import threading
import time

REPO = os.environ.get("CREST_FEEDBACK_REPO", "vincewin/CREST_data")

# var -> (label, max write-age in days). MRMS/PET post within a day or two, so
# a stalled updater shows up fast; NARR-fed TEMP lags weeks at the source, and
# a run with nothing new to add makes no commit at all, so its write-age is
# allowed to drift further before it means anything.
FEEDS = {
    "mrms": ("MRMS Pass2", float(os.environ.get("HEALTH_PASS2_MAX_D", "14"))),
    "pet": ("PET", float(os.environ.get("HEALTH_PET_MAX_D", "14"))),
    "temp": ("TEMP", float(os.environ.get("HEALTH_TEMP_MAX_D", "30"))),
}
# months of archive lag tolerated before coverage itself is called stale: the
# current month's tar may legitimately not exist yet on day 1, so one month
# behind is normal and two is a gap.
MONTHS_MAX = int(os.environ.get("HEALTH_FORCING_MONTHS_MAX", "1"))
TTL_S = float(os.environ.get("HEALTH_FORCING_TTL_S", "1800"))

_lock = threading.Lock()
_cache: dict = {"t": 0.0, "snap": None}


def _months_between(a: tuple[int, int], b: tuple[int, int]) -> int:
    return (b[0] - a[0]) * 12 + (b[1] - a[1])


def _check(api, var: str, now: datetime.datetime) -> dict:
    """Newest month tar for one feed + the commit that wrote it."""
    label, max_d = FEEDS[var]
    year = now.year
    rows = []
    for y in (year, year - 1):                 # January: last year's dir wins
        try:
            rows = [r for r in api.list_repo_tree(
                REPO, path_in_repo=f"{var}/{y}", repo_type="dataset",
                expand=True)
                if re.match(rf"{var}/{y}/{var}_{y}_\d{{2}}\.tar$",
                            getattr(r, "path", ""))]
        except Exception:
            rows = []
        if rows:
            break
    if not rows:
        return {"label": label, "ok": False, "error": "no month tar listed"}
    newest = max(rows, key=lambda r: r.path)
    m = re.search(r"_(\d{4})_(\d{2})\.tar$", newest.path)
    month = (int(m.group(1)), int(m.group(2)))
    behind = _months_between(month, (now.year, now.month))

    lc = getattr(newest, "last_commit", None)
    written = getattr(lc, "date", None)
    age_d = None
    if written is not None:
        age_d = round((now - written.replace(tzinfo=None)).total_seconds()
                      / 86400.0, 1)
    return {"label": label,
            "newest_month": f"{month[0]}-{month[1]:02d}",
            "months_behind": behind,
            "updated": (written.strftime("%Y-%m-%d")
                        if written is not None else None),
            "updated_age_d": age_d,
            "max_age_d": max_d,
            # unknown write date is not proof of trouble (an old client may not
            # expand commits) — coverage still decides in that case
            "ok": (age_d is None or age_d <= max_d) and behind <= MONTHS_MAX}


def snapshot() -> dict:
    """{mrms_pass2, pet, temp, ok} — cached TTL_S; these feeds move weekly."""
    with _lock:
        if _cache["snap"] is not None and time.time() - _cache["t"] < TTL_S:
            return _cache["snap"]
    out: dict = {}
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        now = datetime.datetime.utcnow()
        for var in FEEDS:
            key = "mrms_pass2" if var == "mrms" else var
            try:
                out[key] = _check(api, var, now)
            except Exception as e:
                out[key] = {"ok": False, "error": type(e).__name__}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__}
    out["ok"] = all(v.get("ok", True) for v in out.values()
                    if isinstance(v, dict))
    with _lock:
        _cache.update(t=time.time(), snap=out)
    return out
