"""CREST_updater — HF Space that refreshes every CREST_demo data feed.

The heavy data refresh (NARR temp, MRMS precip, PET, USGS gauge obs) runs HERE,
on HF's own network — next to the CREST_data / CREST_state repos the updaters
write to — instead of on the user's local machine. The weekly local routine only
has to WAKE this Space and poll until the run finishes.

Behavior:
  * On container start, a full update run launches automatically after a short
    delay (UPDATER_AUTO=1, default) — so simply waking a sleeping Space does the
    job. Every updater is idempotent/append-only, so restarts are harmless.
  * POST /api/run  [?feeds=temp,pet,mrms,usgs&key=...]  starts a run explicitly
    (409 if one is already going). If UPDATER_KEY is set (Space secret), the
    key param/header must match it.
  * GET  /api/status  -> {running, started, finished, results, log_tail} — the
    per-feed summary lines the scripts print, plus the freshness block.
  * GET  /  -> tiny status page (also what a wake-up ping hits).

Runs each scripts/update_*.py as a subprocess (same entry points as local runs)
with HF_TOKEN from the Space secret. The freshness check runs last and its
report is included in the status. Nothing here holds state across restarts —
the store repos ARE the state.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, PlainTextResponse

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDS = ["temp", "pet", "mrms", "mrms_recent", "nowcast", "usgs"]   # temp first (matches local routine)
SCRIPTS = {
    "temp": "update_temp_narr.py",
    "pet": "update_pet.py",
    "mrms": "update_mrms.py",
    "mrms_recent": "update_mrms_recent.py",   # rolling Pass1 for AI nowcasting
    "nowcast": "run_nowcast_all.py",          # DI-LSTM nowcasts for all gauges
    "usgs": "update_usgs_obs.py",
    "check": "check_forcing_freshness.py",
}
AUTO = os.environ.get("UPDATER_AUTO", "1") == "1"
# what a wake-up auto-run refreshes. The 6-hourly nowcast wake sets this Space
# variable to "mrms_recent" (light: ~6 small files) so waking is cheap; the
# weekly routine POSTs /api/run explicitly for the full refresh.
AUTO_FEEDS = [f for f in os.environ.get("UPDATER_AUTO_FEEDS", ",".join(FEEDS)).split(",")
              if f in SCRIPTS and f != "check"] or FEEDS
AUTO_DELAY_S = float(os.environ.get("UPDATER_AUTO_DELAY_S", "20"))
KEY = os.environ.get("UPDATER_KEY", "")

app = FastAPI(title="CREST_updater")
_lock = threading.Lock()
_state = {
    "running": False,
    "started": None,          # iso ts of current/last run start
    "finished": None,         # iso ts of last run end
    "results": {},            # feed -> list of summary lines
    "freshness": [],          # lines from check_forcing_freshness
    "log": [],                # rolling combined log (bounded)
    "exit_codes": {},         # feed -> rc
}
_NOISE = ("UserWarning", "warnings.warn", "symlink", "Developer Mode",
          "activate developer", "unauthenticated", "To support", "see this article",
          "Cannot find gdalvrt", "FutureWarning", "DeprecationWarning")


def _log(line: str):
    line = line.rstrip()
    if not line or any(n in line for n in _NOISE):
        return
    _state["log"].append(line)
    del _state["log"][:-400]                      # keep the tail bounded
    print(line, flush=True)


def _run_script(feed: str, extra_args: list[str] | None = None) -> tuple[int, list[str]]:
    """Run one updater subprocess; returns (rc, its non-noise output lines)."""
    script = os.path.join(APP_DIR, "scripts", SCRIPTS[feed])
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_PROGRESS_BARS="1")
    lines: list[str] = []
    _log(f"=== {feed}: {SCRIPTS[feed]} ===")
    p = subprocess.Popen([sys.executable, script] + (extra_args or []),
                         cwd=APP_DIR, env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for raw in p.stdout:
        line = raw.rstrip()
        if line and not any(n in line for n in _NOISE):
            lines.append(line)
            _log(f"  {line}")
    rc = p.wait()
    if rc != 0:
        _log(f"  [{feed}] exited rc={rc}")
    return rc, lines


def _run_all(feeds: list[str]):
    try:
        _state["results"], _state["exit_codes"] = {}, {}
        _state["freshness"] = []
        _state["started"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _state["finished"] = None
        for feed in feeds:
            rc, lines = _run_script(feed)
            _state["results"][feed] = lines
            _state["exit_codes"][feed] = rc
        rc, lines = _run_script("check")          # always end with the freshness report
        _state["freshness"] = lines
        _state["exit_codes"]["check"] = rc
        _log("=== run complete ===")
    except Exception as e:                        # keep the app alive whatever happens
        _log(f"run crashed: {e!r}")
    finally:
        _state["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _state["running"] = False
        _lock.release()


def _start(feeds: list[str]) -> bool:
    if not _lock.acquire(blocking=False):
        return False
    _state["running"] = True
    threading.Thread(target=_run_all, args=(feeds,), daemon=True).start()
    return True


@app.get("/", response_class=PlainTextResponse)
def root():
    s = "running" if _state["running"] else ("idle" if _state["finished"] else "starting")
    return (f"CREST_updater — refreshes CREST_demo data feeds (temp/pet/mrms/mrms_recent/usgs)\n"
            f"state: {s}   started: {_state['started']}   finished: {_state['finished']}\n"
            f"POST /api/run to trigger, GET /api/status for details\n")


@app.get("/api/status")
def status():
    return JSONResponse({k: _state[k] for k in
                         ("running", "started", "finished", "results",
                          "freshness", "exit_codes")} | {"log_tail": _state["log"][-60:]})


@app.post("/api/run")
def run(feeds: str = "", key: str = ""):
    if KEY and key != KEY:
        return JSONResponse({"error": "bad key"}, status_code=403)
    want = [f for f in (feeds.split(",") if feeds else FEEDS) if f in SCRIPTS and f != "check"]
    if not want:
        want = FEEDS
    if not _start(want):
        return JSONResponse({"error": "a run is already in progress"}, status_code=409)
    return {"ok": True, "feeds": want}


@app.on_event("startup")
def _auto():
    if AUTO:
        def kickoff():
            time.sleep(AUTO_DELAY_S)              # let the container settle
            if _start(AUTO_FEEDS):
                _log(f"auto-run started: {','.join(AUTO_FEEDS)} "
                     f"(delay {AUTO_DELAY_S:.0f}s after boot)")
        threading.Thread(target=kickoff, daemon=True).start()

    # self-scheduler: external cron (GitHub Actions ~2.5-3 h effective, local
    # task only while the desktop app is open) can't guarantee hourly cadence,
    # so as long as this container is awake (fleet keep-alive ring pings it)
    # the light feeds run themselves at :58 — right after NCEP posts Pass1.
    if os.environ.get("UPDATER_HOURLY", "1") == "1":
        def hourly():
            while True:
                into = time.time() % 3600         # seconds into the hour
                target = 58 * 60                  # fire at :58
                wait = (target - into) if into < target else (3600 - into + target)
                time.sleep(max(30, wait))
                # only dedup back-to-back runs (e.g. right after a boot auto-run):
                # an external trigger mid-hour must NOT suppress the :58 run, or a
                # new Pass1 hour posted at :50 would wait a whole extra hour.
                fin = _state["finished"]
                recent = False
                if fin:
                    try:
                        t = datetime.fromisoformat(fin)
                        recent = (datetime.now(timezone.utc) - t).total_seconds() < 10 * 60
                    except ValueError:
                        pass
                if not _state["running"] and not recent and _start(AUTO_FEEDS):
                    _log(f"hourly self-run started: {','.join(AUTO_FEEDS)}")
        threading.Thread(target=hourly, daemon=True).start()
