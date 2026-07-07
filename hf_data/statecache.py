"""Result + state cache with temporal-overlap reuse (task #6-A).

Per (gauge, model) we persist the simulated hydrograph rows, the covered time
window, and the times at which EF5 state files were saved. When a new request
overlaps an existing result in time, we reuse the cached portion and only
simulate the missing tail — warm-starting from the saved state at the cache
boundary (so no fresh 3-month warm-up).

State files themselves live in state_dir(gauge, model) and are written/read by
the EF5 binary (STATES=/TIME_STATE=). The JSON record is the planning index.

CACHE_DIR defaults to CREST_demo/_cache; on the Space set CREST_CACHE_DIR to a
persistent volume (storage scaling deferred, per the user).
"""
from __future__ import annotations

import json
import os
from datetime import datetime

TS_FMT = "%Y-%m-%d %H:%M"
STATE_TOL_DAYS = float(os.environ.get("CREST_STATE_TOL_DAYS", "10"))  # reuse a state within +/- this
CACHE_DIR = os.environ.get(
    "CREST_CACHE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache"))


def _rt(s: str) -> datetime:
    return datetime.strptime(s, TS_FMT)


def _key(gauge, model):
    return f"{str(gauge).zfill(8)}_{model}"


def results_path(gauge, model) -> str:
    d = os.path.join(CACHE_DIR, "results")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, _key(gauge, model) + ".json")


def state_dir(gauge, model) -> str:
    d = os.path.join(CACHE_DIR, "states", str(gauge).zfill(8), model)
    os.makedirs(d, exist_ok=True)
    return d


def load_record(gauge, model) -> dict | None:
    p = results_path(gauge, model)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def save_record(gauge, model, rows: list[dict], state_times: list[str],
                variant: str | None = None):
    rec = load_record(gauge, model) or {"gauge": str(gauge).zfill(8), "model": model,
                                        "rows": [], "window": None, "state_times": []}
    if variant is not None and rec.get("variant") != variant:
        # run configuration changed (e.g. different boundary-condition gauges) —
        # rows from the old configuration must not be mixed with the new ones
        rec["rows"], rec["window"] = [], None
        rec["variant"] = variant
    by_time = {r["time"]: r for r in rec["rows"]}
    for r in rows:
        by_time[r["time"]] = r
    merged = sorted(by_time.values(), key=lambda r: r["time"])
    rec["rows"] = merged
    if merged:
        ts = [_rt(r["time"]) for r in merged]
        rec["window"] = [min(ts).strftime(TS_FMT), max(ts).strftime(TS_FMT)]
    rec["state_times"] = sorted(set(rec["state_times"]) | set(state_times))
    with open(results_path(gauge, model), "w") as f:
        json.dump(rec, f)
    return rec


def nearest_state(gauge, model, t: datetime, tol_days: float = STATE_TOL_DAYS):
    """Nearest saved-state time within +/- tol_days of t (exact wins), or None."""
    rec = load_record(gauge, model)
    if not rec:
        return None
    best = None
    for s in rec.get("state_times", []):
        dt = _rt(s)
        delta = abs((dt - t).total_seconds()) / 86400.0
        if delta == 0:
            return (dt, 0.0)                                 # exact match has priority
        if delta <= tol_days and (best is None or delta < best[1]):
            best = (dt, delta)
    return best


def _state_times_on_disk(gauge, model) -> set[datetime]:
    """State-save times parsed from the actual EF5 state files on disk
    (e.g. crestphys_SM_20250704_0000.tif). The JSON index can lag or be
    invalidated (paramstore drops the row cache) while the files remain."""
    import glob as _glob
    import re as _re
    out = set()
    for p in _glob.glob(os.path.join(state_dir(gauge, model), "*.tif")):
        m = _re.search(r"_(\d{8})_(\d{4})\.tif$", os.path.basename(p))
        if m:
            try:
                out.add(datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M"))
            except ValueError:
                pass
    return out


def _state_choice(gauge, model, t: datetime, tol_days: float = STATE_TOL_DAYS):
    """(load_time, warmup_from, need_warmup) for warm-starting a run at t:
        exact state         -> load it directly, no warm-up
        earlier state <=10d -> short warm-up FORWARD over the gap from that state
        else                -> full warm-up (warmup_from=None; caller uses 3 months)
    A state in the *future* of t can't warm forward, so it isn't used here.
    """
    rec = load_record(gauge, model)
    times = _state_times_on_disk(gauge, model)
    for s in (rec or {}).get("state_times", []):
        try:
            times.add(_rt(s))
        except ValueError:
            pass
    if not times:
        return None, None, True
    best_before = None                                       # nearest state at/just before t
    for dt in times:
        gap = (t - dt).total_seconds() / 86400.0             # >0 when dt precedes t
        if gap == 0:
            return t, None, False                            # exact -> no warm-up
        if 0 < gap <= tol_days and (best_before is None or gap < best_before[1]):
            best_before = (dt, gap)
    if best_before:
        return None, best_before[0], True                    # short forward warm-up
    return None, None, True                                  # full warm-up


def plan(gauge, model, a: datetime, b: datetime, variant: str | None = None) -> dict:
    """Decide how to satisfy a request for [a, b]: reuse cache + minimal run,
    warm-starting from the nearest state within +/- STATE_TOL_DAYS if possible.
    `variant` fingerprints the run configuration (boundary-condition gauges) —
    rows cached under a different variant are not reused (states still are)."""
    if os.environ.get("CREST_CACHE", "1") == "0":            # force a fresh full run
        return {"cached_rows": [], "run_start": a, "run_end": b, "load_state_time": None,
                "warmup_from": None, "need_warmup": True, "reason": "cache disabled"}
    rec = load_record(gauge, model)
    if rec and variant is not None and rec.get("variant") != variant:
        rec = {**rec, "rows": [], "window": None}            # states remain usable

    def slice_rows(lo, hi):
        if not rec:
            return []
        return [r for r in rec["rows"] if lo <= _rt(r["time"]) <= hi]

    if rec and rec.get("window"):
        from datetime import timedelta
        slack = timedelta(hours=1)      # EF5's first ts row lands one step AFTER
        c0, c1 = _rt(rec["window"][0]), _rt(rec["window"][1])   # TIME_BEGIN
        if c0 <= a + slack and c1 >= b:                      # fully cached
            return {"cached_rows": slice_rows(a, b), "run_start": None, "run_end": None,
                    "load_state_time": None, "warmup_from": None,
                    "need_warmup": False, "reason": "fully cached"}
        if c0 <= a + slack and a <= c1 < b:                  # extend forward from the cache end
            lt, wf, nw = _state_choice(gauge, model, c1)
            return {"cached_rows": slice_rows(a, c1), "run_start": c1, "run_end": b,
                    "load_state_time": lt, "warmup_from": wf, "need_warmup": nw,
                    "reason": "reuse cache + fill missing tail"}
    lt, wf, nw = _state_choice(gauge, model, a)              # full run
    reason = ("warm start (exact state)" if not nw else
              "short warm-up from nearby state" if wf is not None else "full 3-month warm-up")
    return {"cached_rows": [], "run_start": a, "run_end": b,
            "load_state_time": lt, "warmup_from": wf, "need_warmup": nw, "reason": reason}


if __name__ == "__main__":
    # unit test: cache a window, then request an overlapping one
    import tempfile
    from datetime import timedelta
    globals()["CACHE_DIR"] = tempfile.mkdtemp()
    g, m = "08144500", "crestphys"
    t0 = datetime(2025, 7, 3, 0, 0)
    rows1 = [{"time": (t0 + timedelta(hours=i)).strftime(TS_FMT), "sim_q": i} for i in range(11)]
    save_record(g, m, rows1, state_times=[(t0 + timedelta(hours=10)).strftime(TS_FMT)])
    print("cached window:", load_record(g, m)["window"], "states:", load_record(g, m)["state_times"])

    # request [05:00, 15:00] -> should reuse 05..10 and run 10..15 (warm from saved state @10)
    p = plan(g, m, t0 + timedelta(hours=5), t0 + timedelta(hours=15))
    print("plan:", p["reason"], "| cached", len(p["cached_rows"]), "rows",
          f"({p['cached_rows'][0]['time']}..{p['cached_rows'][-1]['time']})",
          "| run", p["run_start"].strftime(TS_FMT), "->", p["run_end"].strftime(TS_FMT),
          "| load_state", p["load_state_time"].strftime(TS_FMT) if p["load_state_time"] else None,
          "| warmup", p["need_warmup"])

    # request fully inside cache -> no run
    p2 = plan(g, m, t0 + timedelta(hours=2), t0 + timedelta(hours=8))
    print("plan2:", p2["reason"], "| run_start", p2["run_start"])
