"""Incremental routed nowcast for one ungauged point — the per-point unit the
keep-warm Space calls hourly for all 2,676 points.

Two phases, so it never cold-starts and only simulates the newest hour(s):

  Phase A — history / state march: an ordinary CACHED hindcast [t0-hist, t0]
    (observed upstream inflow only, no future injection). The state cache
    reuses the overlap and simulates only the hours added since the last run,
    then saves the end-of-window state at t0. Safe to cache because it holds no
    predicted values. Yields the ~7-day display history.

  Phase B — forecast: a 12-h run [t0, t0+H] warm-STARTED from the t0 state
    Phase A just saved (warmup_days=0), with each upstream cut gauge injected
    with its DI-LSTM nowcast beyond t0 and the future precip left unforced
    (EF5 reads a missing grid as zero → routed inflow drives it). Not cached /
    not saved (nowcast_t0 forces that) so the predicted future never poisons
    the hindcast row cache.

After Phase A, prune_states keeps the newest state + 10-day checkpoints and
deletes the intermediate hourly states.

The single-run 17-day cold approach it replaces (V23) re-simulated everything
on every call; here the steady-state cost is ~1 new hour (Phase A) + 12 h
(Phase B) from warm states.
"""
from __future__ import annotations

from datetime import datetime, timedelta

HIST_DAYS = 7
HORIZON_H = 12
WEST_LON = -105.0            # matches pipeline.WEST_LON (crest vs crestphys)


def _cache_model(lon: float) -> str:
    return ("crest" if lon < WEST_LON else "crestphys") + "-spd"


def compute(gid: str, t0: datetime, hist_days: int = HIST_DAYS,
            horizon_h: int = HORIZON_H) -> dict:
    """Advance one ungauged point to t0 and forecast horizon_h hours.

    Returns {"ok", "gid", "t0", "history": [rows], "forecast": [rows],
             "q": [q1..qH], "reason"?}. history rows are the routed model over
    [t0-hist_days, t0]; forecast rows cover (t0, t0+horizon_h]. Never raises for
    an ordinary data hiccup — returns {"ok": False, "reason": ...}."""
    from hf_data import pipeline, virtualpoints, statecache

    info = virtualpoints.info(str(gid))
    if info is None:
        return {"ok": False, "gid": gid, "reason": "unknown ungauged point"}

    hist_start = t0 - timedelta(days=hist_days)
    t_end = t0 + timedelta(hours=horizon_h)

    status: list[str] = []

    # A point with a qualifying upstream cut gauge runs the truncated SPEED
    # domain (state key "<model>-spd", injection-driven); a headwater point with
    # none falls back to the FULL basin (state key "<model>", MRMS-driven). We
    # don't know which until the run resolves the scheme, so consider both keys.
    base_model = "crest" if info["lon"] < WEST_LON else "crestphys"
    spd_model = base_model + "-spd"

    # Cold-boot warm-start: the app's pipeline skips the fleet fetch for virtual
    # points (it serves them from the parquet store, not EF5), so pull our own
    # last checkpoint from CREST_fleet HERE. On a warm container the state is
    # already on local disk and this is a cheap no-op; after a Space restart it
    # refetches the newest 10-day checkpoint (whichever variant exists) so Phase
    # A short-warms instead of doing a full 3-month cold start.
    try:
        from hf_data import fleetstore
        for m in (spd_model, base_model):
            if fleetstore.ensure_local(gid, m) == "fetched":
                status.append(f"[boot] 🚚 fetched {m} checkpoint from CREST_fleet")
    except Exception:
        pass

    def _drain(gen, tag):
        rows: list[dict] = []
        model = None
        n_up = None
        for kind, ev in gen:
            if kind == "hydro":
                rows += ev.get("rows", [])
            elif kind == "status":
                status.append(f"[{tag}] {ev}")
            elif kind == "params":
                if ev.get("cache_model"):
                    model = ev["cache_model"]       # the key EF5 actually saved state under
                if ev.get("n_upstream") is not None:
                    n_up = ev["n_upstream"]
        return rows, model, n_up

    # Phase A: cached hindcast to t0 (advances + saves state; observed only).
    # Its reported cache_model is the one the end-of-window state is stored under.
    hist, cache_model, n_upstream = _drain(pipeline.run_gauge(
        gid, hist_start, t0, use_mock=False, scheme="speed", grids=False,
        warmup_days=10), "A")
    cache_model = cache_model or spd_model

    # Phase B: 12-h forecast warm-started from the t0 state (no cold warm-up)
    fcst, _, _ = _drain(pipeline.run_gauge(
        gid, t0, t_end, use_mock=False, scheme="speed", grids=False,
        warmup_days=0, nowcast_t0=t0), "B")

    # retention: keep newest + 10-day checkpoints, delete intermediate hourly states
    try:
        statecache.prune_states(gid, cache_model)
    except Exception:
        pass

    q = [round(float(r["sim_q"]), 3) for r in fcst
         if isinstance(r.get("sim_q"), (int, float))][:horizon_h]
    # injected: the truncated speed domain ran, so upstream flow was actually
    # routed to this point (a real routed nowcast). has_upstream: at least one
    # upstream USGS gauge exists at all — False means a headwater with nothing to
    # route, which is nowcast-ineligible by design (still valid for hindcast).
    injected = str(cache_model).endswith("-spd")
    has_upstream = bool(injected or (n_upstream or 0) > 0)
    return {"ok": bool(fcst), "gid": str(gid), "t0": t0.strftime("%Y-%m-%d %H:%M"),
            "history": hist, "forecast": fcst, "q": q, "status": status,
            "cache_model": cache_model, "injected": injected,
            "has_upstream": has_upstream, "lat": info["lat"], "lon": info["lon"],
            "area_km2": info.get("area"),
            "reason": None if fcst else "no forecast rows (no upstream inflow?)"}


if __name__ == "__main__":
    import sys
    from datetime import timezone
    from hf_data import nowcaststore
    gid = sys.argv[1] if len(sys.argv) > 1 else "V70779201"
    t0 = nowcaststore.issue_t0() or datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0, tzinfo=None)
    r = compute(gid, t0.replace(tzinfo=None) if t0.tzinfo else t0)
    print(f"{gid} @ {r['t0']}: ok={r['ok']} hist={len(r['history'])} "
          f"fcst={len(r['forecast'])} q={r['q'][:6]}… {r.get('reason') or ''}")
