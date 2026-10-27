"""CREST_autocal — auto-triggered AI calibration, western gauges first.

Runs forever on its own HF Space (autocal_space/). The lowest-priority compute
class in CREST-AI (user 2026-09-25: user-initiated operations > nowcast >
2-D inundation > data update > fleet runner > this), so it lives on a
dedicated cpu-basic Space, calibrates ONE gauge at a time, and caps how many
LLM-guided calibrations it starts per day.

Priority (PI 2026-10-27: "The auto-calibration, we should prioritize the
western part of the USA, since those areas have more issues"): every
candidate list sorts western gauges (lon < pipeline.WEST_LON) first. An
eligible event beats the backlog. In the WEST an over-prediction beats a
whole-event miss (the western problem is EF5 running far too high); in the
east a miss beats an over-prediction (the original trigger keeps its place).
Then each kind's own keys (miss: more checks, bigger obs peak; oversim:
bigger ratio, bigger obs peak).

Reservoir hold (PI 2026-10-27): a reservoir-affected gauge (GAGES-II
STOR_NOR_2009 >= 10 mm, hf_data/data/gagesii_reservoir_stor.json) is not
calibrated from ANY trigger until AUTOCAL_LAKES_ON=1 — the lake/reservoir
module is being switched on for the west, and calibrating before it simulates
the reservoirs would bake them into the parameters (e.g. 10171000 Jordan
River, 267 mm). Held events are filtered out of eligibility without a Hub
write; they are listed in the state file (held_reservoir) and the phase line.

Triggers:
  * MISS (kind "miss", EF5 under-predicted) — events/missed.json rows written
    by the runner's CREST-miss guard: a streak of >= AUTOCAL_MIN_CHECKS hourly
    checks that ALL missed (a check that passes publishes a map, so the event
    would be in events/index.json — such streaks are skipped), the last check
    at least AUTOCAL_QUIET_H old (the trigger stopped re-checking: the flood
    is over).
  * OVER-PREDICTION (kind "oversim") — a published event (events/index.json)
    whose manifest hydro rows, paired sim_q/obs_q up to t0, peak at
    max(sim)/max(obs) >= AUTOCAL_OVERSIM_RATIO with an obs peak >=
    AUTOCAL_OVERSIM_MIN_OBS m3/s, and whose t_end is at least AUTOCAL_QUIET_H
    old. The guard only rejects UNDER-prediction (sim < 0.3 x obs), so before
    this trigger a western gauge that published at 13-30x obs (09371492 Mud
    Creek, 09337500 Escalante) could never be calibrated. Manifests are
    HF-cached and the verdict is kept per (event, generated, t_end): a poll
    costs no download for an event already judged.
  For both: the gauge was not auto-calibrated within AUTOCAL_COOLDOWN_D days
  and the (gauge, event) was not already handled (events/autocal.json); an
  oversim event that ended before the gauge's last calibration is skipped
  (its sim predates the new parameters).
  * WESTERN BACKLOG (kind "backlog") — only when no event is eligible: the
    next gauge of hf_data/data/autocal_west_backlog.json
    (scripts/build_autocal_backlog.py: tier 1 event-prone, 2 dam-free with a
    suspect parameter set, 3 other dam-free, 4 reservoir-affected; smaller
    basins first). Tier 4 is held back unless AUTOCAL_LAKES_ON=1: calibrated
    before the lake module simulates their reservoirs, the parameters would
    absorb them. Calibrate once: a gauge with any ran=True record is never
    picked again. A backlog gauge that cannot run (obs coverage, calibration
    error) is NOT committed one by one (events/autocal.json is a Hub commit
    per record): it goes on a local skip list in AUTOCAL_STATE (kept across
    loop restarts in this container, retried after AUTOCAL_SKIP_TTL_D days)
    and reaches the Hub only in one "backlog_skips" summary record at most
    every AUTOCAL_SKIP_SUMMARY_H, which also re-seeds the list after a Space
    restart.
Gates for every kind: fewer than AUTOCAL_MAX_PER_DAY calibrations (ran=True,
any kind) started in the last 24 h, the idle gate (busy_reason), and USGS
observations covering >= AUTOCAL_OBS_MIN_FRAC of the window.
Calibration window: AUTOCAL_WINDOW_D days (default 183 — two seasons; user:
"some dry lands don't see anything in 30 days") ending at the event's end
(miss: last check's t0 + 6 h; oversim: the event's t_end; backlog: now),
never later than now - 2 h. Records carry kind and west.
Method: hf_data.calibrate.run_calibration (LLM-proposed multiplier updates
within hydrologic bounds, judged by real EF5 runs), saving the winner only if
it beats the CURRENT parameters on this window (save_vs_baseline) and pushing
it to CREST_state params/ where paramstore.get on every other Space picks it
up. Without an LLM key it waits (AUTOCAL_REQUIRE_LLM=1): heuristic-only
search is not the "AI calibration" this Space exists for.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # CREST_AI checkout root

POLL_S = float(os.environ.get("AUTOCAL_POLL_S", "900"))
MIN_CHECKS = int(os.environ.get("AUTOCAL_MIN_CHECKS", "3"))
QUIET_H = float(os.environ.get("AUTOCAL_QUIET_H", "6"))
COOLDOWN_D = float(os.environ.get("AUTOCAL_COOLDOWN_D", "30"))
MAX_PER_DAY = int(os.environ.get("AUTOCAL_MAX_PER_DAY", "2"))
WINDOW_D = float(os.environ.get("AUTOCAL_WINDOW_D", "183"))
OBS_MIN_FRAC = float(os.environ.get("AUTOCAL_OBS_MIN_FRAC", "0.5"))
ROUNDS = int(os.environ.get("AUTOCAL_ROUNDS", "4"))
K = int(os.environ.get("AUTOCAL_K", "3"))
REQUIRE_LLM = os.environ.get("AUTOCAL_REQUIRE_LLM", "1") == "1"
STREAK_GAP_H = float(os.environ.get("AUTOCAL_STREAK_GAP_H", "3"))
STATE_PATH = os.environ.get("AUTOCAL_STATE", "/tmp/autocal_state.json")
# over-prediction trigger (published events) and the western backlog
OVERSIM_RATIO = float(os.environ.get("AUTOCAL_OVERSIM_RATIO", "3.0"))
OVERSIM_MIN_OBS = float(os.environ.get("AUTOCAL_OVERSIM_MIN_OBS", "1.0"))
LAKES_ON = os.environ.get("AUTOCAL_LAKES_ON", "0") == "1"
BACKLOG_PATH = os.environ.get("AUTOCAL_BACKLOG", os.path.join(
    os.path.dirname(HERE), "hf_data", "data", "autocal_west_backlog.json"))
RESERVOIR_PATH = os.environ.get("AUTOCAL_RESERVOIRS", os.path.join(
    os.path.dirname(HERE), "hf_data", "data", "gagesii_reservoir_stor.json"))
SKIP_TTL_D = float(os.environ.get("AUTOCAL_SKIP_TTL_D", "30"))
SKIP_SUMMARY_H = float(os.environ.get("AUTOCAL_SKIP_SUMMARY_H", "24"))
# Idle gate (user 2026-10-12: "auto_cal has the lowest priority — only when
# there is enough computation"): start, and keep going between rounds, only
# while the dashboard's /api/health shows no user-initiated work, no 2-D
# bundle queued or claimed, and no event EF5 run on the runner's tick. The
# health snapshot is cached 120 s server-side. Fleet runners are NOT a gate:
# they run on their own Spaces with a permanent backlog (would never idle).
HEALTH_URL = os.environ.get("AUTOCAL_HEALTH_URL",
                            "https://vincewin-crest-demo.hf.space/api/health")
IDLE_FOR_S = float(os.environ.get("AUTOCAL_IDLE_FOR_S", "1800"))

TS = "%Y-%m-%dT%H:%M:%SZ"
state = {"phase": "booting", "current": None, "last": None, "runs": 0, "skips": 0,
         "backlog_skips": {}, "skips_summarized": None, "held_reservoir": []}


def log(s: str):
    print(f"{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}Z {s}", flush=True)


def _dump_state():
    try:
        with open(STATE_PATH, "w") as fp:
            json.dump(state, fp)
    except OSError:
        pass


class Preempted(Exception):
    """Higher-priority work appeared mid-calibration: drop it, retry later."""


def busy_reason() -> str | None:
    """None when every higher compute class is quiet, else why not. Fails
    CLOSED: an unreadable health snapshot counts as busy."""
    try:
        import requests
        h = requests.get(HEALTH_URL, timeout=30).json()
    except Exception as e:
        return f"health unreadable ({type(e).__name__})"
    u = h.get("user_ops") or {}
    if "sims" not in u:
        return "health has no user_ops (dashboard not updated?)"
    if u.get("sims") or u.get("cals"):
        return f"user work running ({u.get('sims')} sims, {u.get('cals')} cals)"
    q, w, t = h.get("queue") or {}, h.get("workers") or {}, h.get("tick") or {}
    ev = h.get("events") or {}
    # queue.depth also counts solved-but-unswept bundles (up to 48 h), and
    # re-solves of LISTED episodes count as "published" — so gate on never-
    # published bundles plus any active flood episode (still being re-solved)
    if q.get("unpublished") is None or q.get("unpublished"):
        return f"{q.get('unpublished')} unpublished 2-D bundle(s)"
    if ev.get("active") is None or ev.get("active"):
        return f"{ev.get('active')} active flood episode(s)"
    if w.get("active_claims"):
        return f"{w['active_claims']} 2-D worker claim(s) active"
    if t.get("running"):
        return f"runner tick busy ({t['running']})"
    return None


def _parse(s: str | None) -> dt.datetime | None:
    for fmt in (TS, "%Y-%m-%dT%H:%MZ", "%Y-%m-%d %H:%M"):     # last: manifest hydro rows
        try:
            return dt.datetime.strptime(str(s), fmt)
        except (TypeError, ValueError):
            continue
    return None


def streaks(missed: list, index: dict) -> list[dict]:
    """Group the miss rows by GAUGE into streaks of consecutive hourly checks.

    Each hourly re-check of a still-flagged gauge opens a NEW event id (the
    guard keeps the event off the list, so the next tick detects it afresh
    with the new t0) — the whole-event history lives across rows. A gap of
    more than STREAK_GAP_H between checks starts a new streak. Rows whose
    event id IS listed (CREST caught up) end a streak: that gauge was not
    missed for the whole event."""
    per = {}
    for rec in missed:
        gid = str(rec.get("gauge") or "")
        if not gid:
            continue
        for c in (rec.get("checks") or [rec]):
            at = _parse(c.get("at"))
            if at is None:
                continue
            per.setdefault(gid, []).append({**c, "event": rec.get("event"),
                                            "_at": at, "_listed": rec.get("event") in index,
                                            "obs_peak_m3s": c.get("obs_peak_m3s") or rec.get("obs_peak_m3s")})
    out = []
    for gid, checks in per.items():
        checks.sort(key=lambda c: c["_at"])
        cur = []
        for c in checks:
            if cur and (c["_at"] - cur[-1]["_at"]) > dt.timedelta(hours=STREAK_GAP_H):
                out.append(cur)
                cur = []
            cur.append(c)
        if cur:
            out.append(cur)
    res = []
    for chain in out:
        if any(c["_listed"] for c in chain):
            continue
        res.append({"gauge": chain[0]["event"].split("_")[-1] if chain[0].get("event") else "",
                    "event": chain[-1]["event"], "events": sorted({c["event"] for c in chain if c.get("event")}),
                    "checks": chain, "n_checks": len(chain),
                    "first_at": chain[0]["_at"], "last_at": chain[-1]["_at"],
                    "obs_peak_m3s": max((float(c.get("obs_peak_m3s") or 0) for c in chain), default=0.0),
                    "sim_peak_m3s": max((float(c.get("sim_peak_m3s") or 0) for c in chain), default=0.0)})
    for r in res:
        if not r["gauge"]:
            r["gauge"] = next((str(k) for k, v in per.items() if v and v[0] is r["checks"][0]), "")
    return res


_lons: dict = {}


def gauge_lon(gid: str) -> float | None:
    """Gauge longitude: the basin-extent table row (local file), else the
    GAGES-II catalog. None when neither knows the id."""
    gid = str(gid).zfill(8)
    if gid not in _lons:
        lon = None
        try:
            from hf_data import basinextent
            r = basinextent.row(gid)
            if r is not None:
                lon = float(r["lon"])
        except Exception:
            pass
        if lon is None:
            try:
                from hf_data import gauges
                ll = gauges.get_gauge_coordinates(gid)
                lon = float(ll[1]) if ll else None
            except Exception:
                return None                      # catalog unreachable: not cached
        _lons[gid] = lon
    return _lons[gid]


def is_west(gid: str) -> bool:
    from hf_data.pipeline import WEST_LON
    lon = gauge_lon(gid)
    return lon is not None and lon < WEST_LON


_resv: dict | None = None


def reservoir_stor(gid: str) -> float | None:
    """GAGES-II normal reservoir storage (mm) of a reservoir-affected gauge
    (>= the table's stor_mm_min, 10 mm), else None (dam-free or no GAGES-II
    record). Raises when the committed table is unreadable: the hold must not
    silently lapse (the loop then shows the error and retries)."""
    global _resv
    if _resv is None:
        try:
            with open(RESERVOIR_PATH, encoding="utf-8") as fp:
                d = json.load(fp)
            lim = float(d.get("stor_mm_min", 10.0))
            _resv = {str(g).zfill(8): float(v) for g, v in d["gauges"].items()
                     if float(v) >= lim}
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as e:
            raise RuntimeError(f"reservoir table {RESERVOIR_PATH} unreadable "
                               f"({type(e).__name__}: {e})")
    return _resv.get(str(gid).zfill(8))


def held_stor(gid: str) -> float | None:
    """Reservoir storage (mm) when this gauge is held back from calibration
    (reservoir-affected and AUTOCAL_LAKES_ON off), else None."""
    return None if LAKES_ON else reservoir_stor(gid)


def _history(done: list, now: dt.datetime):
    """(last ran=True time per gauge, handled (gauge, event) pairs, ran=True
    records started in the last 24 h) from events/autocal.json rows."""
    last_run = {}
    for d in done:
        if d.get("ran"):
            t = _parse(d.get("at"))
            if t and (d["gauge"] not in last_run or t > last_run[d["gauge"]]):
                last_run[d["gauge"]] = t
    handled = {(d.get("gauge"), d.get("event")) for d in done}
    today = sum(1 for d in done if d.get("ran") and (_parse(d.get("at")) or now)
                > now - dt.timedelta(hours=24))
    return last_run, handled, today


def _hold(c: dict, held: list | None) -> bool:
    """True (and noted in `held`) when candidate c is a held reservoir gauge."""
    stor = held_stor(c["gauge"])
    if stor is None:
        return False
    if held is not None:
        held.append({"gauge": c["gauge"], "event": c["event"], "kind": c["kind"],
                     "west": c["west"], "stor_mm": stor})
    return True


def miss_candidates(missed: list, index: dict, done: list, now: dt.datetime,
                    held: list | None = None) -> list[dict]:
    """Whole-event miss streaks that are eligible now (kind "miss").
    Reservoir gauges held until AUTOCAL_LAKES_ON=1 go to `held` instead."""
    last_run, handled, _ = _history(done, now)
    cands = []
    for st in streaks(missed, index):
        gid = st["gauge"]
        if not gid or (gid, st["event"]) in handled:
            continue
        if st["n_checks"] < MIN_CHECKS:
            continue
        if now - st["last_at"] < dt.timedelta(hours=QUIET_H):
            continue                             # gauge still being re-checked hourly
        lr = last_run.get(gid)
        if lr and now - lr < dt.timedelta(days=COOLDOWN_D):
            continue
        c = {**st, "kind": "miss", "west": is_west(gid)}
        if not _hold(c, held):
            cands.append(c)
    return cands


def oversim_stats(manifest: dict, t0: dt.datetime | None = None) -> dict | None:
    """Peak EF5 sim vs observed over a published event's 1-D hydro rows up to
    t0 (the manifest's own t0 by default), rows with both sim_q and obs_q only:
    {ratio, sim_peak_m3s, obs_peak_m3s, n_paired}, or None without a pair or
    with a zero observed peak. Rows after t0 are the forecast tail (no obs
    when the event was solved), so they never count."""
    t0 = t0 or _parse(manifest.get("t0"))
    sims, obss = [], []
    for r in manifest.get("hydro") or []:
        t = _parse(r.get("time"))
        try:
            s, o = float(r.get("sim_q")), float(r.get("obs_q"))
        except (TypeError, ValueError):
            continue                             # None: no obs (or no sim) that hour
        if t is None or (t0 and t > t0) or s != s or o != o:
            continue
        sims.append(s)
        obss.append(o)
    if not sims or max(obss) <= 0:
        return None
    ms, mo = max(sims), max(obss)
    return {"ratio": ms / mo, "sim_peak_m3s": round(ms, 2), "obs_peak_m3s": round(mo, 2),
            "n_paired": len(sims)}


_verdicts: dict = {}             # (event, generated, t_end) -> oversim_stats


def _event_stats(ev: str, summary: dict, manifest=None) -> dict | None:
    """oversim_stats of a published event. `manifest` (tests) maps an event id
    to its manifest dict; by default eventstore.load_manifest (HF-cached) with
    the verdict memoised per (event, generated, t_end), so a re-published or
    finalized event is judged again and an unchanged one costs nothing."""
    if manifest is not None:
        m = manifest(ev)
        return oversim_stats(m, _parse(m.get("t0") or summary.get("t0"))) if m else None
    key = (ev, summary.get("generated"), summary.get("t_end"))
    if key not in _verdicts:
        from hf_data import eventstore
        m = eventstore.load_manifest(ev)
        if m is None:
            return None                          # unreadable now: not cached, retried
        _verdicts[key] = oversim_stats(m, _parse(m.get("t0") or summary.get("t0")))
    return _verdicts[key]


def oversim_candidates(index: dict, done: list, now: dt.datetime,
                       manifest=None, held: list | None = None) -> list[dict]:
    """Published events whose EF5 peak was >= OVERSIM_RATIO x the observed
    peak (kind "oversim"). The cheap index-only rules run first, so only
    events that could still qualify ever have their manifest read. Reservoir
    gauges held until AUTOCAL_LAKES_ON=1 go to `held` instead."""
    last_run, handled, _ = _history(done, now)
    for key in [k for k in _verdicts if k[0] not in (index or {})]:
        del _verdicts[key]                       # rolled off the list
    cands = []
    for ev, s in (index or {}).items():
        raw = s.get("gauge") or (s.get("trigger") or {}).get("gauge") or (
            ev.split("_")[-1] if "_" in ev else "")
        if not raw:
            continue
        gid = str(raw).zfill(8)
        if (gid, ev) in handled:
            continue
        end = _parse(s.get("t_end"))
        if end is None or now - end < dt.timedelta(hours=QUIET_H):
            continue                             # still being re-solved (or no end)
        lr = last_run.get(gid)
        if lr and (now - lr < dt.timedelta(days=COOLDOWN_D) or end <= lr):
            continue                             # cooldown / sim predates the new params
        st = _event_stats(ev, s, manifest)
        if (not st or st["obs_peak_m3s"] < OVERSIM_MIN_OBS
                or st["ratio"] < OVERSIM_RATIO):
            continue
        c = {**st, "kind": "oversim", "gauge": gid, "event": ev, "events": [ev],
             "west": is_west(gid), "t0": s.get("t0"), "t_end": s.get("t_end"), "_end": end}
        if not _hold(c, held):
            cands.append(c)
    return cands


def _event_key(c: dict):
    """West first; in the west an over-prediction before a miss, in the east a
    miss before an over-prediction; then the kind's own keys."""
    first = "oversim" if c["west"] else "miss"
    own = ((-c["n_checks"], -c["obs_peak_m3s"]) if c["kind"] == "miss"
           else (-c["ratio"], -c["obs_peak_m3s"]))
    return (not c["west"], c["kind"] != first) + own


def load_backlog(path: str | None = None) -> list:
    """Ranked western backlog entries ([] when the file is missing/bad)."""
    try:
        with open(path or BACKLOG_PATH, encoding="utf-8") as fp:
            return list(json.load(fp).get("gauges") or [])
    except (OSError, ValueError, AttributeError):
        return []


def backlog_skips(done: list, local: dict, now: dt.datetime) -> dict:
    """{gauge: {at, reason}} of backlog gauges skipped without a run within
    SKIP_TTL_D: this container's local list merged with the Hub summaries
    ("backlog_skips" records), so a Space restart does not re-probe them."""
    cut = now - dt.timedelta(days=SKIP_TTL_D)
    rows = [r for d in done if d.get("kind") == "backlog_skips"
            for r in (d.get("skipped") or [])]
    rows += [{"gauge": g, **v} for g, v in (local or {}).items()]
    out = {}
    for r in rows:
        g, t = r.get("gauge"), _parse(r.get("at"))
        if g and t and t > cut and (g not in out or t > _parse(out[g]["at"])):
            out[g] = {"at": r["at"], "reason": r.get("reason")}
    return out


def backlog_candidate(backlog: list, done: list, skips: dict,
                      lakes_on: bool | None = None) -> tuple[dict | None, str]:
    """The next western backlog gauge (kind "backlog") or None, and why.
    Tier, then smaller area. Calibrate once: any ran=True record (any kind,
    any age) retires the gauge. Tier 4 (reservoirs) waits for the lakes."""
    lakes_on = LAKES_ON if lakes_on is None else lakes_on
    ran = {str(d.get("gauge")) for d in done if d.get("ran")}
    held = n_ran = n_skip = 0
    for e in sorted(backlog, key=lambda e: (int(e.get("tier") or 9),
                                            float(e.get("area_km2") or 0),
                                            str(e.get("gauge")))):
        gid = str(e.get("gauge") or "").zfill(8)
        if not e.get("gauge"):
            continue
        if int(e.get("tier") or 9) >= 4 and not lakes_on:
            held += 1
            continue
        if gid in ran:
            n_ran += 1
            continue
        if gid in skips:
            n_skip += 1
            continue
        return ({"kind": "backlog", "gauge": gid, "event": None, "west": is_west(gid),
                 **{k: e.get(k) for k in ("tier", "area_km2", "stor_mm", "donor", "reasons")}},
                f"backlog tier {e.get('tier')}, {n_ran} done, {n_skip} skipped ahead of it")
    return None, (f"western backlog exhausted ({len(backlog)} gauges: {n_ran} done, "
                  f"{n_skip} skipped, {held} reservoir gauges held until AUTOCAL_LAKES_ON=1)")


def pick_candidate(missed: list, index: dict, done: list, now: dt.datetime,
                   backlog: list | None = None, skips: dict | None = None,
                   manifest=None, held: list | None = None):
    """The one candidate to work on next (or None) and a reason string: the
    first eligible event (miss streak or over-prediction, western gauges
    first), else the next western backlog gauge. Event candidates on held
    reservoir gauges are appended to `held` (when given), never picked."""
    # candidates first (cheap: verdicts are memoised) so the held list stays
    # current on the status page even while the daily cap is reached
    held = [] if held is None else held
    cands = (miss_candidates(missed, index, done, now, held=held)
             + oversim_candidates(index, done, now, manifest=manifest, held=held))
    hold = (f"; {len(held)} reservoir-gauge event(s) held until AUTOCAL_LAKES_ON=1"
            if held else "")
    _, _, today = _history(done, now)
    if today >= MAX_PER_DAY:
        return None, f"daily cap reached ({today}/{MAX_PER_DAY} in 24 h){hold}"
    if cands:
        cands.sort(key=_event_key)
        n_west = sum(1 for c in cands if c["west"])
        return cands[0], f"{len(cands)} eligible event(s), {n_west} western{hold}"
    rec, why = backlog_candidate(backlog or [], done, skips or {})
    return rec, f"no eligible event{hold}; {why}"


def skip_summary(done: list, local: dict, now: dt.datetime,
                 summarized_at: str | None = None) -> dict | None:
    """One "backlog_skips" record (ran=False, gauge None) listing the local
    backlog skips newer than the last summary, or None when there is nothing
    new or the last summary (Hub record or `summarized_at`) is younger than
    SKIP_SUMMARY_H. This is the only Hub write a backlog skip ever causes."""
    marks = [_parse(d.get("at")) for d in done if d.get("kind") == "backlog_skips"]
    last = max([t for t in marks + [_parse(summarized_at)] if t], default=None)
    if last and now - last < dt.timedelta(hours=SKIP_SUMMARY_H):
        return None
    new = sorted(({"gauge": g, "at": v.get("at"), "reason": v.get("reason")}
                  for g, v in (local or {}).items()
                  if last is None or (_parse(v.get("at")) or now) > last),
                 key=lambda r: r["at"] or "")
    if not new:
        return None
    return {"kind": "backlog_skips", "gauge": None, "event": None, "ran": False,
            "west": True, "at": now.strftime(TS), "n": len(new),
            "reason": f"{len(new)} western backlog gauge(s) skipped without a run"
                      + (f" since {last.strftime(TS)}" if last else ""),
            "skipped": new}


def window(rec: dict, now: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
    """Calibration window (whole hours): WINDOW_D days ending at the event's
    end — miss: last check's t0 + 6 h; oversim: the event's t_end; backlog:
    now — and never later than now - 2 h."""
    kind = rec.get("kind", "miss")
    if kind == "oversim":
        t_end = rec.get("_end") or _parse(rec.get("t_end")) or now
    elif kind == "backlog":
        t_end = now
    else:
        checks = rec.get("checks") or [rec]
        t0s = ([t for t in (_parse(c.get("t0")) for c in checks) if t]
               or [c["_at"] for c in checks if c.get("_at")])
        t_end = (max(t0s) if t0s else now) + dt.timedelta(hours=6)
    t_end = min(t_end, now - dt.timedelta(hours=2))
    t_start = t_end - dt.timedelta(days=WINDOW_D)
    return (t_start.replace(minute=0, second=0, microsecond=0),
            t_end.replace(minute=0, second=0, microsecond=0))


def calibrate_one(rec: dict, now: dt.datetime) -> dict:
    from hf_data import calibrate, llm, obs
    gid = str(rec["gauge"]).zfill(8)
    kind = rec.get("kind", "miss")
    t_start, t_end = window(rec, now)
    base = {"event": rec.get("event"), "gauge": gid, "kind": kind,
            "west": bool(rec.get("west"))}
    if kind == "miss":
        checks = rec.get("checks") or [rec]
        base.update({"n_checks": len(checks), "events": rec.get("events") or [rec["event"]],
                     "streak": [rec["first_at"].strftime(TS) if rec.get("first_at") else None,
                                rec["last_at"].strftime(TS) if rec.get("last_at") else None],
                     "obs_peak_m3s": rec.get("obs_peak_m3s"),
                     "sim_peak_m3s": rec.get("sim_peak_m3s")})
    elif kind == "oversim":
        base.update({"events": [rec["event"]], "t0": rec.get("t0"), "t_end": rec.get("t_end"),
                     "ratio": round(float(rec["ratio"]), 2), "n_paired": rec.get("n_paired"),
                     "obs_peak_m3s": rec.get("obs_peak_m3s"),
                     "sim_peak_m3s": rec.get("sim_peak_m3s")})
    else:
        base.update({k: rec.get(k) for k in ("tier", "area_km2", "stor_mm", "donor", "reasons")})
    base.update({"window": [t_start.strftime("%Y-%m-%d %H:%M"), t_end.strftime("%Y-%m-%d %H:%M")],
                 "at": now.strftime(TS), "llm": llm.available()})
    # observations must cover the window — an ungauged-in-practice site
    # cannot be calibrated, and a wide gap makes the NSE meaningless
    info = {}
    series = obs.get_series(gid, t_start, t_end, info=info)
    hours = max(1.0, (t_end - t_start).total_seconds() / 3600.0)
    # NWIS rows are 15-min: count the distinct HOURS observed, not the rows
    frac = min(1.0, len({t.replace(minute=0, second=0, microsecond=0)
                         for t, _ in series}) / hours)
    if frac < OBS_MIN_FRAC:
        return {**base, "ran": False,
                "reason": f"obs cover {100 * frac:.0f}% of the window (< {100 * OBS_MIN_FRAC:.0f}%)"
                          + (f"; fetch: {info.get('fetch_error')}" if info.get("fetch_error") else "")}
    what = (f"{kind} event {rec['event']}" if rec.get("event")
            else f"western backlog tier {rec.get('tier')}")
    log(f"calibrating {gid} for {what}: window {base['window'][0]} -> "
        f"{base['window'][1]} ({WINDOW_D:g} d), obs cover {100 * frac:.0f}%, "
        f"{ROUNDS} rounds x {K} candidates, LLM {llm.available()}")
    done = None
    n_runs = 0
    for kind, payload in calibrate.run_calibration(
            gid, t_start, t_end, model="auto", snow="auto", use_mock=False,
            rounds=ROUNDS, k=K, save_vs_baseline=True):
        why = busy_reason() if kind in ("status", "round") else None
        if why:
            raise Preempted(why)
        if kind == "status":
            log(f"  {payload}")
        elif kind == "round":
            n_runs += len(payload.get("tried") or [])
            bn = payload.get("best_nse")
            log(f"  round {payload.get('round')}: best NSE "
                f"{'n/a' if bn is None else f'{bn:.3f}'}")
        elif kind == "done":
            done = payload
    if not done or done.get("error"):
        return {**base, "ran": False, "reason": f"calibration error: {(done or {}).get('error', 'no result')}"}
    bp = done.get("best_params") or {}
    return {**base, "ran": True, "n_runs": n_runs,
            "baseline_nse": done.get("baseline_nse"), "best_nse": done.get("best_nse"),
            "improved": bool(done.get("improved")), "saved": bool(done.get("saved")),
            "extended": bool(done.get("extended")),
            "best_params": {k: round(float(v), 4) for k, v in bp.items()
                            if isinstance(v, (int, float))}}


def _load_local_state():
    """Backlog skip list + last summary time from a previous run of this loop
    in the same container (/tmp survives the runner's loop restarts, not a
    Space restart — the Hub summaries cover that)."""
    try:
        with open(STATE_PATH) as fp:
            old = json.load(fp)
        state["backlog_skips"] = {str(g): v for g, v in (old.get("backlog_skips") or {}).items()
                                  if isinstance(v, dict)}
        state["skips_summarized"] = old.get("skips_summarized")
    except (OSError, ValueError, AttributeError, TypeError):
        pass


def _summarize_skips(done: list, now: dt.datetime):
    """Prune expired local skips; commit a summary record when one is due."""
    from hf_data import eventstore
    cut = now - dt.timedelta(days=SKIP_TTL_D)
    state["backlog_skips"] = {g: v for g, v in state["backlog_skips"].items()
                              if (_parse(v.get("at")) or now) > cut}
    rec = skip_summary(done, state["backlog_skips"], now, state.get("skips_summarized"))
    if rec and eventstore.note_autocal(rec):
        state["skips_summarized"] = rec["at"]
        log(f"backlog skip summary committed: {rec['reason']}")


def _note_held(held: list):
    """Reservoir-gauge event candidates held until AUTOCAL_LAKES_ON=1: kept in
    the state file (status page) and logged when the set changes — no Hub
    write, they are simply not eligible."""
    key = sorted((h["gauge"], h["event"]) for h in held)
    if key != sorted((h["gauge"], h["event"]) for h in state["held_reservoir"]):
        log(f"holding {len(held)} reservoir-gauge event candidate(s) until "
            f"AUTOCAL_LAKES_ON=1: " + (", ".join(
                f"{h['gauge']} ({h['kind']}, {h['stor_mm']:.0f} mm"
                f"{', west' if h['west'] else ''})" for h in held) or "none"))
    state["held_reservoir"] = held


def main():
    from hf_data import eventstore, llm
    _load_local_state()
    if not os.environ.get("HF_TOKEN"):
        state["phase"] = "NO HF_TOKEN — add the secret in Space settings"
        _dump_state()
        log(state["phase"])
        while True:
            time.sleep(3600)
    idle_since = None
    while True:
        now = dt.datetime.utcnow()
        try:
            if REQUIRE_LLM and not llm.available():
                state["phase"] = "waiting for an LLM key (OPENAI_API_KEY secret or VLLM_BASE_URL)"
                _dump_state()
                log(state["phase"])
                time.sleep(POLL_S)
                continue
            busy = busy_reason()
            if busy:
                idle_since = None
            elif idle_since is None:
                idle_since = time.time()
            if busy or time.time() - idle_since < IDLE_FOR_S:
                state["phase"] = ("yielding — " + busy) if busy else (
                    f"idle for {(time.time() - idle_since) / 60:.0f} of "
                    f"{IDLE_FOR_S / 60:.0f} min before starting")
                _dump_state()
                log(state["phase"])
                time.sleep(min(POLL_S, 300))
                continue
            missed = eventstore.load_missed()
            index = eventstore.load_index()
            done = eventstore.load_autocal()
            skips = backlog_skips(done, state["backlog_skips"], now)
            held = []
            rec, why = pick_candidate(missed, index, done, now, backlog=load_backlog(),
                                      skips=skips, held=held)
            _note_held(held)
            if rec is None:
                _summarize_skips(done, now)
                state["phase"] = (f"idle — {why} ({len(missed)} miss rows, "
                                  f"{len(index)} events, {len(done)} records)")
                _dump_state()
                log(state["phase"])
                time.sleep(POLL_S)
                continue
            state["phase"] = f"calibrating {rec['gauge']} ({rec['kind']}; {why})"
            state["current"] = {"event": rec["event"], "gauge": rec["gauge"],
                                "kind": rec["kind"], "west": rec["west"],
                                "started": now.strftime(TS)}
            _dump_state()
            if rec["kind"] == "backlog":
                try:
                    result = calibrate_one(rec, now)
                except Preempted:
                    raise
                except Exception as e:     # a backlog gauge that always throws must
                    log(traceback.format_exc())    # not pin the head of the list
                    result = {"gauge": rec["gauge"], "kind": "backlog", "ran": False,
                              "at": now.strftime(TS),
                              "reason": f"error: {type(e).__name__}: {e}"}
            else:
                result = calibrate_one(rec, now)
            if rec["kind"] == "backlog" and not result.get("ran"):
                # no Hub commit per backlog skip (hundreds of them would each
                # rewrite events/autocal.json): local list + a periodic summary
                state["backlog_skips"][result["gauge"]] = {
                    "at": result["at"], "reason": str(result.get("reason") or "")[:200]}
                _summarize_skips(done, now)
            else:
                eventstore.note_autocal(result)
            state["runs" if result.get("ran") else "skips"] += 1
            state["last"] = result
            state["current"] = None
            log(f"record: {json.dumps(result)[:400]}")
        except Preempted as p:             # nothing recorded: retried when idle
            state["phase"] = f"preempted — {p}"
            state["current"] = None
            idle_since = None
            log(state["phase"])
        except Exception as e:
            state["phase"] = f"error: {type(e).__name__}: {e}"
            state["current"] = None
            log(traceback.format_exc())
            time.sleep(POLL_S)
        _dump_state()
        time.sleep(30)


if __name__ == "__main__":
    main()
