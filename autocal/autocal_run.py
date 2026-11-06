"""CREST_autocal — auto-triggered AI calibration of gauges whose CREST 1-D
simulation scores a low NSE in a flood episode, western gauges first.

Runs forever on its own HF Space (autocal_space/). The lowest-priority compute
class in CREST-AI (user 2026-09-25: user-initiated operations > nowcast >
2-D inundation > data update > fleet runner > this), so it lives on a
dedicated cpu-basic Space, calibrates ONE gauge at a time, and caps how many
LLM-guided calibrations it starts per day. It takes no 2-D compute and never
touches the 2-D queue, a worker or an episode's runs (PI 2026-11-05: "2D run
has priority. Autocal has lower priority than 2D run"; "Never abort a 2D run
as long as the nowcasting put the event in the queue").

Trigger (PI 2026-11-05: "it should be low NSCE score triggers auto call"):
  * LOW NSE (kind "low_nse") — a flood episode in events/index.json (status
    active or ended; a fragment the episode merger absorbed, merged_into, is
    skipped: its flood is judged on the episode it joined) whose CREST 1-D
    hydrograph at the trigger gauge has NSE < AUTOCAL_NSE_MAX (0.1): the
    manifest hydro rows with both sim_q and obs_q, observed part only (time
    <= t0), over the episode window — from its earliest start (sim_start,
    episode_start or the first depth frame: a warm re-run's sim_start is only
    its fresh slice) — with >= AUTOCAL_NSE_MIN_PAIRS (12) paired hours and a
    non-constant observed series. No quiet-period wait: an episode still
    running qualifies ("within an episode of an event, the CREST 1D
    simulation has score lower than 0.1, then the autocal starts").
    Manifests are HF-cached and the verdict is kept per (event, generated):
    a poll costs no download for an episode already judged; each re-publish
    of an active episode is judged again.
  ONE candidate per gauge: a gauge's eligible episodes are grouped, the
  gauge ranks by its worst NSE, and its record lists them all in `events`
  (the worst one in `event`, each one's NSE in episode_nse).
  Eligible episodes: the gauge was not auto-calibrated within
  AUTOCAL_COOLDOWN_D days; the episode is not handled — handled means a
  calibration that RAN (ran=True, saved or not) listed it, or any episode it
  stands for after a merge (its index merged.ids, its manifest's
  merged_from, the entries merged_into it); and it was last published after
  the gauge's last calibration FINISHED (record / ledger `finished`; older
  rows: their start `at`) — an earlier one's hydrograph predates the new
  parameters. The episode merger re-stamps a merge target's `generated`
  with the merge time (provenance.merged.at) without simulating anything:
  such a target counts as published at the latest `generated` of the
  entries it absorbed and its own t0 (published_at).
  Skips: a calibration that did NOT run (obs cover < AUTOCAL_OBS_MIN_FRAC of
  the window or none at all, most of a flood's paired hours missing, a
  calibration error — any exception of the gauge's own calibration,
  run_candidate —, no EF5 NSE, no candidate run with an NSE, or
  AUTOCAL_OUTAGE_STRIKES outages in a row on the gauge) is no handled
  record — the data may come back, and an obs failure the outage checks miss
  (NWIS answering 200 with an empty series, a retired endpoint's 404/410)
  would otherwise retire every eligible episode within minutes. A cover
  under AUTOCAL_OBS_MIN_FRAC decided in part on the obs store's earlier
  answers is asked of NWIS once more first (obs.get_series refresh), and
  obs.get_series caches an empty answer as covered only for a window that
  stored rows bracket on both sides — a quiet NWIS failure (a gap open
  towards now) is asked again. The flood itself must be in the series: the
  hours the runner paired obs at in each episode (flood_spans: its
  nse_span) — NWIS had them then — are asked of NWIS once more when most of
  them are missing, and still missing it is a skip (QUIET_EMPTY, the error
  TTL): never a calibration without the flood's observations. It is a local
  skip (state "skipped": gauge -> at, reason, events) standing
  AUTOCAL_SKIP_TTL_D days for too little obs (some, NWIS answered rows
  without error, the flood's there), about one poll for an outage streak
  and AUTOCAL_SKIP_ERR_TTL_H hours for anything else (skip_ttl); the gauge is
  retried after it, or as soon as a new eligible episode joins it. The Hub
  sees skips only in one "low_nse_skips" summary record at most every
  AUTOCAL_SKIP_SUMMARY_H (no commit per skip), which also re-seeds the list
  after a Space restart; ran=False rows of older kinds count as skips the
  same way. After AUTOCAL_MAX_SKIP_RUN skips in a row the loop pauses until
  the next poll (a failure the checks missed must not skip every gauge).
  Retired triggers: an outlet-Q MISS triggers nothing any more
  (events/missed.json is no longer written: the live runner treats outlet-Q
  misses as advisory and the 2-D run continues), nor do the western
  OVER-PREDICTION and the WESTERN BACKLOG
  (hf_data/data/autocal_west_backlog.json and scripts/build_autocal_backlog.py
  stay in the repo; nothing here reads the backlog). Older autocal.json rows
  keep their kinds (miss, oversim, backlog, backlog_skips).
  No calibrate-once: that rule served the idle-time backlog walk. A gauge
  whose next episode again scores below the threshold after the cooldown is
  calibrated again, on a window holding the new flood — a permanent "once"
  would leave it at a low NSE for good. The ledger events/autocal_done.json
  (eventstore.note_autocal keeps it with every ran=True record) is still
  read: its last calibration per gauge (start and finish) keeps the cooldown
  and the stale-episode rule working after the gauge's rows roll off
  autocal.json (last 500 rows).
While the episode runs (PI 2026-11-05): calibrating its gauge changes nothing
on the 2-D side. A winner that beats the current parameters goes to
CREST_state params/, where the episode's next hourly re-run picks it up
(paramstore.get); whatever the NSE after calibration, the 2-D runs carry on
unchanged ("if after autocal for necessary round of calibration, the NSCE is
still lower than 0.1, then the 2D simulation still starts"), and the episode
is not calibrated again.

Priority (PI 2026-10-27: "The auto-calibration, we should prioritize the
western part of the USA, since those areas have more issues"): western
gauges (lon < pipeline.WEST_LON) first, then the lowest NSE.

Reservoir hold (PI 2026-10-27): a WESTERN reservoir-affected gauge (lon <
WEST_LON and GAGES-II STOR_NOR_2009 >= 10 mm,
hf_data/data/gagesii_reservoir_stor.json) is not calibrated until
AUTOCAL_LAKES_ON=1 — the lake/reservoir module is being switched on
for the west (CREST_LAKES=west), and calibrating before it simulates the
reservoirs would bake them into the parameters (e.g. 10171000 Jordan River,
267 mm; 09337500 Escalante, 11 mm). The flag lifts the hold only while the
lake module is really live here: hf_data.lakes importable, lakes.region()
covering the west and a non-empty lakes.table_version() — otherwise the hold
stays and the phase line says why. Eastern gauges are not held: their lakes
are not simulated any time soon, so a hold would block them indefinitely.
Held episodes are filtered out of eligibility without a Hub write; they are
listed in the state file (held_reservoir) and the phase line.

Idle gate (busy_reason): autocal yields ONLY to user-initiated work — the
dashboard's /api/health user_ops (simulations and calibrations a user
started, running or waiting for a slot) must stay zero for AUTOCAL_IDLE_FOR_S
before a calibration starts, and is read again between rounds: user work
appearing mid-calibration preempts it (Preempted: nothing recorded, retried
when idle). Unreadable health counts as busy (fails closed). The former 2-D
gates (unpublished bundles, active episodes, worker claims, the runner's
tick) are gone (PI 2026-11-05): this Space shares no CPU, GPU or queue with
the 2-D solve, so they protected nothing and kept autocal idle through every
flood — exactly when low-NSE episodes appear. Nowcast, data update and the
fleet runner run on their own Spaces as well and are no gate either.
Outages: an obs fetch that failed for a reason that is not the gauge's
(connection error, timeout, an answer that is no JSON, NWIS 5xx/401/403/429:
fetch_outage — the HTTP codes by is_outage's rule), a network/Hub/memory
exception (is_outage), a calibration whose candidate runs all failed with
such exceptions, or — AUTOCAL_REQUIRE_LLM — no round with LLM candidates (a
provider outage or an exhausted key fails every gauge alike; run_calibration
stops before a heuristic-only search, so nothing is saved or pushed) records
nothing and backs off AUTOCAL_POLL_S; the gauge is retried — until
AUTOCAL_OUTAGE_STRIKES (4) of them in a row on that gauge (counted per
gauge, _outage): then it is a skip of about one poll and the next gauge
gets its turn, so a fault only that gauge triggers cannot pin the queue,
while a global outage, which moves the strikes from gauge to gauge, leaves
the order as it was once it is over. An obs fetch error that is the site's
own (another 4xx, its answer failing to parse) and any other exception of
the gauge's calibration are skips with the error TTL. An exception outside
a calibration is shown as the phase and retried next poll.
History: an unreadable autocal.json or ledger is never "no history" — the
loop waits (phase "history unreadable"). A record whose commit failed stays
pending in AUTOCAL_STATE, counts for the daily cap, handled pairs and
cooldown, and is retried every poll until it is on the Hub. A calibration
whose winner was saved but not pushed to CREST_state (run_calibration's
"pushed" False) is unfinished: kept in AUTOCAL_STATE ("unpushed"), counted
the same way, its push retried every loop; it is recorded (ran=True,
finished at the push) only once the parameters are on CREST_state.
Gates for every calibration: fewer than AUTOCAL_MAX_PER_DAY calibrations
(ran=True, any kind) started in the last 24 h, the idle gate (busy_reason),
and USGS observations covering >= AUTOCAL_OBS_MIN_FRAC of the window.
Calibration window: AUTOCAL_WINDOW_D days (default 183 — two seasons; user:
"some dry lands don't see anything in 30 days") ending at the latest t_end
(t0 without one) of the gauge's eligible episodes — an active one's current
t0 when one is active — and never later than now - 2 h. Records carry kind,
west, nse, n_pairs, nse_max (the threshold), events, episode_nse, obs_cover,
flood_obs_h (each episode's paired hours seen / needed), the status when
picked (active when one of the episodes is) and, when the calibration ran,
its rounds and llm_rounds, n_nse (candidate runs with an NSE), saved,
pushed, model and when it finished.
Method: hf_data.calibrate.run_calibration (LLM-proposed multiplier updates
within hydrologic bounds, judged by real EF5 runs), saving the winner only if
it beats the CURRENT parameters on this window (save_vs_baseline) and pushing
it to CREST_state params/ where paramstore.get on every other Space picks it
up. A calibration in which no EF5 run, or no candidate run (round payloads'
tried[].nse, the baseline aside), produced an NSE is ran=False. Without
an LLM key it waits, and without LLM candidates it does not calibrate
(AUTOCAL_REQUIRE_LLM=1, run_calibration require_llm): heuristic-only
search is not the "AI calibration" this Space exists for.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # CREST_AI checkout root

POLL_S = float(os.environ.get("AUTOCAL_POLL_S", "900"))
COOLDOWN_D = float(os.environ.get("AUTOCAL_COOLDOWN_D", "30"))
MAX_PER_DAY = int(os.environ.get("AUTOCAL_MAX_PER_DAY", "2"))
WINDOW_D = float(os.environ.get("AUTOCAL_WINDOW_D", "183"))
OBS_MIN_FRAC = float(os.environ.get("AUTOCAL_OBS_MIN_FRAC", "0.5"))
ROUNDS = int(os.environ.get("AUTOCAL_ROUNDS", "4"))
K = int(os.environ.get("AUTOCAL_K", "3"))
REQUIRE_LLM = os.environ.get("AUTOCAL_REQUIRE_LLM", "1") == "1"
STATE_PATH = os.environ.get("AUTOCAL_STATE", "/tmp/autocal_state.json")
# low-NSE trigger (PI 2026-11-05): CREST 1-D NSE over a flood episode
NSE_MAX = float(os.environ.get("AUTOCAL_NSE_MAX", "0.1"))
NSE_MIN_PAIRS = int(os.environ.get("AUTOCAL_NSE_MIN_PAIRS", "12"))
# a calibration that did not run is a local skip, never a handled record:
# too little obs stands SKIP_TTL_D days, anything else SKIP_ERR_TTL_H hours
SKIP_TTL_D = float(os.environ.get("AUTOCAL_SKIP_TTL_D", "30"))
SKIP_ERR_TTL_H = float(os.environ.get("AUTOCAL_SKIP_ERR_TTL_H", "24"))
SKIP_SUMMARY_H = float(os.environ.get("AUTOCAL_SKIP_SUMMARY_H", "24"))
MAX_SKIP_RUN = int(os.environ.get("AUTOCAL_MAX_SKIP_RUN", "3"))
# outages in a row on one gauge before it is a skip (error TTL): a fault only
# that gauge triggers must not pin the head of the queue
OUTAGE_STRIKES = int(os.environ.get("AUTOCAL_OUTAGE_STRIKES", "4"))
SKIPS_KIND = "low_nse_skips"                     # the periodic skip summary record
QUIET_EMPTY = "NWIS answered no row for the window"   # skip reason note (skip_ttl)
FLOOD_MISSING = "flood obs missing"                   # skip reason note (skip_ttl)
LAKES_ON = os.environ.get("AUTOCAL_LAKES_ON", "0") == "1"
RESERVOIR_PATH = os.environ.get("AUTOCAL_RESERVOIRS", os.path.join(
    os.path.dirname(HERE), "hf_data", "data", "gagesii_reservoir_stor.json"))
# Idle gate (user 2026-10-12: "auto_cal has the lowest priority — only when
# there is enough computation"): start, and keep going between rounds, only
# while the dashboard's /api/health shows no user-initiated work (ranking,
# user 2026-09-25: user-initiated > nowcast > 2-D inundation > data update >
# fleet runner > autocal). Every class between runs on other Spaces, and
# autocal takes no 2-D compute (PI 2026-11-05: 2-D has priority, never
# aborted), so none of them is a gate. The snapshot is cached 120 s
# server-side.
HEALTH_URL = os.environ.get("AUTOCAL_HEALTH_URL",
                            "https://vincewin-crest-demo.hf.space/api/health")
IDLE_FOR_S = float(os.environ.get("AUTOCAL_IDLE_FOR_S", "1800"))

TS = "%Y-%m-%dT%H:%M:%SZ"
state = {"phase": "booting", "current": None, "last": None, "runs": 0, "skips": 0,
         "skipped": {}, "skips_summarized": None, "held_reservoir": [], "pending": [],
         "unpushed": [], "outage": {}}


def log(s: str):
    print(f"{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}Z {s}", flush=True)


def _dump_state():
    try:
        with open(STATE_PATH, "w") as fp:
            json.dump(state, fp)
    except OSError:
        pass


class Preempted(Exception):
    """User-initiated work appeared mid-calibration: drop it, retry later."""


class Outage(Exception):
    """Trouble that is not the gauge's (obs fetch, network, Hub): record
    nothing, back off, retry later."""


def _outage_code(code: int) -> bool:
    """An HTTP status that says nothing about the gauge: 5xx, 401/403 (auth,
    a WAF block) or 429 (rate limit) — every gauge would get it alike. Any
    other 4xx (no such file / site) is the gauge's own."""
    return code >= 500 or code in (401, 403, 429)


def is_outage(e: BaseException) -> bool:
    """True when an exception says nothing about the gauge: network, Hub,
    auth or memory trouble that would fail every gauge alike (_outage_code
    for an HTTP answer)."""
    if isinstance(e, (ConnectionError, TimeoutError, MemoryError)):
        return True
    code = getattr(getattr(e, "response", None), "status_code", None)
    if isinstance(code, int):
        return _outage_code(code)
    roots = {c.__module__.split(".")[0] for c in type(e).__mro__}
    return bool(roots & {"requests", "urllib3", "httpx", "huggingface_hub", "urllib"})


def fetch_outage(err: BaseException | None) -> bool:
    """True when obs.get_series' failed fetch (its info "fetch_exc") says
    nothing about the gauge: a connection error or timeout (a connection
    broken mid-answer included), an HTTP answer with an outage code
    (_outage_code: 5xx, 401, 403, 429 — is_outage's rule) or an answer that
    is no JSON (a WAF or maintenance page). Anything else is the site's own:
    another 4xx (no such site, a retired endpoint) or its answer failing to
    parse in fetch_usgs_discharge (IndexError, KeyError, a bad timestamp) —
    a skip with the error TTL, not an outage retried every poll."""
    if not isinstance(err, BaseException):
        return False
    code = getattr(getattr(err, "response", None), "status_code", None)
    if isinstance(code, int):
        return _outage_code(code)
    import requests
    rx = requests.exceptions
    return isinstance(err, (ConnectionError, TimeoutError, json.JSONDecodeError,
                            rx.ConnectionError, rx.Timeout, rx.ChunkedEncodingError,
                            getattr(rx, "InvalidJSONError", json.JSONDecodeError)))


def busy_reason() -> str | None:
    """None when no user-initiated work runs on the dashboard, else why not.
    User work is the only class autocal yields to (2-D, nowcast, data update
    and the fleet run on other Spaces). Fails CLOSED: an unreadable health
    snapshot, or one without user_ops, counts as busy."""
    try:
        import requests
        h = requests.get(HEALTH_URL, timeout=30).json()
    except Exception as e:
        return f"health unreadable ({type(e).__name__})"
    u = h.get("user_ops") if isinstance(h, dict) else None
    if not isinstance(u, dict) or "sims" not in u:
        err = u.get("error") if isinstance(u, dict) else None
        return (f"health user_ops unreadable ({err})" if err
                else "health has no user_ops (dashboard not updated?)")
    if u.get("sims") or u.get("cals"):
        return f"user work running ({u.get('sims')} sims, {u.get('cals')} cals)"
    return None


def _num(v) -> bool:
    """True for a finite number (an NSE that was computed)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _parse(s: str | None) -> dt.datetime | None:
    for fmt in (TS, "%Y-%m-%dT%H:%MZ", "%Y-%m-%d %H:%M"):     # last: manifest hydro rows
        try:
            return dt.datetime.strptime(str(s), fmt)
        except (TypeError, ValueError):
            continue
    return None


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


def lake_hold() -> str | None:
    """None when the western reservoir hold is lifted, else why it stays (read
    after "held "). AUTOCAL_LAKES_ON=1 lifts it only while the lake module is
    live here: a flag without hf_data.lakes, without CREST_LAKES covering the
    west or without a lake table would calibrate the reservoirs into the
    parameters just the same."""
    if not LAKES_ON:
        return "until AUTOCAL_LAKES_ON=1"
    try:
        from hf_data import lakes
        reg, ver = lakes.region(), lakes.table_version()
    except Exception as e:
        return f"although AUTOCAL_LAKES_ON=1: hf_data.lakes unavailable ({type(e).__name__})"
    if reg not in ("west", "all"):
        return f"although AUTOCAL_LAKES_ON=1: lakes.region() is {reg!r}, not the west"
    if not ver:
        return "although AUTOCAL_LAKES_ON=1: lakes.table_version() is empty (no lake table)"
    return None


def held_stor(gid: str) -> float | None:
    """Reservoir storage (mm) when this gauge is held back from calibration
    (western, reservoir-affected and the hold not lifted, lake_hold), else
    None. The lake module is switched on for the west only (CREST_LAKES=west)."""
    if not is_west(gid) or lake_hold() is None:
        return None
    return reservoir_stor(gid)


def _history(done: list, now: dt.datetime, ledger: dict | None = None):
    """(last ran=True start per gauge, handled (gauge, event) pairs, ran=True
    records started in the last 24 h) from events/autocal.json rows (plus
    pending ones), the last start also from the ledger (autocal_done.json).
    Handled: every episode of a calibration that RAN (its `event` and
    `events`); a ran=False row is a skip with a TTL (active_skips), never a
    handled episode."""
    last_run = {}
    for g, e in (ledger or {}).items():
        t = _parse(e.get("last")) if isinstance(e, dict) else None
        if t:
            last_run[str(g).zfill(8)] = t
    handled = set()
    for d in done:
        if d.get("ran") and d.get("gauge"):
            g, t = str(d["gauge"]).zfill(8), _parse(d.get("at"))
            if t and (g not in last_run or t > last_run[g]):
                last_run[g] = t
            handled |= {(g, e) for e in [d.get("event"), *(d.get("events") or [])] if e}
    today = sum(1 for d in done if d.get("ran") and (_parse(d.get("at")) or now)
                > now - dt.timedelta(hours=24))
    return last_run, handled, today


def _finished(done: list, ledger: dict | None = None) -> dict:
    """{gauge: when its last calibration (ran=True) FINISHED} — the mark of
    the stale-episode rule: an episode published while a calibration ran was
    still simulated with the parameters it replaced. A record's `finished`
    (older rows: their start `at`, the best they have); the ledger's latest
    `finished`, or its `last` start when that is later (a start without a
    finish recorded)."""
    out = {}
    marks = [(g, (e.get("finished"), e.get("last"))) for g, e in (ledger or {}).items()
             if isinstance(e, dict)]
    marks += [(d["gauge"], (d.get("finished"), d.get("at"))) for d in done
              if d.get("ran") and d.get("gauge")]
    for g, ts in marks:
        ts = [t for t in (_parse(x) for x in ts) if t]
        g = str(g).zfill(8)
        if ts and (g not in out or max(ts) > out[g]):
            out[g] = max(ts)
    return out


def skip_ttl(reason: str | None) -> dt.timedelta:
    """How long a skip (a calibration that did not run) stands: SKIP_TTL_D
    for too little observed flow — some obs, under OBS_MIN_FRAC, fetched
    without error: that does not change within days — and SKIP_ERR_TTL_H for
    anything else: no obs at all, a fetch error or NWIS answering no row for
    a window the store holds rows in (QUIET_EMPTY — an obs failure the
    outage checks missed looks just like that) or most of a flood's paired
    hours missing, a calibration error, no NSE. An outage streak
    (OUTAGE_STRIKES on one gauge) stands about one poll (POLL_S): long
    enough for the next gauge to get its turn, short enough that a global
    outage — every gauge in turn reaches its strikes — does not leave the
    western worst-NSE gauges behind the others once it is over."""
    r = str(reason or "")
    if r.startswith("outage "):
        return dt.timedelta(seconds=POLL_S)
    m = re.match(r"obs cover (\d+)%", r)
    if (m and int(m.group(1)) > 0 and "; fetch: " not in r and QUIET_EMPTY not in r
            and FLOOD_MISSING not in r):
        return dt.timedelta(days=SKIP_TTL_D)
    return dt.timedelta(hours=SKIP_ERR_TTL_H)


def active_skips(done: list, local: dict | None, now: dt.datetime) -> dict:
    """{gauge: {at, reason, events}} of calibrations that did not run, still
    within their skip_ttl: this container's local list, the rows of the Hub's
    "low_nse_skips" summaries (a Space restart re-probes nothing) and
    ran=False records (older kinds, pending ones). `events` is every episode
    a live skip of the gauge covered; the newest one gives at and reason."""
    rows = [r for d in done if d.get("kind") == SKIPS_KIND
            for r in (d.get("skipped") or []) if isinstance(r, dict)]
    rows += [{"gauge": d["gauge"], "events": d.get("events") or [d.get("event")],
              "at": d.get("at"), "reason": d.get("reason")}
             for d in done if not d.get("ran") and d.get("gauge")]
    rows += [{**v, "gauge": g} for g, v in (local or {}).items() if isinstance(v, dict)]
    out = {}
    for r in rows:
        g, t = r.get("gauge"), _parse(r.get("at"))
        if not g or t is None or t <= now - skip_ttl(r.get("reason")):
            continue
        o = out.setdefault(str(g).zfill(8), {"at": r["at"], "reason": r.get("reason"),
                                             "events": []})
        if t > _parse(o["at"]):
            o["at"], o["reason"] = r["at"], r.get("reason")
        o["events"] = sorted(set(o["events"]) | {str(e) for e in r.get("events") or [] if e})
    return out


def skip_summary(done: list, local: dict, now: dt.datetime,
                 summarized_at: str | None = None) -> dict | None:
    """One "low_nse_skips" record (ran=False, gauge None) listing the local
    skips newer than the last summary, or None when there is nothing new or
    the last summary (Hub record or `summarized_at`) is younger than
    SKIP_SUMMARY_H. This is the only Hub write a skip ever causes."""
    marks = [_parse(d.get("at")) for d in done if d.get("kind") == SKIPS_KIND]
    last = max([t for t in marks + [_parse(summarized_at)] if t], default=None)
    if last and now - last < dt.timedelta(hours=SKIP_SUMMARY_H):
        return None
    new = sorted(({"gauge": g, "events": list(v.get("events") or []), "at": v.get("at"),
                   "reason": v.get("reason")}
                  for g, v in (local or {}).items()
                  if last is None or (_parse(v.get("at")) or now) > last),
                 key=lambda r: (r["at"] or "", r["gauge"]))
    if not new:
        return None
    return {"kind": SKIPS_KIND, "gauge": None, "event": None, "ran": False,
            "at": now.strftime(TS), "n": len(new), "nse_max": NSE_MAX,
            "reason": f"{len(new)} low-NSE gauge(s) not calibrated (the calibration did "
                      f"not run)" + (f" since {last.strftime(TS)}" if last else ""),
            "skipped": new}


def _hold(c: dict, held: list | None) -> bool:
    """True (and noted in `held`) when candidate c is a held reservoir gauge."""
    stor = held_stor(c["gauge"])
    if stor is None:
        return False
    if held is not None:
        held.append({"gauge": c["gauge"], "event": c["event"], "events": c["events"],
                     "kind": c["kind"], "west": c["west"], "stor_mm": stor,
                     "nse": round(c["nse"], 3)})
    return True


def episode_start(manifest: dict, summary: dict | None = None) -> dt.datetime | None:
    """Start of an episode's simulation window: the earliest of the
    manifest's sim_start and episode_start, its first depth frame and the
    index summary's sim_start. A warm hourly re-run publishes the sim_start
    of its fresh slice only (frames before it are carried forward and
    episode_start records the first of them), so sim_start alone can be an
    hour before t0. None when none of them parses."""
    ts = [manifest.get("sim_start"), manifest.get("episode_start"),
          (summary or {}).get("sim_start")]
    ts += [f.get("t") for f in manifest.get("frames") or [] if isinstance(f, dict)]
    ts = [t for t in (_parse(x) for x in ts if x) if t]
    return min(ts) if ts else None


def episode_nse(manifest: dict, summary: dict | None = None) -> dict | None:
    """Nash-Sutcliffe efficiency of an episode's CREST 1-D hydrograph at the
    trigger gauge: manifest hydro rows with both sim_q and obs_q (finite),
    observed part only (time <= t0: later rows are the forecast tail) and
    inside the episode window (from episode_start). {nse, n_pairs,
    obs_peak_m3s, sim_peak_m3s, nse_span}, or None when it cannot be judged:
    no t0 or window start, fewer than NSE_MIN_PAIRS paired hours, or a
    constant observed series (NSE undefined)."""
    summary = summary or {}
    t0 = _parse(manifest.get("t0") or summary.get("t0"))
    t_from = episode_start(manifest, summary)
    if t0 is None or t_from is None:
        return None
    pairs = []
    for r in manifest.get("hydro") or []:
        if not isinstance(r, dict):
            continue
        t = _parse(r.get("time"))
        try:
            s, o = float(r.get("sim_q")), float(r.get("obs_q"))
        except (TypeError, ValueError):
            continue                             # None: no obs (or no sim) that hour
        if (t is None or t > t0 or t < t_from
                or not (math.isfinite(s) and math.isfinite(o))):
            continue
        pairs.append((t, s, o))
    if len(pairs) < NSE_MIN_PAIRS:
        return None
    obs = [o for _, _, o in pairs]
    if max(obs) == min(obs):
        return None                              # constant obs: no variance to explain
    mo = sum(obs) / len(obs)
    den = sum((o - mo) ** 2 for o in obs)
    num = sum((s - o) ** 2 for _, s, o in pairs)
    ts = [t for t, _, _ in pairs]
    return {"nse": 1.0 - num / den, "n_pairs": len(pairs),
            "obs_peak_m3s": round(max(obs), 2),
            "sim_peak_m3s": round(max(s for _, s, _ in pairs), 2),
            "nse_span": [min(ts).strftime("%Y-%m-%dT%H:%MZ"),
                         max(ts).strftime("%Y-%m-%dT%H:%MZ")]}


_verdicts: dict = {}             # (event, generated) -> _verdict


def _verdict(ev: str, summary: dict, manifest=None) -> dict | None:
    """What a listed episode's manifest says: {"stats": episode_nse (None:
    it cannot be judged), "merged_at": provenance.merged.at of a merge
    target (else None), "merged_from": the ids it absorbed}; None while the
    manifest is unreadable (not cached: retried). `manifest` (tests) maps an
    event id to its manifest dict; by default eventstore.load_manifest
    (HF-cached) with the verdict memoised per (event, generated), so a
    re-published episode is judged again and an unchanged one costs nothing."""
    def judge(m: dict) -> dict:
        prov = m.get("provenance") if isinstance(m.get("provenance"), dict) else {}
        mg = prov.get("merged") if isinstance(prov.get("merged"), dict) else {}
        mf = m.get("merged_from") if isinstance(m.get("merged_from"), list) else []
        return {"stats": episode_nse(m, summary), "merged_at": mg.get("at"),
                "merged_from": [str(i) for i in mf if i]}
    if manifest is not None:
        m = manifest(ev)
        return judge(m) if m else None
    key = (ev, summary.get("generated"))
    if key not in _verdicts:
        from hf_data import eventstore
        m = eventstore.load_manifest(ev)
        if m is None:
            return None                          # unreadable now: not cached, retried
        _verdicts[key] = judge(m)
    return _verdicts[key]


def published_at(summary: dict, index: dict, merged_at: str | None = None,
                 members=()) -> dt.datetime | None:
    """When an episode's hydrograph was last simulated: its `generated` —
    except on a merge target whose `generated` is still the merge's re-stamp
    (scripts/merge_episodes.py sets it to provenance.merged.at, `merged_at`,
    and simulates nothing): then the latest `generated` of the `members` it
    absorbed (index entries merged_into it keep theirs) and its own t0 (its
    last run came after it). A target re-published since the merge has a
    later `generated`: that one counts. Never later than `generated`."""
    gen = _parse(summary.get("generated"))
    at = _parse(merged_at)
    if at is None or (gen is not None and abs(gen - at) > dt.timedelta(minutes=1)):
        return gen
    ts = [_parse((index.get(m) or {}).get("generated")) for m in members
          if isinstance(index.get(m), dict)]
    ts = [t for t in ts + [_parse(summary.get("t0"))] if t]
    return max(ts) if ts else gen


def _latest(vals) -> str | None:
    """The latest of time strings (as given), None when none parses."""
    ok = [(t, v) for t, v in ((_parse(v), v) for v in vals if v) if t]
    return max(ok)[1] if ok else None


def _gauge_candidate(gid: str, eps: list) -> dict:
    """One gauge's candidate from its eligible episodes: the worst NSE's
    stats and id, every id in `events` (each one's NSE in episode_nse, its
    paired hours' span and count in episode_span / episode_pairs: the flood
    observations calibrate_one checks), status active when one is, and the
    window end _end: the latest t_end (t0 without one) — an active
    episode's current t0 when one is active."""
    eps = sorted(eps, key=lambda e: e["event"])
    worst = min(eps, key=lambda e: (e["nse"], e["event"]))
    active = [e for e in eps if e["status"] == "active"]
    ends = [t for t in (_parse(e["t0"]) for e in active) if t] or [
        t for t in (_parse(e["t_end"]) or _parse(e["t0"]) for e in eps) if t]
    return {**{k: worst[k] for k in ("nse", "n_pairs", "obs_peak_m3s", "sim_peak_m3s",
                                     "nse_span")},
            "kind": "low_nse", "gauge": gid, "event": worst["event"],
            "events": [e["event"] for e in eps],
            "episode_nse": {e["event"]: round(e["nse"], 3) for e in eps},
            "episode_span": {e["event"]: e["nse_span"] for e in eps if e.get("nse_span")},
            "episode_pairs": {e["event"]: e["n_pairs"] for e in eps},
            "west": is_west(gid), "status": "active" if active else "ended",
            "t0": _latest(e["t0"] for e in eps), "t_end": _latest(e["t_end"] for e in eps),
            "_end": max(ends) if ends else None}


def low_nse_candidates(index: dict, done: list, now: dt.datetime,
                       manifest=None, held: list | None = None,
                       ledger: dict | None = None) -> list[dict]:
    """One candidate per gauge (kind "low_nse", any region, _gauge_candidate)
    from its eligible episodes: active or ended, not absorbed by a merge, no
    episode they stand for handled, gauge out of its cooldown, last published
    (published_at) after the gauge's last calibration finished, CREST 1-D
    NSE at the trigger gauge below NSE_MAX. The cheap index-only rules run
    first, so only episodes that could still qualify ever have their
    manifest read. Reservoir gauges held until AUTOCAL_LAKES_ON=1 go to
    `held` instead."""
    index = index or {}
    last_run, handled, _ = _history(done, now, ledger)
    fin = _finished(done, ledger)
    for key in [k for k in _verdicts if k[0] not in index]:
        del _verdicts[key]                       # rolled off the list
    absorbed = {}                                # merge target -> entries merged_into it
    for e, s in index.items():
        if isinstance(s, dict) and s.get("merged_into"):
            absorbed.setdefault(str(s["merged_into"]), []).append(e)
    eps = {}
    for ev, s in index.items():
        if not isinstance(s, dict) or s.get("merged_into"):
            continue                             # a fragment: judged on its episode
        if s.get("status") not in ("active", "ended"):
            continue
        raw = s.get("gauge") or (s.get("trigger") or {}).get("gauge") or (
            ev.split("_")[-1] if "_" in ev else "")
        if not raw:
            continue
        gid = str(raw).zfill(8)
        mrec = s.get("merged") if isinstance(s.get("merged"), dict) else {}
        ids = {ev, *absorbed.get(ev, []), *(str(i) for i in mrec.get("ids") or [] if i)}
        if any((gid, i) in handled for i in ids):
            continue                             # a calibration ran for this flood
        lr, lf = last_run.get(gid), fin.get(gid)
        if lr and now - lr < dt.timedelta(days=COOLDOWN_D):
            continue                             # cooldown
        gen = _parse(s.get("generated"))
        if lf and gen and gen <= lf:
            continue                             # sim predates the new params
        v = _verdict(ev, s, manifest)
        if v is None:
            continue
        if any((gid, i) in handled for i in v["merged_from"]):
            continue
        pub = published_at(s, index, v["merged_at"], absorbed.get(ev, []))
        if lf and pub and pub <= lf:
            continue                             # a merge's re-stamp is no new sim
        st = v["stats"]
        if not st or st["nse"] >= NSE_MAX:
            continue
        eps.setdefault(gid, []).append({**st, "event": ev, "status": s.get("status"),
                                        "t0": s.get("t0"), "t_end": s.get("t_end")})
    cands = []
    for gid, es in eps.items():
        c = _gauge_candidate(gid, es)
        if not _hold(c, held):
            cands.append(c)
    return cands


def _key(c: dict):
    """West first, then the lowest NSE (then the event id: a stable order)."""
    return (not c["west"], c["nse"], c["event"])


def pick_candidate(index: dict, done: list, now: dt.datetime, manifest=None,
                   held: list | None = None, ledger: dict | None = None,
                   local_skips: dict | None = None):
    """The one gauge to work on next (or None) and a reason string: a
    western gauge first, then the worst episode NSE. A gauge whose
    calibration did not run lately (active_skips over `done` and
    `local_skips`, the loop's state "skipped") waits out its skip's TTL —
    unless an episode that skip did not cover joined it. Candidates on held
    (western reservoir) gauges are appended to `held` (when given), never
    picked. `ledger`: eventstore.load_autocal_done()."""
    # candidates first (cheap: verdicts are memoised) so the held list stays
    # current on the status page even while the daily cap is reached
    held = [] if held is None else held
    cands = low_nse_candidates(index, done, now, manifest=manifest, held=held,
                               ledger=ledger)
    why_hold = lake_hold()
    hold = (f"; {len(held)} western reservoir gauge(s) held {why_hold}" if held
            else f"; reservoir hold kept {why_hold}" if LAKES_ON and why_hold else "")
    _, _, today = _history(done, now)
    if today >= MAX_PER_DAY:
        return None, f"daily cap reached ({today}/{MAX_PER_DAY} in 24 h){hold}"
    skips = active_skips(done, local_skips, now)
    live = [c for c in cands
            if not set(c["events"]) <= set((skips.get(c["gauge"]) or {}).get("events") or ())]
    n_skip = len(cands) - len(live)
    skipped = (f"; {n_skip} gauge(s) wait out a skip (calibration did not run)"
               if n_skip else "")
    if live:
        live.sort(key=_key)
        n_west = sum(1 for c in live if c["west"])
        n_ep = sum(len(c["events"]) for c in live)
        return live[0], (f"{len(live)} eligible gauge(s), {n_west} western, with {n_ep} "
                         f"episode(s) of NSE < {NSE_MAX:g}{skipped}{hold}")
    return None, f"no eligible episode with NSE < {NSE_MAX:g}{skipped}{hold}"


def _span(sp) -> tuple[dt.datetime, dt.datetime] | None:
    """[start, end] (time strings or datetimes) -> datetimes, else None."""
    try:
        a, b = (x if isinstance(x, dt.datetime) else _parse(x) for x in sp)
    except (TypeError, ValueError):
        return None
    return (a, b) if a and b else None


def flood_spans(rec: dict, t_start: dt.datetime, t_end: dt.datetime) -> list[tuple]:
    """The hours whose observations the runner paired with the simulation in
    the candidate's episodes, clipped to the calibration window: (event,
    start, end, hours needed) for each — its paired span (episode_span; a
    record without them: its nse_span, the worst episode's; else the 2 days
    before its t0 to its t_end) and its paired hours (episode_pairs /
    n_pairs, at most the span's). NWIS had those observations when the
    runner paired them: a calibration that cannot see them is no
    calibration on the flood."""
    spans = rec.get("episode_span") if isinstance(rec.get("episode_span"), dict) else {}
    pairs = rec.get("episode_pairs") if isinstance(rec.get("episode_pairs"), dict) else {}
    items = [(ev, _span(spans[ev]), pairs.get(ev)) for ev in rec.get("events") or []
             if ev in spans]
    items = [i for i in items if i[1]]
    if not items and _span(rec.get("nse_span") or ()):
        items = [(rec.get("event"), _span(rec["nse_span"]), rec.get("n_pairs"))]
    if not items and _parse(rec.get("t0")):
        t0 = _parse(rec.get("t0"))
        items = [(rec.get("event"), (t0 - dt.timedelta(days=2), _parse(rec.get("t_end")) or t0),
                  rec.get("n_pairs"))]
    out = []
    for ev, (a, b), n in items:
        a = max(a, t_start).replace(minute=0, second=0, microsecond=0)
        b = min(b, t_end)
        if b < a:
            continue
        hours = int((b - a).total_seconds() // 3600) + 1
        try:
            need = min(hours, int(n)) if n else hours
        except (TypeError, ValueError):
            need = hours
        out.append((ev, a, b, need))
    return out


def window(rec: dict, now: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
    """Calibration window (whole hours): WINDOW_D days ending at the
    candidate's _end (_gauge_candidate: the latest t_end of its episodes, an
    active one's t0) — else its t_end, else its t0 — and never later than
    now - 2 h."""
    t_end = rec.get("_end") or _parse(rec.get("t_end")) or _parse(rec.get("t0")) or now
    t_end = min(t_end, now - dt.timedelta(hours=2))
    t_start = t_end - dt.timedelta(days=WINDOW_D)
    return (t_start.replace(minute=0, second=0, microsecond=0),
            t_end.replace(minute=0, second=0, microsecond=0))


def calibrate_one(rec: dict, now: dt.datetime) -> dict:
    from hf_data import calibrate, llm, obs
    gid = str(rec["gauge"]).zfill(8)
    kind = rec.get("kind", "low_nse")
    t_start, t_end = window(rec, now)
    events = list(rec.get("events") or [rec.get("event")])
    base = {"event": rec.get("event"), "gauge": gid, "kind": kind,
            "west": bool(rec.get("west")), "events": events,
            "episode_nse": rec.get("episode_nse"),
            "status": rec.get("status"), "t0": rec.get("t0"), "t_end": rec.get("t_end"),
            "nse": round(float(rec["nse"]), 3), "n_pairs": rec.get("n_pairs"),
            "nse_max": NSE_MAX, "nse_span": rec.get("nse_span"),
            "obs_peak_m3s": rec.get("obs_peak_m3s"), "sim_peak_m3s": rec.get("sim_peak_m3s")}
    base.update({"window": [t_start.strftime("%Y-%m-%d %H:%M"), t_end.strftime("%Y-%m-%d %H:%M")],
                 "at": now.strftime(TS), "llm": llm.available()})
    # observations must cover the window — an ungauged-in-practice site
    # cannot be calibrated, and a wide gap makes the NSE meaningless
    hours = max(1.0, (t_end - t_start).total_seconds() / 3600.0)

    def hours_of(series, a=None, b=None) -> int:
        # NWIS rows are 15-min: count the distinct HOURS observed, not the rows
        return len({t.replace(minute=0, second=0, microsecond=0) for t, _ in series
                    if a is None or a <= t <= b})

    def cover(series) -> float:
        return min(1.0, hours_of(series) / hours)
    info = {}
    series = obs.get_series(gid, t_start, t_end, info=info)
    frac = cover(series)
    quiet = False
    if (frac < OBS_MIN_FRAC and info.get("from_store")
            and not fetch_outage(info.get("fetch_exc"))):
        # decided (in part) on the store's earlier answers — an empty one may
        # have been NWIS failing quietly: ask NWIS for the whole window once
        info = {}
        series = obs.get_series(gid, t_start, t_end, info=info, refresh=True)
        frac = cover(series)
        base["obs_refetched"] = True
        # no row for the whole window, yet the store holds some in it: NWIS
        # is failing quietly again, not a gauge short of obs (skip_ttl)
        quiet = frac > 0 and info.get("empty_windows", 0) >= 1 and not info.get("fetch_error")
    if frac < OBS_MIN_FRAC and fetch_outage(info.get("fetch_exc")):
        # NWIS unreachable, not the gauge
        raise Outage(f"obs fetch for {gid} failed ({str(info['fetch_error'])[:160]}), "
                     f"cover {100 * frac:.0f}%")
    # the flood's own observations: the runner paired them with the
    # simulation, so NWIS has them — most of those hours missing is NWIS
    # failing (quietly, or a fetch error), never a gauge short of obs: ask
    # NWIS for them once more, and without them do not calibrate
    flood, flood_err = {}, None
    for ev, a, b, need in flood_spans(rec, t_start, t_end):
        have = hours_of(series, a, b)
        if 2 * have < need:
            finfo = {}
            got = obs.get_series(gid, a, b, info=finfo, refresh=True)
            if fetch_outage(finfo.get("fetch_exc")):
                raise Outage(f"obs fetch for {gid} failed ({str(finfo['fetch_error'])[:160]}), "
                             f"{have} of the {need} paired hours of {ev}")
            flood_err = finfo.get("fetch_error") or flood_err
            series = sorted({**dict(series), **dict(got)}.items())
            have = hours_of(series, a, b)
            base["flood_refetched"] = True
        flood[str(ev)] = [have, need]
    missing = [(ev, h, n) for ev, (h, n) in flood.items() if 2 * h < n]
    frac = cover(series)
    base["obs_cover"] = round(frac, 3)
    if flood:
        base["flood_obs_h"] = flood
    if frac < OBS_MIN_FRAC or missing:
        # not run: a local skip with a TTL (skip_ttl), never a handled record
        err = info.get("fetch_error") or flood_err
        quiet = (quiet or bool(missing)) and not err
        why = (f"obs cover {100 * frac:.0f}% of the window (< {100 * OBS_MIN_FRAC:.0f}%)"
               if frac < OBS_MIN_FRAC else "")
        if missing:
            # the note first: a skip's reason is kept to 200 characters
            why += ("; " if why else "") + f"{FLOOD_MISSING}: " + ", ".join(
                f"{ev} {h} of {n} paired hours" for ev, h, n in missing[:2]) + (
                f" (+{len(missing) - 2} more episodes)" if len(missing) > 2 else "")
        return {**base, "ran": False,
                "reason": why + (f"; {QUIET_EMPTY}" if quiet else "")
                          + (f"; fetch: {err}" if err else "")}
    log(f"calibrating {gid} for {rec.get('status')} episode(s) {', '.join(map(str, events))} "
        f"(worst CREST 1-D NSE {base['nse']:.2f} < {NSE_MAX:g} over {rec.get('n_pairs')} h "
        f"of {rec['event']}): window {base['window'][0]} -> {base['window'][1]} "
        f"({WINDOW_D:g} d), obs cover {100 * frac:.0f}%, {ROUNDS} rounds x {K} candidates, "
        f"LLM {llm.available()}")
    done = None
    saved = False
    n_runs = n_nse = 0
    errs: list = []                    # candidate runs' exceptions run_calibration swallowed
    for ek, payload in calibrate.run_calibration(
            gid, t_start, t_end, model="auto", snow="auto", use_mock=False,
            rounds=ROUNDS, k=K, save_vs_baseline=True, require_llm=REQUIRE_LLM,
            cand_errors=errs):
        # yield to user work between steps - but never once the winner is
        # saved: a calibration that finished is recorded, not dropped
        if ek == "status" and str(payload).startswith("winner pushed"):
            saved = True
        why = (busy_reason() if ek in ("status", "round") and not saved else None)
        if why:
            raise Preempted(why)
        if ek == "status":
            log(f"  {payload}")
        elif ek == "round":
            tried = [t for t in payload.get("tried") or [] if isinstance(t, dict)]
            n_runs += len(tried)
            n_nse += sum(1 for t in tried if t.get("id") != "baseline" and _num(t.get("nse")))
            bn = payload.get("best_nse")
            log(f"  round {payload.get('round')}: best NSE "
                f"{'n/a' if bn is None else f'{bn:.3f}'}"
                + (f" ({payload['source']} candidates)" if payload.get("source") else ""))
        elif ek == "done":
            done = payload
    if not done or done.get("error"):
        return {**base, "ran": False, "reason": f"calibration error: {(done or {}).get('error', 'no result')}"}
    if REQUIRE_LLM and not done.get("llm_rounds"):
        # the LLM gave no round its candidates (run_calibration stopped
        # before a heuristic-only search: nothing saved or pushed) — a
        # provider outage or an exhausted key fails every gauge alike
        raise Outage(f"no LLM candidates for {gid} "
                     f"({str(done.get('llm_stopped') or 'heuristic rounds only')[:160]})")
    no_nse = ("no EF5 run produced an NSE"
              if done.get("baseline_nse") is None and done.get("best_nse") is None
              else "no candidate run produced an NSE" if not n_nse else None)
    if no_nse:
        # nothing was judged (every candidate failed, or none ran): nothing
        # to start the cooldown with. Candidates that failed for reasons
        # that are not the gauge's (network, Hub, memory) are an outage
        last = f"{type(errs[-1]).__name__}: {str(errs[-1])[:160]}" if errs else ""
        if errs and all(is_outage(e) for e in errs):
            raise Outage(f"{no_nse} for {gid}: {len(errs)} candidate run(s) failed ({last})")
        return {**base, "ran": False, "n_runs": n_runs, "n_nse": n_nse,
                "reason": no_nse + (f" ({len(errs)} candidate run(s) failed; last: {last})"
                                    if errs else "")}
    bp = done.get("best_params") or {}
    out = {**base, "ran": True, "n_runs": n_runs, "n_nse": n_nse,
           "rounds": done.get("rounds"), "llm_rounds": done.get("llm_rounds"),
           "baseline_nse": done.get("baseline_nse"), "best_nse": done.get("best_nse"),
           "improved": bool(done.get("improved")), "saved": bool(done.get("saved")),
           "pushed": done.get("pushed"), "model": done.get("model"),
           "extended": bool(done.get("extended")),
           "best_params": {k: round(float(v), 4) for k, v in bp.items()
                           if isinstance(v, (int, float))}}
    if out["saved"] and out["pushed"] is False:
        # saved here, not on CREST_state: unfinished — main keeps it with the
        # pending records and retries the push every poll (_push_unpushed);
        # it is recorded, and finished, once the winner is there
        return out
    # finished: the winner (if any) is on CREST_state by now — episodes
    # published before this were simulated with the parameters it replaced
    return {**out, "finished": max(now, dt.datetime.utcnow()).strftime(TS)}


def _load_local_state():
    """Pending and unpushed records, the local skip list and the last summary
    time from a previous run of this loop in the same container (/tmp
    survives the runner's loop restarts, not a Space restart — the Hub's
    skip summaries cover that; an unpushed winner is lost with the Space's
    local params store then, and the gauge is calibrated again)."""
    try:
        with open(STATE_PATH) as fp:
            old = json.load(fp)
        state["pending"] = [p for p in (old.get("pending") or []) if isinstance(p, dict)]
        state["unpushed"] = [p for p in (old.get("unpushed") or []) if isinstance(p, dict)]
        state["skipped"] = {str(g): v for g, v in (old.get("skipped") or {}).items()
                            if isinstance(v, dict)}
        state["skips_summarized"] = old.get("skips_summarized")
    except (OSError, ValueError, AttributeError, TypeError):
        pass


def _not_run(rec: dict, now: dt.datetime, reason: str) -> dict:
    """A calibration of candidate `rec` that did not run (ran=False), for a
    failure calibrate_one did not answer itself: the loop puts its gauge on
    the skip list (skip_ttl: the error TTL; an outage streak about one
    poll) and counts it toward the skip run."""
    try:
        nse = round(float(rec.get("nse")), 3)
    except (TypeError, ValueError):
        nse = None
    return {"event": rec.get("event"), "gauge": str(rec["gauge"]).zfill(8),
            "kind": rec.get("kind", "low_nse"), "west": bool(rec.get("west")),
            "events": list(rec.get("events") or [rec.get("event")]),
            "status": rec.get("status"), "nse": nse, "nse_max": NSE_MAX,
            "at": now.strftime(TS), "ran": False, "reason": reason}


def run_candidate(rec: dict, now: dt.datetime) -> dict:
    """calibrate_one, raising Outage for an exception that is not the gauge's
    (is_outage). Any other exception is the gauge's own calibration failing
    — likely again on every try — so it is a calibration that did not run
    ("calibration error: <Type>: <msg>", _not_run): retried after the error
    TTL, while the next gauge gets its turn (re-raised, the same top-ranked
    gauge would be picked and fail every poll)."""
    try:
        return calibrate_one(rec, now)
    except (Preempted, Outage):
        raise
    except Exception as e:
        if is_outage(e):
            raise Outage(f"{type(e).__name__}: {str(e)[:200]}") from e
        log(traceback.format_exc())
        return _not_run(rec, now, f"calibration error: {type(e).__name__}: {str(e)[:200]}")


def _outage(rec: dict, now: dt.datetime, o: Outage) -> dict | None:
    """Nothing is recorded for an outage (None: the loop backs off POLL_S and
    the gauge is retried) — until OUTAGE_STRIKES of them in a row on the
    gauge (state "outage": gauge -> {n, at, reason}; a calibration of the
    gauge that ends in anything but an outage clears its count, an outage
    on another gauge does not): then it is a calibration that did not run
    (_not_run, "outage <n>x in a row: ..."), a skip of about one poll
    (skip_ttl). The next gauge gets its turn, so a fault only that gauge
    triggers cannot pin the head of the queue; a global outage moves the
    strikes from gauge to gauge and each skip lapses within a poll, so the
    order stands when it is over. Counts are kept per gauge, so two gauges
    whose faults alternate at the head reach their strikes in turn, their
    skips overlap and the third gets through."""
    gid = str(rec["gauge"]).zfill(8)
    st = state.setdefault("outage", {})
    for g in [g for g, v in st.items() if not isinstance(v, dict)
              or (_parse(v.get("at")) or now) <= now - dt.timedelta(hours=SKIP_ERR_TTL_H)]:
        del st[g]                                # no outage on it for a day: start over
    n = int((st.get(gid) or {}).get("n") or 0) + 1
    st[gid] = {"n": n, "at": now.strftime(TS), "reason": str(o)[:200]}
    if n >= OUTAGE_STRIKES:
        del st[gid]
        return _not_run(rec, now, f"outage {n}x in a row: {str(o)[:200]}")
    return None


def _rec_key(r: dict) -> tuple:
    return (r.get("gauge"), r.get("event"), r.get("kind"), r.get("at"))


def _push_unpushed():
    """Calibrations whose winner is saved here but whose push to CREST_state
    failed (state "unpushed"): push it again. One that is there now is
    finished (`finished`: now — episodes published before it still ran on
    the parameters it replaces) and joins the pending records, committed by
    _flush_pending. paramstore.get first, as on every Space: a newer record
    another calibration put on CREST_state meanwhile wins the local file, so
    the push never overwrites it with this older winner."""
    if not state["unpushed"]:
        return
    from hf_data import paramstore
    left = []
    for u in state["unpushed"]:
        u = {**u, "push_tries": int(u.get("push_tries") or 1) + 1}
        g, m = u.get("gauge"), u.get("model")
        ok = False
        if g and m:
            try:                                 # never in the loop's way
                paramstore.get(g, m)
                ok = paramstore.push_remote(g, m)
            except Exception as e:
                log(f"push of {g} ({m}) failed: {type(e).__name__}: {str(e)[:160]}")
        if ok:
            u.update({"pushed": True, "finished": dt.datetime.utcnow().strftime(TS)})
            log(f"winner of {g} ({m}) pushed to CREST_state after {u['push_tries']} tries: "
                f"calibration finished")
            state["pending"].append(u)
        else:
            left.append(u)
    state["unpushed"] = left


def _flush_pending(done: list) -> list:
    """Records whose commit failed (state "pending"): drop those now on the
    Hub, retry the rest oldest first (stopping at the first failure), and
    return `done` plus every record not in it — committed just now or still
    pending — so the daily cap, handled pairs and cooldown count them. The
    unpushed ones (state "unpushed": the winner not on CREST_state yet,
    pushed again every loop by _push_unpushed) count the same while they
    wait — they ran; only their record waits for the push."""
    from hf_data import eventstore
    on_hub = {_rec_key(d) for d in done}
    out = [p for p in state["pending"] if _rec_key(p) not in on_hub]
    left = list(out)
    while left and eventstore.note_autocal(left[0]):
        log(f"pending record committed: {left[0].get('gauge')} ({left[0].get('kind')}, "
            f"{left[0].get('at')})")
        left.pop(0)
    state["pending"] = left
    return list(done) + out + [u for u in state["unpushed"] if _rec_key(u) not in on_hub]


def _note_skip(result: dict):
    """A calibration that did not run goes on the local skip list (no Hub
    commit of its own: _summarize_skips) — its gauge waits out skip_ttl."""
    state["skipped"][result["gauge"]] = {
        "at": result["at"], "reason": str(result.get("reason") or "")[:200],
        "events": list(result.get("events") or [result.get("event")])}


def _summarize_skips(done: list, now: dt.datetime):
    """Prune expired local skips; commit a summary record when one is due."""
    from hf_data import eventstore
    state["skipped"] = {g: v for g, v in state["skipped"].items()
                        if (_parse(v.get("at")) or now) > now - skip_ttl(v.get("reason"))}
    rec = skip_summary(done, state["skipped"], now, state.get("skips_summarized"))
    if rec and eventstore.note_autocal(rec):
        state["skips_summarized"] = rec["at"]
        log(f"skip summary committed: {rec['reason']}")


def _note_held(held: list):
    """Reservoir-gauge candidates held until AUTOCAL_LAKES_ON=1: kept in the
    state file (status page) and logged when the set changes — no Hub write,
    they are simply not eligible."""
    key = sorted((h["gauge"], h["event"]) for h in held)
    if key != sorted((h["gauge"], h["event"]) for h in state["held_reservoir"]):
        log(f"holding {len(held)} western reservoir gauge(s) "
            f"{lake_hold() or 'no longer'}: " + (", ".join(
                f"{h['gauge']} ({h['kind']}, {h['stor_mm']:.0f} mm)" for h in held)
                or "none"))
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
    skip_run = 0                       # calibrations in a row that did not run
    while True:
        now = dt.datetime.utcnow()
        try:
            # a saved winner not on CREST_state yet: pushed again whatever the
            # loop does next (a Hub upload takes no compute from user work)
            _push_unpushed()
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
            index = eventstore.load_index()
            done = eventstore.load_autocal()
            ledger = eventstore.load_autocal_done()
            if done is None or ledger is None:
                # an outage is not "no history": picking now could re-calibrate
                # a gauge inside its cooldown or blow through the daily cap
                state["phase"] = (f"history unreadable — events/autocal.json or "
                                  f"autocal_done.json could not be read; retry in "
                                  f"{POLL_S / 60:.0f} min")
                _dump_state()
                log(state["phase"])
                skip_run = 0
                time.sleep(POLL_S)
                continue
            done = _flush_pending(done)
            held = []
            rec, why = pick_candidate(index, done, now, held=held, ledger=ledger,
                                      local_skips=state["skipped"])
            _note_held(held)
            if rec is None:
                _summarize_skips(done, now)
                pend = (f", {len(state['pending'])} pending" if state["pending"] else "") + (
                    f", {len(state['unpushed'])} unpushed" if state["unpushed"] else "")
                state["phase"] = (f"idle — {why} ({len(index)} episodes listed, "
                                  f"{len(done)} records{pend})")
                _dump_state()
                log(state["phase"])
                skip_run = 0
                time.sleep(POLL_S)
                continue
            state["phase"] = f"calibrating {rec['gauge']} ({rec['kind']}; {why})"
            state["current"] = {"event": rec["event"], "events": rec["events"],
                                "gauge": rec["gauge"], "kind": rec["kind"],
                                "west": rec["west"], "nse": round(float(rec["nse"]), 3),
                                "n_pairs": rec["n_pairs"], "status": rec.get("status"),
                                "started": now.strftime(TS)}
            _dump_state()
            gid = str(rec["gauge"]).zfill(8)
            try:
                result = run_candidate(rec, now)
            except Outage as o:
                result = _outage(rec, now, o)
                if result is None:         # nothing recorded: back off, retry
                    state["phase"] = (f"outage — {o}; nothing recorded, retry in "
                                      f"{POLL_S / 60:.0f} min ({state['outage'][gid]['n']} "
                                      f"in a row on {gid}; a skip at {OUTAGE_STRIKES})")
                    state["current"] = None
                    _dump_state()
                    log(state["phase"])
                    skip_run = 0
                    time.sleep(POLL_S)
                    continue
            state["outage"].pop(gid, None)   # a result ends the gauge's outage streak
            unpushed = bool(result.get("ran") and result.get("saved")
                            and result.get("pushed") is False)
            if unpushed:
                skip_run = 0
                # the winner is saved here but not on CREST_state: unfinished,
                # nothing recorded yet — its push is retried every loop
                # (_push_unpushed) and the record committed once it is
                # there; it counts for the cap, the handled pairs and the
                # cooldown meanwhile (_flush_pending)
                state["unpushed"].append(result)
                log(f"push to CREST_state failed — calibration kept unpushed "
                    f"({len(state['unpushed'])})")
            elif result.get("ran"):
                skip_run = 0
                if not eventstore.note_autocal(result):
                    # kept until it is on the Hub; counts for the cap, the
                    # handled pairs and the cooldown meanwhile (_flush_pending)
                    state["pending"].append(result)
                    log(f"commit failed — record kept pending ({len(state['pending'])})")
            else:
                # did not run (too little or no obs, the flood's obs missing,
                # calibration error, no NSE, OUTAGE_STRIKES outages in a row
                # on this gauge): no handled record and no commit per skip —
                # a local skip with a TTL, then the gauge is retried; a
                # summary record at most every SKIP_SUMMARY_H
                _note_skip(result)
                _summarize_skips(done, now)
                skip_run += 1
            state["runs" if result.get("ran") else "skips"] += 1
            state["last"] = result
            state["current"] = None
            log(f"{'unpushed' if unpushed else 'record' if result.get('ran') else 'skip'}: "
                f"{json.dumps(result)[:400]}")
            if skip_run >= MAX_SKIP_RUN:
                # an obs or EF5 failure the outage checks missed must not
                # skip every eligible gauge in one go
                state["phase"] = (f"paused — {skip_run} calibrations in a row did not run "
                                  f"(last: {str(result.get('reason') or '')[:160]}); next "
                                  f"poll in {POLL_S / 60:.0f} min")
                _dump_state()
                log(state["phase"])
                skip_run = 0
                time.sleep(POLL_S)
                continue
        except Preempted as p:             # nothing recorded: retried when idle
            state["phase"] = f"preempted — {p}"
            state["current"] = None
            idle_since = None
            log(state["phase"])
        except Exception as e:
            state["phase"] = f"error: {type(e).__name__}: {e}"
            state["current"] = None
            log(traceback.format_exc())
            skip_run = 0
            time.sleep(POLL_S)
        _dump_state()
        time.sleep(30)


if __name__ == "__main__":
    main()
