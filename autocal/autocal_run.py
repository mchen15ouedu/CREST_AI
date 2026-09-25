"""CREST_autocal — auto-triggered AI calibration after a whole-event CREST miss.

Runs forever on its own HF Space (autocal_space/). The lowest-priority compute
class in CREST-AI (user 2026-09-25: user-initiated operations > nowcast >
2-D inundation > data update > fleet runner > this), so it lives on a
dedicated cpu-basic Space, calibrates ONE gauge at a time, and caps how many
LLM-guided calibrations it starts per day.

Trigger (read from the public event store, written by the runner's
CREST-miss guard):
  * events/missed.json row with n_checks >= AUTOCAL_MIN_CHECKS hourly checks
    that ALL missed (a check that passes publishes a map, so the event would
    be in events/index.json — such events are skipped),
  * the last check is at least AUTOCAL_QUIET_H old (the trigger stopped
    re-checking: the flood is over),
  * the gauge was not auto-calibrated within AUTOCAL_COOLDOWN_D days and the
    event was not already handled (events/autocal.json),
  * fewer than AUTOCAL_MAX_PER_DAY calibrations started in the last 24 h,
  * USGS observations cover >= AUTOCAL_OBS_MIN_FRAC of the window.
Calibration window: AUTOCAL_WINDOW_D days (default 183 — two seasons; user:
"some dry lands don't see anything in 30 days") ending at the event's end.
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

TS = "%Y-%m-%dT%H:%M:%SZ"
state = {"phase": "booting", "current": None, "last": None, "runs": 0, "skips": 0}


def log(s: str):
    print(f"{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}Z {s}", flush=True)


def _dump_state():
    try:
        with open(STATE_PATH, "w") as fp:
            json.dump(state, fp)
    except OSError:
        pass


def _parse(s: str | None) -> dt.datetime | None:
    for fmt in (TS, "%Y-%m-%dT%H:%MZ"):
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


def pick_candidate(missed: list, index: dict, done: list, now: dt.datetime):
    """The one miss streak to work on next (or None) and a reason string."""
    last_run = {}
    for d in done:
        if d.get("ran"):
            t = _parse(d.get("at"))
            if t and (d["gauge"] not in last_run or t > last_run[d["gauge"]]):
                last_run[d["gauge"]] = t
    handled = {(d.get("gauge"), d.get("event")) for d in done}
    today = sum(1 for d in done if d.get("ran") and (_parse(d.get("at")) or now)
                > now - dt.timedelta(hours=24))
    if today >= MAX_PER_DAY:
        return None, f"daily cap reached ({today}/{MAX_PER_DAY} in 24 h)"
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
        cands.append(st)
    if not cands:
        return None, "no eligible whole-event miss"
    cands.sort(key=lambda r: (-r["n_checks"], -r["obs_peak_m3s"]))
    return cands[0], f"{len(cands)} eligible"


def calibrate_one(rec: dict, now: dt.datetime) -> dict:
    from hf_data import calibrate, eventstore, llm, obs
    gid = str(rec["gauge"]).zfill(8)
    checks = rec.get("checks") or [rec]
    t0s = [t for t in (_parse(c.get("t0")) for c in checks) if t] or           [c["_at"] for c in checks if c.get("_at")]
    t_end = (max(t0s) if t0s else now) + dt.timedelta(hours=6)
    t_end = min(t_end, now - dt.timedelta(hours=2))
    t_start = t_end - dt.timedelta(days=WINDOW_D)
    t_start = t_start.replace(minute=0, second=0, microsecond=0)
    t_end = t_end.replace(minute=0, second=0, microsecond=0)
    base = {"event": rec["event"], "gauge": gid, "n_checks": len(checks),
            "events": rec.get("events") or [rec["event"]],
            "streak": [rec["first_at"].strftime(TS) if rec.get("first_at") else None,
                       rec["last_at"].strftime(TS) if rec.get("last_at") else None],
            "obs_peak_m3s": rec.get("obs_peak_m3s"), "sim_peak_m3s": rec.get("sim_peak_m3s"),
            "window": [t_start.strftime("%Y-%m-%d %H:%M"), t_end.strftime("%Y-%m-%d %H:%M")],
            "at": now.strftime(TS), "llm": llm.available()}
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
    log(f"calibrating {gid} for event {rec['event']}: window {base['window'][0]} -> "
        f"{base['window'][1]} ({WINDOW_D:g} d), obs cover {100 * frac:.0f}%, "
        f"{ROUNDS} rounds x {K} candidates, LLM {llm.available()}")
    done = None
    n_runs = 0
    for kind, payload in calibrate.run_calibration(
            gid, t_start, t_end, model="auto", snow="auto", use_mock=False,
            rounds=ROUNDS, k=K, save_vs_baseline=True):
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


def main():
    from hf_data import eventstore, llm
    if not os.environ.get("HF_TOKEN"):
        state["phase"] = "NO HF_TOKEN — add the secret in Space settings"
        _dump_state()
        log(state["phase"])
        while True:
            time.sleep(3600)
    while True:
        now = dt.datetime.utcnow()
        try:
            if REQUIRE_LLM and not llm.available():
                state["phase"] = "waiting for an LLM key (OPENAI_API_KEY secret or VLLM_BASE_URL)"
                _dump_state()
                log(state["phase"])
                time.sleep(POLL_S)
                continue
            missed = eventstore.load_missed()
            index = eventstore.load_index()
            done = eventstore.load_autocal()
            rec, why = pick_candidate(missed, index, done, now)
            if rec is None:
                state["phase"] = f"idle — {why} ({len(missed)} miss rows, {len(done)} records)"
                _dump_state()
                log(state["phase"])
                time.sleep(POLL_S)
                continue
            state["phase"] = f"calibrating {rec['gauge']} ({why})"
            state["current"] = {"event": rec["event"], "gauge": rec["gauge"],
                                "started": now.strftime(TS)}
            _dump_state()
            result = calibrate_one(rec, now)
            eventstore.note_autocal(result)
            state["runs" if result.get("ran") else "skips"] += 1
            state["last"] = result
            state["current"] = None
            log(f"record: {json.dumps(result)[:400]}")
        except Exception as e:
            state["phase"] = f"error: {type(e).__name__}: {e}"
            state["current"] = None
            log(traceback.format_exc())
            time.sleep(POLL_S)
        _dump_state()
        time.sleep(30)


if __name__ == "__main__":
    main()
