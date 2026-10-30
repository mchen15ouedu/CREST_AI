---
title: CREST autocal
emoji: 🎯
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# CREST_autocal

Auto-triggered AI calibration for [CREST_demo](https://huggingface.co/spaces/vincewin/CREST_demo),
**western gauges first** (PI 2026-10-27: "The auto-calibration, we should
prioritize the western part of the USA, since those areas have more issues";
western = longitude < -105). It picks, in order:

1. an **event**, western gauges first —
   * an *over-prediction* (**western gauges only**): a published event
     (`events/index.json`) whose manifest shows EF5 peaking at >=
     `AUTOCAL_OVERSIM_RATIO` x the observed peak (paired hours up to t0, obs
     peak >= `AUTOCAL_OVERSIM_MIN_OBS` m3/s), or
   * a *miss*: the runner's CREST-miss guard recorded that EF5 missed **every
     hourly check of a whole flood event** at a trigger gauge
     (`events/missed.json` on `vincewin/CREST_data`), the streak newer than
     the gauge's last calibration

   (in the west an over-prediction comes before a miss; the east keeps its
   previous behaviour: misses only);
2. when no event is eligible, the next gauge of the **western backlog**
   (`hf_data/data/autocal_west_backlog.json`, built by
   `scripts/build_autocal_backlog.py`: event-prone gauges, then dam-free
   gauges with suspect parameters, then other dam-free gauges, smaller
   basins first), each calibrated once — `events/autocal_done.json`
   (gauge -> first/last/n of its calibrations, never truncated, written with
   every calibration record) remembers that beyond the 500 rows
   `events/autocal.json` keeps.

**Reservoir hold (west only):** a western reservoir-affected gauge (GAGES-II
normal storage >= 10 mm, `hf_data/data/gagesii_reservoir_stor.json`) is not
calibrated from any trigger until `AUTOCAL_LAKES_ON=1` — the lake module is
being switched on for the west (`CREST_LAKES=west`), and calibrating first
would bake the missing reservoirs into the parameters. The flag lifts the
hold only while the lake module is live on this Space (`hf_data.lakes`
importable, `CREST_LAKES` west or all, a lake table with a version);
otherwise the hold stays and the phase line says why. Eastern gauges are not
held. Held events are listed under `held_reservoir` in `/api/status`.

It calibrates that gauge over the preceding six months (LLM-proposed
multiplier updates within hydrologic bounds, judged by real EF5 runs), saves
the winner only if it beats the current parameters on that window, and pushes
it to `vincewin/CREST_state` `params/`, where the dashboard, the event runner
and the fleet pick it up on their next run. Results: `events/autocal.json`
(`kind` miss / oversim / backlog, `west`), shown in the dashboard's Events
panel. Backlog gauges that cannot run are kept on a local skip list
(`AUTOCAL_SKIP_TTL_D` days for too little obs, `AUTOCAL_SKIP_ERR_TTL_H` hours
for errors) and reach the Hub only as one `backlog_skips` summary record a
day; after `AUTOCAL_MAX_SKIP_RUN` skips in a row the walk pauses until the
next poll.

Failures never look like history or skips: an unreadable `autocal.json` /
`autocal_done.json` makes the loop wait ("history unreadable"); an obs-fetch
or network outage records nothing and backs off a poll (a backlog gauge that
hits `AUTOCAL_OUTAGE_STRIKES` in a row gets a short skip); a record whose
commit failed stays pending in the state file — counted for the daily cap
and calibrate-once — and is retried every poll. A calibration in which no
EF5 run produced an NSE is recorded as not run.

Lowest-priority compute class in CREST-AI (user-initiated operations >
nowcast > 2-D inundation > data update > fleet runner > this): one gauge at a
time, at most `AUTOCAL_MAX_PER_DAY` calibrations a day, own Space.

Code lives in [CREST_AI](https://github.com/mchen15ouedu/CREST_AI)
(`autocal/autocal_run.py`, `hf_data/calibrate.py`) and is cloned fresh at
every boot. Secrets: `HF_TOKEN` (write), `OPENAI_API_KEY` (or the
`VLLM_BASE_URL` variable). Variables (defaults): `AUTOCAL_WINDOW_D` 183,
`AUTOCAL_MIN_CHECKS` 3, `AUTOCAL_QUIET_H` 6, `AUTOCAL_COOLDOWN_D` 30,
`AUTOCAL_MAX_PER_DAY` 2, `AUTOCAL_OBS_MIN_FRAC` 0.5, `AUTOCAL_ROUNDS` 4,
`AUTOCAL_K` 3, `AUTOCAL_REQUIRE_LLM` 1, `AUTOCAL_POLL_S` 900,
`AUTOCAL_OVERSIM_RATIO` 3.0, `AUTOCAL_OVERSIM_MIN_OBS` 1.0,
`AUTOCAL_LAKES_ON` 0, `AUTOCAL_SKIP_TTL_D` 30, `AUTOCAL_SKIP_ERR_TTL_H` 24,
`AUTOCAL_SKIP_SUMMARY_H` 24, `AUTOCAL_MAX_SKIP_RUN` 5,
`AUTOCAL_OUTAGE_STRIKES` 4.
