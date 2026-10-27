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
   * a *miss*: the runner's CREST-miss guard recorded that EF5 missed **every
     hourly check of a whole flood event** at a trigger gauge
     (`events/missed.json` on `vincewin/CREST_data`), or
   * an *over-prediction*: a published event (`events/index.json`) whose
     manifest shows EF5 peaking at >= `AUTOCAL_OVERSIM_RATIO` x the observed
     peak (paired hours up to t0, obs peak >= `AUTOCAL_OVERSIM_MIN_OBS` m3/s);
2. when no event is eligible, the next gauge of the **western backlog**
   (`hf_data/data/autocal_west_backlog.json`, built by
   `scripts/build_autocal_backlog.py`: event-prone gauges, then dam-free
   gauges with suspect parameters, then other dam-free gauges, smaller
   basins first; reservoir gauges wait until `AUTOCAL_LAKES_ON=1`), each
   calibrated once.

It calibrates that gauge over the preceding six months (LLM-proposed
multiplier updates within hydrologic bounds, judged by real EF5 runs), saves
the winner only if it beats the current parameters on that window, and pushes
it to `vincewin/CREST_state` `params/`, where the dashboard, the event runner
and the fleet pick it up on their next run. Results: `events/autocal.json`
(`kind` miss / oversim / backlog, `west`), shown in the dashboard's Events
panel. Backlog gauges that cannot run are kept on a local skip list and
reach the Hub only as one `backlog_skips` summary record a day.

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
`AUTOCAL_LAKES_ON` 0, `AUTOCAL_SKIP_TTL_D` 30, `AUTOCAL_SKIP_SUMMARY_H` 24.
