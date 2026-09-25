---
title: CREST autocal
emoji: 🎯
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# CREST_autocal

Auto-triggered AI calibration for [CREST_demo](https://huggingface.co/spaces/vincewin/CREST_demo).
When the runner's CREST-miss guard records that EF5 missed **every hourly
check of a whole flood event** at a trigger gauge (`events/missed.json` on
`vincewin/CREST_data`), this Space calibrates that gauge over the preceding
six months (LLM-proposed multiplier updates within hydrologic bounds, judged
by real EF5 runs), saves the winner only if it beats the current parameters
on that window, and pushes it to `vincewin/CREST_state` `params/`, where the
dashboard, the event runner and the fleet pick it up on their next run.
Results: `events/autocal.json`, shown in the dashboard's Events panel.

Lowest-priority compute class in CREST-AI (user-initiated operations >
nowcast > 2-D inundation > data update > fleet runner > this): one gauge at a
time, at most `AUTOCAL_MAX_PER_DAY` calibrations a day, own Space.

Code lives in [CREST_AI](https://github.com/mchen15ouedu/CREST_AI)
(`autocal/autocal_run.py`, `hf_data/calibrate.py`) and is cloned fresh at
every boot. Secrets: `HF_TOKEN` (write), `OPENAI_API_KEY` (or the
`VLLM_BASE_URL` variable). Variables (defaults): `AUTOCAL_WINDOW_D` 183,
`AUTOCAL_MIN_CHECKS` 3, `AUTOCAL_QUIET_H` 6, `AUTOCAL_COOLDOWN_D` 30,
`AUTOCAL_MAX_PER_DAY` 2, `AUTOCAL_OBS_MIN_FRAC` 0.5, `AUTOCAL_ROUNDS` 4,
`AUTOCAL_K` 3, `AUTOCAL_REQUIRE_LLM` 1, `AUTOCAL_POLL_S` 900.
