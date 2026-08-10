---
title: CREST Updater
emoji: 🔄
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Data-feed refresher for CREST_demo (temp/MRMS/PET/USGS)
---

# CREST_updater — data-feed refresher for CREST_demo

Companion Space to [CREST_demo](https://huggingface.co/spaces/vincewin/CREST_demo).
Keeps every data feed the dashboard depends on refreshed **in the store**, so
simulations read pre-staged data instead of fetching live:

| feed | source | writes to |
|---|---|---|
| TEMP | NARR air.2m via NOAA PSL netCDF mirror | `vincewin/CREST_data` `temp/` month tars |
| PET  | USGS FEWS NET global daily PET | `vincewin/CREST_data` `pet/` month tars |
| MRMS | Iowa State mtarchive hourly QPE grib2 | `vincewin/CREST_data` `mrms/` month tars |
| USGS | NWIS instantaneous discharge | private `vincewin/CREST_state` `obs/` parquet |

All updaters are idempotent and append-only — they scan the recent months,
fetch only what the store is missing, and never touch existing members. A
freshness report (latest stored timestep + lag per feed, runnable-window end)
runs after every update.

**Waking this Space runs the job**: a full update starts automatically ~20 s
after boot. The weekly routine on the local machine just pings the Space and
polls `/api/status` until `running` is false.

- `GET /` — one-line state
- `POST /api/run` (optional `?feeds=temp,pet,mrms,usgs`) — explicit trigger
- `GET /api/status` — per-feed summary lines + freshness block + log tail

Secrets: `HF_TOKEN` (write access to CREST_data + CREST_state). Optional
`UPDATER_KEY` to require a key on `/api/run` (the auto-run on boot doesn't
need it). The USGS obs live-fetch path in the app remains as FALLBACK — the
store is primary, NWIS fills gaps and the provisional last-24 h tail.
