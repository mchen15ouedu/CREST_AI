---
title: CREST_ungauged
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# CREST_ungauged — keep-warm Space

Advances the routed nowcast for all **2,676 ungauged HydroBASINS points** every
hour and publishes one parquet the dashboard serves instantly
(`nowcast/ungauged_latest.parquet` in `vincewin/CREST_data`).

Each point is advanced by **one new hour from a warm state** (never a cold
17-day run): `hf_data.routednow.compute` runs a cached hindcast to `t0` (saving
the state) then a 12-h forecast warm-started from it, with the upstream USGS cut
gauges injected using their observed + DI-LSTM-nowcast flow.

**State storage merges with the gauge fleet.** New 10-day checkpoints are
uploaded to `vincewin/CREST_fleet` as `states/V..._crestphys-spd.pqf` +
`results/...json`, beside the gauge keys; on cold boot each point refetches its
last checkpoint so a restart short-warms instead of cold-starting.

### Settings → Variables
| var | default | meaning |
|-----|---------|---------|
| `UNGAUGED_WORKERS` | `8` | parallel points (size to the vCPUs) |
| `UNGAUGED_SHARD` | — | `K/N` — run only points with index %% N == K (split across sibling Spaces) |
| `UNGAUGED_LIMIT` | `0` | cap points (smoke test) |

### Secret
`HF_TOKEN` — write access (uploads to `CREST_data` + `CREST_fleet`).

Hardware: **cpu-upgrade** (8 vCPU) recommended so a full pass finishes inside
the hour; shard across two Spaces if it runs long.
