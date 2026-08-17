---
title: CREST GPU Worker
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
---

# CREST event worker — ZeroGPU backup (V27 P1.5)

Rung 2 of the CREST-AI event degradation ladder (fork
`docs/DESIGN_V27_PARALLEL.md` Sec. 7): when the HPC GPU worker leaves a 2-D
inundation job unclaimed for `ZERO_DELAY_S` (default 7 min — outage, capacity,
or oversized backlog), this Space claims it and solves in chunked
`@spaces.GPU` slices (state checkpointed to CPU RAM between slices; chunk-vs-
single equivalence verified to <2 mm).

- Queue protocol identical to `crestimap/worker.py` (claim / heartbeat 240 s /
  stale takeover 600 s / failed.json).
- 30 m (`dem_res "1"`) jobs only, cell budget `ZERO_MAX_CELLS` — NEVER
  coarsens; oversized jobs fall through to the Space's CPU window.
- Shadow by default: solves and releases the claim, publishes nothing.
  Set `PUBLISH=1` only after the Space runs `EVENT_QUEUE_MODE=on`.
- Secrets: `HF_TOKEN` (write access to `vincewin/CREST_data`).

Replicate for more capacity: duplicate this Space (worker identity includes
the Space id, so claims never collide).
