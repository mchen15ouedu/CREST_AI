---
title: CREST Demo
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
hf_oauth: true
hf_oauth_expiration_minutes: 43200
pinned: false
license: cc0-1.0
short_description: Agentic flash-flood analysis with live CREST hydrographs
---

# CREST_demo

Agentic flood-analysis dashboard around the CREST/EF5 hydrologic model. A chatbot
drives a multi-agent pipeline (parse query → find basin → pick outlet gauge →
retrieve data → set calibrated params → run CREST → report), with **live**
hydrograph + 2-D streamflow visualization while the model runs.

Integrates [AQUAH](https://github.com/Skyan1002/AQUAH_v0.3) (CrewAI agents +
CREST runner + report writer) with an HF-hosted data layer (`hf_data/`) that
pulls everything from [`vincewin/CREST_data`](https://huggingface.co/datasets/vincewin/CREST_data):

- **basic grids** — HydroSHEDS DEM/flow-dir/flow-acc COGs, clipped per basin
- **forcing** — MRMS/PET Parquet (PQF) month-tars, fetched per event window
- **gauges** — GAGES-II catalog + snap-to-stream
- **params** — calibrated CRESTPHYS+KW multipliers × 1 km grids

## Runtime

Docker SDK Space. The image builds the [mchen15ouedu/EF5](https://github.com/mchen15ouedu/EF5)
fork with Apache Arrow (`./configure --with-arrow`) so EF5 reads PQF forcing
natively, then serves the FastAPI + Leaflet app (`server.py`) on port 7860.

## Configuration

Space **variables**: `CREST_DEMO_MOCK=0` (run the real EF5 binary; `1` = mock).

Space **secrets** (optional — without them query parsing falls back to the
gazetteer and reports to a template):
- `VLLM_BASE_URL` — OpenAI-compatible vLLM endpoint (primary LLM)
- `OPENAI_API_KEY` — paid OpenAI API (last-resort fallback)

No `HF_TOKEN` needed — the data layer
([`vincewin/CREST_data`](https://huggingface.co/datasets/vincewin/CREST_data)) is public.
