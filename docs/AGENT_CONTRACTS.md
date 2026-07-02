# CREST_demo — Agent Contracts

Design of the data passed between agents. Schemas live in
[`agents/contracts.py`](../agents/contracts.py) (Pydantic v2). One `SessionState`
threads every artifact through the pipeline.

## Pipeline

```
user query
   │
   ▼
ACP  Context Parser ───────► EventContext      (metadata + anchor coord)
   │
   ▼
ADR  Dataset Retriever ────► DatasetManifest   (DEM/DDM/FAM + precip/PET parquet; 90-day cache)
   │
   ▼
AP   Perception (vision) ──► BasinPerception   (morphology, drainage, candidate outlets)
   │
   ▼
AOS  Outlet Selector ──────► OutletSelection   ◄── USER CONFIRMS gauge
   │
   ▼
API  Param Initializer ────► ParameterSet      (RAG + Chen 2023 analogs + priors)
   │
   ▼
AO   Operator ─────────────► RunSpec ─► run ─► RunResult   (control file; metrics; FI / F-IDF)
   │        │
   │        └─ streams RunProgress ─► live dashboard (hydrograph + 2-D streamflow)
   ▼
ARW  Report Writer ────────► Report            (viz + diagnostics + findings)
```

## Artifact per agent

| Agent | Output | Key fields |
|---|---|---|
| ACP | `EventContext` | `event_type`, `location_name`, `admin`, `anchor`, `time_window`, `news_refs`, `clarifications_needed` |
| ADR | `DatasetManifest` | `assets[DataAsset]`, `cache[CacheEntry]` (90-day, ref-counted), `preview_maps`, `missing` |
| AP | `BasinPerception` | `morphology`, `drainage_structure`, `candidate_outlets` |
| AOS | `OutletSelection` | `candidate_gauges[USGSGauge]`, `recommended`, `user_confirmed`, `confirmed_gauge` |
| API | `ParameterSet` | `water_balance` (wm,b,im,ke,fc,iwu), `kinematic` (under,leaki,th,isu,alpha,beta,alpha0), `lake`, `citations` |
| AO | `RunSpec`, `RunResult` | control file text; `SkillMetrics` (NSE/KGE/CC/bias/RMSE); `HydroAssessment` (return period, FI, F-IDF) |
| ARW | `Report` | `sections`, `key_findings`, `figures` |

Parameter names mirror EF5 `Models.tbl` exactly, so AO's control-file renderer
is a direct field→key mapping.

## Streaming contract (live viz)

While CREST runs, AO emits `RunProgress` events:
- `hydrograph` → `HydrographPoint` (time, sim_q, obs_q, precip, lake_vol) — one per `ts.*.csv` row.
- `streamflow_2d` → `StreamflowFrame` (time, grid_path) — one raster per timestep.
- `log` / `metric` / `done`.

## Human-in-the-loop gates
- **AOS**: `OutletSelection.user_confirmed` must be `True` before API/AO proceed.
- **ACP**: `clarifications_needed` non-empty → chatbot asks the user before ADR.

## Open questions (blocking downstream agent work)
1. **CREST_data layout** — how is `vincewin/CREST_data` partitioned? (per-variable? per-region/HUC? bbox tiling? parquet schema of the climate data?) Drives `DataAsset` / `ADR`.
2. **Grid formats** — does the model consume BIF, GeoTIFF, or parquet-derived grids at run time? Affects whether ADR converts.
3. **USGS gauge source** — NWIS live API, or a bundled gauge table in the dataset?
4. **News verification** — which source for ACP (web search API, news API)? Optional at first?
5. **LLM providers** — native = HF Inference (`gpt-oss`); OpenAI fallback = GPT-4o. Vision model for AP? Confirm the routing.
6. **F-IDF / FI definitions** — pin exact formulas (Li et al. 2023) for `HydroAssessment`.
