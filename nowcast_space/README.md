---
title: CREST Nowcast
emoji: 🔮
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: AI streamflow nowcasting backend for CREST_demo (DI-LSTM)
---

# CREST_nowcast — AI streamflow nowcasting backend

Companion Space to [CREST_demo](https://huggingface.co/spaces/vincewin/CREST_demo).
Serves minutes-to-hours-ahead streamflow nowcasts from a **DI-LSTM**: an LSTM
whose inputs fuse MRMS radar precipitation with the most recent USGS gauge
observation (the *data integration* idea of Feng, Fang & Shen 2020, WRR,
doi:10.1029/2019WR026793 — reimplemented from the published method; no MHPI
code is used). Where the published DI results are daily-timestep, this model
runs **hourly**, with the observation's age as an input so stale/latent
gauges degrade gracefully.

- **Inference**: CPU, milliseconds per gauge (`/nowcast` API).
- **Training**: ZeroGPU (`@spaces.GPU`), resumable in short bursts.
- **Training data**: hourly USGS IV discharge (NWIS) + basin-mean MRMS
  precipitation extracted from the `vincewin/CREST_data` month tars; prepared
  series cached in the private `vincewin/CREST_nowcast_data` dataset.
- **Weights**: private model repo `vincewin/CREST_nowcast_model`.
