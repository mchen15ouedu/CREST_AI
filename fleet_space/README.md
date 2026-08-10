---
title: CREST fleet runner
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# CREST_fleet_runner

Virtual-user fleet for [CREST_demo](https://huggingface.co/spaces/vincewin/CREST_demo):
continuously pre-simulates USGS gauges (quick-run, 2021–2026, state every
10 days, hourly rows) and uploads the artifacts to the private
`vincewin/CREST_fleet` dataset, from which the dashboard serves users
instantly. The page shows the live fleet log. Code lives in
[CREST_AI](https://github.com/mchen15ouedu/CREST_AI) (`fleet/`) and is cloned
fresh at every boot. Requires the `HF_TOKEN` secret.
