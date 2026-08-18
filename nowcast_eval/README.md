# Nowcast model evaluation harnesses (v1 vs v3-full)

Code + small result tables only; inputs (MRMS tars, NWIS obs, per-month
accumulators, replay predictions) live in `/media/scratch/MengyuChen/nowcast_eval/`.
Full write-up: `../v3_vs_v1_eval_report.txt`; figures: `../eval_figures/`.

| dir | what | status |
|---|---|---|
| `.` (stage1-9) | Jul–Aug 2026 archive replay, 550-gauge stratified sample (433 scorable) | superseded |
| `allgauge/` | same 22-day archive replay on all 9,067 served gauges | superseded |
| `multiyear/` | **the comprehensive eval**: Jan/Apr/Jul/Oct 2016–2026 (43 months), hourly issues, all served gauges, obs scenarios fresh / stale-24 h / no-obs, v1 + v3-full + persistence, exact per-gauge accumulators, LOYO validation of routing | current |

`multiyear/route_pergauge_multiyear.parquet` is the table deployed to
`vincewin/CREST_nowcast_model` as `route_pergauge_v1_v3full.parquet` and read hourly by
`scripts/run_nowcast_all.py` (per gauge x obs regime x horizon family winner).

Run order (multiyear): `my_obs.py` (all months) → `run_multiyear.sh` / `sched.py`
(precip pool + GPU/CPU replays) → `my_score.py` → `my_figures.py` (notebook2 env).
