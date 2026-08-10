# DI-LSTM v2 retraining brief (for Claude Code on the HPC)

You are retraining the DI-LSTM streamflow nowcaster on this cluster. This
document is self-contained: together with the code in this folder
(`CREST_AI/nowcast_space/`) it is everything you need. Environment setup,
SLURM headers, and token locations are in `README_HPC.md` — read that first
if the `nowcast` conda env does not exist yet.

## Why retrain

The deployed v1 model (`dilstm.pt`, feat_version 1) hallucinated flood peaks
at gauges whose observations went dark: on 2026-08-10 it flagged tier-3
(>= Q5) floods at two data-dead gauges with ~0 mm/h rain and ~0 simulated
flow, triggering false 2-D inundation events. The dashboard now demotes
obs-unsupported flood tiers at serving time, but the model itself still
extrapolates freely without observations. v2 fixes the root cause in
training:

1. **feat_version 2** — 5 features instead of 4: an explicit `obs_missing`
   flag, and the obs-age channel capped at 240 h, so "no gauge" is a
   well-defined input state instead of an out-of-distribution 999.
2. **Obs-outage augmentation + obs-channel dropout** — training windows see
   randomized reporting lags, multi-hour outages, and (15% of windows) a
   fully dead gauge while the target stays the true discharge. The model is
   forced to learn a rainfall-driven fallback instead of free extrapolation.
3. **Per-gauge validation skill table** — uploaded next to the checkpoint;
   the dashboard uses it to bar demonstrably unskilled gauges from driving
   flood tiers (v1 recorded only one global val NSE).

## Hard rules

- **NEVER overwrite `dilstm.pt` or `dilstm_h12.pt`** in
  `vincewin/CREST_nowcast_model`. v2 uploads under NEW names
  (`dilstm_v2.pt`, `dilstm_h12_v2.pt`, `skill_dilstm_v2.parquet`,
  `skill_dilstm_h12_v2.parquet`) — the defaults in `train_hpc.py` already do
  this. The serving side prefers the v2 files the moment they exist;
  deleting them from the model repo is the complete rollback.
- **Never print or log the HF token** (it lives in `~/huggingface.txt`).
- Do not push code changes to GitHub without being asked; your job is to run
  the pipeline. If you find a bug, fix it locally, note it clearly in your
  summary, and keep going.
- Long jobs go through SLURM (`slurm_prep.sbatch` / `slurm_train.sbatch`) or
  `nohup` wrappers — never leave heavy work attached to your own process.

## Steps

Work from a fresh `git pull` of `mchen15ouedu/CREST_AI`, in
`nowcast_space/`, with `HF_TOKEN` exported as in README_HPC.md.

### 1. Select the training gauges

One global model serves ~9k CONUS gauges; 4 training gauges cannot support a
per-gauge skill table. Pick a stratified set (area quartiles x region, the 4
starter gauges always included):

    python select_gauges.py --n 256 --out gauges_v2.txt

256 is the intended size. If prep time (step 2) becomes the bottleneck,
128 is an acceptable floor — note the reduction in your summary.

### 2. Prep the data (CPU, resumable, the long pole)

    python prep_hpc.py --gauges-file gauges_v2.txt --months 2023_01-2025_06

- MRMS cost is per MONTH (one CONUS tar pass serves every gauge at once),
  so the gauge count mostly adds USGS requests — cheap. Budget roughly
  10-30 min per month on a login/CPU node; 30 months total. Use
  `slurm_prep.sbatch` (edit `--gauges-file` in) or `run_prep.sh` if the
  login node kills long processes.
- Resumable: (gauge, month) files already in `vincewin/CREST_nowcast_data`
  are skipped. Re-run after any crash. The exit message lists failed months.
- The 2025_01–2025_06 months are the validation set — they are REQUIRED.

### 3. Train the 6-h model (GPU)

    python train_hpc.py --gauges-file gauges_v2.txt --months 2023_01-2024_12 \
        --max-epochs 5000 --patience 300

Defaults: feat_version 2, horizon 6, obs-dropout 0.15, checkpoint
`dilstm_v2.pt`. Uploads the best checkpoint at most every 15 min and at the
end, plus `skill_dilstm_v2.parquet` after the final evaluation. Use
`slurm_train.sbatch` or `run_train.sh` for the actual run; watch
`train.log` / `nowcast_<jobid>.log`.

Dataset scale: ~256 gauges x 2 years is roughly 1.5M training windows
(~2-3 GB as float32 in RAM at N_FEAT=5) — comfortable on any GPU node.
Expect epochs to be ~100x slower than the 4-gauge v1 runs; the
`--patience 300` early stop still applies, so a converged run may take a
day or two of walltime. The periodic 15-min checkpoint uploads make
walltime kills harmless — just resubmit and it resumes.

### 4. Train the 12-h companion

    python train_hpc.py --gauges-file gauges_v2.txt --months 2023_01-2024_12 \
        --horizon 12 --max-epochs 5000 --patience 300

Uploads `dilstm_h12_v2.pt` + `skill_dilstm_h12_v2.parquet`.

### 5. Acceptance criteria (check BEFORE calling it done)

`train_hpc.py` prints these at the end of each run:

- **Persistence baseline**: pooled val NSE of holding the last obs flat.
  The v1 pooled val NSE was ~0.98 against a ~0.983 baseline. The v2
  obs-fresh pooled NSE must be **at or above the persistence baseline**
  printed for its own (larger, harder) gauge set — expect a lower absolute
  number than v1's; that is fine, the gauge set changed.
- **No-obs pooled NSE** must be **positive** (v1 would be strongly negative
  under this test — that is the hallucination).
- **No-obs peak ratio p95 < ~3** (predicted/true window-max flow with the
  obs channel blanked). This is the direct hallucination metric: v1 invents
  >= Q5 peaks from nothing. If p95 is large, raise `--obs-dropout` to 0.25
  and retrain before uploading.
- Spot-check the printed low-skill gauges: a few genuinely hard gauges
  (regulated, snowmelt, tiny arid basins) with low NSE are expected and are
  exactly what the skill table is for — not a failure.

If a criterion fails, do not leave the failing checkpoint as the latest v2
upload: fix and retrain, or delete the v2 files from
`vincewin/CREST_nowcast_model` (that restores v1 serving) and report.

### 6. Handoff

Nothing to deploy: the hourly CONUS precompute prefers `dilstm_v2.pt` /
`dilstm_h12_v2.pt` automatically and joins the skill table into
`nowcast/latest.parquet` (`val_nse_gauge` column), which activates the
dashboard's per-gauge skill gate. In your final summary report: gauge count,
window counts, epochs, pooled val NSE (obs-fresh and no-obs) vs the
persistence baseline, the no-obs peak-ratio p95/p99, the 5 worst gauges,
and anything you changed or reduced.

## Optional follow-up (only if asked)

MC-dropout uncertainty at serving: the precompute honors `NOWCAST_MC_N`
(Space env var, default 0) and writes a per-gauge `mc_cv` spread column;
`train_hpc.py --mc-eval` prints the spread diagnostic. Calibrating a
spread-based tier gate is a separate task — do not tune it as part of this
retraining.
