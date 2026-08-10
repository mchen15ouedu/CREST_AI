# Virtual-user fleet — precompute CREST-AI simulations on the server

Goal: when a real user clicks Simulate, the dashboard serves already-computed
results instantly and warm-starts any re-run (calibration, 2-D maps) from a
state never more than 10 days away.

```
server fleet_run.py ──► vincewin/CREST_fleet (private, read-only for the Space)
                          results/<gid>_<model>.json   hourly rows + state times
                          states/<gid>_<model>.pqf     f16 bundle, state every 10 d
Space (on first touch of a gauge) ──► fleetstore.ensure_local() pulls both,
unpacks states to loose GeoTIFFs → existing cache/warm-start machinery serves it
```

Storage: ~60 MB/gauge avg (states f16 + rows) → all 9,322 GAGES-II gauges x
5 years ≈ 0.6 TB, inside the 1 TB private quota. 2-D Q is NOT stored — the
dashboard re-renders any window on demand from the nearest state (~30-60 s).

## Server setup (once)

```bash
# 1. code (same clone as the dashboard sources)
git clone https://github.com/mchen15ouedu/CREST_AI.git && cd CREST_AI

# 2. python env — caches on scratch, never home
export SCRATCH=/media/scratch/$USER
conda config --add envs_dirs $SCRATCH/conda_envs
conda config --add pkgs_dirs $SCRATCH/conda_pkgs
export PIP_CACHE_DIR=$SCRATCH/pip_cache
conda create -n crestai python=3.11 -y && conda activate crestai
pip install -r requirements.txt truststore

# 3. EF5 binary (fork, PQF reader needs Arrow C++)
conda install -y -c conda-forge arrow-cpp libparquet compilers make autoconf automake libtool
git clone https://github.com/mchen15ouedu/EF5.git && cd EF5
./autogen.sh 2>/dev/null || autoreconf -fi
CXXFLAGS="-std=c++20 -I$CONDA_PREFIX/include" LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib" \
  ./configure --with-arrow && make -j$(nproc)
cd ..          # repo root now has ./EF5/bin/ef5 (fleet_run.py checks this)

# 4. env
export HF_TOKEN=$(tr -d ' \r\n' < ~/huggingface.txt)   # write access (creates CREST_fleet)
export CREST_CACHE_DIR=$SCRATCH/crest_cache            # forcing clips + in-flight states
export CREST_FORCING_CACHE_GB=150                      # LRU cap for the forcing store
export CREST_DEMO_MOCK=0
```

## Run

```bash
# smoke test first: 2 gauges, verify end-to-end + check sizes in the repo
python fleet/fleet_run.py --gauges "08167000, 08144500" --workers 2

# then the real pass (background, survives SSH disconnect)
nohup python fleet/fleet_run.py --workers 8 > fleet.log 2>&1 &
tail -f fleet.log
```

- Defaults: all 9,322 gauges, 2021-07-01..2026-06-30, state every 10 days,
  quick-run (speed) scheme, no AI calibration, no 2-D frames.
- **Resumable**: rerun the same command after any crash/reboot — gauges already
  in the repo are skipped; inside a gauge, chunks whose rows are cached
  fast-forward. `fleet_progress.jsonl` logs one line per gauge.
- Expect very roughly 0.5-1.5 h/gauge (183 chunked EF5 runs incl. one 90-day
  warm-up) → all gauges ≈ 2-5 weeks at 8 workers; scale --workers to the box
  (each worker ≈ 1 core + ~1-2 GB RAM). Start with favorites/flood-prone
  lists via --gauges file if you want priority basins online sooner.
- Be a polite NWIS client: obs are fetched once per gauge then cached; if
  waterservices.usgs.gov throttles, lower --workers.

## Dashboard side (already wired, V20)

`hf_data/fleetstore.py` pulls a gauge's bundle+record from CREST_fleet on
first touch (env `CREST_FLEET_REPO`, empty disables). Rows serve instantly
("served entirely from cache"); 2-D maps re-render from the nearest state;
calibration warm-starts the same way. Fleet data never mixes with
CREST_state uploads/deletions — separate repo, read-only from the Space.
