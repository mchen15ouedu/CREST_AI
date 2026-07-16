# Training the DI-LSTM nowcaster on an HPC GPU

The CREST_nowcast Space stays the inference server; the HPC only trains.
Everything round-trips through two private Hugging Face repos, so no files
move by hand:

```
Space "Data prep"  ──►  vincewin/CREST_nowcast_data   (per-gauge/month parquet)
                                    │  read
HPC  train_hpc.py  ─────────────────┘
      │  upload best checkpoint
      ▼
vincewin/CREST_nowcast_model (dilstm.pt)  ──►  Space "Reload model" serves it
```

## One-time setup on the HPC

```bash
# 1. copy these four files to the cluster (scp or git clone of CREST_AI/nowcast_space)
#    model.py  train.py  data.py  train_hpc.py   (+ slurm_train.sbatch)

# 2. environment (CUDA build of torch — pick the wheel matching the cluster's driver)
conda create -n nowcast python=3.11 -y
conda activate nowcast
pip install torch numpy pandas pyarrow huggingface_hub requests

# 3. HF token with read+write on the two private repos, OUTSIDE any repo dir
echo hf_xxxxxxxx > ~/.hf_token && chmod 600 ~/.hf_token
```

Compute nodes need outbound HTTPS to huggingface.co (and usgs.gov only if you
ever run prep here). If your compute nodes are offline, run the script once on
a login node — it caches every download under `$HF_HOME` — then submit the
batch job; the cached copies are used automatically. The data is small (a few
MB of parquet per gauge-year).

## Run

```bash
sbatch slurm_train.sbatch
# or interactively:
export HF_TOKEN=$(cat ~/.hf_token)
python train_hpc.py --gauges "01011000, 08166200, 08167000, 08144500" \
    --months 2023_01-2024_12 --max-epochs 5000 --patience 300
```

- Resumes from the current repo checkpoint by default (`--fresh` to restart).
- Prints the **persistence-baseline val NSE** first — the number the model
  must beat for the AI to have real skill.
- Early-stops on pooled val NSE (2025_01–2025_06 held out), keeps the best
  epoch, uploads it at most every `--upload-every-min` (default 15) and at
  the end. `--no-upload` writes `dilstm_best.pt` locally instead.

## After training

Open the Space → Status tab → **Reload model** (or REST `api_name=reload`).
The dashboard's next nowcast call uses the new checkpoint automatically.

## Adding gauges / months

New gauges or months must be prepped first (Space → Data prep tab, resumable,
CPU-only) so the parquet files exist in `CREST_nowcast_data`; then pass the
same `--gauges/--months` here.
