# Migrating the DI-LSTM nowcaster pipeline to an HPC (SLURM)

What moves where:

| Piece | Runs on | Why |
|---|---|---|
| Data prep (`prep_hpc.py`) | HPC CPU node *or* Space | either works; both write the same repo |
| Training (`train_hpc.py`) | **HPC GPU node** | no ZeroGPU time cap or quota |
| Inference `/nowcast` API | **stays on the Space** | dashboard needs an always-on public endpoint; HPC nodes can't serve one |
| CREST_demo dashboard | stays on its Space | public web app |

Everything round-trips through two private Hugging Face repos, so no files
move by hand and Space/ZeroGPU remains a working fallback:

```
prep (HPC or Space)  ──►  vincewin/CREST_nowcast_data   (per-gauge/month parquet)
                                     │  read
HPC  train_hpc.py    ────────────────┘
      │  upload best checkpoint
      ▼
vincewin/CREST_nowcast_model (dilstm.pt)  ──►  Space "Reload model" serves it
```

## 1. Copy the code

```bash
ssh <you>@<hpc>
git clone https://github.com/mchen15ouedu/CREST_AI.git
cd CREST_AI/nowcast_space        # model.py train.py data.py train_hpc.py prep_hpc.py + sbatch files
```

(or `scp -r` the local `pythonscripts/CREST_nowcast` folder — same files.)

## 2. Environment (once)

```bash
module load Anaconda3            # or Mamba/Miniconda — whatever the cluster provides
conda create -n nowcast python=3.11 -y
conda activate nowcast
pip install torch numpy pandas pyarrow huggingface_hub requests
python -c "import torch; print(torch.cuda.is_available())"   # on a GPU node: True
```

If the default `torch` wheel doesn't see the GPU, install the build matching
the cluster's CUDA driver, e.g. `pip install torch --index-url
https://download.pytorch.org/whl/cu121`.

## 3. Token + cache locations (once)

```bash
echo hf_xxxxxxxx > ~/.hf_token && chmod 600 ~/.hf_token     # read+write on the two private repos
echo 'export HF_HOME=/scratch/$USER/hf_cache' >> ~/.bashrc  # month-tars are GB-scale; keep off home quota
```

Compute nodes need outbound HTTPS to huggingface.co (prep also needs
waterservices.usgs.gov). If compute nodes are offline, run each script once on
the login node — it fills `$HF_HOME` — then submit; cached files are reused.

## 4. Data prep (only for gauges/months not already in the data repo)

The 4 starter gauges × 2023_01–2025_06 are already prepped. For new gauges or
periods:

```bash
sbatch slurm_prep.sbatch                 # edit --gauges/--months inside first
# or interactively:
export HF_TOKEN=$(cat ~/.hf_token)
python prep_hpc.py --gauges "07331600, 07316000" --months 2023_01-2025_06
```

Resumable: (gauge, month) files already in the repo are skipped, so rerun
freely after a timeout. Budget very roughly 2–10 min per month (one CONUS
MRMS tar download dominates; HPC networks are slower than HF-internal).

## 5. Train

```bash
sbatch slurm_train.sbatch
squeue -u $USER                          # watch the queue
tail -f nowcast_<jobid>.log              # watch epochs / val NSE
```

or interactively (e.g. `srun -p gpu --gres=gpu:1 --mem=16G -t 2:00:00 --pty bash`):

```bash
export HF_TOKEN=$(cat ~/.hf_token)
python train_hpc.py --gauges "01011000, 08166200, 08167000, 08144500" \
    --months 2023_01-2024_12 --max-epochs 5000 --patience 300
```

- Resumes from the current repo checkpoint (`--fresh` to restart).
- Prints the **persistence-baseline val NSE** first — the number the model
  must beat for real skill (currently 0.9833 pooled; the model must win on
  flood-rise events, not just pooled NSE).
- Early-stops on pooled val NSE (2025_01–2025_06 held out), keeps the best
  epoch, uploads at most every `--upload-every-min` (15) and at the end.
  `--no-upload` writes `dilstm_best.pt` locally instead.
- Job killed by walltime? The periodic uploads mean at most the last
  15 minutes of progress is lost — just resubmit.

## 6. Serve the new model

Space → Status tab → **Reload model** (or REST `api_name=reload`). The
dashboard's next nowcast call uses the new checkpoint automatically. Nothing
else to deploy.

## SLURM knobs to check on your cluster

- Partition names: `sinfo -o "%P %G %l"` — edit `--partition` (and
  `--gres`, some clusters want `--gres=gpu:a100:1`).
- `module avail anaconda` / `mamba` — edit the `module load` line.
- Scratch path convention for `HF_HOME`.
