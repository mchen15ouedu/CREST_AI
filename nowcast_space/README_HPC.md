# Migrating the DI-LSTM nowcaster pipeline to an HPC (SLURM)

The sbatch files are cluster-agnostic — three lines to adapt on any SLURM
system (partition, module load, scratch path). Discover your cluster's values:

```bash
sinfo -o "%P %G %l"        # partitions, their GPUs (gres), walltime limits
module avail 2>&1 | grep -iE "mamba|conda"
echo $SCRATCH; df -h /scratch 2>/dev/null   # where the big filesystem lives
```

(For reference, OU OSCER Sooner used `sooner_test` / `sooner_gpu_test` and
`module load Mamba`.)

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

Conda envs and caches are what silently fill home quotas — put them on
scratch from the start:

```bash
module load Mamba                              # or Anaconda3/Miniconda
export SCRATCH=/scratch/$USER                  # adjust to your cluster
conda config --add envs_dirs $SCRATCH/conda_envs
conda config --add pkgs_dirs $SCRATCH/conda_pkgs
export PIP_CACHE_DIR=$SCRATCH/pip_cache

mamba create -n nowcast python=3.11 -y
source activate nowcast                        # (avoid `mamba init` on shared clusters)
pip install torch numpy pandas pyarrow huggingface_hub requests
python -c "import torch; print(torch.cuda.is_available())"   # on a GPU node: True
```

Caveat: many clusters purge scratch (2–4 weeks untouched). Everything here is
re-creatable (env from this README, data/checkpoints live in the HF repos),
so a purge costs minutes, not work.

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

or interactively (`srun -p sooner_gpu_test --gres=gpu:1 --mem=16G -t 2:00:00 --pty bash`):

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

## Claude Code on the cluster (optional collaborator)

To have Claude Code work alongside you on the HPC without the interactive
login (no browser / paste there):

```bash
# on your LAPTOP (logged-in Claude Code, working clipboard):
claude setup-token            # browser flow → prints a 1-year token; save to a file
scp claude_token.txt <you>@<hpc>:~/.claude_code_token

# on the HPC (type these short lines by hand):
chmod 600 ~/.claude_code_token
echo 'export CLAUDE_CODE_OAUTH_TOKEN=$(cat ~/.claude_code_token)' >> ~/.bashrc
echo 'export CLAUDE_CONFIG_DIR=$SCRATCH/claude_config' >> ~/.bashrc   # keep ~/.claude off home quota
source ~/.bashrc && claude    # starts authenticated, no paste needed
```

Run it on the login node inside the cloned `CREST_AI/nowcast_space` folder —
this README plus the code is enough context for it to submit and monitor the
SLURM jobs.

## SLURM knobs to check on your cluster

- Partition names: `sinfo -o "%P %G %l"` — edit `--partition` (and
  `--gres`, some clusters want `--gres=gpu:a100:1`).
- `module avail anaconda` / `mamba` — edit the `module load` line.
- Scratch path convention for `HF_HOME`.
