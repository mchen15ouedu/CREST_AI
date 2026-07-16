#!/bin/bash
# DI-LSTM nowcaster training — OSCER Sooner (OU). Submit from the directory
# holding model.py/train.py/data.py/train_hpc.py:
#   sbatch slurm_train.sbatch
#
#SBATCH --job-name=crest_nowcast
#SBATCH --partition=sooner_gpu_test    # general-access GPU pool (L40S/H100/Ada), 48 h max
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=nowcast_%j.log

module load Mamba                      # OSCER: never run `mamba init`; use source activate
source activate nowcast                # see README_HPC.md for the env

# token lives OUTSIDE the repo:  echo hf_... > ~/.hf_token && chmod 600 ~/.hf_token
export HF_TOKEN=$(cat ~/.hf_token)
# /scratch is auto-purged (~2 weeks) — fine, the HF cache is re-downloadable
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

cd "$SLURM_SUBMIT_DIR"
python train_hpc.py --max-epochs 5000 --patience 300
