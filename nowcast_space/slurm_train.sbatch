#!/bin/bash
# DI-LSTM nowcaster training on a SLURM GPU node. Submit from the directory
# holding model.py/train.py/data.py/train_hpc.py:
#   sbatch slurm_train.sbatch
#
# EDIT FOR YOUR CLUSTER (discover with:  sinfo -o "%P %G %l"  and  module avail):
#   --partition        your GPU partition (e.g. OSCER Sooner: sooner_gpu_test)
#   --gres             some clusters want a type, e.g. --gres=gpu:a100:1
#   module load line   Mamba / Anaconda3 / Miniconda — whatever exists
#   SCRATCH            your cluster's big non-home filesystem
#
#SBATCH --job-name=crest_nowcast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=nowcast_%j.log

module load Mamba 2>/dev/null || module load Anaconda3 2>/dev/null || true
source activate nowcast                # see README_HPC.md for the env

# token file: run once  chmod 600 ~/huggingface.txt
# (tr strips stray spaces/CRLF in case the file was written on Windows)
export HF_TOKEN=$(tr -d ' \r\n' < ~/huggingface.txt)
# keep ALL caches off the (small) home quota
SCRATCH=${SCRATCH:-/media/scratch/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}

cd "$SLURM_SUBMIT_DIR"
python train_hpc.py --max-epochs 5000 --patience 300
