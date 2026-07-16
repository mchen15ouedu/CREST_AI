#!/bin/bash
# Data prep on a SLURM CPU node (no GPU needed). Downloads one CONUS MRMS
# month-tar per month (GB-scale) — caches MUST live on scratch, not home.
#   sbatch slurm_prep.sbatch
#
# EDIT FOR YOUR CLUSTER: --partition (any CPU queue), module load line, SCRATCH.
#
#SBATCH --job-name=nowcast_prep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=prep_%j.log

module load Mamba 2>/dev/null || module load Anaconda3 2>/dev/null || true
source activate nowcast

export HF_TOKEN=$(tr -d ' \r\n' < ~/huggingface.txt)
SCRATCH=${SCRATCH:-/media/scratch/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}

cd "$SLURM_SUBMIT_DIR"
python prep_hpc.py --months 2023_01-2024_12
python prep_hpc.py --months 2025_01-2025_06          # validation months
