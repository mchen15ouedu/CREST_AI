#!/bin/bash
# Data prep on a CPU node (no GPU needed). Downloads one CONUS MRMS month-tar
# per month, so give it scratch space + generous time for many months.
#   sbatch slurm_prep.sbatch
#
#SBATCH --job-name=nowcast_prep
#SBATCH --partition=normal         # any CPU partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=prep_%j.log

module load Anaconda3 2>/dev/null || module load Mamba 2>/dev/null || true
source activate nowcast

export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}   # month-tars are GB-scale

cd "$SLURM_SUBMIT_DIR"
python prep_hpc.py --months 2023_01-2024_12
python prep_hpc.py --months 2025_01-2025_06          # validation months
