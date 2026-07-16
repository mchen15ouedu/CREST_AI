#!/bin/bash
# DI-LSTM nowcaster training on an HPC GPU node (e.g. OU OSCER).
# Submit from the directory holding model.py/train.py/data.py/train_hpc.py:
#   sbatch slurm_train.sbatch
#
#SBATCH --job-name=crest_nowcast
#SBATCH --partition=gpu            # check your cluster: sinfo -o "%P %G"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=nowcast_%j.log

module load Anaconda3 2>/dev/null || module load Mamba 2>/dev/null || true
source activate nowcast            # see README_HPC.md for the env

# token lives OUTSIDE the repo:  echo hf_... > ~/.hf_token && chmod 600 ~/.hf_token
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

cd "$SLURM_SUBMIT_DIR"
python train_hpc.py --max-epochs 5000 --patience 300
