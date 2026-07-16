#!/bin/bash
# Plain GPU server (no SLURM). Activate the env first, then:
#   ./run_train.sh                        # defaults (4 starter gauges, 2023-2024)
#   ./run_train.sh --max-epochs 10000     # extra args pass through to train_hpc.py
# Runs in the background via nohup, so it survives SSH disconnects.
cd "$(dirname "$0")"

export HF_TOKEN=$(tr -d ' \r\n' < ~/huggingface.txt)
export SCRATCH=${SCRATCH:-/media/scratch/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}

nohup python train_hpc.py --max-epochs 5000 --patience 300 "$@" > train.log 2>&1 &
echo "training started, PID $!  —  watch with:  tail -f train.log"
