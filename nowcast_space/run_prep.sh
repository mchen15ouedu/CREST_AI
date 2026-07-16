#!/bin/bash
# Data prep on a plain server (no SLURM, no GPU needed). Activate the env, then:
#   ./run_prep.sh --gauges "07331600, 07316000" --months 2023_01-2025_06
# Resumable — rerun freely; months already in the data repo are skipped.
cd "$(dirname "$0")"

export HF_TOKEN=$(tr -d ' \r\n' < ~/huggingface.txt)
export SCRATCH=${SCRATCH:-/media/scratch/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}

nohup python prep_hpc.py "$@" > prep.log 2>&1 &
echo "prep started, PID $!  —  watch with:  tail -f prep.log"
