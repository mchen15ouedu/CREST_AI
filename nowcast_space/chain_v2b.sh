#!/bin/bash
# v2 retrain round 2: obs-dropout 0.15 -> 0.25 because the first round
# failed the no-obs peak-ratio acceptance criterion (p95 5.33/5.57 vs < ~3;
# RETRAIN_V2_BRIEF.md step 5). Data already prepped — training only.
#   nohup ./chain_v2b.sh > chain_v2b.log 2>&1 &
cd "$(dirname "$0")"
export HF_TOKEN=$(tr -d ' \r\n' < ~/huggingface.txt)
export SCRATCH=${SCRATCH:-/media/scratch/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
PY=$SCRATCH/conda_envs/nowcast/bin/python

echo "$(date): training v2 6h, obs-dropout 0.25 (dilstm_v2.pt)"
$PY -u train_hpc.py --gauges-file gauges_v2.txt --months 2023_01-2024_12 \
    --obs-dropout 0.25 --max-epochs 5000 --patience 300 --fresh \
    > train_v2b.log 2>&1
echo "$(date): v2 6h exited $? — training v2 12h, obs-dropout 0.25 (dilstm_h12_v2.pt)"
$PY -u train_hpc.py --gauges-file gauges_v2.txt --months 2023_01-2024_12 \
    --horizon 12 --obs-dropout 0.25 --max-epochs 5000 --patience 300 --fresh \
    > train_h12_v2b.log 2>&1
echo "$(date): v2 12h exited $? — round-2 v2 runs finished"
