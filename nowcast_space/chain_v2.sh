#!/bin/bash
# DI-LSTM v2 retraining chain (RETRAIN_V2_BRIEF.md): prep the stratified
# 256-gauge set, then train the v2 6h model and the 12h companion, each
# uploading its checkpoint + per-gauge skill table under NEW v2 names.
#   nohup ./chain_v2.sh > chain_v2.log 2>&1 &
cd "$(dirname "$0")"
export HF_TOKEN=$(tr -d ' \r\n' < ~/huggingface.txt)
export SCRATCH=${SCRATCH:-/media/scratch/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
PY=$SCRATCH/conda_envs/nowcast/bin/python

echo "$(date): prep starting (256 gauges, 2023_01-2025_06)"
$PY -u prep_hpc.py --gauges-file gauges_v2.txt --months 2023_01-2025_06 \
    > prep_v2.log 2>&1
if ! grep -q "prep DONE" prep_v2.log; then
    echo "$(date): prep did not finish cleanly — NOT training. See prep_v2.log;"
    echo "rerun this script after fixing (prep is resumable)."
    exit 1
fi
echo "$(date): prep DONE — training v2 6h (dilstm_v2.pt)"
$PY -u train_hpc.py --gauges-file gauges_v2.txt --months 2023_01-2024_12 \
    --max-epochs 5000 --patience 300 > train_v2.log 2>&1
echo "$(date): v2 6h exited $? — training v2 12h (dilstm_h12_v2.pt)"
$PY -u train_hpc.py --gauges-file gauges_v2.txt --months 2023_01-2024_12 \
    --horizon 12 --max-epochs 5000 --patience 300 > train_h12_v2.log 2>&1
echo "$(date): v2 12h exited $? — all v2 runs finished"
