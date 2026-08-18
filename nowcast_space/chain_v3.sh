#!/bin/bash
# DI-LSTM v3: feat_version 3 (16 HydroBASINS-L7 static attributes appended,
# from build_gauge_statics.py) + per-gauge variance-normalized NSE loss
# (Kratzert et al. 2019). Waits for the round-2 v2 chain to release the GPU,
# then trains 6h and 12h companions -> dilstm_v3.pt / dilstm_h12_v3.pt.
# Serving ignores v3 names until the Space is upgraded, so this is a pure
# experiment alongside the live v2 models.
#   nohup ./chain_v3.sh <v2b_chain_pid> > chain_v3.log 2>&1 &
cd "$(dirname "$0")"
V2B_PID=${1:?usage: chain_v3.sh <v2b_chain_pid>}
export HF_TOKEN=$(tr -d ' \r\n' < ~/huggingface.txt)
export SCRATCH=${SCRATCH:-/media/scratch/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
PY=$SCRATCH/conda_envs/nowcast/bin/python

echo "$(date): waiting for v2 round-2 chain (PID $V2B_PID) to finish"
while kill -0 "$V2B_PID" 2>/dev/null; do sleep 60; done
sleep 5

echo "$(date): GPU free — training v3 6h (statics + NSE loss, dilstm_v3.pt)"
$PY -u train_hpc.py --gauges-file gauges_v2.txt --months 2023_01-2024_12 \
    --feat-version 3 --loss nse --obs-dropout 0.25 \
    --max-epochs 5000 --patience 300 --fresh > train_v3.log 2>&1
echo "$(date): v3 6h exited $? — training v3 12h (dilstm_h12_v3.pt)"
$PY -u train_hpc.py --gauges-file gauges_v2.txt --months 2023_01-2024_12 \
    --horizon 12 --feat-version 3 --loss nse --obs-dropout 0.25 \
    --max-epochs 5000 --patience 300 --fresh > train_h12_v3.log 2>&1
echo "$(date): v3 12h exited $? — v3 chain finished"
