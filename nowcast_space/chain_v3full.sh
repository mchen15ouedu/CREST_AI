#!/bin/bash
# Full-scale DI-LSTM v3: ALL 6,036 gauges, feat_version 3 (statics appended
# per batch — windows stay 5-channel, ~45 GB instead of ~190 GB) + per-gauge
# NSE loss + obs-dropout 0.25. Uploads under NEW names (dilstm_v3_full.pt /
# dilstm_h12_v3_full.pt) so the live-routed dilstm_v3.pt is untouched until
# the archive-replay eval judges the new checkpoints.
#
# Attempt 2 (Aug 13): attempt 1 diverged with best at epoch 1 — flat lr 1e-3
# at ~60k steps/epoch plus ~100x NSE weights on near-flat gauges. Fixes:
# lr 3e-4 + ReduceLROnPlateau, NSE weight cap 25, validation every 10k
# batches (patience 120 is now 120 evals ≈ 17 epochs of stagnation).
#   nohup ./chain_v3full.sh > chain_v3full.log 2>&1 &
cd "$(dirname "$0")"
export HF_TOKEN=$(tr -d ' \r\n' < ~/huggingface.txt)
export SCRATCH=${SCRATCH:-/media/scratch/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
PY=$SCRATCH/conda_envs/nowcast/bin/python

echo "$(date): v3 full-scale 6h, attempt 2 (dilstm_v3_full.pt)"
$PY -u train_hpc.py --gauges-file gauges_all_6036.txt --months 2023_01-2024_12 \
    --feat-version 3 --loss nse --obs-dropout 0.25 \
    --lr 3e-4 --val-every 10000 \
    --max-epochs 5000 --patience 120 --fresh \
    --ckpt-name dilstm_v3_full.pt > train_v3full.log 2>&1
echo "$(date): 6h exited $? — v3 full-scale 12h (dilstm_h12_v3_full.pt)"
$PY -u train_hpc.py --gauges-file gauges_all_6036.txt --months 2023_01-2024_12 \
    --horizon 12 --feat-version 3 --loss nse --obs-dropout 0.25 \
    --lr 3e-4 --val-every 10000 \
    --max-epochs 5000 --patience 120 --fresh \
    --ckpt-name dilstm_h12_v3_full.pt > train_h12_v3full.log 2>&1
echo "$(date): 12h exited $? — full-scale v3 chain finished"
