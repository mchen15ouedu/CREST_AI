#!/bin/bash
# Sequential horizon retraining: 12 h ahead, then 24 h ahead, all 6,036 gauges.
#   nohup ./chain_train_horizons.sh > chain_horizons.log 2>&1 &
# Each run uploads to its own file in the model repo (dilstm_h12.pt /
# dilstm_h24.pt), so the Space's 6 h checkpoint (dilstm.pt) is untouched.
# Uses the same prepped data — windows are rebuilt per horizon at train time.
cd "$(dirname "$0")"

export HF_TOKEN=$(tr -d ' \r\n' < ~/huggingface.txt)
export SCRATCH=${SCRATCH:-/media/scratch/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
PY=$SCRATCH/conda_envs/nowcast/bin/python

for HZ in 12 24; do
    echo "$(date): starting H=${HZ} training"
    NOWCAST_HORIZON=$HZ NOWCAST_CKPT="dilstm_h${HZ}.pt" \
        $PY -u train_hpc.py --fresh --gauges-file gauges_all_6036.txt \
        --max-epochs 5000 --patience 300 > "train_h${HZ}.log" 2>&1
    echo "$(date): H=${HZ} training exited with status $?"
done
echo "$(date): all horizon runs finished"
