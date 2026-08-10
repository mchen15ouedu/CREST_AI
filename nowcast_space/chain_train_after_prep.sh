#!/bin/bash
# Waits for the running full prep to finish, then starts the scaled training
# run — so training kicks off even with nobody logged in.
#   nohup ./chain_train_after_prep.sh <prep_pid> > chain.log 2>&1 &
# Refuses to start training unless prep.log ends with a clean "prep DONE"
# (if prep died or reported failed months, rerun ./run_prep.sh first — it
# resumes — then rerun this script).
cd "$(dirname "$0")"
PREP_PID=${1:?usage: chain_train_after_prep.sh <prep_pid>}

echo "$(date): waiting for prep (PID $PREP_PID) to finish"
while kill -0 "$PREP_PID" 2>/dev/null \
      && ! grep -qE "prep DONE|prep incomplete" prep.log; do
    sleep 60
done
sleep 5   # let the last log lines flush

if ! grep -q "prep DONE" prep.log; then
    echo "$(date): prep did not finish cleanly — NOT starting training."
    echo "Fix/resume with ./run_prep.sh --gauges-file gauges_all_6036.txt --months 2023_01-2025_06, then rerun this script."
    exit 1
fi

echo "$(date): prep DONE — starting scaled training (--fresh, all gauges)"
export HF_TOKEN=$(tr -d ' \r\n' < ~/huggingface.txt)
export SCRATCH=${SCRATCH:-/media/scratch/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
exec python -u train_hpc.py --fresh --gauges-file gauges_all_6036.txt \
    --max-epochs 5000 --patience 300 > train.log 2>&1
