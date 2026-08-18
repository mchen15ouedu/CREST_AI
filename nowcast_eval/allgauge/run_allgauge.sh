#!/bin/bash
# all-gauge replay: obs -> precip -> replay v1/v3f -> route table
cd "$(dirname "$0")"
PY=/media/scratch/MengyuChen/conda_envs/nowcast/bin/python
export HF_HOME=/media/scratch/MengyuChen/hf_cache
set -e
echo "$(date): stage3 obs"; $PY -u stage3_obs.py > stage3.log 2>&1
echo "$(date): stage4 precip"; $PY -u stage4_precip.py > stage4.log 2>&1
echo "$(date): stage5 replay"; $PY -u stage5_replay.py > stage5.log 2>&1
echo "$(date): stage6 route"; $PY -u stage6_route.py > stage6.log 2>&1
echo "$(date): all-gauge chain finished"
# post-processing (figures need the notebook2 env)
NB=/home/MengyuChen/.conda/envs/notebook2/bin/python
echo "$(date): stage7 report stats"; $PY -u stage7_report_all.py > stage7.log 2>&1
echo "$(date): figures"; $NB stage9_scatter_all.py > stage9.log 2>&1; $NB stage10_map_winners.py > stage10.log 2>&1
echo "$(date): post-processing finished"
