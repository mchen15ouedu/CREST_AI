#!/bin/bash
cd "$(dirname "$0")"
PY=/media/scratch/MengyuChen/conda_envs/nowcast/bin/python
NB=/home/MengyuChen/.conda/envs/notebook2/bin/python
export HF_HOME=/media/scratch/MengyuChen/hf_cache
set -e
echo "$(date): stage6 route"; $PY -u stage6_route.py > stage6.log 2>&1
echo "$(date): stage7 report stats"; $PY -u stage7_report_all.py > stage7.log 2>&1
echo "$(date): stage8 addendum"; $PY -u stage8_addendum.py > stage8.log 2>&1
echo "$(date): figures"; $NB stage9_scatter_all.py > stage9.log 2>&1; $NB stage10_map_winners.py > stage10.log 2>&1
echo "$(date): post-processing finished"
