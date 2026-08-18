#!/bin/bash
# Re-run replay eval with the full-scale v3 checkpoints added (v3f_6 / v3f_12),
# then rescore and redraw the per-gauge scatters (fig6 unchanged, fig7 new).
cd "$(dirname "$0")"
PY=/media/scratch/MengyuChen/conda_envs/nowcast/bin/python
PYFIG=/home/MengyuChen/.conda/envs/notebook2/bin/python
echo "$(date): stage5 replay (8 checkpoints)"
$PY -u stage5_replay.py > stage5.log 2>&1 || { echo "stage5 FAILED"; exit 1; }
echo "$(date): stage6 score"
$PY -u stage6_score.py > stage6.log 2>&1 || { echo "stage6 FAILED"; exit 1; }
echo "$(date): scatters"
$PYFIG -u stage8_scatter.py > stage8.log 2>&1 || { echo "stage8 FAILED"; exit 1; }
$PYFIG -u stage9_scatter_full.py > stage9.log 2>&1 || { echo "stage9 FAILED"; exit 1; }
echo "$(date): eval chain done"
