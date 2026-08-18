#!/bin/bash
# Multi-year all-gauge eval driver: precip pool (5 procs) + replay scheduler
# (2 concurrent GPU replays, each month as soon as its obs+precip exist).
# my_obs.py must already be running / done (it writes obs/obs_<ym>.parquet).
cd "$(dirname "$0")"
PY=/media/scratch/MengyuChen/conda_envs/nowcast/bin/python
export HF_HOME=/media/scratch/MengyuChen/hf_cache
MONTHS=$($PY -c "import common; print(' '.join(common.MONTHS))")
echo "$(date): driver start"
# replay scheduler (background)
(
  while true; do
    alldone=1
    for ym in $MONTHS; do
      [ -f acc/acc_$ym.npz ] && continue
      if [ -f logs/replay_$ym.log ] && grep -q "Traceback" logs/replay_$ym.log && [ ! -f logs/replay_$ym.retry ]; then
        echo "$(date): replay $ym FAILED once, retrying"; touch logs/replay_$ym.retry; rm -f logs/replay_$ym.log
      fi
      if [ -f logs/replay_$ym.log ] && grep -q "Traceback" logs/replay_$ym.log; then continue; fi   # failed twice: skip
      alldone=0
      [ -f obs/obs_$ym.parquet ] && [ -f precip/precip_$ym.npz ] || continue
      pgrep -f "my_replay.py $ym\$" >/dev/null && continue
      if [ "$(pgrep -fc 'my_replay.py')" -lt 2 ]; then
        echo "$(date): replay $ym start"
        nohup $PY -u my_replay.py $ym > logs/replay_$ym.log 2>&1 &
        sleep 20
      fi
    done
    [ $alldone = 1 ] && break
    sleep 60
  done
  echo "$(date): all replays finished"
) &
SCHED=$!
# precip pool
printf "%s\n" $MONTHS | xargs -P 8 -I{} sh -c "$PY -u my_precip.py {} > logs/precip_{}.log 2>&1"
echo "$(date): precip pool finished"
wait $SCHED
echo "$(date): driver finished (replays done) — run my_score.py"
