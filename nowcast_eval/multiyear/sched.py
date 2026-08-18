"""Replay scheduler: GPU slots take ready months from the front of MONTHS,
CPU slots (CUDA hidden, 8 threads) from the back. A month is ready when its
obs + precip exist; done when acc/acc_<ym>.npz exists; retried once on
failure. Runs until every month is done or failed twice."""
import os
import subprocess
import sys
import time
from datetime import datetime

from common import BASE, MONTHS

PY = "/media/scratch/MengyuChen/conda_envs/nowcast/bin/python"
GPU_SLOTS = int(os.environ.get("GPU_SLOTS", "2"))
CPU_SLOTS = int(os.environ.get("CPU_SLOTS", "3"))
running = {}                    # ym -> (Popen, kind)
fails = {}


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}: {msg}", flush=True)


def done(ym):
    return os.path.exists(os.path.join(BASE, "acc", f"acc_{ym}.npz"))


def ready(ym):
    return (os.path.exists(os.path.join(BASE, "obs", f"obs_{ym}.parquet"))
            and os.path.exists(os.path.join(BASE, "precip", f"precip_{ym}.npz")))


def launch(ym, kind):
    env = dict(os.environ)
    if kind == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["REPLAY_THREADS"] = "8"
    lg = open(os.path.join(BASE, "logs", f"replay_{ym}.log"), "w")
    p = subprocess.Popen([PY, "-u", "my_replay.py", ym], cwd=BASE, env=env, stdout=lg, stderr=subprocess.STDOUT)
    running[ym] = (p, kind)
    log(f"launch {ym} on {kind}")


while True:
    for ym in list(running):
        p, kind = running[ym]
        if p.poll() is not None:
            del running[ym]
            if done(ym):
                log(f"done {ym} ({kind})")
            else:
                fails[ym] = fails.get(ym, 0) + 1
                log(f"FAILED {ym} ({kind}) attempt {fails[ym]}")
    pending = [m for m in MONTHS if not done(m) and m not in running and fails.get(m, 0) < 2]
    if not pending and not running:
        break
    rdy = [m for m in pending if ready(m)]
    n_gpu = sum(1 for _, k in running.values() if k == "gpu")
    n_cpu = sum(1 for _, k in running.values() if k == "cpu")
    for m in rdy:
        if n_gpu >= GPU_SLOTS:
            break
        launch(m, "gpu"); n_gpu += 1; rdy.remove(m)
    for m in reversed(rdy):
        if n_cpu >= CPU_SLOTS:
            break
        launch(m, "cpu"); n_cpu += 1
    time.sleep(30)
log(f"scheduler finished; failed: {[m for m, k in fails.items() if k >= 2]}")
