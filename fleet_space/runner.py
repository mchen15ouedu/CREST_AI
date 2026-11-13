"""CREST_fleet_runner — runs the virtual-user fleet on this Space, forever.

Boot: clone the current CREST_AI sources (so the runner always uses the same
fleet/pipeline code as the dashboard, no redeploys here), link the EF5 binary,
start fleet/fleet_run.py as a subprocess, and serve a status page on 7860.
When a pass finishes (or the process dies) it re-runs after a pause — the
fleet is resumable, so a finished catalog makes re-runs cheap no-ops.

Space variables (Settings → Variables):
  FLEET_WORKERS  parallel gauges (default 2 — matches free cpu-basic vCPUs)
  FLEET_GAUGES   "all" (default) or comma list of USGS ids
  FLEET_START / FLEET_END   period (defaults 2021-07-01 / 2026-06-30)
  FLEET_ORDER    "forward" (default) or "reverse" — run two runners (e.g. this
                 Space + the user's server) from opposite ends of the catalog
  FLEET_SHARD    "K/N" — this runner takes gauges with catalog-index % N == K;
                 N sibling Spaces with shards 0/N..N-1/N split the work disjointly
Secret: HF_TOKEN (write access — uploads to vincewin/CREST_fleet).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

SRC = "/app/src"
LOG = "/tmp/fleet.log"
CACHE = "/tmp/crest_cache"
CREST_AI_GIT = "https://github.com/mchen15ouedu/CREST_AI.git"
state = {"phase": "booting", "passes": 0}

# free Spaces sleep without HTTP traffic (which freezes the fleet) — sibling
# runners ping each other so the ring stays awake; a GET also WAKES a sleeper
PEERS_DEFAULT = ("https://vincewin-crest-fleet-runner.hf.space,"
                 "https://vincewin-crest-fleet-runner2.hf.space,"
                 "https://vincewin-crest-fleet-runner3.hf.space")


def clone_sources():
    """The one boot step that precedes the shared runtime: the Dockerfile
    copies only this file, so hf_data/spaceruntime.py (boot, keep-awake,
    status page, child env) exists only after this clone."""
    shutil.rmtree(SRC, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "1", CREST_AI_GIT, SRC], check=True)
    sys.path.insert(0, SRC)


def purge_tar_cache():
    """The CONUS month-tars are huge and single-use here — drop old hub-cache
    blobs so the Space's ephemeral disk never fills mid-pass. The basic/
    terrain COGs are kept: every gauge clips from those local copies (windowed
    /vsicurl reads get rate-limited fleet-wide)."""
    root = os.path.join(CACHE, "hub")
    keep = {os.path.realpath(p) for p in glob.glob(
        os.path.join(root, "datasets--*", "snapshots", "*", "basic", "*"))}
    now = time.time()
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            try:
                if os.path.realpath(p) in keep:
                    continue
                if os.path.getsize(p) > 200e6 and now - os.path.getmtime(p) > 2 * 3600:
                    os.remove(p)
            except OSError:
                pass


def fleet_loop():
    if not os.environ.get("HF_TOKEN"):
        state["phase"] = "NO HF_TOKEN — add the secret in Space settings"
        return
    from hf_data import spaceruntime as rt
    env = rt.child_env(CACHE, forcing_cache_gb="25")
    args = ["python3", "fleet/fleet_run.py",
            "--workers", os.environ.get("FLEET_WORKERS", "2"),
            "--gauges", os.environ.get("FLEET_GAUGES", "all"),
            "--start", os.environ.get("FLEET_START", "2021-07-01"),
            "--end", os.environ.get("FLEET_END", "2026-06-30")]
    if os.environ.get("FLEET_ORDER", "forward") == "reverse":
        args.append("--reverse")
    if os.environ.get("FLEET_SHARD"):              # "K/N" — multi-Space split
        args += ["--shard", os.environ["FLEET_SHARD"]]
    while True:
        state["phase"] = "running"
        state["passes"] += 1
        with open(LOG, "a") as lf:
            lf.write(f"\n===== fleet pass {state['passes']} @ "
                     f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} =====\n")
            lf.flush()
            p = subprocess.Popen(args, cwd=SRC, env=env, stdout=lf,
                                 stderr=subprocess.STDOUT)
            while p.poll() is None:
                time.sleep(1800)
                purge_tar_cache()
        state["phase"] = f"pass {state['passes']} ended (rc={p.returncode}) — next in 1 h"
        time.sleep(3600)


def fleet_summary():
    """This container's gauge tally, from the fleet's progress ledger."""
    ok = fail = 0
    try:
        for ln in open(os.path.join(SRC, "fleet_progress.jsonl")):
            try:
                if json.loads(ln).get("ok"):
                    ok += 1
                else:
                    fail += 1
            except Exception:
                pass
    except OSError:
        pass
    return f"this container: {ok} gauges ok, {fail} failed"


if __name__ == "__main__":
    clone_sources()
    from hf_data import spaceruntime as rt
    rt.boot_sources(SRC)
    rt.start_thread(fleet_loop)
    rt.KeepAwake(rt.peers_from_env("FLEET_PEERS", PEERS_DEFAULT),
                 state=state, first_delay_s=60).start()
    rt.serve_status("CREST_fleet_runner", state, log_path=LOG, tail=60,
                    summary=fleet_summary)
