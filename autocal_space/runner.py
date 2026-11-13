"""CREST_autocal — boots the auto-calibration loop on this Space, forever.

Boot: clone the current CREST_AI sources (same calibrate/pipeline code as the
dashboard, no redeploys here), link the EF5 binary, start
autocal/autocal_run.py as a subprocess (restarted after a pause if it dies),
and serve a status page on 7860. Secrets: HF_TOKEN (write), OPENAI_API_KEY.
"""
import json
import os
import shutil
import subprocess
import sys
import time

SRC = "/app/src"
LOG = "/tmp/autocal.log"
CACHE = "/tmp/crest_cache"
STATE = "/tmp/autocal_state.json"
CREST_AI_GIT = "https://github.com/mchen15ouedu/CREST_AI.git"
state = {"phase": "booting", "restarts": 0}

# free Spaces sleep without HTTP traffic; this Space pings the fleet ring so
# nobody dozes off, and keeps ITSELF awake with the shared runtime's
# authenticated self-ping (the ring's anonymous pings cannot reach a private
# Space: no calibration from ~09-25 until e3a17b8)
PEERS_DEFAULT = ("https://vincewin-crest-fleet-runner.hf.space,"
                 "https://vincewin-crest-updater.hf.space")
# one authenticated self-visit per 6 h: 8 tries inside the 48 h sleep window
SELF_PING_S = float(os.environ.get("AUTOCAL_SELF_PING_S", "21600"))


def clone_sources():
    """The one boot step that precedes the shared runtime: the Dockerfile
    copies only this file, so hf_data/spaceruntime.py (boot, keep-awake,
    status page, child env) exists only after this clone."""
    shutil.rmtree(SRC, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "1", CREST_AI_GIT, SRC], check=True)
    sys.path.insert(0, SRC)


def autocal_loop():
    from hf_data import spaceruntime as rt
    env = rt.child_env(CACHE, forcing_cache_gb="20", AUTOCAL_STATE=STATE)
    while True:
        state["phase"] = "running"
        with open(LOG, "a") as lf:
            lf.write(f"\n===== autocal start #{state['restarts'] + 1} @ "
                     f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} =====\n")
            lf.flush()
            p = subprocess.Popen(["python3", "autocal/autocal_run.py"], cwd=SRC,
                                 env=env, stdout=lf, stderr=subprocess.STDOUT)
            p.wait()
        state["restarts"] += 1
        state["phase"] = f"loop exited (rc={p.returncode}) — restart in 10 min"
        time.sleep(600)


def _loop_state() -> dict:
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:
        return {}


def autocal_summary():
    st = _loop_state()
    return (f"loop: {st.get('phase', '?')} | calibrations run {st.get('runs', 0)}, "
            f"skipped {st.get('skips', 0)} | current {st.get('current')}")


if __name__ == "__main__":
    clone_sources()
    from hf_data import spaceruntime as rt
    rt.boot_sources(SRC)
    rt.start_thread(autocal_loop)
    rt.KeepAwake(rt.peers_from_env("AUTOCAL_PEERS", PEERS_DEFAULT),
                 self_ping_every=round(SELF_PING_S / rt.PEER_INTERVAL_S),
                 state=state, first_delay_s=60).start()
    rt.serve_status("CREST_autocal", state, log_path=LOG, tail=80,
                    summary=autocal_summary,
                    api_extra=lambda: {"autocal": _loop_state()})
