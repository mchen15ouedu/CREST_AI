"""CREST_autocal — boots the auto-calibration loop on this Space, forever.

Boot: clone the current CREST_AI sources (same calibrate/pipeline code as the
dashboard, no redeploys here), link the EF5 binary, start
autocal/autocal_run.py as a subprocess (restarted after a pause if it dies),
and serve a status page on 7860. Secrets: HF_TOKEN (write), OPENAI_API_KEY.
"""
import http.server
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request

SRC = "/app/src"
LOG = "/tmp/autocal.log"
CACHE = "/tmp/crest_cache"
STATE = "/tmp/autocal_state.json"
started = time.time()
state = {"phase": "booting", "restarts": 0}

# free Spaces sleep without HTTP traffic; the fleet ring pings this Space
# (FLEET_PEERS) and this Space pings the ring back so nobody dozes off
PEERS = [u.strip() for u in os.environ.get("AUTOCAL_PEERS",
    "https://vincewin-crest-fleet-runner.hf.space,"
    "https://vincewin-crest-updater.hf.space").split(",") if u.strip()]


def keepalive_loop():
    while True:
        for u in PEERS:
            try:
                urllib.request.urlopen(u, timeout=20).read(100)
            except Exception:
                pass
        time.sleep(1200)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, check=True, **kw)


def boot_sources():
    shutil.rmtree(SRC, ignore_errors=True)
    sh(f"git clone --depth 1 https://github.com/mchen15ouedu/CREST_AI.git {SRC}")
    if not os.path.exists(os.path.join(SRC, "EF5")):
        os.symlink("/EF5", os.path.join(SRC, "EF5"))


def autocal_loop():
    env = dict(os.environ,
               CREST_DEMO_MOCK="0",
               CREST_CACHE_DIR=CACHE,
               CREST_FORCING_CACHE_GB=os.environ.get("CREST_FORCING_CACHE_GB", "20"),
               HF_HOME=os.path.join(CACHE, "hub"),
               AUTOCAL_STATE=STATE,
               GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2",
               PYTHONUNBUFFERED="1")
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


class Status(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            with open(LOG) as f:
                tail = f.readlines()[-80:]
        except OSError:
            tail = ["(no log yet)"]
        try:
            with open(STATE) as f:
                st = json.load(f)
        except Exception:
            st = {}
        if self.path.startswith("/api/status"):
            data = json.dumps({"runner": state, "autocal": st,
                               "up_h": round((time.time() - started) / 3600, 2)}).encode()
            ctype = "application/json"
        else:
            du = shutil.disk_usage("/tmp")
            body = (f"CREST_autocal — {state['phase']} | loop: {st.get('phase', '?')}\n"
                    f"up {(time.time() - started) / 3600:.1f} h | calibrations run "
                    f"{st.get('runs', 0)}, skipped {st.get('skips', 0)} | current "
                    f"{st.get('current')} | disk /tmp {du.used / 1e9:.0f}/{du.total / 1e9:.0f} GB\n"
                    + "=" * 70 + "\n" + "".join(tail))
            data = body.encode("utf-8", "replace")
            ctype = "text/plain; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    boot_sources()
    threading.Thread(target=autocal_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    http.server.ThreadingHTTPServer(("0.0.0.0", 7860), Status).serve_forever()
