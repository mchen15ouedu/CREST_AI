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
Secret: HF_TOKEN (write access — uploads to vincewin/CREST_fleet).
"""
import http.server
import json
import os
import shutil
import subprocess
import threading
import time

SRC = "/app/src"
LOG = "/tmp/fleet.log"
CACHE = "/tmp/crest_cache"
started = time.time()
state = {"phase": "booting", "passes": 0}


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, check=True, **kw)


def boot_sources():
    shutil.rmtree(SRC, ignore_errors=True)
    sh(f"git clone --depth 1 https://github.com/mchen15ouedu/CREST_AI.git {SRC}")
    if not os.path.exists(os.path.join(SRC, "EF5")):
        os.symlink("/EF5", os.path.join(SRC, "EF5"))


def purge_tar_cache():
    """The CONUS month-tars are huge and single-use here — drop old hub-cache
    blobs so the Space's ephemeral disk never fills mid-pass."""
    root = os.path.join(CACHE, "hub")
    now = time.time()
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            try:
                if os.path.getsize(p) > 200e6 and now - os.path.getmtime(p) > 2 * 3600:
                    os.remove(p)
            except OSError:
                pass


def fleet_loop():
    if not os.environ.get("HF_TOKEN"):
        state["phase"] = "NO HF_TOKEN — add the secret in Space settings"
        return
    env = dict(os.environ,
               CREST_DEMO_MOCK="0",
               CREST_CACHE_DIR=CACHE,
               CREST_FORCING_CACHE_GB=os.environ.get("CREST_FORCING_CACHE_GB", "25"),
               HF_HOME=os.path.join(CACHE, "hub"),
               PYTHONUNBUFFERED="1")
    args = ["python3", "fleet/fleet_run.py",
            "--workers", os.environ.get("FLEET_WORKERS", "2"),
            "--gauges", os.environ.get("FLEET_GAUGES", "all"),
            "--start", os.environ.get("FLEET_START", "2021-07-01"),
            "--end", os.environ.get("FLEET_END", "2026-06-30")]
    if os.environ.get("FLEET_ORDER", "forward") == "reverse":
        args.append("--reverse")
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


class Status(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            with open(LOG) as f:
                tail = f.readlines()[-60:]
        except OSError:
            tail = ["(no log yet)"]
        prog = os.path.join(SRC, "fleet_progress.jsonl")
        ok = fail = 0
        try:
            for ln in open(prog):
                try:
                    ok += 1 if json.loads(ln).get("ok") else 0
                    fail += 0 if json.loads(ln).get("ok") else 1
                except Exception:
                    pass
        except OSError:
            pass
        du = shutil.disk_usage("/tmp")
        body = (f"CREST_fleet_runner — {state['phase']}\n"
                f"up {(time.time() - started) / 3600:.1f} h | this container: "
                f"{ok} gauges ok, {fail} failed | disk /tmp "
                f"{du.used / 1e9:.0f}/{du.total / 1e9:.0f} GB\n"
                + "=" * 70 + "\n" + "".join(tail))
        data = body.encode("utf-8", "replace")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):                     # keep server chatter out of stdout
        pass


if __name__ == "__main__":
    boot_sources()
    threading.Thread(target=fleet_loop, daemon=True).start()
    http.server.ThreadingHTTPServer(("0.0.0.0", 7860), Status).serve_forever()
