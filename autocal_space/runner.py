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
import urllib.error
import urllib.request

SRC = "/app/src"
LOG = "/tmp/autocal.log"
CACHE = "/tmp/crest_cache"
STATE = "/tmp/autocal_state.json"
started = time.time()
state = {"phase": "booting", "restarts": 0}

# free Spaces sleep without HTTP traffic; this Space pings the fleet ring so
# nobody dozes off, and keeps ITSELF awake with an authenticated self-ping
# (see self_ping: the ring's anonymous pings cannot reach a private Space)
PEERS = [u.strip() for u in os.environ.get("AUTOCAL_PEERS",
    "https://vincewin-crest-fleet-runner.hf.space,"
    "https://vincewin-crest-updater.hf.space").split(",") if u.strip()]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None        # a 3xx raises HTTPError: the token never leaves this host


_OPENER = urllib.request.build_opener(_NoRedirect)
# one authenticated self-visit per 6 h: 8 tries inside the 48 h sleep window,
# far from keep-alive-spam territory (the 20-min ring loop runs it every Nth pass)
SELF_PING_EVERY = max(1, round(float(os.environ.get("AUTOCAL_SELF_PING_S", "21600")) / 1200))


def _own_host():
    host = os.environ.get("SPACE_HOST", "").strip()
    sid = os.environ.get("SPACE_ID", "")
    if not host and "/" in sid:
        host = sid.replace("/", "-").replace("_", "-").replace(".", "-").lower() + ".hf.space"
    return host


def self_ping():
    """GET this Space's own status page through the public URL WITH the
    HF_TOKEN secret. This Space is private: HF's proxy turns away anonymous
    requests before they reach the container, so the fleet ring's plain
    pings never counted as traffic and it slept from ~10-15 (no calibration
    since 09-25). An authenticated visit is what the 48 h sleep timer sees.
    Each attempt prints one line (never the token) to the run log, which the
    Hub API reads without visiting — so a later sleep can be diagnosed."""
    host, tok = _own_host(), os.environ.get("HF_TOKEN", "").strip()
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not host or not tok:
        rec = {"ok": False, "at": at, "why": "no SPACE_HOST/HF_TOKEN"}
    else:
        req = urllib.request.Request(f"https://{host}/api/status",
                                     headers={"Authorization": f"Bearer {tok}"})
        try:
            with _OPENER.open(req, timeout=30) as r:
                body = r.read(200)
                rec = {"ok": 200 <= r.status < 300 and body.lstrip().startswith(b"{"),
                       "at": at, "http": r.status}
        except urllib.error.HTTPError as e:    # 401/403/404 = token or scope
            rec = {"ok": False, "at": at, "err": "HTTPError", "http": e.code}
        except Exception as e:                 # never the token: type only
            rec = {"ok": False, "at": at, "err": type(e).__name__}
    state["self_ping"] = rec
    print("self_ping " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)


def keepalive_loop():
    time.sleep(60)                             # let the status server start
    n = 0
    while True:
        if n % SELF_PING_EVERY == 0:
            self_ping()
        n += 1
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
