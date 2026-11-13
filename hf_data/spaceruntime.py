"""Shared startup runtime for the deployable Spaces (fleet, autocal and
ungauged runners; the GPU worker uses the keep-awake part).

A Space declares only its work loop and its status fields; this module owns
the rest, once:

  * boot_sources  — link the EF5 binary into the cloned CREST_AI sources and
                    put them on sys.path. (The `git clone` itself has to stay
                    in each runner.py: the Dockerfiles only COPY runner.py, so
                    this module does not exist on disk until that clone.)
  * KeepAwake     — the keep-awake policy: an anonymous peer ring (a GET also
                    WAKES a sleeping sibling) plus an authenticated self-ping.
                    HF's proxy turns away anonymous requests to a private
                    Space before they reach the container, so only a visit
                    carrying the Space's own HF_TOKEN counts against the
                    sleep timer (the failure fixed for autocal in e3a17b8).
  * serve_status  — the status-page server: text page on every path, JSON on
                    /api/status (the self-ping's target).
  * child_env     — the environment every Space hands its EF5/pipeline work.

Stdlib only: it is imported right after the clone, before anything else.
Never logs the token.
"""
from __future__ import annotations

import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

CREST_AI_GIT = "https://github.com/mchen15ouedu/CREST_AI.git"
PEER_INTERVAL_S = 1200.0                 # one ring pass every 20 min
SELF_PING_S = 21600.0                    # one authenticated self-visit per 6 h


# --------------------------------------------------------------------------- #
# keep-awake
# --------------------------------------------------------------------------- #
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None        # a 3xx raises HTTPError: the token never leaves this host


def auth_opener():
    """Opener for the authenticated self-ping: refuses redirects, so the
    Authorization header is only ever sent to the Space's own host."""
    return urllib.request.build_opener(_NoRedirect)


def own_host(environ=None) -> str:
    """This Space's public host: SPACE_HOST, else derived from SPACE_ID."""
    env = os.environ if environ is None else environ
    host = env.get("SPACE_HOST", "").strip()
    sid = env.get("SPACE_ID", "")
    if not host and "/" in sid:
        host = (sid.replace("/", "-").replace("_", "-").replace(".", "-").lower()
                + ".hf.space")
    return host


def peers_from_env(name: str, default: str = "") -> list:
    return [u.strip() for u in os.environ.get(name, default).split(",") if u.strip()]


class KeepAwake:
    """Keep-awake policy. Each tick(n) is one pass: an authenticated self-ping
    on every `self_ping_every`-th pass, then an anonymous GET of each peer
    except this Space itself. run() loops tick() every `interval_s`."""

    def __init__(self, peers=(), self_host=None, token=None, *,
                 self_ping_every=None, interval_s=PEER_INTERVAL_S,
                 self_path="/api/status", expect_json=True,
                 opener=None, peer_opener=None, log=None,
                 sleep=time.sleep, first_delay_s=0.0, state=None):
        self.state = state                       # status dict: gets "self_ping"
        self.peers = list(peers)
        self.self_host = own_host() if self_host is None else self_host
        self.token = (os.environ.get("HF_TOKEN", "") if token is None else token).strip()
        self.interval_s = float(interval_s)
        if self_ping_every is None:
            self_ping_every = round(SELF_PING_S / self.interval_s)
        self.self_ping_every = max(1, int(self_ping_every))
        self.self_path = self_path
        self.expect_json = expect_json
        self.opener = opener or auth_opener()
        self.peer_opener = peer_opener or opener or urllib.request.build_opener()
        self.log = log or (lambda line: print(line, flush=True))
        self.sleep = sleep
        self.first_delay_s = first_delay_s
        self.last_self_ping = None

    # -- one pass ----------------------------------------------------------
    def tick(self, n: int):
        if n % self.self_ping_every == 0:
            self.self_ping()
        self.ping_peers()

    def self_ping(self) -> dict:
        at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.self_host or not self.token:
            rec = {"ok": False, "at": at, "why": "no SPACE_HOST/HF_TOKEN"}
        else:
            req = urllib.request.Request(
                f"https://{self.self_host}{self.self_path}",
                headers={"Authorization": f"Bearer {self.token}"})
            try:
                with self.opener.open(req, timeout=30) as r:
                    body = r.read(200)
                    ok = 200 <= r.status < 300
                    if self.expect_json:
                        ok = ok and body.lstrip().startswith(b"{")
                    rec = {"ok": ok, "at": at, "http": r.status}
            except urllib.error.HTTPError as e:    # 3xx/401/403/404 = token or scope
                rec = {"ok": False, "at": at, "err": "HTTPError", "http": e.code}
            except Exception as e:                 # never the token: type only
                rec = {"ok": False, "at": at, "err": type(e).__name__}
        self.last_self_ping = rec
        if self.state is not None:
            self.state["self_ping"] = rec
        self.log("self_ping " + " ".join(f"{k}={v}" for k, v in rec.items()))
        return rec

    def ping_peers(self):
        for u in self.peers:
            if urllib.parse.urlparse(u).hostname == self.self_host:
                continue
            try:
                with self.peer_opener.open(u, timeout=20) as r:
                    r.read(100)
            except Exception:
                pass                 # a sleeping peer is still woken by the request

    # -- forever -----------------------------------------------------------
    def run(self):
        if self.first_delay_s:
            self.sleep(self.first_delay_s)      # let the status server start
        n = 0
        while True:
            self.tick(n)
            n += 1
            self.sleep(self.interval_s)

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, daemon=True, name="keep-awake")
        t.start()
        return t


def start_thread(target, *args) -> threading.Thread:
    """Run a Space's work loop beside the status server (daemon thread)."""
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


# --------------------------------------------------------------------------- #
# boot
# --------------------------------------------------------------------------- #
def boot_sources(src: str, ef5: str = "/EF5", *, symlink=os.symlink):
    """Finish booting freshly cloned sources: link the image's EF5 build to
    <src>/EF5 (pipelines run ./EF5/bin/ef5 relative to the sources) and put the
    sources first on sys.path. Idempotent."""
    link = os.path.join(src, "EF5")
    if not os.path.lexists(link):
        symlink(ef5, link)
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)


# --------------------------------------------------------------------------- #
# child-process environment
# --------------------------------------------------------------------------- #
def child_env(cache: str, forcing_cache_gb: str | None = None, **extra) -> dict:
    """Environment for a Space's pipeline work: the real (non-mock) pipeline,
    every cache under `cache` (HF hub blobs in <cache>/hub), GDAL /vsicurl
    retries on 429/5xx, unbuffered logs. `forcing_cache_gb` is the Space's
    default forcing-cache cap; a CREST_FORCING_CACHE_GB Space variable wins.
    Secrets (HF_TOKEN, ...) are inherited from os.environ."""
    env = dict(os.environ,
               CREST_DEMO_MOCK="0",
               CREST_CACHE_DIR=cache,
               HF_HOME=os.path.join(cache, "hub"),
               GDAL_HTTP_MAX_RETRY="5",
               GDAL_HTTP_RETRY_DELAY="2",
               PYTHONUNBUFFERED="1")
    if forcing_cache_gb is not None:
        env["CREST_FORCING_CACHE_GB"] = os.environ.get("CREST_FORCING_CACHE_GB",
                                                       forcing_cache_gb)
    env.update({k: str(v) for k, v in extra.items()})
    return env


# --------------------------------------------------------------------------- #
# status server
# --------------------------------------------------------------------------- #
def _tail(path, n):
    if not path:
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except OSError:
        return ["(no log yet)\n"]


def _disk(path="/tmp"):
    try:
        du = shutil.disk_usage(path)
        return f"disk {path} {du.used / 1e9:.0f}/{du.total / 1e9:.0f} GB"
    except OSError:
        return f"disk {path} ?"


def status_handler(title, state, *, log_path=None, tail=60, summary=None,
                   api_extra=None, started=None, rule=70):
    """Request handler for a Space status page. GET /api/status -> JSON
    {"runner": state, "up_h": ..., **api_extra()}; any other path -> text:
    "<title> — <phase>", an "up | <summary> | disk" line, and the log tail."""
    t0 = time.time() if started is None else started

    class Status(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            up_h = (time.time() - t0) / 3600
            if self.path.startswith("/api/status"):
                doc = {"runner": state, "up_h": round(up_h, 2)}
                doc.update(api_extra() if api_extra else {})
                data = json.dumps(doc, default=str).encode()
                ctype = "application/json"
            else:
                mid = summary() if summary else ""
                body = (f"{title} — {state.get('phase', '?')}\n"
                        f"up {up_h:.1f} h | " + (f"{mid} | " if mid else "")
                        + _disk() + "\n" + "=" * rule + "\n"
                        + "".join(_tail(log_path, tail)))
                data = body.encode("utf-8", "replace")
                ctype = "text/plain; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):             # keep server chatter out of stdout
            pass

    return Status


def serve_status(title, state, *, port=7860, host="0.0.0.0", background=False,
                 **kw):
    """Serve the status page (see status_handler). Blocks forever unless
    background=True, which returns the running server."""
    srv = http.server.ThreadingHTTPServer((host, port), status_handler(title, state, **kw))
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    srv.serve_forever()
