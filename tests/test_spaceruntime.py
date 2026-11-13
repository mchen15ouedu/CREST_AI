"""hf_data.spaceruntime: the shared startup module of the deployable Spaces
(fleet, autocal, ungauged runners and the GPU worker). Offline: HTTP goes
through a fake opener or a localhost server."""
from __future__ import annotations

import http.server
import io
import json
import os
import sys
import threading
import urllib.error
import urllib.request

import pytest

from hf_data import spaceruntime as rt

TOKEN = "hf_SECRETtoken123"


class FakeResp:
    def __init__(self, status=200, body=b'{"ok": true}'):
        self.status, self._body = status, body

    def read(self, n=-1):
        return self._body if n < 0 else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    """Records every request; answers from a {url: response-or-exception} map."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.requests = []

    def open(self, req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        headers = {} if isinstance(req, str) else dict(req.header_items())
        self.requests.append((url, headers))
        a = self.answers.get(url, FakeResp())
        if isinstance(a, Exception):
            raise a
        return a


def make(peers=(), self_host="me-space.hf.space", token=TOKEN, every=1, **kw):
    opener = FakeOpener(kw.pop("answers", None))
    logs = []
    ka = rt.KeepAwake(peers=list(peers), self_host=self_host, token=token,
                      self_ping_every=every, opener=opener, log=logs.append, **kw)
    return ka, opener, logs


def test_self_ping_is_authenticated_get_of_own_status_page():
    ka, opener, _ = make()
    ka.tick(0)
    (url, headers), = opener.requests
    assert url == "https://me-space.hf.space/api/status"
    assert headers.get("Authorization") == f"Bearer {TOKEN}"
    assert ka.last_self_ping["ok"] is True
    assert ka.last_self_ping["http"] == 200


def test_self_ping_result_is_published_into_the_space_state():
    st = {"phase": "running"}
    ka, _, _ = make(state=st)
    ka.tick(0)
    assert st["self_ping"]["ok"] is True


def test_peer_ring_is_anonymous_and_skips_this_space():
    peers = ["https://a.hf.space", "https://me-space.hf.space", "https://b.hf.space"]
    ka, opener, _ = make(peers=peers, every=10)
    ka.tick(1)                                     # not a self-ping pass
    assert [u for u, _ in opener.requests] == ["https://a.hf.space", "https://b.hf.space"]
    assert all("Authorization" not in h for _, h in opener.requests)


def test_a_sleeping_peer_does_not_stop_the_ring():
    peers = ["https://a.hf.space", "https://b.hf.space"]
    ka, opener, _ = make(peers=peers, every=10,
                         answers={"https://a.hf.space": TimeoutError()})
    ka.tick(1)
    assert [u for u, _ in opener.requests] == peers


def test_self_ping_runs_on_every_nth_pass_only():
    ka, opener, _ = make(peers=["https://a.hf.space"], every=3)
    for n in range(7):
        ka.tick(n)
    selfs = [u for u, _ in opener.requests if "me-space" in u]
    assert len(selfs) == 3                         # passes 0, 3, 6
    assert len(opener.requests) == 7 + 3


def test_default_cadence_is_one_self_ping_per_six_hours_of_ring_passes():
    ka, _, _ = make(every=None)
    assert ka.self_ping_every == 18                # 6 h / 20 min


def test_rejected_self_ping_is_recorded_and_logged_without_the_token():
    url = "https://me-space.hf.space/api/status"
    err = urllib.error.HTTPError(url, 401, "Unauthorized", {}, io.BytesIO(b""))
    ka, _, logs = make(answers={url: err})
    rec = ka.self_ping()
    assert rec["ok"] is False and rec["http"] == 401
    assert len(logs) == 1 and logs[0].startswith("self_ping ")
    assert TOKEN not in logs[0] and TOKEN not in repr(rec)


def test_a_200_that_is_not_the_status_json_is_not_ok():
    url = "https://me-space.hf.space/api/status"
    ka, _, _ = make(answers={url: FakeResp(200, b"<html>login</html>")})
    assert ka.self_ping()["ok"] is False


def test_html_target_counts_any_2xx_when_json_is_not_expected():
    url = "https://me-space.hf.space/"
    ka, opener, _ = make(self_path="/", expect_json=False,
                         answers={url: FakeResp(200, b"<html>gradio</html>")})
    assert ka.self_ping()["ok"] is True
    assert opener.requests[0][1]["Authorization"] == f"Bearer {TOKEN}"


def test_without_token_no_request_is_made():
    ka, opener, _ = make(token="")
    rec = ka.self_ping()
    assert rec["ok"] is False and opener.requests == []


def test_own_host_from_space_id():
    assert rt.own_host({"SPACE_ID": "vincewin/CREST_fleet.runner"}) == \
        "vincewin-crest-fleet-runner.hf.space"
    assert rt.own_host({"SPACE_HOST": "x.hf.space", "SPACE_ID": "a/b"}) == "x.hf.space"


def test_run_waits_then_ticks_every_interval():
    sleeps = []

    class Stop(Exception):
        pass

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:
            raise Stop

    ka, opener, _ = make(peers=["https://a.hf.space"], every=1,
                         sleep=fake_sleep, first_delay_s=60, interval_s=1200)
    try:
        ka.run()
    except Stop:
        pass
    assert sleeps == [60, 1200, 1200]
    assert len(opener.requests) == 4               # 2 passes x (self + 1 peer)


# --------------------------------------------------------------------------- #
# status server (real localhost server, port 0)
# --------------------------------------------------------------------------- #
@pytest.fixture
def status_server():
    servers = []

    def start(**kw):
        srv = rt.serve_status(port=0, host="127.0.0.1", background=True, **kw)
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_status_page_is_text_with_the_space_header_and_log_tail(status_server, tmp_path):
    log = tmp_path / "run.log"
    log.write_text("".join(f"line {i}\n" for i in range(100)))
    base = status_server(title="CREST_test", state={"phase": "running"},
                         log_path=str(log), tail=5,
                         summary=lambda: "3 gauges ok")
    code, ctype, body = _get(base + "/")
    text = body.decode()
    assert code == 200 and ctype.startswith("text/plain")
    assert text.startswith("CREST_test — running\n")
    assert "3 gauges ok" in text and "disk /tmp" in text
    assert text.rstrip().endswith("line 99") and "line 94" not in text


def test_api_status_is_json_with_runner_state_and_extra_fields(status_server):
    st = {"phase": "booting", "passes": 2}
    base = status_server(title="CREST_test", state=st,
                         api_extra=lambda: {"autocal": {"runs": 4}})
    code, ctype, body = _get(base + "/api/status")
    doc = json.loads(body)
    assert code == 200 and ctype == "application/json"
    assert doc["runner"] == st and doc["autocal"] == {"runs": 4}
    assert isinstance(doc["up_h"], float)


def test_missing_log_still_serves(status_server, tmp_path):
    base = status_server(title="T", state={"phase": "x"},
                         log_path=str(tmp_path / "none.log"))
    assert b"(no log yet)" in _get(base + "/")[2]


def test_self_ping_reaches_status_json_and_counts_ok(status_server):
    base = status_server(title="T", state={"phase": "x"})
    host = base[len("http://"):]

    class LocalOpener:                       # the real no-redirect opener, over http
        def open(self, req, timeout=None):
            req.full_url = req.full_url.replace("https://", "http://")
            return rt.auth_opener().open(req, timeout=timeout)

    ka = rt.KeepAwake(self_host=host, token=TOKEN, opener=LocalOpener(),
                      log=lambda s: None)
    assert ka.self_ping()["ok"] is True


def test_auth_opener_does_not_follow_redirects(tmp_path):
    hits = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append((self.path, self.headers.get("Authorization")))
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/status",
                                     headers={"Authorization": f"Bearer {TOKEN}"})
        with pytest.raises(urllib.error.HTTPError) as ei:
            rt.auth_opener().open(req, timeout=5)
        assert ei.value.code == 302
        assert hits == [("/api/status", f"Bearer {TOKEN}")]    # never re-sent
    finally:
        srv.shutdown()
        srv.server_close()


# --------------------------------------------------------------------------- #
# child-process environment and boot
# --------------------------------------------------------------------------- #
def test_child_env_points_caches_at_the_space_cache(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.delenv("CREST_FORCING_CACHE_GB", raising=False)
    env = rt.child_env("/tmp/crest_cache", forcing_cache_gb="25", AUTOCAL_STATE="/tmp/s")
    assert env["CREST_CACHE_DIR"] == "/tmp/crest_cache"
    assert env["HF_HOME"] == os.path.join("/tmp/crest_cache", "hub")
    assert env["CREST_DEMO_MOCK"] == "0"
    assert env["GDAL_HTTP_MAX_RETRY"] == "5" and env["GDAL_HTTP_RETRY_DELAY"] == "2"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["CREST_FORCING_CACHE_GB"] == "25"
    assert env["AUTOCAL_STATE"] == "/tmp/s"
    assert env["HF_TOKEN"] == "x"                  # secrets inherited


def test_child_env_respects_a_space_variable_forcing_cap(monkeypatch):
    monkeypatch.setenv("CREST_FORCING_CACHE_GB", "40")
    assert rt.child_env("/c", forcing_cache_gb="25")["CREST_FORCING_CACHE_GB"] == "40"


def test_boot_sources_links_ef5_and_puts_sources_on_path(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    links = []

    def fake_symlink(target, path):            # Windows CI lacks symlink rights
        links.append((target, path))
        os.mkdir(path)

    monkeypatch.setattr(sys, "path", list(sys.path))
    rt.boot_sources(str(src), ef5="/EF5", symlink=fake_symlink)
    assert links == [("/EF5", os.path.join(str(src), "EF5"))]
    assert sys.path[0] == str(src)
    rt.boot_sources(str(src), ef5="/EF5", symlink=fake_symlink)   # idempotent
    assert len(links) == 1 and sys.path.count(str(src)) == 1


RUNNERS = ["fleet_space", "autocal_space", "ungauged_space"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("space", RUNNERS)
def test_runner_loads_alone_as_its_dockerfile_deploys_it(space, tmp_path):
    """The Dockerfile COPYs only runner.py: before the clone, the module must
    load with nothing else on disk (ungauged's spawn workers re-import it too),
    and its boot step must be the shared clone command."""
    import shutil
    import subprocess
    shutil.copy(os.path.join(ROOT, space, "runner.py"), tmp_path / "runner.py")
    code = ("import sys; sys.path.insert(0, '.'); import runner; "
            "assert 'hf_data' not in sys.modules; "
            "print(runner.CREST_AI_GIT)")
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                       capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PYTHONPATH": ""})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == rt.CREST_AI_GIT


@pytest.mark.parametrize("space", RUNNERS + ["gpu_worker_space"])
def test_spaces_use_the_shared_runtime_not_copies(space):
    name = "app.py" if space == "gpu_worker_space" else "runner.py"
    src = open(os.path.join(ROOT, space, name), encoding="utf-8").read()
    assert "spaceruntime" in src
    for copied in ("def keepalive_loop", "def _keep_awake", "def self_ping",
                   "def boot_sources", "class Status(", "def _own_host"):
        assert copied not in src, f"{space} still defines {copied}"


def test_clone_command_is_a_shallow_clone_of_crest_ai():
    assert rt.clone_command("/app/src") == [
        "git", "clone", "--depth", "1",
        "https://github.com/mchen15ouedu/CREST_AI.git", "/app/src"]
