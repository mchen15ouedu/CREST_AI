"""CREST_ungauged — keep-warm Space for the 2,676 ungauged routed nowcasts.

Every hour, for each HydroBASINS ungauged point, this advances the routed
nowcast by ONE new hour from a warm state (never a cold 17-day run) and writes
the whole set to one parquet the dashboard serves instantly:

  per point  ->  hf_data.routednow.compute(vp, t0)
                    Phase A: cached hindcast to t0 (advances + saves state)
                    Phase B: 12-h forecast warm-started from the t0 state
  all points ->  ungaugednow_store.build_table + upload_latest
                    -> nowcast/ungauged_latest.parquet in CREST_data

State bookkeeping (matches the gauge fleet exactly, so the two coexist in one
long-term store):
  * on cold boot each point refetches its last 10-day checkpoint from
    CREST_fleet (routednow.compute -> fleetstore.ensure_local), so a Space
    restart short-warms instead of cold-starting;
  * routednow.compute -> statecache.prune_states keeps the newest state (next
    hour's warm start) + one checkpoint per 10 days locally, deletes the rest;
  * when a NEW 10-day checkpoint is minted, this uploads the point's state
    bundle + record to vincewin/CREST_fleet as states/<key>.pqf +
    results/<key>.json (key "V..._crestphys-spd") — right beside the gauge keys.

t0 is the issue time of the current gauged DI-LSTM nowcast (nowcaststore), so
the upstream cut-gauge injection has nowcast values to route downstream. A pass
runs only once per issue time.

Space variables (Settings -> Variables):
  UNGAUGED_WORKERS   parallel points (default 8 — sized for cpu-upgrade)
  UNGAUGED_SHARD     "K/N": run only points with catalog-index %% N == K, so N
                     sibling Spaces split the 2,676 disjointly
  UNGAUGED_LIMIT     cap points (0 = all) — for smoke tests
Secret: HF_TOKEN (write access — uploads to CREST_data and CREST_fleet).
"""
import http.server
import json
import os
import shutil
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

SRC = "/app/src"
CACHE = "/tmp/crest_cache"
LOG = "/tmp/ungauged.log"
REPO = os.environ.get("CREST_FEEDBACK_REPO", "vincewin/CREST_data")   # nowcast parquet + no-upstream list
CONUS = (-125.5, 24.0, -66.5, 50.0)          # (w, s, e, n)
started = time.time()
state = {"phase": "booting", "passes": 0, "t0": None, "ok": 0, "fail": 0,
         "uploaded": 0, "last_min": 0.0}

PEERS = [u.strip() for u in os.environ.get("UNGAUGED_PEERS", "").split(",") if u.strip()]
SELF_HOST = (os.environ.get("SPACE_ID", "").replace("/", "-").replace("_", "-")
             .lower() + ".hf.space")


def _log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def keepalive_loop():
    while True:
        for u in PEERS:
            if urllib.parse.urlparse(u).hostname == SELF_HOST:
                continue
            try:
                urllib.request.urlopen(u, timeout=20).read(100)
            except Exception:
                pass
        time.sleep(1200)


def sh(cmd):
    import subprocess
    subprocess.run(cmd, shell=True, check=True)


def boot_sources():
    shutil.rmtree(SRC, ignore_errors=True)
    sh(f"git clone --depth 1 https://github.com/mchen15ouedu/CREST_AI.git {SRC}")
    if not os.path.exists(os.path.join(SRC, "EF5")):
        os.symlink("/EF5", os.path.join(SRC, "EF5"))


# --------------------------------------------------------------------------- #
# per-point work (child process)
# --------------------------------------------------------------------------- #
def _checkpoints(vp, model) -> set:
    """Non-head state times = the 10-day checkpoints (all but the newest)."""
    from hf_data import statecache
    rec = statecache.load_record(vp, model) or {}
    st = sorted(rec.get("state_times", []))
    return set(st[:-1])                        # drop the newest (warm-start head)


def run_one(vp: str, t0_iso: str, force_upload: bool = False) -> dict:
    """Advance one ungauged point and, when a new checkpoint appears (or the
    parent flags it as not-yet-in-fleet), persist it to CREST_fleet. Returns the
    compact row for the nowcast parquet."""
    from hf_data import routednow
    t0 = datetime.fromisoformat(t0_iso)
    model = None
    try:
        from hf_data import virtualpoints
        info = virtualpoints.info(vp)
        model = routednow._cache_model(info["lon"]) if info else None
    except Exception:
        info = None

    before = _checkpoints(vp, model) if model else set()

    r = routednow.compute(vp, t0)              # ensure_local (cached) + both phases + prune
    if not r.get("ok"):
        return {"vp": vp, "ok": False, "reason": r.get("reason")}

    model = r.get("cache_model") or model
    key = f"{str(vp).zfill(8)}_{model}" if model else None
    injected = bool(r.get("injected"))          # truncated speed domain -> real routed nowcast
    has_upstream = bool(r.get("has_upstream"))  # any upstream USGS gauge exists at all
    after = _checkpoints(vp, model) if model else set()
    # only persist checkpoints for injected (nowcast-relevant) runs; a transient
    # full-basin hour warm-starts from local disk and needs no fleet copy
    new_ckpt = injected and (bool(after - before) or force_upload)
    uploaded = False
    if new_ckpt and model and os.environ.get("HF_TOKEN"):
        try:
            uploaded = _upload_fleet(vp, model)
        except Exception as e:
            _safe_note(vp, f"fleet upload failed: {type(e).__name__}: {e}")

    return {"vp": vp, "ok": True, "uploaded": uploaded, "key": key,
            "injected": injected, "has_upstream": has_upstream,
            "lat": r.get("lat"), "lon": r.get("lon"), "area_km2": r.get("area_km2"),
            "q": r.get("q"), "history": r.get("history")}


def _upload_fleet(vp, model, attempts: int = 6) -> bool:
    """Pack the (pruned) state dir + record and commit to CREST_fleet, keeping
    the local copies for next hour's warm start.

    Many workers commit to the same repo ref, so a racing commit loses the
    parent-oid check (HTTP 412 / "reference update failed"). Retry with a
    pid-jittered backoff so the burst serializes instead of dropping states."""
    import random
    import time as _t
    from hf_data import statebundle, statecache, fleetstore
    from huggingface_hub import CommitOperationAdd, HfApi
    key = f"{str(vp).zfill(8)}_{model}"
    sdir = statecache.state_dir(vp, model)
    blob = statebundle.pack_dir(sdir)
    if blob is None:
        return False
    rec_path = statecache.results_path(vp, model)
    ops = [CommitOperationAdd(f"states/{key}.pqf", blob)]
    if os.path.exists(rec_path):
        ops.append(CommitOperationAdd(f"results/{key}.json", rec_path))
    api = HfApi(token=os.environ["HF_TOKEN"])
    for i in range(attempts):
        try:
            api.create_commit(repo_id=fleetstore.REPO, repo_type="dataset",
                              operations=ops,
                              commit_message=f"ungauged checkpoint {key}")
            return True
        except Exception:
            if i == attempts - 1:
                raise
            _t.sleep(min(30.0, 1.5 ** i + random.uniform(0, 1.5)))
    return False


def _safe_note(vp, msg):
    try:
        _log(f"  [{vp}] {msg}")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# hourly pass (parent)
# --------------------------------------------------------------------------- #
def _all_points() -> list[dict]:
    from hf_data import virtualpoints
    w, s, e, n = CONUS
    pts = virtualpoints.for_bbox(w, s, e, n, limit=100000)
    shard = os.environ.get("UNGAUGED_SHARD", "")
    if shard:
        k, nn = map(int, shard.split("/"))
        pts = [p for i, p in enumerate(pts) if i % nn == k]
    pts = [p for p in pts if p["id"] not in _ineligible]   # skip known headwaters
    lim = int(os.environ.get("UNGAUGED_LIMIT", "0") or 0)
    return pts[:lim] if lim else pts


_uploaded_keys: set = set()          # fleet keys known-present (seeded at boot)


def _seed_uploaded_keys():
    """List the ungauged checkpoints already in CREST_fleet so we don't
    re-upload them, and so any point NOT listed is (re)attempted every pass
    until it lands — self-healing past commit-collision drops."""
    try:
        from hf_data import fleetstore
        from huggingface_hub import HfApi
        files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
            fleetstore.REPO, repo_type="dataset")
        for f in files:
            if f.startswith("states/V") and f.endswith(".pqf"):
                _uploaded_keys.add(f[len("states/"):-len(".pqf")])
        _log(f"seeded {len(_uploaded_keys)} existing ungauged checkpoints from CREST_fleet")
    except Exception as e:
        _log(f"could not seed fleet keys: {type(e).__name__}: {e}")


_ineligible: set = set()             # ungauged ids with NO upstream gauge (headwaters)


def _shard_tag() -> str:
    sh = os.environ.get("UNGAUGED_SHARD", "")
    return ("__" + sh.replace("/", "of")) if sh else ""    # "0/3" -> "__0of3"


def _no_upstream_path() -> str:
    return f"nowcast/ungauged_no_upstream{_shard_tag()}.json"


def _seed_ineligible():
    """Load this shard's known headwater (no-upstream) ids from CREST_data so
    they're skipped from the start — the routed-injection design can't serve
    them, so they're nowcast-ineligible (they remain valid on the hindcast
    side, which is served on demand, not here)."""
    try:
        import json
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO, _no_upstream_path(), repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"))
        _ineligible.update(json.load(open(p)))
        _log(f"seeded {len(_ineligible)} nowcast-ineligible (no-upstream) points")
    except Exception:
        _log("no prior no-upstream list (first run)")


def _save_ineligible():
    try:
        import io
        import json
        from huggingface_hub import CommitOperationAdd, HfApi
        buf = io.BytesIO(json.dumps(sorted(_ineligible)).encode())
        HfApi(token=os.environ["HF_TOKEN"]).create_commit(
            repo_id=REPO, repo_type="dataset",
            operations=[CommitOperationAdd(_no_upstream_path(), buf)],
            commit_message=f"no-upstream list {_shard_tag() or 'all'} ({len(_ineligible)})")
    except Exception as e:
        _log(f"could not save no-upstream list: {type(e).__name__}: {e}")


def _keys_for(vp, lon):
    """Both candidate fleet keys for a point — the truncated speed domain
    ("<model>-spd") and the full-basin fallback ("<model>"). We don't know which
    a point uses until it runs, so a point is "already checkpointed" if EITHER
    is in the ledger."""
    from hf_data import routednow
    base = routednow._cache_model(lon)[:-4]         # strip "-spd"
    g = str(vp).zfill(8)
    return f"{g}_{base}-spd", f"{g}_{base}"


def _current_t0():
    """Issue time of the gauged nowcast (so upstream injection has predictions),
    or the current hour if the gauged store isn't reachable."""
    from hf_data import nowcaststore
    t0 = nowcaststore.issue_t0()
    if t0 is None:
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        return now.replace(tzinfo=None)
    return t0.replace(tzinfo=None) if t0.tzinfo else t0


def run_pass(t0):
    from hf_data import ungaugednow_store
    pts = _all_points()
    workers = int(os.environ.get("UNGAUGED_WORKERS", "8"))
    _log(f"pass {state['passes'] + 1}: {len(pts)} points @ t0={t0} "
         f"| {workers} workers | {len(_uploaded_keys)} checkpoints already in fleet")
    t_start = time.time()
    rows, ok, fail, up, new_headwater = [], 0, 0, 0, 0
    t0_iso = t0.isoformat()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {}
        for p in pts:
            k_spd, k_base = _keys_for(p["id"], p["lon"])
            force = k_spd not in _uploaded_keys and k_base not in _uploaded_keys
            futs[pool.submit(run_one, p["id"], t0_iso, force)] = p["id"]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                r = {"vp": futs[fu], "ok": False, "reason": f"{type(e).__name__}: {e}"}
            if r.get("ok"):
                ok += 1
                if r.get("uploaded"):
                    up += 1
                    if r.get("key"):
                        _uploaded_keys.add(r["key"])
                if r.get("injected"):
                    rows.append(r)                  # nowcast parquet: routed points only
                if not r.get("has_upstream") and r["vp"] not in _ineligible:
                    _ineligible.add(r["vp"])         # headwater: drop from nowcast for good
                    new_headwater += 1
            else:
                fail += 1
            if i % 200 == 0:
                _log(f"  {i}/{len(pts)} done ({ok} ok, {fail} fail, {up} up, "
                     f"{len(rows)} routed, {new_headwater} new headwater)")

    if rows:
        try:
            tbl = ungaugednow_store.build_table(rows, t0)
            ungaugednow_store.upload_latest(tbl, tag=_shard_tag())
            _log(f"  uploaded ungauged_latest{_shard_tag()}.parquet ({len(rows)} routed points)")
        except Exception as e:
            _log(f"  PARQUET UPLOAD FAILED: {type(e).__name__}: {e}")
    if new_headwater:
        _save_ineligible()
        _log(f"  {new_headwater} new headwater points -> no-upstream list "
             f"({len(_ineligible)} total, skipped next pass)")

    mins = (time.time() - t_start) / 60
    state.update(passes=state["passes"] + 1, ok=ok, fail=fail, uploaded=up,
                 last_min=round(mins, 1), t0=t0.strftime("%Y-%m-%d %H:%M"))
    _log(f"pass done: {ok} ok, {fail} fail, {up} checkpoints, {len(rows)} routed nowcasts, "
         f"{len(_ineligible)} headwaters skipped, {mins:.1f} min")


def warm_loop():
    if not os.environ.get("HF_TOKEN"):
        state["phase"] = "NO HF_TOKEN — add the secret in Space settings"
        _log(state["phase"])
        return
    os.environ.update(CREST_DEMO_MOCK="0", CREST_CACHE_DIR=CACHE,
                      HF_HOME=os.path.join(CACHE, "hub"),
                      GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2",
                      PYTHONUNBUFFERED="1")
    os.chdir(SRC)                              # EF5 runs relative to ./EF5/bin/ef5
    import sys
    sys.path.insert(0, SRC)
    _seed_uploaded_keys()                       # so we don't re-upload what's already there
    _seed_ineligible()                          # skip known headwaters from the start
    last_t0 = None
    while True:
        try:
            t0 = _current_t0()
            if t0 == last_t0:
                state["phase"] = f"idle — waiting for a new issue time (last {t0})"
                time.sleep(300)
                continue
            state["phase"] = "running"
            run_pass(t0)
            last_t0 = t0
        except Exception as e:
            _log(f"PASS ERROR: {type(e).__name__}: {e}")
            time.sleep(120)
        state["phase"] = "sleeping until next issue time"
        time.sleep(300)


# --------------------------------------------------------------------------- #
# status page
# --------------------------------------------------------------------------- #
class Status(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            with open(LOG) as f:
                tail = f.readlines()[-60:]
        except OSError:
            tail = ["(no log yet)\n"]
        du = shutil.disk_usage("/tmp")
        body = (f"CREST_ungauged (keep-warm) — {state['phase']}\n"
                f"up {(time.time() - started) / 3600:.1f} h | pass {state['passes']} "
                f"@ t0={state['t0']} | {state['ok']} ok, {state['fail']} fail, "
                f"{state['uploaded']} checkpoints | last pass {state['last_min']} min | "
                f"disk /tmp {du.used / 1e9:.0f}/{du.total / 1e9:.0f} GB\n"
                + "=" * 72 + "\n" + "".join(tail))
        data = body.encode("utf-8", "replace")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    boot_sources()
    threading.Thread(target=warm_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    http.server.ThreadingHTTPServer(("0.0.0.0", 7860), Status).serve_forever()
