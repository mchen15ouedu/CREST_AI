"""CREST_gpu_worker — ZeroGPU BACKUP event worker (V27 P1.5 + P1.6 sessions,
degradation-ladder rung 2). Claims events/queue/ jobs the HPC worker has left
unclaimed for ZERO_DELAY_S, solves them in <=120 s GPU slices, and — only
when PUBLISH=1 after the user flips the Space to EVENT_QUEUE_MODE=on —
publishes exactly like the HPC worker.

P1.6: solver state is a resident crestimap.session.EventSession per episode
(depth AND momentum held in RAM between bundles), so returning to an episode
this Space already solved is a minutes-long catch-up visit from the last
obs-supported junction, not an hours-long anchored re-solve. Sessions for
episodes this worker has visited get scheduling priority over new events.

Queue protocol: identical to crestimap/worker.py (claim/heartbeat/stale
takeover/failed.json). 30 m (dem_res "1") jobs only; never coarsens — an
oversized or 10 m job is simply not claimed.

ZeroGPU note: @spaces.GPU functions run in a forked child with the GPU;
arguments/returns are pickled, and parent-side mutations inside the child are
lost. gpu_chunk() therefore RETURNS session.state_tuple() and the parent
reapplies it. Files the child writes (depth frames) land on the shared fs.

Never prints or logs the HF token.
"""
from __future__ import annotations

import datetime
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback

# ---------------------------------------------------------------- bootstrap
APP = os.path.dirname(os.path.abspath(__file__))
FORK = "https://github.com/mchen15ouedu/CREST-iMAP.git"
CREST_AI = "https://huggingface.co/spaces/vincewin/CREST_demo"


def _bootstrap():
    """Blobless sparse clones: crestimap/ from the fork's v2, hf_data/ from
    the CREST_demo Space (eventsim/eventstore reuse — nothing reimplemented)."""
    if not os.path.isdir(os.path.join(APP, "crestimap")):
        d = os.path.join(APP, "_fork")
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "-b", "v2", FORK, d], check=True)
        subprocess.run(["git", "checkout", "HEAD", "--", "crestimap"],
                       cwd=d, check=True)
        shutil.copytree(os.path.join(d, "crestimap"),
                        os.path.join(APP, "crestimap"))
        shutil.rmtree(d, ignore_errors=True)
    ai = os.path.join(APP, "_crest_ai")
    if not os.path.isdir(os.path.join(ai, "hf_data")):
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        CREST_AI, ai], check=True)
        subprocess.run(["git", "checkout", "HEAD", "--", "hf_data"],
                       cwd=ai, check=True)
    if ai not in sys.path:
        sys.path.insert(0, ai)


_bootstrap()

import gradio as gr                                          # noqa: E402

try:
    import spaces                                            # noqa: E402
    HAS_ZERO = True
except Exception:                                            # local dev
    HAS_ZERO = False

TS = "%Y-%m-%dT%H:%M:%SZ"
TMIN = "%Y-%m-%dT%H:%MZ"
QPREFIX = "events/queue"
REPO = os.environ.get("CREST_FEEDBACK_REPO", "vincewin/CREST_data")
ZERO_DELAY_S = float(os.environ.get("ZERO_DELAY_S", "420"))   # HPC first shot
# Backup -> active drain under pressure (user directive 2026-08-14): with
# more than this many jobs pending, skip the polite deferral and claim the
# next unclaimed event immediately — one solver per event, so this Space
# always works a DIFFERENT event than the HPC.
PRESSURE_N = int(os.environ.get("ZERO_PRESSURE_N", "4"))
STALE_S = float(os.environ.get("ZERO_STALE_S", "600"))
HB_S = float(os.environ.get("ZERO_HB_S", "240"))
POLL_S = float(os.environ.get("ZERO_POLL_S", "60"))
MAX_CELLS = int(os.environ.get("ZERO_MAX_CELLS", "2000000"))
PUBLISH = os.environ.get("PUBLISH", "0") == "1"
CHUNK0_SIM_S = float(os.environ.get("ZERO_CHUNK0_SIM_S", "300"))
GPU_TARGET_S = float(os.environ.get("ZERO_GPU_TARGET_S", "80"))
DEM_RES_DEG = {"1": 1.0 / 3600.0, "13": 1.0 / 10800.0}
IDENT = f"zerogpu:{os.environ.get('SPACE_ID', 'local')}:{os.getpid()}"
os.environ.setdefault("EVENT_ENGINE_IDENT", IDENT)   # provenance stamp

_tok = os.environ.get("HF_TOKEN", "").strip()
if _tok:
    os.environ["HF_TOKEN"] = _tok                    # eventstore reads env

MAX_SESSIONS = int(os.environ.get("ZERO_MAX_SESSIONS", "6"))
EVICT_IDLE_H = float(os.environ.get("ZERO_EVICT_IDLE_H", "12"))

LOG: list[str] = []
W = {"phase": "idle", "job": None, "runner": None, "chunk_sim_s": CHUNK0_SIM_S,
     "last_poll": 0.0, "last_hb": 0.0, "queued": "", "workdir": None,
     "done": set(), "solved": [], "t_wall0": 0.0,
     # P1.6 resident sessions: ev -> {"session", "bbox", "visits", "last",
     # "root"}. RAM-only; a Space rebuild recovers via the bundle's
     # init_depth (state-hierarchy level 2).
     "sessions": {}}
_tick_lock = threading.Lock()


def log(msg: str):
    LOG.append(f"[{datetime.datetime.utcnow():%H:%M:%S}Z] {msg}")
    del LOG[:-300]
    print(LOG[-1], flush=True)


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=_tok or None)


def _read_json(path: str):
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(REPO, path, repo_type="dataset", token=_tok or None,
                            force_download=True)
        with open(p, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def _commit(ops, message: str) -> bool:
    try:
        _api().create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                             commit_message=message)
        return True
    except Exception as e:
        log(f"commit failed ({message}): {type(e).__name__}")
        return False


def _commit_claim(ev: str, queued: str) -> bool:
    from huggingface_hub import CommitOperationAdd
    body = json.dumps({"worker": IDENT,
                       "hb": datetime.datetime.utcnow().strftime(TS),
                       "queued": queued})
    return _commit([CommitOperationAdd(f"{QPREFIX}/{ev}.claim",
                                       io.BytesIO(body.encode()))],
                   f"claim {ev} [{IDENT}]")


def _claim_fresh(claim) -> bool:
    if not claim:
        return False
    try:
        hb = datetime.datetime.strptime(claim.get("hb", ""), TS)
    except ValueError:
        return False
    return (datetime.datetime.utcnow() - hb).total_seconds() < STALE_S


def _scan():
    """Oldest BACKUP-eligible job: bundle present, 30 m, inside cell budget,
    unclaimed (or stale), and queued more than ZERO_DELAY_S ago (the HPC gets
    first shot — this Space is ladder rung 2, not a race participant)."""
    try:
        files = _api().list_repo_files(REPO, repo_type="dataset")
    except Exception as e:
        log(f"list_repo_files failed: {type(e).__name__}")
        return None
    qf = [f for f in files if f.startswith(QPREFIX + "/")]
    ids = sorted({os.path.basename(f)[:-5] for f in qf
                  if f.endswith(".json") and not f.endswith(".failed.json")})
    now = datetime.datetime.utcnow()
    # queue pressure: backlog > PRESSURE_N promotes this Space from deferred
    # backup to active drain (claim immediately; claims keep events disjoint)
    backlog = sum(1 for ev in ids if f"{QPREFIX}/{ev}.forcing.tar.gz" in qf)
    delay = 0.0 if backlog > PRESSURE_N else ZERO_DELAY_S
    if backlog > PRESSURE_N and W.get("pressure") != backlog:
        W["pressure"] = backlog
        log(f"queue pressure: {backlog} pending jobs > {PRESSURE_N} — "
            f"claiming without deferral")
    cands = []
    for ev in ids:
        if f"{QPREFIX}/{ev}.forcing.tar.gz" not in qf:
            continue
        spec = _read_json(f"{QPREFIX}/{ev}.json")
        if not spec or (ev, spec.get("queued", "")) in W["done"]:
            continue
        if str(spec.get("dem_res", "1")) != "1":
            continue                              # 30 m only on this rung
        try:
            age = (now - datetime.datetime.strptime(spec["queued"], TS)
                   ).total_seconds()
        except Exception:
            age = 0.0
        if age < delay:
            continue
        if not spec.get("domain"):
            # pre-basin bundle (rectangular-window era): never solved as a
            # rectangle — the runner re-enqueues it with a HUC12 basin domain
            log(f"skip {ev}: no basin domain in spec — awaiting re-enqueue")
            continue
        w, s, e, n = spec["bbox_basin"]
        res = DEM_RES_DEG["1"]
        cells = int((spec.get("domain") or {}).get("n_bbox") or
                    int(round((n - s) / res)) * int(round((e - w) / res)))
        if cells > MAX_CELLS:
            log(f"skip {ev}: {cells / 1e6:.1f} M cells > "
                f"{MAX_CELLS / 1e6:.1f} M budget (never coarsen)")
            continue
        if f"{QPREFIX}/{ev}.claim" in qf:
            claim = _read_json(f"{QPREFIX}/{ev}.claim")
            if _claim_fresh(claim) and claim.get("worker") != IDENT:
                continue
        cands.append({"id": ev, "spec": spec})
    # P1.6: episodes with a resident session first — a catch-up visit is
    # minutes; new events (full anchored solve) fill the remaining time
    cands.sort(key=lambda j: (j["id"] not in W["sessions"],
                              j["spec"].get("queued", "")))
    return cands[0] if cands else None


# ------------------------------------------------------------------ GPU work
def _gpu_chunk_impl(session, sim_s: float, device: str):
    done = session.chunk(sim_s, device=device)
    return done, session.state_tuple()


if HAS_ZERO:
    @spaces.GPU(duration=120)
    def gpu_chunk(session, sim_s: float):
        return _gpu_chunk_impl(session, sim_s, "cuda")
else:
    def gpu_chunk(session, sim_s: float):
        return _gpu_chunk_impl(session, sim_s, "cpu")


# ------------------------------------------------------------------ lifecycle
def _start_job(job):
    from crestimap import EventConfig
    from crestimap.session import EventSession

    ev, spec = job["id"], job["spec"]
    # CLAIM-LESS SHADOW: an unvalidated backup must never block the ladder.
    # In "on" mode the Space (and the HPC's scan) treat any fresh claim as
    # "a worker is on it" — a shadow claim would make the Space wait 45 min
    # for a result we are forbidden to publish, and stop the HPC claiming.
    # So we only write claims when we're allowed to deliver (PUBLISH=1).
    if PUBLISH:
        if not _commit_claim(ev, spec.get("queued", "")):
            return False
        time.sleep(3)
        claim = _read_json(f"{QPREFIX}/{ev}.claim")
        if not claim or claim.get("worker") != IDENT:
            log(f"lost claim race on {ev}")
            return False

    rec = W["sessions"].get(ev)
    if rec and (rec["bbox"] != spec["bbox_basin"]
                or rec.get("cold") != spec.get("cold")):
        log(f"{ev}: {'cold re-run requested' if rec.get('cold') != spec.get('cold') else 'bbox changed'}"
            f" — dropping resident session")
        shutil.rmtree(rec["root"], ignore_errors=True)
        W["sessions"].pop(ev, None)
        rec = None
    if rec is None:
        rec = {"session": None, "root": tempfile.mkdtemp(prefix=f"sess_{ev}_"),
               "bbox": spec["bbox_basin"], "cold": spec.get("cold"),
               "visits": 0, "last": time.time()}
    W["sessions"][ev] = rec        # registered now so _fail_job cleans it up
    n = rec["visits"] + 1
    forcing = os.path.join(rec["root"], f"forcing_{n}")
    out_dir = os.path.join(rec["root"], f"out_{n}")
    os.makedirs(forcing, exist_ok=True)
    from huggingface_hub import hf_hub_download
    tar = hf_hub_download(REPO, f"{QPREFIX}/{ev}.forcing.tar.gz",
                          repo_type="dataset", token=_tok or None,
                          force_download=True)
    with tarfile.open(tar) as tf:
        tf.extractall(forcing)
    log(f"{ev}: forcing extracted ({len(os.listdir(forcing))} grids)")

    t0 = datetime.datetime.strptime(spec["t0"], TMIN)
    t_end = datetime.datetime.strptime(spec["t_end"], TMIN)
    if rec["session"] is None:
        cfg = EventConfig(
            event_id=ev, bbox=tuple(spec["bbox_basin"]),
            t0=t0, t_end=t_end,
            sim_start=datetime.datetime.strptime(spec["sim_start"], TMIN),
            ef5_output_dir=forcing, out_dir=out_dir,
            model=spec.get("model", "crest"), dem_res="1",
            dem_cache=os.path.join(tempfile.gettempdir(), "dem_cache"),
            max_cells=MAX_CELLS, trigger=spec.get("trigger"),
            device="cpu", progress=log)      # module-level fn: fork-picklable
        session = EventSession(cfg, resume=True)   # CPU: DEM + pre-wet/resume
        if max(session.grid.dx, session.grid.dy) > 45.0:   # ~30 m native
            raise RuntimeError(f"grid dx {session.grid.dx:.0f} m — load_dem "
                               f"coarsened; refusing (resolution-first)")
        rec["session"] = session
    session = rec["session"]
    session.begin_visit(forcing, out_dir, t0, t_end)
    ny, nx = session.grid.z.shape
    W.update(job=job, runner=session, workdir=None, phase="solving",
             queued=spec.get("queued", ""), chunk_sim_s=CHUNK0_SIM_S,
             t_wall0=time.time(), last_hb=time.time())
    log(f"{ev}: visit {n} begins ({ny}x{nx}, "
        f"{(session.visit['t_end_rel'] - session.t) / 3600:.1f} sim-h from "
        f"{session.state_time():%m-%d %H:%M})")
    return True


def _finish_job():
    job, session = W["job"], W["runner"]
    ev, spec = job["id"], job["spec"]
    queued = W["queued"]
    rec = W["sessions"][ev]
    out_dir = session.visit["out_dir"]
    n = rec["visits"] + 1
    from huggingface_hub import CommitOperationDelete
    try:
        manifest = session.end_visit()            # rolls back to the junction
        manifest["gauge"] = spec["gauge"]["id"]
        manifest["hydro"] = spec.get("hydro") or []
        manifest["status"] = "active"
        manifest["basin"] = spec.get("basin")
        manifest["huc12s"] = spec.get("huc12s")
        manifest["domain"] = {**{k: v for k, v in (spec.get("domain") or {}).items()
                                 if k != "hucs"},
                              **(manifest.get("domain") or {})}
        from hf_data import eventsim, eventstore
        eventsim._update_archive(out_dir, manifest, log)
        if not manifest.get("archive_frames"):
            manifest["status"] = "ended"
            manifest["dry"] = True
        eventsim._make_pngs(out_dir, manifest)
        wall = time.time() - W["t_wall0"]
        log(f"{ev}: visit {n} done — {len(manifest['frames'])} frames, "
            f"wall {wall / 60:.1f} min")
        if PUBLISH:
            ok = eventstore.publish_event(out_dir, manifest)
            log(f"{ev}: publish {'OK' if ok else 'FAILED'}")
            ops = [CommitOperationDelete(f"{QPREFIX}/{ev}.claim")]
            cur = _read_json(f"{QPREFIX}/{ev}.json")
            if ok and cur and cur.get("queued", "") == queued:
                ops += [CommitOperationDelete(f"{QPREFIX}/{ev}.json"),
                        CommitOperationDelete(f"{QPREFIX}/{ev}.forcing.tar.gz")]
            _commit(ops, f"zero-worker done {ev}")
        else:
            log(f"{ev}: shadow done — no claim was held, nothing to release")
        W["solved"].append({"id": ev, "visit": n,
                            "frames": len(manifest["frames"]),
                            "wall_min": round(wall / 60, 1),
                            "published": PUBLISH})
        rec["visits"] = n
        rec["last"] = time.time()
        # bound disk: only the just-published visit's dirs remain
        for stale in os.listdir(rec["root"]):
            if stale not in (f"forcing_{n}", f"out_{n}"):
                shutil.rmtree(os.path.join(rec["root"], stale),
                              ignore_errors=True)
    finally:
        W["done"].add((ev, queued))
        W.update(job=None, runner=None, workdir=None, phase="idle")


def _fail_job(exc):
    job = W["job"]
    ev, queued = job["id"], W["queued"]
    tb = "\n".join(traceback.format_exc().strip().splitlines()[-8:])
    log(f"{ev}: FAILED — {type(exc).__name__}: {exc}")
    if PUBLISH:                                   # shadow held no claim
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        ops = [CommitOperationDelete(f"{QPREFIX}/{ev}.claim")]
        cur = _read_json(f"{QPREFIX}/{ev}.json")
        if cur and cur.get("queued", "") == queued:
            body = json.dumps({"worker": IDENT,
                               "failed": datetime.datetime.utcnow().strftime(TS),
                               "error": f"{type(exc).__name__}: {exc}"[:2000],
                               "traceback_tail": tb[:4000]})
            ops += [CommitOperationAdd(f"{QPREFIX}/{ev}.failed.json",
                                       io.BytesIO(body.encode())),
                    CommitOperationDelete(f"{QPREFIX}/{ev}.json"),
                    CommitOperationDelete(f"{QPREFIX}/{ev}.forcing.tar.gz")]
        _commit(ops, f"zero-worker failed {ev}")
    W["done"].add((ev, queued))
    # mid-visit state is not trustworthy — drop the session; the next
    # bundle rebuilds via init_depth (state-hierarchy level 2)
    rec = W["sessions"].pop(ev, None)
    if rec:
        shutil.rmtree(rec["root"], ignore_errors=True)
    W.update(job=None, runner=None, workdir=None, phase="idle")


def _evict_sessions():
    """Idle and LRU eviction of parked sessions (never the active one)."""
    now = time.time()
    active = (W["job"] or {}).get("id")
    for ev in list(W["sessions"]):
        if ev == active:
            continue
        idle_h = (now - W["sessions"][ev]["last"]) / 3600.0
        if idle_h > EVICT_IDLE_H:
            log(f"evicting session {ev} (idle {idle_h:.0f} h)")
            shutil.rmtree(W["sessions"][ev]["root"], ignore_errors=True)
            del W["sessions"][ev]
    while len(W["sessions"]) > MAX_SESSIONS:
        parked = [e for e in W["sessions"] if e != active]
        if not parked:
            break
        ev = min(parked, key=lambda e: W["sessions"][e]["last"])
        log(f"evicting session {ev} (LRU, cap {MAX_SESSIONS})")
        shutil.rmtree(W["sessions"][ev]["root"], ignore_errors=True)
        del W["sessions"][ev]


def tick():
    """One state-machine step; driven by gr.Timer (single-flight)."""
    if not _tick_lock.acquire(blocking=False):
        return _status(), "\n".join(LOG[-40:])
    try:
        now = time.time()
        if W["phase"] == "idle":
            if now - W["last_poll"] >= POLL_S:
                W["last_poll"] = now
                _evict_sessions()
                job = _scan()
                if job:
                    kind = ("catch-up" if job["id"] in W["sessions"]
                            else "new")
                    log(f"claiming {job['id']} (backup rung, {kind}, "
                        f"queued {job['spec'].get('queued')})")
                    try:
                        _start_job(job)
                    except Exception as e:
                        W["job"] = job
                        W["queued"] = job["spec"].get("queued", "")
                        _fail_job(e)
        elif W["phase"] == "solving":
            ev = W["job"]["id"]
            if PUBLISH and now - W["last_hb"] >= HB_S:
                _commit_claim(ev, W["queued"])
                W["last_hb"] = now
            session = W["runner"]
            t0 = time.time()
            try:
                done, state = gpu_chunk(session, W["chunk_sim_s"])
                session.set_state(state)
            except Exception as e:
                _fail_job(e)
                return _status(), "\n".join(LOG[-40:])
            wall = max(time.time() - t0, 1e-3)
            rate = W["chunk_sim_s"] / wall                  # sim-s per wall-s
            W["chunk_sim_s"] = float(min(max(GPU_TARGET_S * rate, 60.0), 14400.0))
            total_h = session.visit["t_end_rel"] / 3600.0
            log(f"{ev}: sim t={session.t / 3600:.2f}/{total_h:.0f} h "
                f"(chunk {wall:.0f} s wall, next {W['chunk_sim_s'] / 60:.0f} sim-min)")
            if done:
                try:
                    _finish_job()
                except Exception as e:
                    _fail_job(e)
        return _status(), "\n".join(LOG[-40:])
    finally:
        _tick_lock.release()


def _status():
    s = W["runner"]
    return {"ident": IDENT, "phase": W["phase"], "publish": PUBLISH,
            "job": (W["job"] or {}).get("id"),
            "sim_h": round(s.t / 3600, 2) if s else None,
            "total_h": (round(s.visit["t_end_rel"] / 3600, 1)
                        if s and s.visit else None),
            "chunk_sim_min": round(W["chunk_sim_s"] / 60, 1),
            "sessions": {ev: {"visits": r["visits"],
                              "held_at": (r["session"].state_time()
                                          .strftime("%m-%d %H:%MZ")
                                          if r["session"] else None)}
                         for ev, r in W["sessions"].items()},
            "solved": W["solved"][-5:], "zero_gpu": HAS_ZERO}


def _keep_awake():
    import requests
    host = os.environ.get("SPACE_HOST", "")
    while host:
        time.sleep(900)
        try:
            requests.get(f"https://{host}", timeout=30)
        except Exception:
            pass


threading.Thread(target=_keep_awake, daemon=True).start()
log(f"boot: {IDENT} publish={PUBLISH} zero={HAS_ZERO} "
    f"delay={ZERO_DELAY_S:.0f}s budget={MAX_CELLS / 1e6:.1f}M cells")

with gr.Blocks(title="CREST GPU worker (ZeroGPU backup)") as demo:
    gr.Markdown("## CREST event worker — ZeroGPU backup rung\n"
                "Claims 2-D inundation jobs the HPC leaves unclaimed; "
                "shadow mode unless PUBLISH=1.")
    st = gr.JSON(label="worker state")
    lg = gr.Textbox(label="log", lines=18, max_lines=18)
    t = gr.Timer(5)
    t.tick(tick, outputs=[st, lg], concurrency_limit=1)
    demo.load(tick, outputs=[st, lg])

demo.queue().launch()
