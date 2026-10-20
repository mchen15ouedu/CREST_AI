"""CREST_gpu_worker — ZeroGPU event worker (V27 P1.5 + P1.6 sessions). Since
the HPC worker was released (2026-10-16) it is the only 2-D engine: it claims
events/queue/ jobs, solves them in <=120 s GPU slices within a daily GPU
budget (authenticated calls draw on the account's ZeroGPU quota), and — when
PUBLISH=1 — publishes exactly like the HPC worker did.

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
    the CREST_demo Space (eventsim/eventstore reuse — nothing reimplemented).

    crestimap is imported straight from the clone (kept as a git checkout), so
    provenance.stamp() records the fork SHA on every manifest; a copied-out
    package has no .git and stamped only "2.0.0.dev0"."""
    fork = os.path.join(APP, "_fork")
    if not os.path.isdir(os.path.join(fork, "crestimap")):
        shutil.rmtree(fork, ignore_errors=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "-b", "v2", FORK, fork], check=True)
        subprocess.run(["git", "checkout", "HEAD", "--", "crestimap"],
                       cwd=fork, check=True)
    if fork not in sys.path:
        sys.path.insert(0, fork)
    ai = os.path.join(APP, "_crest_ai")
    if not os.path.isdir(os.path.join(ai, "hf_data")):
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        CREST_AI, ai], check=True)
        subprocess.run(["git", "checkout", "HEAD", "--", "hf_data"],
                       cwd=ai, check=True)
    if ai not in sys.path:
        sys.path.insert(0, ai)


_bootstrap()
try:
    _FORK_SHA = subprocess.run(
        ["git", "-C", os.path.join(APP, "_fork"), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=15).stdout.strip()[:12] or None
except Exception:
    _FORK_SHA = None

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
# The HPC worker was released 2026-10-16; this Space is now the only 2-D
# engine, so there is no reason to leave a job unclaimed for anyone else.
ZERO_DELAY_S = float(os.environ.get("ZERO_DELAY_S", "0"))
# GPU calls must be AUTHENTICATED to get the account's ZeroGPU quota (PRO:
# 40 min/day, fixed 24 h window from first use). Self-calls over 127.0.0.1
# carry no X-IP-Token and land in a shared anonymous "Space app" pool that
# gave ~0 GPU-s/day from 2026-09-23 to 10-16 ("Space app has reached its GPU
# limit" on every claim). ZERO_AUTH=1 self-drives through the public Space
# URL with HF_TOKEN instead.
ZERO_AUTH = os.environ.get("ZERO_AUTH", "1") == "1"
# Our own daily cap, below the 40-min PRO quota: the quota is shared with
# every other ZeroGPU use under the same account, and past it ZeroGPU bills
# prepaid credits automatically ($1 per 10 min). 0 disables the cap.
DAILY_BUDGET_S = float(os.environ.get("ZERO_DAILY_BUDGET_S", "1800"))
GPU_CALL_S = 120.0          # declared @spaces.GPU duration (admission needs it)
if 0 < DAILY_BUDGET_S < GPU_CALL_S:
    DAILY_BUDGET_S = GPU_CALL_S                 # a smaller cap could never run
# a NEW event (full anchored solve) is only claimed with this much budget
# left; otherwise it would be claimed, paused ~a day, and solved stale
MIN_NEW_S = float(os.environ.get("ZERO_MIN_NEW_S", "600"))
# "No GPU was available after 60s" is platform capacity, not this job's
# fault and not quota: wait and retry the same chunk.
CAPACITY_BACKOFF_S = float(os.environ.get("ZERO_CAPACITY_BACKOFF_S", "120"))
# Backup -> active drain under pressure (user directive 2026-08-14): with
# more than this many jobs pending, skip the polite deferral and claim the
# next unclaimed event immediately — one solver per event, so this Space
# always works a DIFFERENT event than the HPC.
PRESSURE_N = int(os.environ.get("ZERO_PRESSURE_N", "4"))
STALE_S = float(os.environ.get("ZERO_STALE_S", "600"))
HB_S = float(os.environ.get("ZERO_HB_S", "240"))
POLL_S = float(os.environ.get("ZERO_POLL_S", "60"))
# ZeroGPU quota exhaustion is NOT a poison job: release the claim (the HPC
# or a later quota window takes the bundle) and stop claiming for a while.
# Without this (seen 2026-09-11) the backup rung claimed a bundle, hit
# "Space app has reached its GPU limit", and CONSUMED it as failed.
QUOTA_BACKOFF_S = float(os.environ.get("ZERO_QUOTA_BACKOFF_S", "3600"))
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
# > one daily budget pause (~23.5 h), so parked sessions survive the pause and
# the next bundle is a cheap catch-up rather than a full re-solve
EVICT_IDLE_H = float(os.environ.get("ZERO_EVICT_IDLE_H", "36"))
# cadence of the server-side tick loop (see _self_drive); matches the old
# gr.Timer(5) so idle polling behaviour is unchanged
SELF_TICK_S = float(os.environ.get("ZERO_SELF_TICK_S", "5"))

LOG: list[str] = []
W = {"phase": "idle", "job": None, "runner": None, "chunk_sim_s": CHUNK0_SIM_S,
     "last_poll": 0.0, "last_hb": 0.0, "queued": "", "workdir": None,
     "done": set(), "solved": [], "t_wall0": 0.0, "last_tick": 0.0,
     # daily GPU budget: [(start, seconds), ...] of GPU calls in a SLIDING
     # 24 h window, so use stays under the cap in any 24 h (hence in any of
     # ZeroGPU's own windows); RAM-only — keep the account's prepaid credit
     # balance at $0 so an over-run after a restart becomes a quota error
     "gpu_calls": [], "capacity_until": 0.0, "quota_until": 0.0,
     "paused_note": "", "auth_suspect": False,
     # /tick calls by outcome (see tick_api); "accepted" must keep rising
     "tick_api": {"accepted": 0, "rejected": 0, "empty": 0,
                  "last_reject_log": 0.0},
     # what the last _scan saw (status "scan"): time, queue ids, backlog,
     # candidates and per-reason skip counts
     "scan": {},
     # P1.6 resident sessions: ev -> {"session", "bbox", "visits", "last",
     # "root"}. RAM-only; a Space rebuild recovers via the bundle's
     # init_depth (state-hierarchy level 2).
     "sessions": {}}
_tick_lock = threading.Lock()
# secret the self-drive loop passes to /tick (in-process only; never logged)
_SELF_KEY = __import__("secrets").token_hex(16)


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


def _used_s(now: float) -> float:
    """GPU seconds booked in the last 24 h (old calls are forgotten)."""
    W["gpu_calls"] = [(t, s) for t, s in W["gpu_calls"] if now - t < 86400.0]
    return sum(s for _, s in W["gpu_calls"])


def _budget_left(now: float) -> float:
    """Seconds of daily GPU budget left (inf when uncapped)."""
    if DAILY_BUDGET_S <= 0:
        return float("inf")
    return DAILY_BUDGET_S - _used_s(now)


def _budget_free_at(now: float, need: float) -> float:
    """When at least `need` seconds of budget will be free again."""
    left = _budget_left(now)
    for t, s in sorted(W["gpu_calls"]):
        if left >= need:
            break
        left += s
        now = t + 86400.0
    return now


def _charge(seconds: float, started: float):
    """Book a GPU call that began at `started` and lasted `seconds` (wall,
    incl. fork/pickle overhead — conservative)."""
    if DAILY_BUDGET_S > 0:
        W["gpu_calls"].append((started, max(0.0, seconds)))


def _gpu_wait_reason(now: float) -> str | None:
    """Why no GPU call may start now (None = go)."""
    if now < W["quota_until"]:
        return f"ZeroGPU quota exhausted until {_hhmm(W['quota_until'])}"
    if now < W["capacity_until"]:
        return "no ZeroGPU capacity — retrying shortly"
    if _budget_left(now) < GPU_CALL_S:
        return (f"daily GPU budget used ({_used_s(now) / 60:.0f} of "
                f"{DAILY_BUDGET_S / 60:.0f} min in 24 h) — resumes "
                f"{_hhmm(_budget_free_at(now, GPU_CALL_S))}")
    return None


def _hhmm(t: float) -> str:
    return datetime.datetime.utcfromtimestamp(t).strftime("%m-%d %H:%MZ")


def _quota_retry_s(exc) -> float | None:
    """Seconds until ZeroGPU says to retry ('... try again in H:MM:SS'), or
    None when the error names no retry time (e.g. the anonymous pool's
    'Space app has reached its GPU limit')."""
    import re
    m = re.search(r"try again in (\d+):(\d{2}):(\d{2})", str(exc), re.I)
    if m:
        h, mi, s = (int(x) for x in m.groups())
        return float(h * 3600 + mi * 60 + s) + 60.0
    return None


def _auth_suspect(exc) -> bool:
    """The error text of the ANONYMOUS pool: our calls were not attributed
    to the account (ZERO_AUTH did not take effect)."""
    m = f"{exc}".lower()
    return "space app has reached its gpu limit" in m or "unlogged" in m


def _is_capacity_error(exc) -> bool:
    return "no gpu was available" in f"{exc}".lower()


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
    skip = W["scan"]["skipped"] = {}
    W["scan"].update(at=now.strftime(TS), ids=len(ids), backlog=backlog)

    def _skip(why):
        skip[why] = skip.get(why, 0) + 1

    for ev in ids:
        if f"{QPREFIX}/{ev}.forcing.tar.gz" not in qf:
            _skip("no_forcing")
            continue
        spec = _read_json(f"{QPREFIX}/{ev}.json")
        if not spec:
            _skip("spec_unreadable")
            continue
        if (ev, spec.get("queued", "")) in W["done"]:
            _skip("done")
            continue
        if str(spec.get("dem_res", "1")) != "1":
            _skip("not_30m")
            continue                              # 30 m only on this rung
        try:
            age = (now - datetime.datetime.strptime(spec["queued"], TS)
                   ).total_seconds()
        except Exception:
            age = 0.0
        if age < delay:
            _skip("deferred")
            continue
        if not spec.get("domain"):
            # pre-basin bundle (rectangular-window era): never solved as a
            # rectangle — the runner re-enqueues it with a HUC12 basin domain
            log(f"skip {ev}: no basin domain in spec — awaiting re-enqueue")
            _skip("no_domain")
            continue
        w, s, e, n = spec["bbox_basin"]
        res = DEM_RES_DEG["1"]
        cells = int((spec.get("domain") or {}).get("n_bbox") or
                    int(round((n - s) / res)) * int(round((e - w) / res)))
        if cells > MAX_CELLS:
            log(f"skip {ev}: {cells / 1e6:.1f} M cells > "
                f"{MAX_CELLS / 1e6:.1f} M budget (never coarsen)")
            _skip("over_budget")
            continue
        if f"{QPREFIX}/{ev}.claim" in qf:
            claim = _read_json(f"{QPREFIX}/{ev}.claim")
            if _claim_fresh(claim) and claim.get("worker") != IDENT:
                _skip("claimed")
                continue
        cands.append({"id": ev, "spec": spec})
    W["scan"]["candidates"] = len(cands)
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


def _is_quota_error(exc) -> bool:
    m = f"{type(exc).__name__}: {exc}".lower()
    return "gpu limit" in m or "quota" in m


def _fail_job(exc):
    job = W["job"]
    ev, queued = job["id"], W["queued"]
    tb = "\n".join(traceback.format_exc().strip().splitlines()[-8:])
    if _is_quota_error(exc):
        # not this job's fault: hand it back untouched and back off
        W["quota_until"] = time.time() + QUOTA_BACKOFF_S
        log(f"{ev}: ZeroGPU quota exhausted — releasing the claim, no "
            f"claims for {QUOTA_BACKOFF_S / 60:.0f} min ({exc})")
        if PUBLISH:
            from huggingface_hub import CommitOperationDelete
            _commit([CommitOperationDelete(f"{QPREFIX}/{ev}.claim")],
                    f"zero-worker quota — released {ev}")
        rec = W["sessions"].pop(ev, None)      # mid-visit state is untrustworthy
        if rec:
            shutil.rmtree(rec["root"], ignore_errors=True)
        W.update(job=None, runner=None, workdir=None, phase="idle")
        return                                  # NOT added to done: retryable
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


def view():
    """What a browser sees: status + log only. It never advances the state
    machine — a visitor's request would run @spaces.GPU under the VISITOR's
    ZeroGPU identity, and an anonymous visitor's quota error could stall the
    only 2-D engine for a day."""
    return _status(), "\n".join(LOG[-40:])


def tick_api(key: str = ""):
    """The /tick endpoint: advances the worker only for the self-drive loop,
    which passes this process's secret; anyone else just gets the status.

    Every call is counted (status "tick_api"): from 10-19 to 10-20 the worker
    sat idle next to an eligible job with no log line at all, which means
    either the key never arrived or tick() found nothing. These counters tell
    the two apart from outside."""
    c = W["tick_api"]
    if not key:
        c["empty"] += 1
        return view()
    if key != _SELF_KEY:
        c["rejected"] += 1
        if time.time() - c["last_reject_log"] > 3600:
            c["last_reject_log"] = time.time()
            log(f"tick_api: rejected a key (len {len(key)}; "
                f"{c['rejected']} so far)")
        return view()
    c["accepted"] += 1
    if c["accepted"] == 1:
        log("tick_api: self-drive key accepted — worker advancing")
    return tick()


def tick():
    """One state-machine step (single-flight); called via tick_api by the
    self-drive loop."""
    if not _tick_lock.acquire(blocking=False):
        return _status(), "\n".join(LOG[-40:])
    try:
        now = time.time()
        wait = _gpu_wait_reason(now)
        if (wait or "") != W["paused_note"]:
            W["paused_note"] = wait or ""
            log(f"paused: {wait}" if wait else "GPU available — resuming")
        if W["phase"] == "idle":
            # never claim a job we couldn't start computing on right away
            if now - W["last_poll"] >= POLL_S and wait is None:
                W["last_poll"] = now
                _evict_sessions()
                job = _scan()
                if (job and job["id"] not in W["sessions"]
                        and _budget_left(now) < MIN_NEW_S):
                    job = None          # a new event waits for a fuller budget
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
            if wait is not None:
                # paused mid-visit: the session keeps its state in RAM and
                # the claim stays fresh (heartbeat above); resume later
                return _status(), "\n".join(LOG[-40:])
            t0 = time.time()
            try:
                done, state = gpu_chunk(session, W["chunk_sim_s"])
                session.set_state(state)
            except Exception as e:
                # admission failures leave the session untouched (the chunk
                # never ran): hold the job and retry the same chunk later —
                # but only when ZeroGPU names a retry time; a quota error
                # without one (the anonymous pool: our identity was not
                # recognised) takes the old release-and-back-off path
                retry = _quota_retry_s(e) if _is_quota_error(e) else None
                if _is_capacity_error(e):
                    W["capacity_until"] = time.time() + CAPACITY_BACKOFF_S
                    log(f"{ev}: no ZeroGPU capacity — retry in "
                        f"{CAPACITY_BACKOFF_S:.0f} s ({e})")
                elif retry is not None:
                    W["quota_until"] = time.time() + retry
                    log(f"{ev}: ZeroGPU quota exhausted — holding the job "
                        f"until {_hhmm(W['quota_until'])} ({e})")
                else:
                    if _auth_suspect(e):
                        W["auth_suspect"] = True
                        log(f"{ev}: anonymous-pool quota error — GPU calls are "
                            f"NOT attributed to the account (ZERO_AUTH failed?)")
                    elif not _is_quota_error(e):
                        _charge(time.time() - t0, t0)   # it ran, then failed
                    _fail_job(e)
                return _status(), "\n".join(LOG[-40:])
            _charge(time.time() - t0, t0)
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
            # seconds since the server-side loop last completed a tick — the
            # liveness signal: if this grows without bound the worker is idle
            # even though the container reads RUNNING (the 2026-08-24 failure)
            "tick_age_s": (round(time.time() - W["last_tick"], 1)
                           if W["last_tick"] else None),
            "sessions": {ev: {"visits": r["visits"],
                              "held_at": (r["session"].state_time()
                                          .strftime("%m-%d %H:%MZ")
                                          if r["session"] else None)}
                         for ev, r in W["sessions"].items()},
            "solved": W["solved"][-5:], "zero_gpu": HAS_ZERO,
            "tick_api": {k: W["tick_api"][k]
                         for k in ("accepted", "rejected", "empty")},
            "scan": W["scan"],
            "fork_sha": _FORK_SHA,
            "gpu_budget": {"used_min_24h": round(_used_s(time.time()) / 60, 1),
                           "budget_min": round(DAILY_BUDGET_S / 60, 1),
                           "authenticated": bool(ZERO_AUTH and _tok),
                           "auth_suspect": W["auth_suspect"],
                           "paused": W["paused_note"] or None}}


def _keep_awake():
    import requests
    host = os.environ.get("SPACE_HOST", "")
    while host:
        time.sleep(900)
        try:
            requests.get(f"https://{host}", timeout=30)
        except Exception:
            pass


def _self_drive():
    """Server-side tick loop — the worker's actual engine.

    gr.Timer is a FRONTEND timer: it only fires while a browser session has
    the Space open, and demo.load only fires on a page load. _keep_awake's
    plain GET keeps the container off sleep but opens no Gradio session, so
    with nobody watching this worker sat RUNNING and completely idle — found
    2026-08-24: 59 min of silence between boot and the first externally
    triggered /tick, and every past ZeroGPU solve coincided with someone
    polling it. The fleet/ungauged workers self-drive with a plain thread;
    this brings the GPU worker in line.

    The tick is driven through the Space's own /tick API over localhost
    rather than by calling tick() inline, so the state machine advances on
    exactly the request path it was built for (queue single-flight, and the
    @spaces.GPU allocation happens inside a normal Gradio request)."""
    from gradio_client import Client
    base = f"http://127.0.0.1:{os.environ.get('GRADIO_SERVER_PORT', '7860')}"
    space = os.environ.get("SPACE_ID", "")
    if ZERO_AUTH and _tok and space:
        # through HF's proxy with the account token: the proxy attaches the
        # caller identity (X-IP-Token), so @spaces.GPU draws on the account's
        # ZeroGPU quota instead of the anonymous per-Space pool
        src, kw, how = space, {"hf_token": _tok}, "authenticated (account quota)"
    else:
        src, kw, how = base, {}, "localhost, unauthenticated (anonymous pool)"
    client = None
    while True:
        try:
            if client is None:
                # token positionally: gradio_client 1.x names it hf_token,
                # 2.x names it token
                client = (Client(src, kw["hf_token"], verbose=False) if kw
                          else Client(src, verbose=False))
                log(f"self-drive: tick loop connected — {how}, "
                    f"{SELF_TICK_S:.0f} s cadence")
            # bounded wait: a half-open proxy connection must not hang the
            # only engine forever (a healthy tick is <= ~3 min)
            client.submit(_SELF_KEY, api_name="/tick").result(timeout=900)
            W["last_tick"] = time.time()
        except Exception as e:                    # server still booting, etc.
            try:
                if client is not None:
                    client.close()
            except Exception:
                pass
            client = None
            log(f"self-drive: {type(e).__name__}: {str(e)[:200]} — retrying")
            time.sleep(15)
            continue
        time.sleep(SELF_TICK_S)


threading.Thread(target=_keep_awake, daemon=True).start()
threading.Thread(target=_self_drive, daemon=True).start()
log(f"boot: {IDENT} publish={PUBLISH} zero={HAS_ZERO} "
    f"delay={ZERO_DELAY_S:.0f}s budget={MAX_CELLS / 1e6:.1f}M cells "
    f"gpu/day={DAILY_BUDGET_S / 60:.0f} min auth={bool(ZERO_AUTH and _tok)} "
    f"crestimap={_FORK_SHA}")

with gr.Blocks(title="CREST GPU worker (ZeroGPU)") as demo:
    gr.Markdown("## CREST event worker — ZeroGPU\n"
                "The 2-D inundation engine since the HPC worker was released "
                "(2026-10-16). Works within a daily GPU budget; shadow mode "
                "unless PUBLISH=1.")
    st = gr.JSON(label="worker state")
    lg = gr.Textbox(label="log", lines=18, max_lines=18)
    # the page only DISPLAYS (view); the worker advances through /tick,
    # which needs the in-process key the self-drive loop sends
    t = gr.Timer(5)
    t.tick(view, outputs=[st, lg], api_name=False)
    demo.load(view, outputs=[st, lg], api_name=False)
    key_in = gr.Textbox(visible=False)
    tick_btn = gr.Button(visible=False)
    tick_btn.click(tick_api, inputs=[key_in], outputs=[st, lg],
                   api_name="tick", concurrency_limit=1)

demo.queue().launch()
