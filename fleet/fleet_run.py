"""Virtual-user fleet: precompute CREST-AI simulations for many gauges.

Runs on the offline server (NOT the Space; needs the EF5 binary + HF_TOKEN).
For each gauge it drives hf_data.pipeline.run_gauge in CHUNK_DAYS windows so
EF5 saves a model state at every chunk boundary (10 days by default) and the
hourly rows accumulate in the statecache record — exactly the artifacts the
dashboard's cache/warm-start machinery consumes. When a gauge's whole period
is done, its states are packed to one float16 bundle and uploaded with the
record to the read-only fleet repo; local files are then deleted.

Resumable at every level: gauges already in the repo are skipped, and within
a gauge statecache.plan() fast-forwards chunks whose rows are already cached.

Outlets: every USGS gauge plus every HydroBASINS level-07 pour point (virtual
point), each on its basin outline (hf_data/basinextent: never a box; an outlet
without one is skipped with the reason). --size splits them at FLEET_BIG_KM2:
the HF runners take the small class (default), the Jetstream2 fleet the big.
Each outlet's speed-scheme cut set is fixed once for the whole period
(pipeline.plan_cut_set), so no chunk re-keys its record.

Run from the repo root (CWD must contain ./EF5/bin/ef5):
    python fleet/fleet_run.py --workers 8
    python fleet/fleet_run.py --gauges "08167000, 08144500"     # subset
See fleet/README_FLEET.md for server setup.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

FLEET_REPO = os.environ.get("CREST_FLEET_REPO", "vincewin/CREST_fleet")
DEF_START, DEF_END = "2021-07-01", "2026-06-30"      # last 5 y of MRMS coverage
PROGRESS = "fleet_progress.jsonl"
# size split (PI 2026-10-23): basins at/above this drainage area run on the
# Jetstream2 fleet (FLEET_SIZE=big); the HF runners take the rest (default
# "small") - a 20,000 km2 basin needs ~15 h on a cpu-basic runner, which a
# Space restart throws away
BIG_KM2 = float(os.environ.get("FLEET_BIG_KM2", "5000"))
# pour points (HydroBASINS level-07 virtual points) upload under their own
# prefix: CREST_fleet's states/V*, results/V* are the ungauged nowcast's
# rolling checkpoints, which a 5-year hindcast must never overwrite
VP_PREFIX = "vp/"


def _token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("HF_TOKEN env var not set")
    return tok


def _prefix(gid: str) -> str:
    return VP_PREFIX if str(gid).upper().startswith("V") else ""


def _done_keys() -> set[str]:
    """Outlet ids already fully uploaded (results + states present). By id,
    not by key: rows are keyed by model+scheme+parameter tag, state grids by
    model+scheme only, so the two names differ for a calibrated gauge.
    A listing failure is retried, then fatal: treating it as "nothing done"
    would re-run and re-upload every finished outlet of the shard."""
    from huggingface_hub import HfApi
    for attempt in range(4):
        try:
            files = set(HfApi(token=_token()).list_repo_files(FLEET_REPO, repo_type="dataset"))
            break
        except Exception as e:
            if attempt == 3:
                sys.exit(f"cannot list {FLEET_REPO}: {type(e).__name__}: {e}")
            time.sleep(30 * (attempt + 1))
    done = set()
    for pre in ("", VP_PREFIX):
        res = {f[len(pre + "results/"):].split("_", 1)[0] for f in files
               if f.startswith(pre + "results/")}
        st = {f[len(pre + "states/"):].split("_", 1)[0] for f in files
              if f.startswith(pre + "states/")}
        ids = res & st
        # gauges under the bare prefix; pour points only under vp/
        done |= {g for g in ids if g.upper().startswith("V") == (pre == VP_PREFIX)}
    return {g + "_" for g in done}


_SHARED_KEYS = None            # per worker process


def _keys(bbox) -> tuple[str, str]:
    """(terrain clip key, forcing store key) of an extent: basic.store_key and
    the formula forcing.store_dir uses (without their mkdir)."""
    from hf_data import basic
    return (basic.store_key(bbox),
            "_".join(f"{v:.3f}" for v in bbox).replace("-", "m").replace(".", "p"))


def _shared_store_keys() -> set:
    """Clip-store keys that more than one outlet maps to (nested basins share
    an extent). Never pruned: a sibling, possibly in another shard on the same
    cache, may be running on them right now. Extents come from the
    basin-extent table, never computed at runtime."""
    global _SHARED_KEYS
    if _SHARED_KEYS is None:
        from collections import Counter
        from hf_data import basinextent
        df, _ = basinextent._load()
        n = Counter()
        for r in df[df.status == "ok"].itertuples():
            for k in _keys((r.w, r.s, r.e, r.n)):
                n[k] += 1
        _SHARED_KEYS = {k for k, c in n.items() if c > 1}
    return _SHARED_KEYS


def _prune_basin_store(gid: str) -> None:
    """Drop the outlet's per-basin clip stores (hourly forcing ~89k files over
    5 years, plus clipped terrain/params) when it finishes, uploaded or failed:
    a retry re-clips them from the locally cached month tars.
    Opt-in (FLEET_PRUNE_FORCING=1) for hosts with a PERSISTENT cache, e.g. the
    Jetstream2 VM: nothing else prunes these there (the datamgr janitor runs
    only in server.py), while HF Spaces lose /tmp at every restart anyway.
    Same extent as the run: basinextent.ef5_extent(gid)."""
    from hf_data import basic, basinextent, forcing
    try:
        bbox = basinextent.ef5_extent(gid)["bounds"]
    except basinextent.BasinExtentError:
        return                         # never ran: nothing was clipped
    kb, kf = _keys(bbox)
    shared = _shared_store_keys()
    if kf not in shared:
        for var in ("mrms", "pet", "temp"):
            shutil.rmtree(forcing.store_dir(var, bbox), ignore_errors=True)
    if kb not in shared:
        shutil.rmtree(basic.store_dir(bbox), ignore_errors=True)


def select_outlets(ids: list[str], size: str):
    """(runnable ids in input order, Counter of skip reasons). Outlets without
    a basin outline are skipped once, with the reason - never retried in a
    loop, never boxed; the rest are kept if they fall in the size class."""
    from collections import Counter
    from hf_data import basinextent
    skipped = Counter()
    runnable = []
    for g in ids:
        try:
            basinextent.ef5_extent(g)
        except basinextent.BasinExtentError as e:
            skipped[e.reason] += 1
            continue
        a = basinextent.size_km2(g) or 0.0
        if size == "all" or (a >= BIG_KM2) == (size == "big"):
            runnable.append(g)
    return runnable, skipped


def _check_code_rev(gid: str) -> None:
    """On a persistent cache a gauge's partial rows/states survive restarts,
    and a restart pulls new CREST_AI code: one uploaded record could then mix
    two code versions. FLEET_CODE_REV (set by the VM's run-fleet.sh to the
    hf_data/ + fleet/ tree hashes) is stamped per gauge; on a mismatch the
    gauge's partial work is dropped and it starts over. Unset on HF: no-op."""
    rev = os.environ.get("FLEET_CODE_REV", "").strip()
    if not rev:
        return
    from hf_data import statecache
    g8 = str(gid).zfill(8)
    sroot = os.path.join(statecache.CACHE_DIR, "states", g8)
    stamp = os.path.join(sroot, ".code_rev")
    try:
        old = open(stamp).read().strip()
    except OSError:
        old = None
    if old != rev:
        partial = glob.glob(os.path.join(statecache.CACHE_DIR, "results", f"{g8}_*.json"))
        if old is not None or partial or os.path.isdir(sroot):
            shutil.rmtree(sroot, ignore_errors=True)
            for p in partial:
                os.remove(p)
        os.makedirs(sroot, exist_ok=True)
        with open(stamp, "w") as fh:
            fh.write(rev + "\n")


def run_one_gauge(gid: str, t0: datetime, t1: datetime, chunk_days: int) -> dict:
    """Child process: chunked run -> bundle -> upload -> local cleanup."""
    _check_code_rev(gid)
    try:
        return _run_one_gauge(gid, t0, t1, chunk_days)
    finally:
        if os.environ.get("FLEET_PRUNE_FORCING") == "1":
            _prune_basin_store(gid)


def _run_one_gauge(gid: str, t0: datetime, t1: datetime, chunk_days: int) -> dict:
    import tempfile
    from hf_data import pipeline, statecache, statebundle
    from huggingface_hub import HfApi, CommitOperationAdd

    from hf_data import basinextent
    t_start = time.time()
    last_status = ""
    rows_model = state_model = None
    lakes_prov = None
    try:
        ext = basinextent.ef5_extent(gid)
        # M3: one cut set for the whole period, so no chunk re-keys the record
        cuts = pipeline.plan_cut_set(gid, t0, t1)
    except basinextent.BasinExtentError as e:
        return {"gauge": gid, "ok": False, "rc": -2, "no_basin": True,
                "last": str(e)[-300:]}
    cur = t0
    while cur < t1:
        ce = min(cur + timedelta(days=chunk_days), t1)
        rc = None
        for kind, payload in pipeline.run_gauge(gid, cur, ce, use_mock=False,
                                                grids=False, scheme="speed",
                                                cut_ids=cuts):
            if kind == "status":
                last_status = payload
            elif kind == "params":
                rows_model = payload.get("cache_model") or rows_model
                state_model = payload.get("state_model") or state_model
                lakes_prov = payload.get("lakes") or lakes_prov
            elif kind == "done":
                rc = payload.get("returncode")
        if rc != 0:
            return {"gauge": gid, "ok": False, "at": cur.strftime("%Y-%m-%d"),
                    "rc": rc, "last": last_status[-300:]}
        cur = ce

    # the run resolved model/scheme itself and reported its keys: rows under
    # model+scheme(+parameter tag), state grids under model+scheme
    if rows_model:
        rec_path = statecache.results_path(gid, rows_model)
        recs = [rec_path] if os.path.exists(rec_path) else []
    else:
        recs = sorted(glob.glob(os.path.join(statecache.CACHE_DIR, "results",
                                             f"{str(gid).zfill(8)}_*.json")),
                      key=os.path.getmtime)
    if not recs:
        return {"gauge": gid, "ok": False, "rc": -1, "last": "no result record"}
    rec_path = recs[-1]
    key = os.path.splitext(os.path.basename(rec_path))[0]
    skey = f"{str(gid).zfill(8)}_{state_model}" if state_model else key

    sdir = statecache.state_dir(gid, skey.split("_", 1)[1])
    # provenance: which basin outline (and table version) these rows were
    # simulated on - box-era records lack it
    try:
        rec = json.load(open(rec_path))
        rec["extent"] = basinextent.provenance(ext)
        rec["cut_ids"] = cuts
        rec["lakes"] = lakes_prov                      # lake set (None: lake module off)
        with open(rec_path, "w") as fh:
            json.dump(rec, fh)
    except (OSError, ValueError):
        pass
    pre = _prefix(gid)
    # bundle to a temp FILE: streamed pack, and the compressed bundle of a big
    # basin is never held in RAM (statebundle.pack_dir)
    fd, bpath = tempfile.mkstemp(prefix=f"fleet_{skey}_", suffix=".pqf")
    os.close(fd)
    try:
        if statebundle.pack_dir(sdir, out_path=bpath) is None:
            return {"gauge": gid, "ok": False, "rc": -1, "last": "no state grids"}
        mb = os.path.getsize(bpath) / 1e6
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_commit(repo_id=FLEET_REPO, repo_type="dataset",
                          operations=[
                              CommitOperationAdd(f"{pre}states/{skey}.pqf", bpath),
                              CommitOperationAdd(f"{pre}results/{key}.json", rec_path)],
                          commit_message=f"fleet {pre}{key}")
    finally:
        try:
            os.remove(bpath)
        except OSError:
            pass
    n_states = len(json.load(open(rec_path)).get("state_times", []))
    n_rows = len(json.load(open(rec_path)).get("rows", []))

    # bound local disk: everything just uploaded is re-fetchable
    shutil.rmtree(os.path.join(statecache.CACHE_DIR, "states", str(gid).zfill(8)),
                  ignore_errors=True)
    for p in glob.glob(os.path.join(statecache.CACHE_DIR, "results",
                                    f"{str(gid).zfill(8)}_*.json")):
        os.remove(p)
    return {"gauge": gid, "ok": True, "key": key, "rows": n_rows,
            "states": n_states, "mb": round(mb, 1),
            "min": round((time.time() - t_start) / 60, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=DEF_START)
    ap.add_argument("--end", default=DEF_END)
    ap.add_argument("--chunk-days", type=int, default=10,
                    help="state-save interval (matches CREST_STATE_TOL_DAYS x2)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gauges", default="all",
                    help='"all" (USGS gauges + HydroBASINS pour points), comma list, '
                         "or path to a file of outlet ids")
    ap.add_argument("--size", default=None, choices=("small", "big", "all"),
                    help=f"basins < FLEET_BIG_KM2 ({BIG_KM2:,.0f} km2: the HF runners), "
                         ">= it (the Jetstream2 fleet), or all. Default: FLEET_SIZE, "
                         "else small for --gauges all and all for an explicit list")
    ap.add_argument("--limit", type=int, default=0, help="stop after N gauges (0=all)")
    ap.add_argument("--reverse", action="store_true",
                    help="walk the catalog backwards — lets two runners (Space + "
                         "server) work from opposite ends without collisions")
    ap.add_argument("--shard", default="",
                    help='"K/N": run only gauges with catalog-index %% N == K — '
                         "N parallel runners split the catalog disjointly")
    args = ap.parse_args()
    if args.size is None:
        # an explicit list (e.g. a pilot) runs as given; the full roster takes
        # the HF runners' class unless FLEET_SIZE says otherwise
        args.size = os.environ.get("FLEET_SIZE", "").strip() or \
            ("small" if args.gauges == "all" else "all")
    if args.size not in ("small", "big", "all"):
        sys.exit(f"FLEET_SIZE must be small|big|all, got {args.size!r}")

    if not os.path.exists(os.path.join("EF5", "bin", "ef5")):
        sys.exit("run from the repo root: ./EF5/bin/ef5 not found (see README_FLEET.md)")
    _token()
    os.environ.setdefault("CREST_DEMO_MOCK", "0")

    from huggingface_hub import HfApi
    HfApi(token=_token()).create_repo(FLEET_REPO, repo_type="dataset",
                                      private=True, exist_ok=True)

    # thousands of gauges doing anonymous /vsicurl range-reads is exactly what
    # HF rate-limits (fleet-wide "RasterioIOError: Read failed" storm) — fetch
    # the three CONUS terrain COGs once so every worker clips a local copy
    from hf_data import basic
    print("prefetching CONUS terrain COGs (~7 GB, one-time per container)...",
          flush=True)
    basic.prefetch()

    def _norm(s):
        s = s.strip()
        return s if s.upper().startswith("V") else s.zfill(8)

    if args.gauges == "all":
        # gauges in catalog order (their shard membership never changes), then
        # the pour points - seamless coverage of rivers without a gauge
        from hf_data import gauges as G, virtualpoints
        cat = G.load_catalog()
        ids = [str(s).zfill(8) for s in cat["STAID"].tolist()]
        vp = virtualpoints._load()
        if vp is None:
            sys.exit("virtual-point catalog unavailable")
        ids += [str(v) for v in vp["vp"].tolist()]
    elif os.path.exists(args.gauges):
        ids = [_norm(ln) for ln in open(args.gauges) if ln.strip()]
    else:
        ids = [_norm(s) for s in args.gauges.split(",") if s.strip()]

    if args.shard:
        k, n = map(int, args.shard.split("/"))
        ids = [g for i, g in enumerate(ids) if i % n == k]
        print(f"shard {k}/{n}: {len(ids)} outlets in this slice", flush=True)

    runnable, skipped = select_outlets(ids, args.size)
    if skipped:
        print(f"no basin boundary (not simulated, never boxed): {dict(skipped)}", flush=True)
    print(f"size class {args.size} (split at {BIG_KM2:,.0f} km2): "
          f"{len(runnable)} of {len(ids) - sum(skipped.values())} outlets", flush=True)
    if args.gauges != "all" and len(runnable) < len(ids) - sum(skipped.values()):
        out = sorted(set(ids) - set(runnable))
        print(f"outside size class {args.size}: {', '.join(out[:40])}"
              f"{' ...' if len(out) > 40 else ''}", flush=True)
    ids = runnable
    done = _done_keys()
    todo = [g for g in ids if not any(k.startswith(g + "_") for k in done)]
    if args.reverse:
        todo.reverse()
    if args.limit:
        todo = todo[:args.limit]
    print(f"fleet: {len(ids)} requested, {len(ids) - len(todo)} already done, "
          f"{len(todo)} to run | {args.start}..{args.end}, state every "
          f"{args.chunk_days} d, {args.workers} workers", flush=True)

    t0 = datetime.fromisoformat(args.start)
    t1 = datetime.fromisoformat(args.end)
    ok = fail = 0
    # spawn, not fork: the parent's hf-xet transfer runtime (its threads and
    # locks, started by the COG prefetch) must not be copied into workers - a
    # forked worker deadlocked on it (futex) on the Jetstream2 VM, 2026-10-27
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool, \
            open(PROGRESS, "a") as prog:
        futs = {pool.submit(run_one_gauge, g, t0, t1, args.chunk_days): g
                for g in todo}
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except Exception as e:
                r = {"gauge": futs[fu], "ok": False, "rc": -1,
                     "last": f"{type(e).__name__}: {e}"}
            r["when"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            prog.write(json.dumps(r) + "\n")
            prog.flush()
            ok += r["ok"]; fail += (not r["ok"])
            print(("OK " if r["ok"] else "FAIL ") + json.dumps(r), flush=True)
    print(f"fleet pass complete: {ok} ok, {fail} failed "
          f"(rerun to retry failures — done gauges are skipped)", flush=True)


if __name__ == "__main__":
    main()
