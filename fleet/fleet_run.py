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

Run from the repo root (CWD must contain ./EF5/bin/ef5):
    python fleet/fleet_run.py --workers 8
    python fleet/fleet_run.py --gauges "08167000, 08144500"     # subset
See fleet/README_FLEET.md for server setup.
"""
from __future__ import annotations

import argparse
import glob
import json
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


def _token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("HF_TOKEN env var not set")
    return tok


def _done_keys() -> set[str]:
    """gauge keys already fully uploaded (results + states present)."""
    from huggingface_hub import HfApi
    try:
        files = set(HfApi(token=_token()).list_repo_files(FLEET_REPO, repo_type="dataset"))
    except Exception:
        return set()
    res = {f[len("results/"):-len(".json")] for f in files if f.startswith("results/")}
    st = {f[len("states/"):-len(".pqf")] for f in files if f.startswith("states/")}
    return res & st


def run_one_gauge(gid: str, t0: datetime, t1: datetime, chunk_days: int) -> dict:
    """Child process: chunked run -> bundle -> upload -> local cleanup."""
    from hf_data import pipeline, statecache, statebundle
    from huggingface_hub import HfApi, CommitOperationAdd

    t_start = time.time()
    last_status = ""
    cur = t0
    while cur < t1:
        ce = min(cur + timedelta(days=chunk_days), t1)
        rc = None
        for kind, payload in pipeline.run_gauge(gid, cur, ce, use_mock=False,
                                                grids=False, scheme="speed"):
            if kind == "status":
                last_status = payload
            elif kind == "done":
                rc = payload.get("returncode")
        if rc != 0:
            return {"gauge": gid, "ok": False, "at": cur.strftime("%Y-%m-%d"),
                    "rc": rc, "last": last_status[-300:]}
        cur = ce

    # the run resolved model/scheme itself — find the record it wrote
    recs = sorted(glob.glob(os.path.join(statecache.CACHE_DIR, "results",
                                         f"{str(gid).zfill(8)}_*.json")),
                  key=os.path.getmtime)
    if not recs:
        return {"gauge": gid, "ok": False, "rc": -1, "last": "no result record"}
    rec_path = recs[-1]
    key = os.path.splitext(os.path.basename(rec_path))[0]
    model = key.split("_", 1)[1]

    sdir = statecache.state_dir(gid, model)
    blob = statebundle.pack_dir(sdir)
    if blob is None:
        return {"gauge": gid, "ok": False, "rc": -1, "last": "no state grids"}
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_commit(repo_id=FLEET_REPO, repo_type="dataset",
                      operations=[
                          CommitOperationAdd(f"states/{key}.pqf", blob),
                          CommitOperationAdd(f"results/{key}.json", rec_path)],
                      commit_message=f"fleet {key}")
    n_states = len(json.load(open(rec_path)).get("state_times", []))
    n_rows = len(json.load(open(rec_path)).get("rows", []))

    # bound local disk: everything just uploaded is re-fetchable
    shutil.rmtree(os.path.join(statecache.CACHE_DIR, "states", str(gid).zfill(8)),
                  ignore_errors=True)
    for p in glob.glob(os.path.join(statecache.CACHE_DIR, "results",
                                    f"{str(gid).zfill(8)}_*.json")):
        os.remove(p)
    return {"gauge": gid, "ok": True, "key": key, "rows": n_rows,
            "states": n_states, "mb": round(len(blob) / 1e6, 1),
            "min": round((time.time() - t_start) / 60, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=DEF_START)
    ap.add_argument("--end", default=DEF_END)
    ap.add_argument("--chunk-days", type=int, default=10,
                    help="state-save interval (matches CREST_STATE_TOL_DAYS x2)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gauges", default="all",
                    help='"all", comma list, or path to a file of gauge ids')
    ap.add_argument("--limit", type=int, default=0, help="stop after N gauges (0=all)")
    ap.add_argument("--reverse", action="store_true",
                    help="walk the catalog backwards — lets two runners (Space + "
                         "server) work from opposite ends without collisions")
    args = ap.parse_args()

    if not os.path.exists(os.path.join("EF5", "bin", "ef5")):
        sys.exit("run from the repo root: ./EF5/bin/ef5 not found (see README_FLEET.md)")
    _token()
    os.environ.setdefault("CREST_DEMO_MOCK", "0")

    from huggingface_hub import HfApi
    HfApi(token=_token()).create_repo(FLEET_REPO, repo_type="dataset",
                                      private=True, exist_ok=True)

    if args.gauges == "all":
        from hf_data import gauges as G
        cat = G.load_catalog()
        ids = [str(s).zfill(8) for s in cat["STAID"].tolist()]
    elif os.path.exists(args.gauges):
        ids = [ln.strip().zfill(8) for ln in open(args.gauges) if ln.strip()]
    else:
        ids = [s.strip().zfill(8) for s in args.gauges.split(",") if s.strip()]

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
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
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
