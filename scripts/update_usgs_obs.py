"""Proactively refresh + store USGS observed-discharge so runs don't fetch it live.

The app already caches USGS instantaneous discharge coverage-aware in
CACHE_DIR/obs/<site>.parquet (hf_data/obs.get_series) and syncs it to the private
CREST_state dataset (hf_data/persist), but that store only fills LAZILY — a gauge
is fetched the first time someone runs it, and only its last CREST_OBS_LAG_H
hours refresh afterwards. This job keeps the store WARM: it pulls the CREST_state
obs snapshot, extends every tracked gauge's series up to now over a rolling
window, and pushes the updated parquet back — so at run time the Space serves
observations from the store and only ever hits NWIS for the unavoidable
provisional tail (the most recent LAG_H hours, which are still being revised).

Gauge set (union):
  * every gauge already in the obs store (keeps prior runs' gauges fresh), plus
  * --gauges 08144500,07103700         explicit ids, and/or
  * --catalog-bbox W,S,E,N             all catalog gauges in a bbox, and/or
  * --gauges-file path                 one id per line.
New gauges enter the store naturally the first time a user runs them; seed extra
ones here if you want them pre-warmed.

Reuses obs.get_series verbatim (same coverage bookkeeping, same provisional-tail
policy), so nothing about how the app reads obs changes. Needs the HF token (read
+ write to the private CREST_state repo).

Run:
    python scripts/update_usgs_obs.py [--back-days 45] [--gauges ...]
                                      [--catalog-bbox W,S,E,N] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import truststore
truststore.inject_into_ssl()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forcing_update_common import hf_token as _token  # noqa: E402

STATE_REPO = os.environ.get("CREST_STATE_REPO", "vincewin/CREST_state")


def _restore_obs(api, cache_dir: str) -> list[str]:
    """Pull the CREST_state obs/ snapshot into cache_dir/obs; return stored ids."""
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(STATE_REPO, repo_type="dataset", local_dir=cache_dir,
                          allow_patterns=["obs/**"], token=api.token)
    except Exception as e:
        print(f"  [obs] restore note: {e}")
    obs_dir = os.path.join(cache_dir, "obs")
    return sorted(os.path.basename(p)[:-8]           # strip ".parquet"
                  for p in glob.glob(os.path.join(obs_dir, "*.parquet")))


def _target_gauges(stored: list[str], args) -> list[str]:
    ids = set(stored)
    if args.gauges:
        ids.update(g.strip().zfill(8) for g in args.gauges.split(",") if g.strip())
    if args.gauges_file and os.path.exists(args.gauges_file):
        with open(args.gauges_file) as f:
            ids.update(ln.strip().zfill(8) for ln in f if ln.strip())
    if args.catalog_bbox:
        try:
            from hf_data import gauges
            bbox = tuple(float(x) for x in args.catalog_bbox.split(","))
            cat = gauges.load_catalog(bbox)
            col = next((c for c in ("STAID", "staid", "id", "site") if c in cat.columns), None)
            if col:
                ids.update(str(s).zfill(8) for s in cat[col].tolist())
        except Exception as e:
            print(f"  [obs] catalog-bbox note: {e}")
    return sorted(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--back-days", type=int, default=45,
                    help="rolling window refreshed up to now (default 45)")
    ap.add_argument("--gauges", help="comma list of site ids to also warm")
    ap.add_argument("--gauges-file", help="file with one site id per line")
    ap.add_argument("--catalog-bbox", help="W,S,E,N — warm all catalog gauges inside")
    ap.add_argument("--dry-run", action="store_true", help="fetch but don't upload")
    args = ap.parse_args()

    tok = _token()
    if not tok:
        print("no HF token — cannot reach the private CREST_state store; aborting")
        return 1
    from huggingface_hub import HfApi
    api = HfApi(token=tok)

    # a dedicated working cache so obs.get_series reads/writes the restored store
    cache_dir = os.path.join(tempfile.gettempdir(), "crest_obs_sync")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["CREST_CACHE_DIR"] = cache_dir        # BEFORE importing statecache
    os.environ["HF_TOKEN"] = tok

    stored = _restore_obs(api, cache_dir)
    gauges = _target_gauges(stored, args)
    print(f"obs refresh @ {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC — "
          f"{len(stored)} in store, {len(gauges)} to refresh, window {args.back_days} d")
    if not gauges:
        print("no gauges to refresh (store empty and no --gauges/--catalog-bbox seed)")
        return 0

    from hf_data import obs
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    t0 = now - timedelta(days=args.back_days)
    fetched_total, updated, errors, latest = 0, 0, 0, None
    for i, site in enumerate(gauges, 1):
        info: dict = {}
        try:
            series = obs.get_series(site, t0, now, info=info)
        except Exception as e:
            errors += 1
            print(f"  {site}: ERROR {e}")
            continue
        if info.get("fetch_error"):
            errors += 1
        n = info.get("fetched_windows", 0)
        fetched_total += n
        if n:
            updated += 1
        if series and (latest is None or series[-1][0] > latest):
            latest = series[-1][0]
        if i % 25 == 0:
            print(f"  ... {i}/{len(gauges)} processed")

    lag = f"{(now - latest).total_seconds()/3600:.1f} h" if latest else "n/a"
    print(f"fetched {fetched_total} NWIS window(s) across {updated} gauge(s); "
          f"newest obs {latest} UTC (lag {lag}); {errors} error(s)")

    if args.dry_run:
        print("dry-run: not uploading obs store")
        return 0
    if updated == 0:
        print("no store changes to upload")
        return 0
    api.upload_folder(repo_id=STATE_REPO, repo_type="dataset", folder_path=cache_dir,
                      allow_patterns=["obs/*.parquet"],
                      commit_message=f"obs: refresh {updated} gauge(s) up to "
                                     f"{now:%Y-%m-%d %H:%M} UTC")
    print(f"uploaded refreshed obs for {updated} gauge(s) to {STATE_REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
