"""One-time prep: IANA time zone for every GAGES-II gauge (timezonefinder).

Writes gauges/gauge_tz.parquet (gid -> tz, e.g. "America/Chicago") to
vincewin/CREST_data. The dashboard shows hydrograph times in the GAUGE'S
local zone (frontend formats UTC stamps via Intl with this tz — DST handled
by the browser's own tz database), with a UTC toggle for emergency users.

    python scripts/prep_gauge_tz.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import truststore
truststore.inject_into_ssl()

from forcing_update_common import HF_REPO, hf_token                             # noqa: E402

TZ_PATH = "gauges/gauge_tz.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    token = hf_token()

    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    from timezonefinder import TimezoneFinder
    cat = pq.read_table(hf_hub_download(HF_REPO, "gauges/gagesII_9322.parquet",
                                        repo_type="dataset", token=token))
    gid = [str(s).zfill(8) for s in cat.column("STAID").to_pylist()]
    lat = cat.column("LAT_GAGE").to_numpy().astype("float64")
    lon = cat.column("LNG_GAGE").to_numpy().astype("float64")

    tf = TimezoneFinder(in_memory=True)
    tzs = []
    for la, lo in zip(lat, lon):
        tz = tf.timezone_at(lat=float(la), lng=float(lo))
        tzs.append(tz or "UTC")
    from collections import Counter
    top = Counter(tzs).most_common(6)
    print(f"{len(gid):,} gauges ->", ", ".join(f"{t} x{c}" for t, c in top))

    tbl = pa.table({"gid": gid, "tz": tzs})
    if args.dry_run:
        print("[dry-run: no upload]")
        return 0
    tmp = os.path.join(tempfile.mkdtemp(), "gauge_tz.parquet")
    pq.write_table(tbl, tmp, compression="zstd")
    HfApi(token=token).create_commit(
        repo_id=HF_REPO, repo_type="dataset",
        operations=[CommitOperationAdd(TZ_PATH, tmp)],
        commit_message=f"gauge time zones ({len(gid)} gauges, timezonefinder)")
    print("uploaded", TZ_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
