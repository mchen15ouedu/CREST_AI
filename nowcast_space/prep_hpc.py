"""Data prep on the HPC (CPU-only) — same resumable prep the Space runs,
as a CLI. Fills the private dataset repo (vincewin/CREST_nowcast_data) with
obs/<gid>/<YYYY_MM>.parquet and mrms/<gid>/<YYYY_MM>.parquet; months and
gauges already present are skipped, so re-running after a crash is safe.

Each month downloads one CONUS MRMS tar (~GB-scale) from vincewin/CREST_data,
extracts every gauge's basin mean in one pass, then deletes it — point
HF_HOME at scratch, not your small home quota.

    export HF_TOKEN=$(cat ~/.hf_token)
    python prep_hpc.py --gauges "01011000, 08167000" --months 2023_01-2024_12
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from train_hpc import DEFAULT_GAUGES, DEFAULT_MONTHS, gauge_meta, months_range
import data as D


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gauges", default=DEFAULT_GAUGES)
    ap.add_argument("--months", default=DEFAULT_MONTHS,
                    help="YYYY_MM-YYYY_MM (remember the 2025 val months too)")
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN env var not set")

    gauges = gauge_meta(args.gauges)
    if not gauges:
        sys.exit("no valid gauges")
    months = months_range(args.months)
    print(f"prep: {[g['id'] for g in gauges]} × {len(months)} months")
    failed = []
    for ym in months:
        y, m = map(int, ym.split("_"))
        try:
            rep = D.prep_month(gauges, y, m)
            print(f"{ym}: +{rep['obs_added']} obs, +{rep['mrms_added']} mrms "
                  f"({rep['skipped']} already present)", flush=True)
        except Exception as e:
            failed.append(ym)
            print(f"{ym}: FAILED {type(e).__name__}: {e}", flush=True)
    if failed:
        sys.exit(f"prep incomplete — re-run for: {', '.join(failed)}")
    print("prep DONE")


if __name__ == "__main__":
    main()
