"""NWIS IV discharge for all served gauges, one file per eval month.

Window per month: month start - 4 d (lookback + slack) .. month end + 1 d
(12-h leads). Per-(month, chunk) pickle cache; exponential backoff on 503;
never gives up on a chunk (an HTTP 200 with zero series is a real empty
chunk). Output obs/obs_<ym>.parquet (gid, dt[UTC naive], q_cms).
usage: my_obs.py [ym ...]   (default: all MONTHS)
"""
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from common import BASE, MONTHS, gauges

CFS_TO_CMS = 0.0283168
gids, _, _, _ = gauges()
CHUNKS = [gids[j:j + 40] for j in range(0, len(gids), 40)]
CDIR = os.path.join(BASE, "obs_chunks")


def fetch(job):
    ym, ci, sites = job
    cp = os.path.join(CDIR, f"{ym}_chunk_{ci:04d}.pkl")
    if os.path.exists(cp):
        return pickle.load(open(cp, "rb"))
    y, m = int(ym[:4]), int(ym[4:])
    start = datetime(y, m, 1) - timedelta(days=4)
    end = datetime(y + (m == 12), m % 12 + 1, 1) + timedelta(days=1)
    delay = 5.0
    for attempt in range(60):
        try:
            r = requests.get("https://waterservices.usgs.gov/nwis/iv/",
                             params={"sites": ",".join(sites), "parameterCd": "00060",
                                     "format": "json", "siteStatus": "all",
                                     "startDT": start.strftime("%Y-%m-%dT%H:%MZ"),
                                     "endDT": end.strftime("%Y-%m-%dT%H:%MZ")},
                             timeout=300)
            if r.status_code == 404:          # no data for any site in range
                out = []
            else:
                r.raise_for_status()
                out = []
                for ts in r.json().get("value", {}).get("timeSeries", []):
                    sid = ts["sourceInfo"]["siteCode"][0]["value"].zfill(8)
                    for v in ts["values"][0]["value"]:
                        try:
                            cfs = float(v["value"])
                        except (TypeError, ValueError):
                            continue
                        if cfs < 0:
                            continue
                        dt = (datetime.fromisoformat(v["dateTime"].replace("Z", "+00:00"))
                              .astimezone(timezone.utc).replace(tzinfo=None))
                        out.append((sid, dt, cfs * CFS_TO_CMS))
            pickle.dump(out, open(cp, "wb"))
            return out
        except Exception as e:
            print(f"  {ym} chunk {ci} retry {attempt} ({str(e)[:50]}) sleep {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 1.6, 180.0)
    raise RuntimeError(f"{ym} chunk {ci} failed")


months = sys.argv[1:] or MONTHS
for ym in months:
    outp = os.path.join(BASE, "obs", f"obs_{ym}.parquet")
    if os.path.exists(outp):
        continue
    t = time.time()
    jobs = [(ym, ci, s) for ci, s in enumerate(CHUNKS)]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for got in ex.map(fetch, jobs):
            rows.extend(got)
    df = pd.DataFrame(rows, columns=["gid", "dt", "q_cms"]).sort_values(["gid", "dt"])
    df = df.drop_duplicates(["gid", "dt"])
    df.to_parquet(outp, index=False)
    print(f"{ym}: {len(df)} rows, {df['gid'].nunique()} gauges, {time.time() - t:.0f}s", flush=True)
print("obs done", flush=True)
