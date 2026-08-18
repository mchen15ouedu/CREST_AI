"""Stage 3 (all-gauge): post-hoc USGS truth, robust to NWIS 503 throttling.

Same request as the 433-gauge eval (IV 00060, Jul 18 - Aug 12 2026), but:
2 workers, per-chunk exponential backoff, retries until every chunk has
succeeded (an HTTP 200 with zero series is a legitimate empty chunk),
per-chunk results cached in obs_chunks/ so a rerun resumes.
"""
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
CFS_TO_CMS = 0.0283168
START = "2026-07-18T00:00Z"
END = "2026-08-12T15:00Z"
CDIR = os.path.join(BASE, "obs_chunks")
os.makedirs(CDIR, exist_ok=True)

gids = [x.strip() for x in open(os.path.join(BASE, "eval_gauges.txt")) if x.strip()]
print(f"{len(gids)} gauges", flush=True)


def fetch(ci_sites):
    ci, sites = ci_sites
    cp = os.path.join(CDIR, f"chunk_{ci:04d}.pkl")
    if os.path.exists(cp):
        return pickle.load(open(cp, "rb"))
    out = []
    delay = 5.0
    for attempt in range(40):
        try:
            r = requests.get("https://waterservices.usgs.gov/nwis/iv/",
                             params={"sites": ",".join(sites), "parameterCd": "00060",
                                     "format": "json", "siteStatus": "all",
                                     "startDT": START, "endDT": END},
                             timeout=180)
            r.raise_for_status()
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
            print(f"  chunk {ci} retry {attempt} ({str(e)[:60]}) sleep {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 1.6, 120.0)
    raise RuntimeError(f"chunk {ci} failed after 40 attempts")


chunks = [(i, gids[j:j + 40]) for i, j in enumerate(range(0, len(gids), 40))]
rows = []
with ThreadPoolExecutor(max_workers=2) as ex:
    for i, got in enumerate(ex.map(fetch, chunks)):
        rows.extend(got)
        if i % 10 == 0 or i == len(chunks) - 1:
            print(f"  chunk {i + 1}/{len(chunks)}: total rows {len(rows)}", flush=True)

df = pd.DataFrame(rows, columns=["gid", "dt", "q_cms"]).sort_values(["gid", "dt"])
df = df.drop_duplicates(["gid", "dt"])
df.to_parquet(os.path.join(BASE, "obs_rows.parquet"), index=False)
print(f"saved {len(df)} rows, {df['gid'].nunique()} gauges with data", flush=True)
