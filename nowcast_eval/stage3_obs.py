"""Stage 3: post-hoc USGS truth for the eval gauges.

Batched NWIS instantaneous-values fetch for Jul 18 - Aug 12 2026 (72 h of
lookback before the first archived issue, through the last scored lead).
Post-hoc data serves two roles: (a) truth at t0+k, (b) the replay obs stream,
truncated per issue at the archived obs_last_time so the model sees exactly
the staleness the operational run saw.

Output: obs_rows.parquet (gid, dt, q_cms) — raw event list, sorted.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
CFS_TO_CMS = 0.0283168
START = "2026-07-18T00:00Z"
END = "2026-08-12T15:00Z"

gids = [x.strip() for x in open(os.path.join(BASE, "eval_gauges.txt")) if x.strip()]
print(f"{len(gids)} gauges", flush=True)


def fetch(sites):
    out = []
    for attempt in range(3):
        try:
            r = requests.get("https://waterservices.usgs.gov/nwis/iv/",
                             params={"sites": ",".join(sites), "parameterCd": "00060",
                                     "format": "json", "siteStatus": "all",
                                     "startDT": START, "endDT": END},
                             timeout=120)
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
            return out
        except Exception as e:
            print(f"  chunk retry {attempt}: {e}", flush=True)
    return out


chunks = [gids[i:i + 40] for i in range(0, len(gids), 40)]
rows = []
with ThreadPoolExecutor(max_workers=6) as ex:
    for i, got in enumerate(ex.map(fetch, chunks)):
        rows.extend(got)
        print(f"  chunk {i + 1}/{len(chunks)}: total rows {len(rows)}", flush=True)

df = pd.DataFrame(rows, columns=["gid", "dt", "q_cms"]).sort_values(["gid", "dt"])
df = df.drop_duplicates(["gid", "dt"])
df.to_parquet(os.path.join(BASE, "obs_rows.parquet"), index=False)
print(f"saved {len(df)} rows, {df['gid'].nunique()} gauges with data", flush=True)
