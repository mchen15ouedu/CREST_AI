"""Stage 1: pull the operational nowcast archive + choose eval gauges.

Downloads every nowcast/archive parquet from vincewin/CREST_data, records
per-issue metadata (which model file served it), builds a per-gauge obs-age
profile over the window, and picks an out-of-sample eval set (~500 gauges not
in gauges_v2.txt) stratified so the stale/dead regimes are represented.

Outputs (in this dir):
  archive/           mirrored nowcast/archive tree
  issues.csv         t0, model_file, model12_file per issue
  gauge_profile.csv  per-gauge obs-age stats across issues
  eval_gauges.txt    selected gauge ids
"""
import os, glob, collections
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

BASE = os.path.dirname(os.path.abspath(__file__))
TOK = open(os.path.expanduser("~/huggingface.txt")).read().strip()
TRAIN_GAUGES = "/home/MengyuChen/CREST_AI/nowcast_space/gauges_v2.txt"
T0_MAX = "2026081200"          # cap: 12-h leads need truth obs through t0+12

print("downloading archive ...", flush=True)
snapshot_download("vincewin/CREST_data", repo_type="dataset", token=TOK,
                  allow_patterns=["nowcast/archive/*"],
                  local_dir=os.path.join(BASE, "archive"))

files = sorted(glob.glob(os.path.join(BASE, "archive", "nowcast/archive/*/nc_*.parquet")))
print(f"{len(files)} archive files", flush=True)

rows, prof = [], collections.defaultdict(lambda: {"n": 0, "fresh": 0, "stale": 0,
                                                  "dead": 0, "age_max": 0.0})
for i, f in enumerate(files):
    t0 = os.path.basename(f)[3:13]
    if t0 > T0_MAX:
        continue
    pf = pq.ParquetFile(f)
    md = {k.decode(): v.decode() for k, v in (pf.schema_arrow.metadata or {}).items()
          if not k.startswith(b"ARROW")}
    rows.append({"t0": t0, "model_file": md.get("model_file", "?"),
                 "model12_file": md.get("model12_file", "?"),
                 "path": os.path.relpath(f, BASE)})
    t = pf.read(["gid", "obs_age_h"])
    gids = t.column("gid").to_pylist()
    age = t.column("obs_age_h").to_numpy()
    for g, a in zip(gids, age):
        p = prof[g]
        p["n"] += 1
        if a <= 6:
            p["fresh"] += 1
        elif a <= 240:
            p["stale"] += 1
        else:
            p["dead"] += 1
        p["age_max"] = max(p["age_max"], float(a))
    if i % 100 == 0:
        print(f"  scanned {i}", flush=True)

issues = pd.DataFrame(rows).sort_values("t0")
issues.to_csv(os.path.join(BASE, "issues.csv"), index=False)
print(issues["model_file"].value_counts(), flush=True)
print(issues["model12_file"].value_counts(), flush=True)

gp = pd.DataFrame([{"gid": g, **v} for g, v in prof.items()])
gp["frac_stale"] = (gp["stale"] + gp["dead"]) / gp["n"]
gp.to_csv(os.path.join(BASE, "gauge_profile.csv"), index=False)

train = set(x.strip() for x in open(TRAIN_GAUGES) if x.strip())
pool = gp[~gp["gid"].isin(train)].copy()
print(f"pool: {len(pool)} gauges not in training set", flush=True)

rng = np.random.default_rng(42)
# mostly-fresh gauges: random sample; stale/dead-experiencing gauges: take more
always_fresh = pool[pool["frac_stale"] <= 0.02]
sometimes_stale = pool[(pool["frac_stale"] > 0.02) & (pool["frac_stale"] <= 0.8)]
mostly_dead = pool[pool["frac_stale"] > 0.8]
pick = []
for df, k in ((always_fresh, 300), (sometimes_stale, 150), (mostly_dead, 100)):
    take = df["gid"].to_list()
    if len(take) > k:
        take = list(rng.choice(take, k, replace=False))
    pick.extend(take)
    print(f"  bucket size {len(df)} -> picked {len(take)}", flush=True)
with open(os.path.join(BASE, "eval_gauges.txt"), "w") as f:
    f.write("\n".join(sorted(pick)) + "\n")
print(f"eval set: {len(pick)} gauges", flush=True)
