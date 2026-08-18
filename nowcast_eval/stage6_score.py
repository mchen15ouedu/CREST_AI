"""Stage 6: score archived v1 + replayed v1/v2/v3 against post-hoc truth.

Main comparison runs on the v1-era issues (t0 <= 2026-08-11 00Z, 487 issues)
so archived-operational v1 and the three replayed models are scored on
identical (issue, gauge, lead) samples. The v2-era tail (24 issues) is used
only as a second replay-fidelity check (archived v2 vs replayed v2).

Outputs:
  eval_long_6h.parquet / eval_long_12h.parquet   sample-level table
  nse_by_lead.csv        pooled + per-gauge-median NSE, model x lead
  nse_by_regime.csv      pooled NSE, model x obs-age regime x lead group
  pergauge_nse.csv       per-gauge NSE + lat/lon (for CDFs + CONUS map)
  fidelity.txt           v1/v2 replay-vs-archive agreement
"""
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
V1_ERA_MAX = "2026081100"

rp = np.load(os.path.join(BASE, "replay_preds.npz"), allow_pickle=True)
ae = np.load(os.path.join(BASE, "archive_eval.npz"), allow_pickle=True)
gids = [str(g) for g in rp["gids"]]
t0s = [str(t) for t in rp["t0"]]
nG, nI = len(gids), len(t0s)
t0_dt = pd.to_datetime(pd.Series(t0s), format="%Y%m%d%H").values.astype("datetime64[s]")

# ---- truth: hourly series (obs within the trailing hour) ---------------------
od = pd.read_parquet(os.path.join(BASE, "obs_rows.parquet"))
truth = {}
for g, sub in od.groupby("gid"):
    T = sub["dt"].to_numpy().astype("datetime64[s]")
    V = sub["q_cms"].to_numpy().astype("float64")
    truth[g] = (T, V)


def truth_at(g, when):                       # when: (n,) datetime64[s]
    tv = truth.get(g)
    if tv is None:
        return np.full(len(when), np.nan)
    T, V = tv
    idx = np.searchsorted(T, when, side="right") - 1
    ok = idx >= 0
    ic = np.clip(idx, 0, None)
    fresh = ok & ((when - T[ic]) <= np.timedelta64(3600, "s"))
    return np.where(fresh, V[ic], np.nan)


# ---- assemble sample-level tables --------------------------------------------
v1_era = np.array([t <= V1_ERA_MAX for t in t0s])
age = ae["obs_age"]                                             # (nI, nG)
regime = np.where(age <= 6, "fresh", np.where(age <= 240, "stale", "dead"))

# gauge lat/lon for the map
arch0 = sorted(glob.glob(os.path.join(BASE, "archive/nowcast/archive/*/nc_*.parquet")))[0]
t = pq.read_table(arch0, columns=["gid", "lat", "lon"])
ll = {str(g).zfill(8): (la, lo) for g, la, lo in zip(
    t.column("gid").to_pylist(), t.column("lat").to_numpy(), t.column("lon").to_numpy())}


def build_long(hor, arch_key, models):
    rows = []
    for k in range(hor):                       # lead k+1
        when = t0_dt + np.timedelta64(3600 * (k + 1), "s")     # (nI,)
        for gi, g in enumerate(gids):
            y = truth_at(g, when)
            ok = np.isfinite(y) & v1_era
            if not ok.any():
                continue
            jj = np.where(ok)[0]
            d = {"gid": g, "t0": np.array(t0s, dtype=object)[jj],
                 "lead": k + 1, "truth": y[jj],
                 "obs_age": age[jj, gi], "regime": regime[jj, gi],
                 "arch": ae[arch_key][jj, gi, k],
                 "persist": rp["persist"][jj, gi]}
            for m in models:
                d[m] = rp[m][jj, gi, k]
            rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


print("building 6h table ...", flush=True)
L6 = build_long(6, "q6", ["v1_6", "v2_6", "v3_6", "v3f_6"])
print(f"  {len(L6)} samples", flush=True)
L6.to_parquet(os.path.join(BASE, "eval_long_6h.parquet"), index=False)
print("building 12h table ...", flush=True)
L12 = build_long(12, "q12", ["v1_12", "v2_12", "v3_12", "v3f_12"])
print(f"  {len(L12)} samples", flush=True)
L12.to_parquet(os.path.join(BASE, "eval_long_12h.parquet"), index=False)


def nse(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    if m.sum() < 30:
        return np.nan
    p, o = p[m], o[m]
    den = ((o - o.mean()) ** 2).sum()
    if den <= 0:
        return np.nan
    return 1.0 - ((p - o) ** 2).sum() / den


# ---- pooled + per-gauge-median NSE by lead -----------------------------------
recs = []
for tbl, fam, models in ((L6, "6h", ["arch", "persist", "v1_6", "v2_6", "v3_6", "v3f_6"]),
                         (L12, "12h", ["arch", "persist", "v1_12", "v2_12", "v3_12", "v3f_12"])):
    for m in models:
        for lead, sub in tbl.groupby("lead"):
            gm = sub.groupby("gid").apply(
                lambda s: nse(s[m].to_numpy(), s["truth"].to_numpy()),
                include_groups=False)
            recs.append({"family": fam, "model": m, "lead": lead,
                         "pooled_nse": nse(sub[m].to_numpy(), sub["truth"].to_numpy()),
                         "median_gauge_nse": float(np.nanmedian(gm)),
                         "n": len(sub)})
bl = pd.DataFrame(recs)
bl.to_csv(os.path.join(BASE, "nse_by_lead.csv"), index=False)
print(bl.pivot_table(index=["family", "lead"], columns="model",
                     values="median_gauge_nse").round(3), flush=True)

# ---- obs-age regime stratification -------------------------------------------
recs = []
for tbl, fam, models in ((L6, "6h", ["arch", "persist", "v1_6", "v2_6", "v3_6", "v3f_6"]),
                         (L12, "12h", ["arch", "persist", "v1_12", "v2_12", "v3_12", "v3f_12"])):
    for m in models:
        for reg, sub in tbl.groupby("regime"):
            recs.append({"family": fam, "model": m, "regime": reg,
                         "pooled_nse": nse(sub[m].to_numpy(), sub["truth"].to_numpy()),
                         "n": len(sub)})
pd.DataFrame(recs).to_csv(os.path.join(BASE, "nse_by_regime.csv"), index=False)

# ---- per-gauge NSE (all leads pooled) for CDF + map --------------------------
recs = []
for tbl, fam, models in ((L6, "6h", ["arch", "persist", "v1_6", "v2_6", "v3_6", "v3f_6"]),
                         (L12, "12h", ["arch", "persist", "v1_12", "v2_12", "v3_12", "v3f_12"])):
    for g, sub in tbl.groupby("gid"):
        la, lo = ll.get(g, (np.nan, np.nan))
        r = {"family": fam, "gid": g, "lat": la, "lon": lo, "n": len(sub),
             "frac_stale": float((sub["obs_age"] > 6).mean())}
        for m in models:
            r[f"nse_{m}"] = nse(sub[m].to_numpy(), sub["truth"].to_numpy())
        recs.append(r)
pg = pd.DataFrame(recs)
pg.to_csv(os.path.join(BASE, "pergauge_nse.csv"), index=False)

# ---- exports for the figure stage (notebook2 env reads CSV only) -------------
pd.DataFrame({"gid": list(ll), "lat": [v[0] for v in ll.values()],
              "lon": [v[1] for v in ll.values()]}).to_csv(
    os.path.join(BASE, "allgauges_latlon.csv"), index=False)

# hydrograph examples: gauges with real flow variation during stale spells
cand = []
for g, sub in L6[L6["lead"] == 6].groupby("gid"):
    st = sub[sub["regime"] != "fresh"]
    tr = st["truth"].to_numpy()
    if len(st) >= 24 and np.isfinite(tr).sum() >= 24:
        cv = np.nanstd(tr) / (np.nanmean(tr) + 1.0)
        cand.append((cv * np.isfinite(tr).sum() ** 0.5, g))
for _, g in sorted(cand, reverse=True)[:4]:
    sub = L6[(L6["gid"] == g) & (L6["lead"] == 6)].sort_values("t0")
    sub.to_csv(os.path.join(BASE, f"hydro_{g}.csv"), index=False)
print("figure exports done:", [g for _, g in sorted(cand, reverse=True)[:4]],
      flush=True)

# ---- replay fidelity ---------------------------------------------------------
lines = []
m6 = np.isfinite(L6["arch"]) & np.isfinite(L6["v1_6"])
r = np.corrcoef(np.log1p(L6.loc[m6, "arch"]), np.log1p(L6.loc[m6, "v1_6"]))[0, 1]
lines.append(f"v1 6h replay vs archived (v1 era, log space): r={r:.4f}, "
             f"NSE={nse(L6.loc[m6, 'v1_6'].to_numpy(), L6.loc[m6, 'arch'].to_numpy()):.4f}, "
             f"n={int(m6.sum())}")
m12 = np.isfinite(L12["arch"]) & np.isfinite(L12["v1_12"])
r = np.corrcoef(np.log1p(L12.loc[m12, "arch"]), np.log1p(L12.loc[m12, "v1_12"]))[0, 1]
lines.append(f"v1 12h replay vs archived: r={r:.4f}, "
             f"NSE={nse(L12.loc[m12, 'v1_12'].to_numpy(), L12.loc[m12, 'arch'].to_numpy()):.4f}, "
             f"n={int(m12.sum())}")
open(os.path.join(BASE, "fidelity.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
print("stage 6 done", flush=True)
