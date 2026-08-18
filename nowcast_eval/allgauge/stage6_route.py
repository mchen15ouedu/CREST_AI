"""Stage 6 (all-gauge): per-gauge v1 vs v3-full NSE -> serving route table.

For every served gauge with post-hoc truth, score the replayed v1 and
v3-full predictions (all leads pooled, same NSE definition as the 433-gauge
eval) and pick the winner per horizon family. Also scores the current
obs-age rule (v1 fresh / v3 stale+dead) for comparison.

Outputs: route_pergauge.parquet / .csv   gid, n_*, nse_v1_*, nse_v3f_*,
                                          nse_agerule_*, winner_6, winner_12,
                                          frac_stale
         route_summary.txt
"""
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
rp = np.load(os.path.join(BASE, "replay_preds.npz"), allow_pickle=True)
ae = np.load(os.path.join(BASE, "archive_eval.npz"), allow_pickle=True)
gids = [str(g) for g in rp["gids"]]
t0s = [str(t) for t in rp["t0"]]
nG, nI = len(gids), len(t0s)
t0_dt = pd.to_datetime(pd.Series(t0s), format="%Y%m%d%H").values.astype("datetime64[s]")
age = ae["obs_age"]                                          # (nI, nG)
PR = {k: rp[k] for k in ("v1_6", "v3f_6", "v1_12", "v3f_12")}  # load once (npz access re-reads!)
print(f"{nG} gauges, {nI} issues ({t0s[0]}..{t0s[-1]})", flush=True)

od = pd.read_parquet(os.path.join(BASE, "obs_rows.parquet"))
truth = {g: (s["dt"].to_numpy().astype("datetime64[s]"),
             s["q_cms"].to_numpy().astype("float64"))
         for g, s in od.groupby("gid")}
print(f"truth for {len(truth)} gauges", flush=True)


def truth_at(g, when):
    tv = truth.get(g)
    if tv is None:
        return np.full(when.shape, np.nan)
    T, V = tv
    idx = np.searchsorted(T, when.ravel(), side="right") - 1
    ok = idx >= 0
    ic = np.clip(idx, 0, None)
    fresh = ok & ((when.ravel() - T[ic]) <= np.timedelta64(3600, "s"))
    return np.where(fresh, V[ic], np.nan).reshape(when.shape)


def nse(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    if m.sum() < 30:
        return np.nan
    p, o = p[m], o[m]
    den = ((o - o.mean()) ** 2).sum()
    if den <= 0:
        return np.nan
    return 1.0 - ((p - o) ** 2).sum() / den


recs = []
for gi, g in enumerate(gids):
    r = {"gid": g}
    for fam, hor, m1, m3 in (("6", 6, "v1_6", "v3f_6"), ("12", 12, "v1_12", "v3f_12")):
        when = t0_dt[:, None] + (np.arange(1, hor + 1)[None, :] * 3600).astype("timedelta64[s]")
        y = truth_at(g, when)                                # (nI, hor)
        p1 = PR[m1][:, gi, :]
        p3 = PR[m3][:, gi, :]
        stale = (age[:, gi] > 6.0)[:, None] & np.ones_like(y, bool)
        pr = np.where(stale, p3, p1)
        ok = np.isfinite(y)
        r[f"n_{fam}"] = int(ok.sum())
        r[f"nse_v1_{fam}"] = nse(p1, y)
        r[f"nse_v3f_{fam}"] = nse(p3, y)
        r[f"nse_agerule_{fam}"] = nse(pr, y)
        if fam == "6":
            r["frac_stale"] = float(stale[ok].mean()) if ok.any() else np.nan
        a, b = r[f"nse_v1_{fam}"], r[f"nse_v3f_{fam}"]
        r[f"winner_{fam}"] = ("v1" if a > b else "v3") if np.isfinite(a) and np.isfinite(b) else ""
    recs.append(r)
    if gi % 1000 == 0:
        print(f"  {gi}/{nG}", flush=True)

df = pd.DataFrame(recs)
df.to_csv(os.path.join(BASE, "route_pergauge.csv"), index=False)
md = {b"source": b"archive replay eval, all served gauges",
      b"issues": f"{t0s[0]}..{t0s[-1]} ({nI})".encode(),
      b"models": b"v1=dilstm.pt/dilstm_h12.pt v3=dilstm_v3_full.pt/dilstm_h12_v3_full.pt",
      b"rule": b"winner_* = higher per-gauge NSE (all leads pooled, >=30 samples); '' = undecided -> obs-age fallback"}
pq.write_table(pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(md),
               os.path.join(BASE, "route_pergauge.parquet"))

lines = [f"all-gauge route table: {nG} served gauges, {nI} issues {t0s[0]}..{t0s[-1]}"]
for fam in ("6", "12"):
    d = df[df[f"winner_{fam}"] != ""]
    v1w = int((d[f"winner_{fam}"] == "v1").sum())
    best = np.where(d[f"winner_{fam}"] == "v1", d[f"nse_v1_{fam}"], d[f"nse_v3f_{fam}"])
    lines += [f"{fam}h: decided {len(d)}/{nG} gauges -> v1 {v1w}, v3 {len(d) - v1w}; undecided {nG - len(d)} (obs-age fallback)",
              f"     median per-gauge NSE: v1-only {d[f'nse_v1_{fam}'].median():.3f} | v3-only {d[f'nse_v3f_{fam}'].median():.3f} "
              f"| obs-age rule {d[f'nse_agerule_{fam}'].median():.3f} | per-gauge winner {np.median(best):.3f}",
              f"     frac gauges NSE>0: v1 {(d[f'nse_v1_{fam}']>0).mean():.2f} v3 {(d[f'nse_v3f_{fam}']>0).mean():.2f} "
              f"age-rule {(d[f'nse_agerule_{fam}']>0).mean():.2f} winner {(best>0).mean():.2f}",
              f"     frac gauges NSE>0.5: v1 {(d[f'nse_v1_{fam}']>0.5).mean():.2f} v3 {(d[f'nse_v3f_{fam}']>0.5).mean():.2f} "
              f"age-rule {(d[f'nse_agerule_{fam}']>0.5).mean():.2f} winner {(best>0.5).mean():.2f}"]
    sp = d["frac_stale"] > 0.1
    lines.append(f"     stale-prone (frac_stale>0.1) n={int(sp.sum())}: v1 wins {int((d.loc[sp, f'winner_{fam}']=='v1').sum())}; "
                 f"fresh-feed n={int((~sp).sum())}: v1 wins {int((d.loc[~sp, f'winner_{fam}']=='v1').sum())}")
open(os.path.join(BASE, "route_summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
print("stage 6 (route) done", flush=True)
