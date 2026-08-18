"""Stage 7 (all-gauge): the v1 vs v3-full evaluation numbers on ALL served
gauges — regime-pooled NSE, per-lead pooled + per-gauge-median NSE, fresh-feed
vs stale-prone per-gauge medians, persistence baseline. Writes
report_stats.txt (consumed by the CREST_AI report) and eval_long-free numpy
tables (nothing sample-level is stored; everything is recomputed here)."""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
rp = np.load(os.path.join(BASE, "replay_preds.npz"), allow_pickle=True)
ae = np.load(os.path.join(BASE, "archive_eval.npz"), allow_pickle=True)
gids = [str(g) for g in rp["gids"]]
t0s = [str(t) for t in rp["t0"]]
nG, nI = len(gids), len(t0s)
t0_dt = pd.to_datetime(pd.Series(t0s), format="%Y%m%d%H").values.astype("datetime64[s]")
age = ae["obs_age"]                                            # (nI, nG)
regime = np.where(age <= 6, 0, np.where(age <= 240, 1, 2))     # fresh/stale/dead
REG = ["fresh", "stale", "dead"]

od = pd.read_parquet(os.path.join(BASE, "obs_rows.parquet"))
truth = {g: (s["dt"].to_numpy().astype("datetime64[s]"), s["q_cms"].to_numpy().astype("float64"))
         for g, s in od.groupby("gid")}

# truth Y (nI, nG, 12) at t0 + lead
Y = np.full((nI, nG, 12), np.nan, "float32")
when = t0_dt[:, None] + (np.arange(1, 13)[None, :] * 3600).astype("timedelta64[s]")   # (nI,12)
wf = when.ravel()
for gi, g in enumerate(gids):
    tv = truth.get(g)
    if tv is None:
        continue
    T, V = tv
    idx = np.searchsorted(T, wf, side="right") - 1
    ok = idx >= 0
    ic = np.clip(idx, 0, None)
    fresh = ok & ((wf - T[ic]) <= np.timedelta64(3600, "s"))
    Y[:, gi, :] = np.where(fresh, V[ic], np.nan).reshape(nI, 12)
print("truth built", flush=True)

P = {"v1_6": rp["v1_6"], "v3f_6": rp["v3f_6"], "v1_12": rp["v1_12"], "v3f_12": rp["v3f_12"]}
persist = rp["persist"].astype("float32")                      # (nI, nG)


def nse_pooled(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    if m.sum() < 30:
        return np.nan
    p, o = p[m], o[m]
    den = ((o - o.mean()) ** 2).sum()
    return np.nan if den <= 0 else 1.0 - ((p - o) ** 2).sum() / den


def nse_pergauge(p, o):
    """p, o: (nI, nG, k) -> per-gauge NSE over all (issue, lead) samples."""
    m = np.isfinite(p) & np.isfinite(o)
    n = m.sum((0, 2))
    oo = np.where(m, o, 0.0)
    om = oo.sum((0, 2)) / np.maximum(n, 1)
    den = (np.where(m, (o - om[None, :, None]) ** 2, 0.0)).sum((0, 2))
    num = (np.where(m, (p - o) ** 2, 0.0)).sum((0, 2))
    r = 1.0 - num / np.where(den > 0, den, np.nan)
    r[n < 30] = np.nan
    return r


lines = [f"ALL-GAUGE replay: {nG} served gauges, {nI} issues {t0s[0]}..{t0s[-1]}; "
         f"{sum(1 for g in gids if g in truth)} gauges with post-hoc truth"]
for fam, hor in (("6", 6), ("12", 12)):
    y = Y[:, :, :hor]
    p1, p3 = P[f"v1_{fam}"], P[f"v3f_{fam}"]
    pp = np.repeat(persist[:, :, None], hor, axis=2)
    stale = (age > 6.0)[:, :, None]
    pr = np.where(stale, p3, p1)
    lines.append(f"\n== {fam}h family ==")
    # regime pooled
    for ri, rn in enumerate(REG):
        m = (regime == ri)[:, :, None] & np.ones_like(y, bool)
        n = int((m & np.isfinite(y)).sum())
        lines.append(f"pooled NSE {rn:5s} (n={n:>9d}): v1 {nse_pooled(p1[m], y[m]):.3f}  v3f {nse_pooled(p3[m], y[m]):.3f}  "
                     f"persist {nse_pooled(pp[m], y[m]):.3f}  age-rule {nse_pooled(pr[m], y[m]):.3f}")
    # by lead
    lines.append("lead  pooled(v1,v3f,persist)   median-per-gauge(v1,v3f,persist)")
    for k in range(hor):
        yk, ok = y[:, :, k:k + 1], None
        g1 = nse_pergauge(p1[:, :, k:k + 1], yk); g3 = nse_pergauge(p3[:, :, k:k + 1], yk)
        gp = nse_pergauge(pp[:, :, k:k + 1], yk)
        lines.append(f"{k + 1:>4d}  {nse_pooled(p1[:, :, k], y[:, :, k]):.3f} {nse_pooled(p3[:, :, k], y[:, :, k]):.3f} "
                     f"{nse_pooled(pp[:, :, k], y[:, :, k]):.3f}   {np.nanmedian(g1):.3f} {np.nanmedian(g3):.3f} {np.nanmedian(gp):.3f}")
    # per-gauge, all leads pooled
    g1 = nse_pergauge(p1, y); g3 = nse_pergauge(p3, y); gr = nse_pergauge(pr, y); gp = nse_pergauge(pp, y)
    okg = np.isfinite(g1) & np.isfinite(g3)
    fs = np.where(np.isfinite(y).any(2).sum(0) > 0,
                  ((age > 6) & np.isfinite(y).any(2)).sum(0) / np.maximum(np.isfinite(y).any(2).sum(0), 1), np.nan)
    sp = fs > 0.1
    best = np.where(g1 > g3, g1, g3)
    lines.append(f"per-gauge (all leads), decided gauges n={int(okg.sum())}: median v1 {np.nanmedian(g1[okg]):.3f} "
                 f"v3f {np.nanmedian(g3[okg]):.3f} persist {np.nanmedian(gp[okg]):.3f} age-rule {np.nanmedian(gr[okg]):.3f} "
                 f"per-gauge-winner {np.nanmedian(best[okg]):.3f}; v3f wins {int((g3[okg] > g1[okg]).sum())}/{int(okg.sum())}")
    for lab, m in (("fresh-feed (frac_stale<=0.1)", okg & ~sp), ("stale-prone (frac_stale>0.1)", okg & sp)):
        lines.append(f"  {lab:30s} n={int(m.sum()):>5d}: median v1 {np.nanmedian(g1[m]):.3f} v3f {np.nanmedian(g3[m]):.3f} "
                     f"persist {np.nanmedian(gp[m]):.3f}; v3f wins {int((g3[m] > g1[m]).sum())}; "
                     f"frac NSE>0 v1 {np.mean(g1[m] > 0):.2f} v3f {np.mean(g3[m] > 0):.2f}; "
                     f"frac NSE>0.5 v1 {np.mean(g1[m] > 0.5):.2f} v3f {np.mean(g3[m] > 0.5):.2f}")
    lines.append(f"  frac gauges NSE>0: v1 {np.mean(g1[okg] > 0):.2f} v3f {np.mean(g3[okg] > 0):.2f} winner {np.mean(best[okg] > 0):.2f}; "
                 f"NSE>0.5: v1 {np.mean(g1[okg] > 0.5):.2f} v3f {np.mean(g3[okg] > 0.5):.2f} winner {np.mean(best[okg] > 0.5):.2f}")
open(os.path.join(BASE, "report_stats.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
