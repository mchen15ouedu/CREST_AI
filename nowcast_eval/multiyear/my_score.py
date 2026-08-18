"""Score the multi-year all-gauge replay from the per-month accumulators.

Outputs (in this dir):
  score_pergauge.parquet  per gauge x scenario x family: NSE of v1, v3f,
                          persist over out-of-sample months, in-sample months,
                          all months; winner per (scenario, family)
  route_pergauge_multiyear.parquet  serving table: gid, winner_{6,12}_{fresh,stale,noobs}
  loyo_cv.csv             leave-one-year-out validation of the routing rule
  by_lead.csv, by_period.csv, peaks.csv, report_stats.txt
"""
import glob
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import BASE, MODELS, SCEN, SEASON, in_sample

files = sorted(glob.glob(os.path.join(BASE, "acc", "acc_*.npz")))
months = [os.path.basename(f)[4:10] for f in files]
A = {m: np.load(f) for m, f in zip(months, files)}
gids = [str(g) for g in A[months[0]]["gids"]]
nG = len(gids)
nS, nM, LE = len(SCEN), len(MODELS), 12
KEYS = ("n", "so", "so2", "se2")
print(f"{len(months)} months: {months}", flush=True)


def combine(sel):
    out = {k: np.zeros((nS, nM, nG, LE), "float64") for k in KEYS}
    for m in sel:
        for k in KEYS:
            out[k] += A[m][k]
    return out


def nse_from(acc, leads):
    """per-gauge NSE (nS, nM, nG) pooling the given lead indices."""
    n = acc["n"][..., leads].sum(-1)
    so = acc["so"][..., leads].sum(-1)
    so2 = acc["so2"][..., leads].sum(-1)
    se2 = acc["se2"][..., leads].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = so2 - so * so / np.maximum(n, 1)
        r = 1.0 - se2 / np.where(den > 0, den, np.nan)
    r[n < 30] = np.nan
    return r


def pooled_nse(acc, leads):
    n = acc["n"][..., leads].sum(-1).sum(-1)      # (nS, nM)
    so = acc["so"][..., leads].sum(-1).sum(-1)
    so2 = acc["so2"][..., leads].sum(-1).sum(-1)
    se2 = acc["se2"][..., leads].sum(-1).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = so2 - so * so / np.maximum(n, 1)
        return 1.0 - se2 / np.where(den > 0, den, np.nan), n


FAM = {"6": (list(range(6)), "v1_6", "v3f_6"), "12": (list(range(12)), "v1_12", "v3f_12")}
mi = {m: i for i, m in enumerate(MODELS)}
si = {s: i for i, s in enumerate(SCEN)}
oos = [m for m in months if not in_sample(m)]
ins = [m for m in months if in_sample(m)]
years = sorted({m[:4] for m in months})
lines = [f"multi-year all-gauge eval: {len(months)} months ({len(oos)} out-of-sample, {len(ins)} in-sample "
         f"[training window 2023-01..2025-06]); {nG} served gauges; hourly issues; "
         f"scenarios {SCEN}; models {MODELS}"]

# ---- 1) per-gauge table --------------------------------------------------------
recs = {"gid": gids}
for tag, sel in (("oos", oos), ("ins", ins), ("all", months)):
    if not sel:
        continue
    acc = combine(sel)
    for fam, (leads, m1, m3) in FAM.items():
        r = nse_from(acc, leads)
        for sc in SCEN:
            recs[f"nse_v1_{fam}_{sc}_{tag}"] = r[si[sc], mi[m1]]
            recs[f"nse_v3f_{fam}_{sc}_{tag}"] = r[si[sc], mi[m3]]
            recs[f"nse_persist_{fam}_{sc}_{tag}"] = r[si[sc], mi["persist"]]
pg = pd.DataFrame(recs)
# obs availability: fraction of issues with fresh obs at t0 (all months)
nf = sum(A[m]["n_fresh_t0"] for m in months); ni = sum(int(A[m]["n_issues"]) for m in months)
pg["frac_fresh_t0"] = nf / ni
pg.to_parquet(os.path.join(BASE, "score_pergauge.parquet"), index=False)
pg.to_csv(os.path.join(BASE, "score_pergauge.csv"), index=False)

# ---- 2) headline per-gauge stats ------------------------------------------------
def summarize(tag, sel):
    out = [f"\n=== {tag}: {len(sel)} months {sel[0]}..{sel[-1]} ==="]
    acc = combine(sel)
    for fam, (leads, m1, m3) in FAM.items():
        r = nse_from(acc, leads)
        for sc in SCEN:
            a, b, p = r[si[sc], mi[m1]], r[si[sc], mi[m3]], r[si[sc], mi["persist"]]
            ok = np.isfinite(a) & np.isfinite(b)
            best = np.where(a > b, a, b)
            out.append(f"{fam:>2s}h {sc:8s} n={int(ok.sum()):>5d} gauges | median NSE v1 {np.nanmedian(a[ok]):.3f} "
                       f"v3f {np.nanmedian(b[ok]):.3f} persist {np.nanmedian(p[ok]) if np.isfinite(p[ok]).any() else float('nan'):.3f} "
                       f"best-of-two {np.nanmedian(best[ok]):.3f} | v3f wins {int((b[ok] > a[ok]).sum())} "
                       f"({100 * (b[ok] > a[ok]).mean():.0f}%) | frac NSE>0 v1 {np.mean(a[ok] > 0):.2f} v3f {np.mean(b[ok] > 0):.2f} "
                       f"| frac NSE>0.5 v1 {np.mean(a[ok] > 0.5):.2f} v3f {np.mean(b[ok] > 0.5):.2f}")
        pn, n = pooled_nse(acc, leads)
        out.append(f"{fam:>2s}h pooled NSE (all leads): " + "; ".join(
            f"{sc}: v1 {pn[si[sc], mi[m1]]:.3f} v3f {pn[si[sc], mi[m3]]:.3f} persist {pn[si[sc], mi['persist']]:.3f} (n={int(n[si[sc], mi[m1]])})"
            for sc in SCEN))
    return out


lines += summarize("OUT-OF-SAMPLE", oos)
if ins:
    lines += summarize("IN-SAMPLE (training window)", ins)

# ---- 3) by lead (out-of-sample) ------------------------------------------------
acc = combine(oos)
rows = []
for sc in SCEN:
    for k in range(LE):
        r = nse_from(acc, [k])
        pn, n = pooled_nse(acc, [k])
        row = {"scenario": sc, "lead": k + 1, "n": int(n[si[sc], mi["v1_12"]])}
        for m in MODELS:
            if k >= 6 and m.endswith("_6"):
                continue
            row[f"pooled_{m}"] = pn[si[sc], mi[m]]
            row[f"median_{m}"] = np.nanmedian(r[si[sc], mi[m]])
        rows.append(row)
bl = pd.DataFrame(rows)
bl.to_csv(os.path.join(BASE, "by_lead.csv"), index=False)
lines.append("\n--- out-of-sample, by lead: median per-gauge NSE (pooled NSE) ---")
for sc in SCEN:
    lines.append(f"[{sc}]  lead:  v1_6 / v3f_6 / v1_12 / v3f_12 / persist")
    for _, r in bl[bl.scenario == sc].iterrows():
        f = lambda m: (f"{r[f'median_{m}']:.3f}({r[f'pooled_{m}']:.3f})" if f"median_{m}" in r and np.isfinite(r[f"median_{m}"]) else "   -   ")
        lines.append(f"   {int(r['lead']):>2d}: {f('v1_6')} / {f('v3f_6')} / {f('v1_12')} / {f('v3f_12')} / {f('persist')}")

# ---- 4) by period (year, season) — fresh + noobs, 6h & 12h medians -------------
rows = []
for m in months:
    acc1 = combine([m])
    row = {"month": m, "year": m[:4], "season": SEASON[m[4:]], "in_sample": in_sample(m)}
    for fam, (leads, m1, m3) in FAM.items():
        r = nse_from(acc1, leads)
        for sc in SCEN:
            a, b = r[si[sc], mi[m1]], r[si[sc], mi[m3]]
            ok = np.isfinite(a) & np.isfinite(b)
            row[f"n_{fam}_{sc}"] = int(ok.sum())
            row[f"med_v1_{fam}_{sc}"] = np.nanmedian(a[ok]) if ok.any() else np.nan
            row[f"med_v3f_{fam}_{sc}"] = np.nanmedian(b[ok]) if ok.any() else np.nan
            row[f"v3f_winfrac_{fam}_{sc}"] = float((b[ok] > a[ok]).mean()) if ok.any() else np.nan
    rows.append(row)
bp = pd.DataFrame(rows)
bp.to_csv(os.path.join(BASE, "by_period.csv"), index=False)
lines.append("\n--- by month: median per-gauge NSE v1 / v3f (fresh 6h | fresh 12h | noobs 6h | noobs 12h), v3f win-frac fresh 6h ---")
for _, r in bp.iterrows():
    lines.append(f"  {r['month']} {r['season']:6s}{' *in-sample*' if r['in_sample'] else '            '} n={int(r['n_6_fresh']):>5d}: "
                 f"{r['med_v1_6_fresh']:.3f}/{r['med_v3f_6_fresh']:.3f} | {r['med_v1_12_fresh']:.3f}/{r['med_v3f_12_fresh']:.3f} | "
                 f"{r['med_v1_6_noobs']:.3f}/{r['med_v3f_6_noobs']:.3f} | {r['med_v1_12_noobs']:.3f}/{r['med_v3f_12_noobs']:.3f} | "
                 f"{r['v3f_winfrac_6_fresh']:.2f}")

# ---- 5) peak-ratio (hallucination) in noobs: max pred / max obs per gauge -------
rows = []
for sc in ("fresh", "noobs"):
    for m in ("v1_6", "v3f_6", "v1_12", "v3f_12"):
        mp = np.full(nG, -np.inf); mo = np.full(nG, -np.inf)
        for mm in oos:
            mp = np.maximum(mp, A[mm]["maxp"][si[sc], mi[m]].max(-1))
            mo = np.maximum(mo, A[mm]["maxo"][si[sc], mi[m]].max(-1))
        ok = np.isfinite(mp) & np.isfinite(mo) & (mo > 0)
        ratio = mp[ok] / mo[ok]
        rows.append({"scenario": sc, "model": m, "n": int(ok.sum()),
                     "peak_ratio_p50": np.median(ratio), "peak_ratio_p95": np.percentile(ratio, 95),
                     "peak_ratio_p99": np.percentile(ratio, 99), "frac_ratio_gt3": float((ratio > 3).mean())})
pk = pd.DataFrame(rows)
pk.to_csv(os.path.join(BASE, "peaks.csv"), index=False)
lines.append("\n--- peak ratio max(pred)/max(obs) per gauge, out-of-sample ---")
for _, r in pk.iterrows():
    lines.append(f"  {r['scenario']:6s} {r['model']:7s}: p50 {r['peak_ratio_p50']:.2f} p95 {r['peak_ratio_p95']:.2f} p99 {r['peak_ratio_p99']:.2f} frac>3 {r['frac_ratio_gt3']:.3f}")

# ---- 6) routing table (all months) + leave-one-year-out CV ---------------------
def winners(sel):
    acc = combine(sel)
    w = {}
    for fam, (leads, m1, m3) in FAM.items():
        r = nse_from(acc, leads)
        for sc in SCEN:
            a, b = r[si[sc], mi[m1]], r[si[sc], mi[m3]]
            w[(fam, sc)] = np.where(np.isfinite(a) & np.isfinite(b), np.where(a > b, "v1", "v3"), "")
    return w


W = winners(months)
rt = {"gid": gids}
for (fam, sc), w in W.items():
    rt[f"winner_{fam}_{ {'fresh': 'fresh', 'stale24': 'stale', 'noobs': 'noobs'}[sc] }"] = w
rt = pd.DataFrame(rt)
md = {b"source": b"multi-year all-gauge replay (Jan/Apr/Jul/Oct 2016-2026, hourly issues)",
      b"months": ",".join(months).encode(),
      b"rule": b"winner_{fam}_{regime} = higher per-gauge NSE (all leads pooled, >=30 samples); '' = undecided",
      b"models": b"v1=dilstm.pt/dilstm_h12.pt v3=dilstm_v3_full.pt/dilstm_h12_v3_full.pt"}
pq.write_table(pa.Table.from_pandas(rt, preserve_index=False).replace_schema_metadata(md),
               os.path.join(BASE, "route_pergauge_multiyear.parquet"))
rt.to_csv(os.path.join(BASE, "route_pergauge_multiyear.csv"), index=False)
lines.append("\n--- routing table (all months) ---")
for c in [c for c in rt.columns if c != "gid"]:
    v = rt[c].value_counts()
    lines.append(f"  {c}: v1 {int(v.get('v1', 0))}, v3 {int(v.get('v3', 0))}, undecided {int(v.get('', 0))}")

# LOYO: choose winners on all months except year Y (per scenario/family), score on Y
rows = []
for Y in years:
    test = [m for m in months if m[:4] == Y]
    train = [m for m in months if m[:4] != Y]
    Wt = winners(train)
    acc = combine(test)
    for fam, (leads, m1, m3) in FAM.items():
        r = nse_from(acc, leads)
        for sc in SCEN:
            a, b = r[si[sc], mi[m1]], r[si[sc], mi[m3]]
            w = Wt[(fam, sc)]
            ok = np.isfinite(a) & np.isfinite(b) & (w != "")
            routed = np.where(w == "v1", a, b)
            rows.append({"year": Y, "family": fam, "scenario": sc, "n": int(ok.sum()),
                         "med_v1": np.nanmedian(a[ok]), "med_v3f": np.nanmedian(b[ok]),
                         "med_routed": np.nanmedian(routed[ok]),
                         "routed_beats_v1_frac": float((routed[ok] >= a[ok]).mean()),
                         "frac_gt0_v1": float((a[ok] > 0).mean()), "frac_gt0_v3f": float((b[ok] > 0).mean()),
                         "frac_gt0_routed": float((routed[ok] > 0).mean())})
cv = pd.DataFrame(rows)
cv.to_csv(os.path.join(BASE, "loyo_cv.csv"), index=False)
lines.append("\n--- leave-one-year-out CV of per-gauge routing (winner chosen on other years, scored on held-out year): median per-gauge NSE v1 / v3f / routed ---")
for sc in SCEN:
    for fam in FAM:
        sub = cv[(cv.scenario == sc) & (cv.family == fam)]
        lines.append(f"  [{sc} {fam}h] " + "  ".join(f"{r.year}: {r.med_v1:.3f}/{r.med_v3f:.3f}/{r.med_routed:.3f}" for r in sub.itertuples()))
        lines.append(f"      mean over years: v1 {sub.med_v1.mean():.3f}  v3f {sub.med_v3f.mean():.3f}  routed {sub.med_routed.mean():.3f}; "
                     f"frac gauges NSE>0: v1 {sub.frac_gt0_v1.mean():.2f} v3f {sub.frac_gt0_v3f.mean():.2f} routed {sub.frac_gt0_routed.mean():.2f}")

open(os.path.join(BASE, "report_stats.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
