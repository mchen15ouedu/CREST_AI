"""Figures for the multi-year all-gauge eval (notebook2 env: matplotlib+geopandas).
fig9  per-gauge scatter v1 vs v3-full, out-of-sample, fresh & noobs, 6h/12h
fig10 CONUS winner maps: fresh-obs (6h, 12h) and no-obs (6h, 12h)
fig11 per-gauge NSE CDFs by scenario (v1, v3f, persistence), out-of-sample
fig12 by-month medians (fresh 6h) v1 vs v3f across the 43 months
"""
import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = "/home/MengyuChen/CREST_AI/eval_figures"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # v1 / v3 / persistence
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "font.family": "sans-serif", "text.color": INK, "axes.edgecolor": BASELINE,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "font.size": 10,
                     "axes.titlesize": 11, "figure.titlesize": 13})
pg = pd.read_csv(os.path.join(BASE, "score_pergauge.csv"), dtype={"gid": str})
rt = pd.read_csv(os.path.join(BASE, "route_pergauge_multiyear.csv"), dtype={"gid": str}).fillna("")
ll = pd.read_csv(os.path.join(BASE, "..", "allgauges_latlon.csv"), dtype={"gid": str})
pg = pg.merge(ll, on="gid", how="left")
rt = rt.merge(ll, on="gid", how="left")
tag = "oos"

# ---- fig9 scatter ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10.6, 10.6), constrained_layout=True)
for row, sc in enumerate(("fresh", "noobs")):
    for col, fam in enumerate(("6", "12")):
        ax = axes[row, col]
        c1, c3 = f"nse_v1_{fam}_{sc}_{tag}", f"nse_v3f_{fam}_{sc}_{tag}"
        d = pg.dropna(subset=[c1, c3])
        x, y = np.clip(d[c1], -1, 1), np.clip(d[c3], -1, 1)
        w = int((d[c3] > d[c1]).sum())
        ax.scatter(x, y, s=5, color=C1, alpha=0.45, lw=0)
        ax.plot([-1, 1], [-1, 1], "--", color=MUTED, lw=1.2, zorder=0)
        ax.set_xlim(-1.04, 1.04); ax.set_ylim(-1.04, 1.04); ax.set_aspect("equal")
        ax.set_title(f"{fam}-h family, {'fresh obs' if sc == 'fresh' else 'no obs (dead feed / ungauged)'} — "
                     f"v3-full better at {w}/{len(d)} ({100 * w / len(d):.0f}%)\n"
                     f"median NSE v1 {d[c1].median():.3f}, v3-full {d[c3].median():.3f}", fontsize=9.5)
        ax.annotate("v3-full better", (-0.93, 0.88), color=INK2, fontsize=9)
        ax.annotate("v1 better", (0.55, -0.93), color=INK2, fontsize=9)
        ax.set_xlabel("v1 per-gauge NSE (clipped at −1)"); ax.set_ylabel("v3-full per-gauge NSE (clipped at −1)")
fig.suptitle("v1 vs full-scale v3 — one dot per gauge, out-of-sample months (2016–2022, 2025-07–2026-07)\n"
             "hourly issues, all served gauges, identical inputs")
fig.savefig(os.path.join(OUT, "fig9_multiyear_scatter_v1_vs_v3full.png"), dpi=150)

# ---- fig10 winner maps ----------------------------------------------------------
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
usa = world[world["name"] == "United States of America"]
fig, axes = plt.subplots(2, 2, figsize=(15, 9.6), constrained_layout=True)
for row, (sc, lab) in enumerate((("fresh", "fresh obs at issue time"), ("noobs", "no obs (dead feed / ungauged)"))):
    for col, fam in enumerate(("6", "12")):
        ax = axes[row, col]
        c = f"winner_{fam}_{sc}"
        d = rt[(rt.lon > -125.5) & (rt.lon < -66) & (rt.lat > 24) & (rt.lat < 50)]
        usa.boundary.plot(ax=ax, color=BASELINE, linewidth=0.8, zorder=0)
        und, v1, v3 = d[d[c] == ""], d[d[c] == "v1"], d[d[c] == "v3"]
        ax.scatter(und.lon, und.lat, s=3, color="#c3c2b7", alpha=0.6, lw=0, zorder=1)
        ax.scatter(v1.lon, v1.lat, s=6, color=C1, alpha=0.85, lw=0, zorder=2)
        ax.scatter(v3.lon, v3.lat, s=6, color=C2, alpha=0.85, lw=0, zorder=3)
        ax.set_xlim(-125.5, -66); ax.set_ylim(24, 50); ax.set_aspect(1.25); ax.set_axis_off()
        n = len(v1) + len(v3)
        ax.set_title(f"{fam}-h family, {lab}: v1 best at {len(v1)} ({100 * len(v1) / max(n, 1):.0f}%), "
                     f"v3-full best at {len(v3)} ({100 * len(v3) / max(n, 1):.0f}%), {len(und)} undecided", loc="left", fontsize=9.5)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=6, color=C1, label="v1 better"),
                           Line2D([], [], marker="o", ls="", ms=6, color=C2, label="v3-full better"),
                           Line2D([], [], marker="o", ls="", ms=4, color="#c3c2b7", label="undecided (no truth)")],
                  loc="lower left", fontsize=8.5)
fig.suptitle("Best nowcast model per gauge — all 43 evaluation months (Jan/Apr/Jul/Oct 2016–2026), per-gauge NSE over all leads")
fig.savefig(os.path.join(OUT, "fig10_multiyear_winner_maps.png"), dpi=150)

# ---- fig11 CDFs -----------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True, sharey=True)
for row, fam in enumerate(("6", "12")):
    for col, sc in enumerate(("fresh", "stale24", "noobs")):
        ax = axes[row, col]
        for m, color, name in (("v1", C1, "v1"), ("v3f", C2, "v3-full"), ("persist", C3, "persistence")):
            c = f"nse_{m}_{fam}_{sc}_{tag}"
            v = pg[c].dropna().to_numpy()
            if not len(v):
                continue
            v = np.sort(np.clip(v, -1, 1))
            ax.plot(v, np.linspace(0, 1, len(v)), color=color, lw=2, label=f"{name} (median {np.median(v):.2f})")
        ax.axvline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(-1, 1); ax.set_title(f"{fam}-h family, {sc}"); ax.set_xlabel("per-gauge NSE (clipped at −1)")
        ax.legend(loc="upper left", fontsize=8.5)
    axes[row, 0].set_ylabel("fraction of gauges ≤ NSE")
fig.suptitle("Per-gauge NSE distributions, out-of-sample months, all served gauges")
fig.savefig(os.path.join(OUT, "fig11_multiyear_nse_cdf.png"), dpi=150)

# ---- fig12 by month -------------------------------------------------------------
bp = pd.read_csv(os.path.join(BASE, "by_period.csv"), dtype={"month": str})
fig, ax = plt.subplots(figsize=(13, 4.6), constrained_layout=True)
x = np.arange(len(bp))
ax.plot(x, bp["med_v1_6_fresh"], "-o", color=C1, ms=4, lw=1.6, label="v1 (fresh obs)")
ax.plot(x, bp["med_v3f_6_fresh"], "-o", color=C2, ms=4, lw=1.6, label="v3-full (fresh obs)")
ax.plot(x, bp["med_v1_6_noobs"], "--s", color=C1, ms=3.5, lw=1.2, alpha=0.7, label="v1 (no obs)")
ax.plot(x, bp["med_v3f_6_noobs"], "--s", color=C2, ms=3.5, lw=1.2, alpha=0.7, label="v3-full (no obs)")
for i, r in bp.iterrows():
    if r["in_sample"]:
        ax.axvspan(i - 0.5, i + 0.5, color="#f0efe9", zorder=0)
ax.set_xticks(x); ax.set_xticklabels([f"{m[:4]}-{m[4:]}" for m in bp["month"]], rotation=90, fontsize=8)
ax.set_ylabel("median per-gauge NSE (6-h family, all leads)"); ax.axhline(0, color=MUTED, lw=0.8, ls=":")
ax.set_title("Median per-gauge NSE by evaluation month (shaded = inside the models' training window)")
ax.legend(ncol=4, fontsize=9, loc="lower left")
fig.savefig(os.path.join(OUT, "fig12_multiyear_by_month.png"), dpi=150)
print("figures saved")
