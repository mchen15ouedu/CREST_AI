"""Fig 6: per-gauge NSE, v1 vs v3 head-to-head on common gauges.

One dot per gauge (identical issues/leads for both models). Split by the
gauge's real-time obs reliability over the window: mostly-fresh feeds are
v1's home turf (and v1's training set); stale/dead-prone feeds are the
regime the v3 routing serves.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = "/home/MengyuChen/CREST_AI/eval_figures"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
C_FRESH, C_STALE = "#2a78d6", "#eb6834"      # categorical slots 1-2

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "text.color": INK, "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.size": 10, "axes.titlesize": 11,
    "figure.titlesize": 13,
})

pg = pd.read_csv(os.path.join(BASE, "pergauge_nse.csv"), dtype={"gid": str})
fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4), constrained_layout=True)
for ax, fam, c1, c3 in ((axes[0], "6h", "nse_v1_6", "nse_v3_6"),
                        (axes[1], "12h", "nse_v1_12", "nse_v3_12")):
    d = pg[(pg["family"] == fam)].dropna(subset=[c1, c3]).copy()
    d["x"] = np.clip(d[c1], -1, 1)
    d["y"] = np.clip(d[c3], -1, 1)
    stale_prone = d["frac_stale"] > 0.1
    for m, color, label in ((~stale_prone, C_FRESH, "reliable obs feed"),
                            (stale_prone, C_STALE, "stale/dead-prone feed")):
        s = d[m]
        w3 = int((s[c3] > s[c1]).sum())
        ax.scatter(s["x"], s["y"], s=16, color=color, lw=0.4,
                   edgecolors=SURFACE, alpha=0.85,
                   label=f"{label}  (v3 better at {w3}/{len(s)})")
    ax.plot([-1, 1], [-1, 1], "--", color=MUTED, lw=1.2, zorder=0)
    ax.annotate("v3 better", (-0.93, 0.88), color=INK2, fontsize=9)
    ax.annotate("v1 better", (0.55, -0.93), color=INK2, fontsize=9)
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.set_aspect("equal")
    ax.set_xlabel("v1 per-gauge NSE (clipped at −1)")
    ax.set_title(f"{fam}-horizon models")
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)
    ax.legend(loc="lower left", fontsize=8.5, markerscale=1.3)
    med = (float(np.median(d[c1])), float(np.median(d[c3])))
    print(fam, "medians v1/v3:", np.round(med, 3),
          "| v3 wins overall:", int((d[c3] > d[c1]).sum()), "/", len(d),
          "| among stale-prone:", int((d.loc[stale_prone, c3] > d.loc[stale_prone, c1]).sum()),
          "/", int(stale_prone.sum()))
axes[0].set_ylabel("v3 per-gauge NSE (clipped at −1)")
fig.suptitle("v1 vs v3, same gauges, same issues — one dot per gauge\n"
             "(eval gauges are in v1's training set and NOT in v3's: "
             "v1 has home-field advantage on every dot)")
fig.savefig(os.path.join(OUT, "fig6_v1_vs_v3_pergauge.png"), dpi=150)
print("saved fig6")
