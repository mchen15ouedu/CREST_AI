"""Fig 8: CONUS map of which model wins at each served gauge (v1 vs v3-full).

One dot per gauge, from the all-gauge archive replay (route_pergauge.csv):
blue = v1 better, orange = v3-full better, grey = undecided (no post-hoc
truth in the window, i.e. dead feed -> obs-age fallback -> v3 in practice).
Two panels: 6-h and 12-h horizon families.
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
C_V1, C_V3, C_UND = "#2a78d6", "#eb6834", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "text.color": INK, "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "font.size": 10, "axes.titlesize": 11, "figure.titlesize": 13,
})

rt = pd.read_csv(os.path.join(BASE, "route_pergauge.csv"), dtype={"gid": str})
ll = pd.read_csv(os.path.join(BASE, "..", "allgauges_latlon.csv"), dtype={"gid": str})
d = rt.merge(ll, on="gid", how="left")
d = d[(d["lon"] > -125.5) & (d["lon"] < -66) & (d["lat"] > 24) & (d["lat"] < 50)]
for c in ("winner_6", "winner_12"):
    d[c] = d[c].fillna("")

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
usa = world[world["name"] == "United States of America"]

fig, axes = plt.subplots(2, 1, figsize=(11, 11.6), constrained_layout=True)
for ax, fam, col in ((axes[0], "6-h", "winner_6"), (axes[1], "12-h", "winner_12")):
    usa.boundary.plot(ax=ax, color=BASELINE, linewidth=0.8, zorder=0)
    und = d[d[col] == ""]
    v1 = d[d[col] == "v1"]
    v3 = d[d[col] == "v3"]
    ax.scatter(und["lon"], und["lat"], s=5, color=C_UND, alpha=0.7, lw=0, zorder=1)
    ax.scatter(v1["lon"], v1["lat"], s=9, color=C_V1, alpha=0.85, lw=0, zorder=2)
    ax.scatter(v3["lon"], v3["lat"], s=9, color=C_V3, alpha=0.85, lw=0, zorder=3)
    ax.set_xlim(-125.5, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect(1.25)
    ax.set_axis_off()
    n = len(v1) + len(v3)
    ax.set_title(f"{fam} horizon — best model per gauge  "
                 f"(v1 at {len(v1)} = {100 * len(v1) / n:.0f}%,  "
                 f"v3-full at {len(v3)} = {100 * len(v3) / n:.0f}%,  "
                 f"{len(und)} undecided)", loc="left")
    handles = [Line2D([], [], marker="o", ls="", ms=7, color=C_V1, label=f"v1 better ({len(v1)})"),
               Line2D([], [], marker="o", ls="", ms=7, color=C_V3, label=f"v3-full better ({len(v3)})"),
               Line2D([], [], marker="o", ls="", ms=5, color=C_UND,
                      label=f"undecided: no truth in window, dead feed → v3 ({len(und)})")]
    ax.legend(handles=handles, loc="lower left", fontsize=9)
fig.suptitle("Which nowcast model wins at each gauge — v1 vs full-scale v3\n"
             "archive replay, 511 issues 2026-07-21 .. 08-12, per-gauge NSE over all leads; "
             "this table drives live routing since 2026-08-17")
fig.savefig(os.path.join(OUT, "fig8_winner_map_v1_vs_v3full.png"), dpi=150)
print("saved fig8")
