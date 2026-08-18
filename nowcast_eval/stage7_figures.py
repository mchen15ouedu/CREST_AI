"""Stage 7: evaluation figures -> /home/MengyuChen/CREST_AI/eval_figures/.

Runs in the notebook2 env (matplotlib; inputs are CSV because that env has no
pyarrow). Palette: dataviz reference instance, light mode — categorical slots
1-3 (blue/orange/aqua = v1/v2/v3, all-pairs validated), muted ink for the
persistence baseline, blue<->red diverging with gray midpoint for the NSE maps.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = "/home/MengyuChen/CREST_AI/eval_figures"
os.makedirs(OUT, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_V1, C_V2, C_V3 = "#2a78d6", "#eb6834", "#1baf7a"   # slots 1-3, validated

MODELS_6 = [("persist", "persistence", MUTED, ":"),
            ("arch", "v1 (operational archive)", C_V1, "--"),
            ("v1_6", "v1 (replay)", C_V1, "-"),
            ("v2_6", "v2 (replay)", C_V2, "-"),
            ("v3_6", "v3 (replay)", C_V3, "-")]
MODELS_12 = [("persist", "persistence", MUTED, ":"),
             ("arch", "v1 (operational archive)", C_V1, "--"),
             ("v1_12", "v1 (replay)", C_V1, "-"),
             ("v2_12", "v2 (replay)", C_V2, "-"),
             ("v3_12", "v3 (replay)", C_V3, "-")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "text.color": INK, "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "font.size": 10, "axes.titlesize": 11, "figure.titlesize": 13,
})


def style(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


# ---------------------------------------------------------------- fig 1: leads
bl = pd.read_csv(os.path.join(BASE, "nse_by_lead.csv"))
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
for ax, fam, models in ((axes[0], "6h", MODELS_6), (axes[1], "12h", MODELS_12)):
    sub = bl[bl["family"] == fam]
    for key, label, color, ls in models:
        s = sub[sub["model"] == key].sort_values("lead")
        ax.plot(s["lead"], s["median_gauge_nse"], ls, color=color, lw=2,
                marker="o", ms=4, label=label)
    ax.set_title(f"{fam}-horizon models")
    ax.set_xlabel("lead time (h)")
    ax.set_xticks(sub["lead"].unique())
    style(ax)
axes[0].set_ylabel("median per-gauge NSE")
axes[0].legend(loc="lower left", fontsize=8.5)
fig.suptitle("Nowcast skill vs lead time — out-of-sample gauges, "
             "487 operational issues (Jul 21 – Aug 11 2026)")
fig.savefig(os.path.join(OUT, "fig1_nse_by_lead.png"), dpi=150)
plt.close(fig)
print("fig1 done", flush=True)

# ------------------------------------------------------------- fig 2: regimes
rg = pd.read_csv(os.path.join(BASE, "nse_by_regime.csv"))
REGIMES = ["fresh", "stale", "dead"]
RLAB = {"fresh": "fresh obs (≤6 h)", "stale": "stale obs (6–240 h)",
        "dead": "no obs (>240 h)"}
FLOOR = -0.55
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
for ax, fam, models in ((axes[0], "6h", MODELS_6), (axes[1], "12h", MODELS_12)):
    sub = rg[rg["family"] == fam]
    shown = [m for m in models if m[0] != "arch"]      # keep bars readable
    w = 0.8 / len(shown)
    for i, (key, label, color, _) in enumerate(shown):
        vals, xs = [], []
        for j, reg in enumerate(REGIMES):
            v = sub[(sub["model"] == key) & (sub["regime"] == reg)]["pooled_nse"]
            v = float(v.iloc[0]) if len(v) else np.nan
            xs.append(j + (i - (len(shown) - 1) / 2) * w)
            vals.append(v)
        clipped = np.maximum(vals, FLOOR)
        ax.bar(xs, clipped, width=w * 0.92, color=color, label=label, zorder=3)
        for x, v, c in zip(xs, vals, clipped):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (x, max(c, 0) + 0.02 if v >= 0 else c - 0.02),
                            ha="center", va="bottom" if v >= 0 else "top",
                            fontsize=7, color=INK2)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels([RLAB[r] for r in REGIMES])
    ax.set_ylim(FLOOR - 0.15, 1.12)
    ax.set_title(f"{fam}-horizon models")
    style(ax)
axes[0].set_ylabel("pooled NSE")
axes[0].legend(loc="lower left", fontsize=8.5)
fig.suptitle("Skill by observation freshness at issue time "
             "(bars clipped at NSE − 0.55; labels give true values)")
fig.savefig(os.path.join(OUT, "fig2_nse_by_obs_age.png"), dpi=150)
plt.close(fig)
print("fig2 done", flush=True)

# ---------------------------------------------------------------- fig 3: CDFs
pg = pd.read_csv(os.path.join(BASE, "pergauge_nse.csv"), dtype={"gid": str})
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
for ax, fam, models in ((axes[0], "6h", MODELS_6), (axes[1], "12h", MODELS_12)):
    sub = pg[pg["family"] == fam]
    for key, label, color, ls in models:
        v = np.sort(sub[f"nse_{key}"].dropna().to_numpy())
        if not len(v):
            continue
        ax.plot(np.clip(v, -1, 1), np.arange(1, len(v) + 1) / len(v), ls,
                color=color, lw=2, label=label)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("per-gauge NSE (clipped at −1)")
    ax.set_title(f"{fam}-horizon models")
    style(ax)
axes[0].set_ylabel("fraction of gauges below")
axes[0].legend(loc="upper left", fontsize=8.5)
fig.suptitle("Per-gauge NSE distribution (all leads pooled) — lower curve is better")
fig.savefig(os.path.join(OUT, "fig3_pergauge_nse_cdf.png"), dpi=150)
plt.close(fig)
print("fig3 done", flush=True)

# --------------------------------------------------------- fig 4: hydrographs
import glob
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
hfiles = sorted(glob.glob(os.path.join(BASE, "hydro_*.csv")))[:4]
if hfiles:
    fig, axes = plt.subplots(len(hfiles), 1, figsize=(10.5, 2.6 * len(hfiles)),
                             constrained_layout=True, squeeze=False)
    for ax, hf in zip(axes[:, 0], hfiles):
        d = pd.read_csv(hf, dtype={"gid": str, "t0": str})
        tt_dt = pd.to_datetime(d["t0"], format="%Y%m%d%H") + pd.Timedelta(hours=6)
        tt = pd.Series(mdates.date2num(tt_dt.dt.to_pydatetime()))
        stale = (d["obs_age"] > 6).to_numpy()
        # shade stale spells
        in_run = False
        for i in range(len(d)):
            if stale[i] and not in_run:
                s0, in_run = tt.iloc[i], True
            if in_run and (not stale[i] or i == len(d) - 1):
                ax.axvspan(s0, tt.iloc[i],
                           color=GRID, alpha=0.55, zorder=0)
                in_run = False
        ax.plot(tt, d["truth"], color=INK, lw=2.2, label="observed (post-hoc)")
        ax.plot(tt, d["v1_6"], color=C_V1, lw=1.6, label="v1")
        ax.plot(tt, d["v2_6"], color=C_V2, lw=1.6, label="v2")
        ax.plot(tt, d["v3_6"], color=C_V3, lw=1.6, label="v3")
        ax.set_title(f"gauge {d['gid'].iloc[0]} — 6-h-ahead predictions "
                     "(gray spans: real-time obs were stale/absent)", fontsize=10)
        ax.set_ylabel("Q (m³/s)")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        style(ax)
    axes[0, 0].legend(loc="upper right", fontsize=8.5, ncol=4)
    fig.suptitle("Stale-gauge events: what each model predicted with degraded obs")
    fig.savefig(os.path.join(OUT, "fig4_stale_event_hydrographs.png"), dpi=150)
    plt.close(fig)
    print("fig4 done", flush=True)

# ---------------------------------------------------------- fig 5: CONUS maps
allg = pd.read_csv(os.path.join(BASE, "allgauges_latlon.csv"))
div = LinearSegmentedColormap.from_list(
    "nse_div", ["#7a2423", "#e34948", "#f0efec", "#5598e7", "#0d366b"])
p6 = pg[pg["family"] == "6h"]
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), constrained_layout=True)
panels = [("nse_v1_6", "v1 (replay)"), ("nse_v2_6", "v2 (replay)"),
          ("nse_v3_6", "v3 (replay)")]
norm = Normalize(vmin=-1, vmax=1)
for ax, (col, title) in zip(axes.flat[:3], panels):
    ax.scatter(allg["lon"], allg["lat"], s=2, color=GRID, lw=0, zorder=1)
    d = p6.dropna(subset=[col])
    sc = ax.scatter(d["lon"], d["lat"], c=np.clip(d[col], -1, 1), cmap=div,
                    norm=norm, s=16, lw=0.3, edgecolors=SURFACE, zorder=2)
    ax.set_title(f"per-gauge NSE — {title}")
# difference panel
ax = axes.flat[3]
ax.scatter(allg["lon"], allg["lat"], s=2, color=GRID, lw=0, zorder=1)
d = p6.dropna(subset=["nse_v1_6", "nse_v3_6"]).copy()
diff = np.clip(d["nse_v3_6"], -1, 1) - np.clip(d["nse_v1_6"], -1, 1)
nd = Normalize(vmin=-0.5, vmax=0.5)
sc2 = ax.scatter(d["lon"], d["lat"], c=np.clip(diff, -0.5, 0.5), cmap=div,
                 norm=nd, s=16, lw=0.3, edgecolors=SURFACE, zorder=2)
ax.set_title("v3 − v1 (clipped NSE difference)")
for ax in axes.flat:
    ax.set_aspect(1.25)
    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 50)
    ax.grid(False)
    ax.tick_params(labelsize=8)
cb = fig.colorbar(sc, ax=axes[:, 0], shrink=0.55, aspect=28, pad=0.01)
cb.set_label("NSE (clipped at −1)", color=INK2, fontsize=9)
cb2 = fig.colorbar(sc2, ax=axes[:, 1], shrink=0.55, aspect=28, pad=0.01)
cb2.set_label("ΔNSE (v3 − v1)", color=INK2, fontsize=9)
fig.suptitle("Per-gauge 6-h nowcast NSE across CONUS — 438 scored out-of-sample "
             "gauges (gray: full gauge network)")
fig.savefig(os.path.join(OUT, "fig5_conus_nse_map.png"), dpi=150)
plt.close(fig)
print("fig5 done", flush=True)
print("all figures written to", OUT, flush=True)
