import os, numpy as np, pandas as pd, geopandas as gpd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BASE=os.path.dirname(os.path.abspath(__file__)); OUT="/home/MengyuChen/CREST_AI/eval_figures"
SURFACE,INK,INK2,BASELINE="#fcfcfb","#0b0b0b","#52514e","#c3c2b7"
plt.rcParams.update({"figure.facecolor":SURFACE,"axes.facecolor":SURFACE,"savefig.facecolor":SURFACE,"font.family":"sans-serif","text.color":INK,"axes.titlesize":11,"figure.titlesize":13})
d=pd.read_csv(os.path.join(BASE,"pergauge_with_gages2.csv"),dtype={"gid":str})
ll=pd.read_csv(os.path.join(BASE,"..","allgauges_latlon.csv"),dtype={"gid":str}); d=d.merge(ll,on="gid")
d=d[(d.lon>-125.5)&(d.lon<-66)&(d.lat>24)&(d.lat<50)]
usa=gpd.read_file(gpd.datasets.get_path("naturalearth_lowres")); usa=usa[usa.name=="United States of America"]
fig,axes=plt.subplots(2,1,figsize=(11,11.6),constrained_layout=True)
# panel 1: v3-full no-obs skill classes (sequential-ish, 3 classes + fail)
ax=axes[0]; usa.boundary.plot(ax=ax,color=BASELINE,linewidth=0.8,zorder=0)
v=d["nse_v3f_6_noobs_oos"]
cls=[("NSE ≥ 0.4",v>=0.4,"#0b3d91",8),("0.1 – 0.4",(v>=0.1)&(v<0.4),"#3987e5",6),("0 – 0.1",(v>=0)&(v<0.1),"#a9c9f0",5),("< 0 (worse than mean)",v<0,"#eb6834",4)]
for lab,m,c,s in cls[::-1]:
    ax.scatter(d.lon[m],d.lat[m],s=s,color=c,lw=0,alpha=0.85,label=f"{lab} ({int(m.sum())})")
ax.set_xlim(-125.5,-66); ax.set_ylim(24,50); ax.set_aspect(1.25); ax.set_axis_off()
ax.legend(loc="lower left",fontsize=9,markerscale=1.5); ax.set_title("v3-full skill WITHOUT observations (6-h family, per-gauge NSE, out-of-sample months) — the ungauged / dead-feed regime",loc="left")
# panel 2: model gain over persistence with fresh obs (v1, 6h)
ax=axes[1]; usa.boundary.plot(ax=ax,color=BASELINE,linewidth=0.8,zorder=0)
g=d["dp_fresh6"]
cls=[("gain ≥ 0.05",g>=0.05,"#0b3d91",8),("0.01 – 0.05",(g>=0.01)&(g<0.05),"#3987e5",6),("|gain| < 0.01 (≈ persistence)",(g>-0.01)&(g<0.01),"#c3c2b7",4),("worse than persistence",g<=-0.01,"#eb6834",6)]
for lab,m,c,s in cls[::-1]:
    ax.scatter(d.lon[m],d.lat[m],s=s,color=c,lw=0,alpha=0.85,label=f"{lab} ({int(m.sum())})")
ax.set_xlim(-125.5,-66); ax.set_ylim(24,50); ax.set_aspect(1.25); ax.set_axis_off()
ax.legend(loc="lower left",fontsize=9,markerscale=1.5); ax.set_title("Where the nowcast adds value over persistence WITH fresh obs (v1 6-h family, per-gauge NSE minus persistence NSE)",loc="left")
fig.savefig(os.path.join(OUT,"fig13_noobs_skill_and_persistence_gain_maps.png"),dpi=150); print("saved fig13")
