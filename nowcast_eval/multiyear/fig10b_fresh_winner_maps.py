"""fig10b: only the fresh-obs winner maps (6-h and 12-h), same styling as fig10."""
import os, geopandas as gpd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
BASE=os.path.dirname(os.path.abspath(__file__)); OUT="/home/MengyuChen/CREST_AI/eval_figures"
SURFACE,INK,INK2,BASELINE,C1,C2="#fcfcfb","#0b0b0b","#52514e","#c3c2b7","#2a78d6","#eb6834"
plt.rcParams.update({"figure.facecolor":SURFACE,"axes.facecolor":SURFACE,"savefig.facecolor":SURFACE,"font.family":"sans-serif","text.color":INK,"axes.titlesize":11,"figure.titlesize":13})
rt=pd.read_csv(os.path.join(BASE,"route_pergauge_multiyear.csv"),dtype={"gid":str}).fillna("")
ll=pd.read_csv(os.path.join(BASE,"..","allgauges_latlon.csv"),dtype={"gid":str}); rt=rt.merge(ll,on="gid",how="left")
d=rt[(rt.lon>-125.5)&(rt.lon<-66)&(rt.lat>24)&(rt.lat<50)]
world=gpd.read_file(gpd.datasets.get_path("naturalearth_lowres")); usa=world[world.name=="United States of America"]
fig,axes=plt.subplots(1,2,figsize=(15,5.4),constrained_layout=True)
for ax,fam in zip(axes,("6","12")):
    c=f"winner_{fam}_fresh"
    usa.boundary.plot(ax=ax,color=BASELINE,linewidth=0.8,zorder=0)
    und,v1,v3=d[d[c]==""],d[d[c]=="v1"],d[d[c]=="v3"]
    ax.scatter(und.lon,und.lat,s=3,color="#c3c2b7",alpha=0.6,lw=0,zorder=1)
    ax.scatter(v1.lon,v1.lat,s=6,color=C1,alpha=0.85,lw=0,zorder=2)
    ax.scatter(v3.lon,v3.lat,s=6,color=C2,alpha=0.85,lw=0,zorder=3)
    ax.set_xlim(-125.5,-66); ax.set_ylim(24,50); ax.set_aspect(1.25); ax.set_axis_off()
    n=len(v1)+len(v3)
    ax.set_title(f"{fam}-h family, fresh obs at issue time: v1 best at {len(v1)} ({100*len(v1)/n:.0f}%), v3-full best at {len(v3)} ({100*len(v3)/n:.0f}%), {len(und)} undecided",loc="left",fontsize=9.5)
    ax.legend(handles=[Line2D([],[],marker="o",ls="",ms=6,color=C1,label="v1 better"),Line2D([],[],marker="o",ls="",ms=6,color=C2,label="v3-full better"),Line2D([],[],marker="o",ls="",ms=4,color="#c3c2b7",label="undecided (no truth)")],loc="lower left",fontsize=8.5)
fig.suptitle("Best nowcast model per gauge with fresh observations — all 43 evaluation months (Jan/Apr/Jul/Oct 2016–2026), per-gauge NSE over all leads")
fig.savefig(os.path.join(OUT,"fig10b_multiyear_winner_maps_fresh_only.png"),dpi=150); print("saved fig10b")
