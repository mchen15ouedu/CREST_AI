import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
G="/media/scratch/MengyuChen/nowcast_eval/multiyear/gages2/./"
def rd(f, cols=None):
    d=pd.read_csv(G+f, dtype={"STAID":str}, encoding="latin1", low_memory=False)
    d["STAID"]=d["STAID"].str.zfill(8)
    return d if cols is None else d[["STAID"]+[c for c in cols if c in d.columns]]
a=rd("conterm_bas_classif.txt",["CLASS","AGGECOREGION","HYDRO_DISTURB_INDX"])
a=a.merge(rd("conterm_basinid.txt",["DRAIN_SQKM","HUC02"]),on="STAID",how="left")
a=a.merge(rd("conterm_climate.txt",["PPTAVG_BASIN","T_AVG_BASIN","SNOW_PCT_PRECIP","PRECIP_SEAS_IND","PET","WD_BASIN"]),on="STAID",how="left")
a=a.merge(rd("conterm_hydro.txt",["BFI_AVE","STREAMS_KM_SQ_KM","RUNAVE7100","PERDUN","TOPWET","CONTACT"]),on="STAID",how="left")
a=a.merge(rd("conterm_topo.txt",["ELEV_MEAN_M_BASIN","SLOPE_PCT","RRMEAN"]),on="STAID",how="left")
a=a.merge(rd("conterm_hydromod_dams.txt",["NDAMS_2009","STOR_NOR_2009","MAJ_NDAMS_2009","pre1940_STOR"]),on="STAID",how="left")
a=a.merge(rd("conterm_lc06_basin.txt",["DEVNLCD06","FORESTNLCD06","PLANTNLCD06","WATERNLCD06","IMPNLCD06"]),on="STAID",how="left")
a=a.merge(rd("conterm_soils.txt",["PERMAVE","CLAYAVE","SANDAVE","AWCAVE"]),on="STAID",how="left")
a["aridity"]=a["PET"]/a["PPTAVG_BASIN"]           # PET (mm) / mean annual precip (cm*10?) check units
sc=pd.read_csv("score_pergauge.csv",dtype={"gid":str}); rt=pd.read_csv("route_pergauge_multiyear.csv",dtype={"gid":str}).fillna("")
d=sc.merge(rt,on="gid").merge(a,left_on="gid",right_on="STAID",how="inner")
print("joined", len(d), "of", len(sc), "; PPT units sample:", a["PPTAVG_BASIN"].describe()[["min","50%","max"]].round(1).to_dict(), "PET:", a["PET"].describe()[["min","50%","max"]].round(0).to_dict())
d["dv_fresh6"]=d["nse_v3f_6_fresh_oos"]-d["nse_v1_6_fresh_oos"]; d["dv_fresh12"]=d["nse_v3f_12_fresh_oos"]-d["nse_v1_12_fresh_oos"]
d["dv_stale6"]=d["nse_v3f_6_stale24_oos"]-d["nse_v1_6_stale24_oos"]; d["dv_noobs6"]=d["nse_v3f_6_noobs_oos"]-d["nse_v1_6_noobs_oos"]
d["dp_fresh6"]=d["nse_v1_6_fresh_oos"]-d["nse_persist_6_fresh_oos"]   # model gain over persistence
d["v3noobs6"]=d["nse_v3f_6_noobs_oos"]; d["v3stale6"]=d["nse_v3f_6_stale24_oos"]
d["ff6"]=(d["winner_6_fresh"]=="v3"); d["ff12"]=(d["winner_12_fresh"]=="v3"); d["fs6"]=(d["winner_6_stale"]=="v3"); d["fn6"]=(d["winner_6_noobs"]=="v3")
d=d[d["winner_6_fresh"]!=""]
def tab(by, label=None, order=None):
    g=d.groupby(by)
    t=pd.DataFrame({"n":g.size(),
        "v3win% fresh6":(100*g["ff6"].mean()).round(0),"v3win% fresh12":(100*g["ff12"].mean()).round(0),
        "v3win% stale":(100*g["fs6"].mean()).round(0),"v3win% noobs":(100*g["fn6"].mean()).round(0),
        "med dNSE fresh6":g["dv_fresh6"].median().round(3),"med dNSE fresh12":g["dv_fresh12"].median().round(3),
        "med v3 NSE stale":g["v3stale6"].median().round(2),"med v3 NSE noobs":g["v3noobs6"].median().round(2),"noobs>0 %":(100*(g["v3noobs6"].apply(lambda s:(s>0).mean()))).round(0),
        "med v1 fresh6":g["nse_v1_6_fresh_oos"].median().round(3),"model-persist fresh6":g["dp_fresh6"].median().round(3)})
    if order is not None: t=t.reindex([o for o in order if o in t.index])
    print(f"\n### by {label or by}"); print(t.to_string())
d["area_bin"]=pd.cut(d["DRAIN_SQKM"],[0,50,200,1000,5000,20000,1e7],labels=["<50","50-200","200-1k","1k-5k","5k-20k",">20k"])
tab("area_bin","drainage area km2")
tab("AGGECOREGION","aggregated ecoregion")
tab("HUC02","HUC2 region")
tab("CLASS","GAGES-II class")
d["snow_bin"]=pd.cut(d["SNOW_PCT_PRECIP"],[-1,5,15,30,50,100],labels=["<5%","5-15%","15-30%","30-50%",">50%"]); tab("snow_bin","snow % of precip")
d["ppt_bin"]=pd.cut(d["PPTAVG_BASIN"],[0,40,70,100,130,400],labels=["<40cm","40-70","70-100","100-130",">130"]); tab("ppt_bin","mean annual precip (cm)")
d["bfi_bin"]=pd.cut(d["BFI_AVE"],[0,30,45,60,100],labels=["<30","30-45","45-60",">60"]); tab("bfi_bin","baseflow index")
d["stor_bin"]=pd.cut(d["STOR_NOR_2009"],[-1,0,50,500,5000,1e9],labels=["0","0-50","50-500","500-5000",">5000"]); tab("stor_bin","dam storage (Ml/km2? NOR storage)")
d["dev_bin"]=pd.cut(d["DEVNLCD06"],[-1,5,15,35,100],labels=["<5%","5-15%","15-35%",">35%"]); tab("dev_bin","developed land %")
d["for_bin"]=pd.cut(d["FORESTNLCD06"],[-1,20,50,80,100],labels=["<20%","20-50%","50-80%",">80%"]); tab("for_bin","forest %")
d["slope_bin"]=pd.cut(d["SLOPE_PCT"],[-1,2,5,10,20,100],labels=["<2","2-5","5-10","10-20",">20"]); tab("slope_bin","basin slope %")
d["hdi_bin"]=pd.cut(d["HYDRO_DISTURB_INDX"],[-1,5,10,15,20,50],labels=["<=5","6-10","11-15","16-20",">20"]); tab("hdi_bin","hydro disturbance index")
d["perm_bin"]=pd.cut(d["PERMAVE"],[-1,2,4,8,50],labels=["<2","2-4","4-8",">8"]); tab("perm_bin","soil permeability (in/hr)")
# correlations of dNSE with continuous attributes (Spearman)
cols=["DRAIN_SQKM","PPTAVG_BASIN","T_AVG_BASIN","SNOW_PCT_PRECIP","BFI_AVE","ELEV_MEAN_M_BASIN","SLOPE_PCT","STOR_NOR_2009","DEVNLCD06","FORESTNLCD06","PLANTNLCD06","PERMAVE","HYDRO_DISTURB_INDX","STREAMS_KM_SQ_KM","RUNAVE7100","frac_fresh_t0"]
print("\n### Spearman rho with attributes")
print(pd.DataFrame({k:d[cols+[k]].corr(method="spearman")[k].drop(k).round(2) for k in ["dv_fresh6","dv_fresh12","dv_stale6","v3noobs6","dp_fresh6","nse_v1_6_fresh_oos"]}).to_string())
d.to_csv("pergauge_with_gages2.csv",index=False)
