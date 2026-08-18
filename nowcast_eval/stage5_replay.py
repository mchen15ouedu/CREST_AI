"""Stage 5: replay all six checkpoints over the archived issue times.

For every archived issue t0 and eval gauge, rebuild the 72-h input window the
operational writer would have built:
  - precip: the same Pass1-first basin means (stage 4), NaN->0 like the writer
  - obs: post-hoc NWIS rows TRUNCATED at the archived obs_last_time, so the
    model sees exactly the staleness the operational run saw
then run v1/v2/v3 x 6h/12h checkpoints. Also extracts the archived
operational predictions for the same (issue, gauge) pairs.

Outputs: replay_preds.npz   preds[name] (n_issue, n_gauge, horizon)
         archive_eval.npz   archived q1..q6 / q12_*, obs_age_h, cutoffs
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "/home/MengyuChen/CREST_AI/nowcast_space")
BASE = os.path.dirname(os.path.abspath(__file__))
AGE_CAP_H = 240.0

# ---- inputs ------------------------------------------------------------------
z = np.load(os.path.join(BASE, "precip.npz"))
gids = [str(g) for g in z["gids"]]
hours = [str(h) for h in z["hours"]]
pmat = z["pmat"]                                   # (nG, nH)
hour_dt = pd.to_datetime(hours, format="%Y%m%d%H").values.astype("datetime64[s]")
hour_pos = {h: i for i, h in enumerate(hours)}
nG = len(gids)
g2col = {g: k for k, g in enumerate(gids)}

od = pd.read_parquet(os.path.join(BASE, "obs_rows.parquet"))
obs_t, obs_v = {}, {}
for g, sub in od.groupby("gid"):
    obs_t[g] = sub["dt"].to_numpy().astype("datetime64[s]")
    obs_v[g] = sub["q_cms"].to_numpy().astype("float64")
print(f"{nG} gauges, obs for {len(obs_t)}", flush=True)

iss = pd.read_csv(os.path.join(BASE, "issues.csv"), dtype=str).sort_values("t0")
iss = iss[iss["t0"].isin(hour_pos)].reset_index(drop=True)
nI = len(iss)
t0_pos = np.array([hour_pos[t] for t in iss["t0"]])
print(f"{nI} issues", flush=True)

# ---- archived cutoffs + operational predictions ------------------------------
q6c = [f"q{k}" for k in range(1, 7)]
q12c = [f"q12_{k}" for k in range(1, 13)]
cut = np.full((nI, nG), np.datetime64("NaT"), "datetime64[s]")
a_age = np.full((nI, nG), np.nan, "float32")
a_q6 = np.full((nI, nG, 6), np.nan, "float32")
a_q12 = np.full((nI, nG, 12), np.nan, "float32")
for j, path in enumerate(iss["path"]):
    t = pq.read_table(os.path.join(BASE, path),
                      columns=["gid", "obs_last_time", "obs_age_h"] + q6c + q12c)
    ag = [str(x).zfill(8) for x in t.column("gid").to_pylist()]
    sel = np.array([g2col.get(g, -1) for g in ag])
    m = sel >= 0
    cols = sel[m]
    olt = pd.to_datetime(pd.Series(t.column("obs_last_time").to_pylist()),
                         format="%Y-%m-%d %H:%M", errors="coerce").values
    cut[j, cols] = olt.astype("datetime64[s]")[m]
    a_age[j, cols] = t.column("obs_age_h").to_numpy()[m]
    for k, c in enumerate(q6c):
        a_q6[j, cols, k] = t.column(c).to_numpy()[m]
    for k, c in enumerate(q12c):
        a_q12[j, cols, k] = t.column(c).to_numpy()[m]
    if j % 100 == 0:
        print(f"  archive {j}/{nI}", flush=True)

# ---- obs stream per (issue, gauge, window hour) ------------------------------
W = t0_pos[:, None] - 71 + np.arange(72)[None, :]           # (nI, 72) hour idx
OBS = np.zeros((nG, nI, 72), "float32")
AGE = np.full((nG, nI, 72), 999.0, "float32")
for k, g in enumerate(gids):
    T, V = obs_t.get(g), obs_v.get(g)
    if T is None or not len(T):
        continue
    last_le = np.searchsorted(T, hour_dt, side="right") - 1   # (nH,)
    cnat = np.isnat(cut[:, k])
    ci = np.full(nI, -1, np.int64)
    ci[~cnat] = np.searchsorted(T, cut[~cnat, k], side="right") - 1
    idx = np.minimum(last_le[W], ci[:, None])                 # (nI, 72)
    ok = idx >= 0
    ic = np.clip(idx, 0, None)
    OBS[k] = np.where(ok, V[ic], 0.0)
    AGE[k] = np.where(ok, (hour_dt[W] - T[ic]) / np.timedelta64(3600, "s"), 999.0)
print("obs streams built", flush=True)

# ---- per-checkpoint feature assembly + inference -----------------------------
from model import DILSTM, n_feat_for  # noqa: E402

arch0 = sorted(glob.glob(os.path.join(BASE, "archive/nowcast/archive/*/nc_*.parquet")))[0]
t = pq.read_table(arch0, columns=["gid", "area_km2"])
amap = {str(g).zfill(8): a for g, a in zip(t.column("gid").to_pylist(),
                                           t.column("area_km2").to_numpy())}
area = np.array([amap[g] for g in gids], "float64")

sdf = pd.read_parquet(os.path.join(BASE, "ckpts", "gauges", "gauge_statics.parquet"))
sdf = sdf.set_index("STAID")
scols = [c for c in sdf.columns if c != "hybas_id"]
raw_statics = {str(i).zfill(8): sdf.loc[i, scols].to_numpy("float64")
               for i in sdf.index}

CKPTS = {"v1_6": "dilstm.pt", "v2_6": "dilstm_v2.pt", "v3_6": "dilstm_v3.pt",
         "v1_12": "dilstm_h12.pt", "v2_12": "dilstm_h12_v2.pt",
         "v3_12": "dilstm_h12_v3.pt",
         "v3f_6": "dilstm_v3_full.pt", "v3f_12": "dilstm_h12_v3_full.pt"}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P = np.log1p(np.nan_to_num(np.maximum(pmat, 0.0)))          # writer semantics

out = {}
for name, fn in CKPTS.items():
    ck = torch.load(os.path.join(BASE, "ckpts", fn), map_location="cpu",
                    weights_only=False)
    fv = int(ck.get("feat_version", 1))
    hor = int(ck.get("horizon", 6))
    nf = n_feat_for(fv)
    stats = ck["stats"]
    la = ((np.log10(np.maximum(area, 1.0)) - stats["la_mean"])
          / max(stats["la_std"], 1e-6)).astype("float32")
    stz = None
    if fv >= 3:
        mu = np.array(stats["static_mean"]); sd = np.array(stats["static_std"])
        med = np.array(stats["static_median"])
        rows = []
        for g in gids:
            r = raw_statics.get(g, med.copy())
            r = np.where(np.isfinite(r), r, med)
            rows.append((r - mu) / sd)
        stz = np.asarray(rows, "float32")
    model = DILSTM(n_feat=nf, horizon=hor).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    preds = np.full((nI, nG, hor), np.nan, "float32")
    with torch.no_grad():
        for j in range(nI):
            f = np.zeros((nG, 72, nf), "float32")
            f[:, :, 0] = P[:, W[j]]
            f[:, :, 3] = la[:, None]
            obs_j, age_j = OBS[:, j], AGE[:, j]
            if fv >= 2:
                miss = (age_j > AGE_CAP_H) | (age_j >= 999.0)
                f[:, :, 1] = np.where(miss, 0.0, np.log1p(np.maximum(obs_j, 0.0)))
                f[:, :, 2] = np.minimum(age_j, AGE_CAP_H) / 24.0
                f[:, :, 4] = miss
                if fv >= 3:
                    f[:, :, 5:] = stz[:, None, :]
            else:
                f[:, :, 1] = np.log1p(np.maximum(obs_j, 0.0))
                f[:, :, 2] = age_j / 24.0
            y = model(torch.from_numpy(f).to(dev)).cpu().numpy()
            preds[j] = np.maximum(np.expm1(y), 0.0)
            if j % 100 == 0:
                print(f"  {name}: issue {j}/{nI}", flush=True)
    out[name] = preds
    print(f"{name}: done (fv{fv}, h{hor}, epoch {ck.get('epoch')})", flush=True)

out["persist"] = OBS[:, :, 71].T.copy()      # last-known obs at t0 (nI, nG)
np.savez_compressed(os.path.join(BASE, "replay_preds.npz"),
                    gids=np.array(gids), t0=iss["t0"].to_numpy(),
                    **out)
np.savez_compressed(os.path.join(BASE, "archive_eval.npz"),
                    gids=np.array(gids), t0=iss["t0"].to_numpy(),
                    obs_age=a_age, q6=a_q6, q12=a_q12,
                    cutoff=cut.astype("datetime64[s]").astype("int64"))
print("saved replay_preds.npz + archive_eval.npz", flush=True)
