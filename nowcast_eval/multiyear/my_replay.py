"""Replay one eval month: every hour of the month is an issue; every served
gauge; three obs scenarios (fresh = all obs up to t0 as NWIS holds them;
stale24 = obs cut off 24 h before t0; noobs = no obs at all, i.e. dead
feed / ungauged proxy); models v1 6h/12h, v3-full 6h/12h, persistence.
Saves EXACT per-gauge accumulators (n, sum o, sum o^2, sum (p-o)^2, sum p,
max p, max o) per scenario x model x lead so any subset of months can be
scored later without re-running.   usage: my_replay.py <ym>"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/MengyuChen/CREST_AI/nowcast_space")
from model import DILSTM, n_feat_for  # noqa: E402
from common import (AGE_CAP_H, BASE, CKPTS, LEADS, LOOKBACK, MODELS, SCEN,  # noqa: E402
                    gauges, month_hours)

ym = sys.argv[1]
outp = os.path.join(BASE, "acc", f"acc_{ym}.npz")
if os.path.exists(outp):
    print(f"{ym}: exists"); sys.exit(0)
T0 = time.time()
gids, lat, lon, area = gauges()
nG = len(gids)
hours, i0 = month_hours(ym)
nH0 = len(hours)
# extended hour grid: + LEADS hours for truth of the last issues
from datetime import timedelta  # noqa: E402
hours_ext = hours + [hours[-1] + timedelta(hours=k) for k in range(1, LEADS + 1)]
nH = len(hours_ext)
hour_dt = np.array(hours_ext, dtype="datetime64[s]")

pz = np.load(os.path.join(BASE, "precip", f"precip_{ym}.npz"))
assert list(pz["gids"]) == gids and len(pz["hours"]) == nH0
P = np.zeros((nG, nH), "float32")
P[:, :nH0] = np.log1p(np.nan_to_num(np.maximum(pz["pmat"], 0.0)))
n_pmiss = int(np.isnan(pz["pmat"]).all(0).sum())

od = pd.read_parquet(os.path.join(BASE, "obs", f"obs_{ym}.parquet"))
OBSH = np.zeros((nG, nH), "float32")          # last obs value at/before hour
AGEH = np.full((nG, nH), 999.0, "float32")    # its age (h)
TR = np.full((nG, nH), np.nan, "float32")     # truth: obs within trailing hour
g2k = {g: k for k, g in enumerate(gids)}
for g, sub in od.groupby("gid"):
    k = g2k.get(g, -1)
    if k < 0:
        continue
    T = sub["dt"].to_numpy().astype("datetime64[s]")
    V = sub["q_cms"].to_numpy().astype("float32")
    idx = np.searchsorted(T, hour_dt, side="right") - 1
    ok = idx >= 0
    ic = np.clip(idx, 0, None)
    ag = (hour_dt - T[ic]) / np.timedelta64(3600, "s")
    OBSH[k] = np.where(ok, V[ic], 0.0)
    AGEH[k] = np.where(ok, ag, 999.0)
    TR[k] = np.where(ok & (ag <= 1.0), V[ic], np.nan)
print(f"{ym}: obs streams built ({od['gid'].nunique()} gauges w/ obs), {time.time() - T0:.0f}s", flush=True)
del od

# ---- models -----------------------------------------------------------------
sdf = pd.read_parquet(os.path.join(BASE, "ckpts", "gauges", "gauge_statics.parquet")).set_index("STAID")
scols = [c for c in sdf.columns if c != "hybas_id"]
raw_statics = {str(i).zfill(8): sdf.loc[i, scols].to_numpy("float64") for i in sdf.index}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dev.type == "cpu":
    torch.set_num_threads(int(os.environ.get("REPLAY_THREADS", "8")))
CHUNK = 2048                                   # gauges per forward (GPU memory)
print(f"{ym}: device {dev}", flush=True)
nets = {}
for name, fn in CKPTS.items():
    ck = torch.load(os.path.join(BASE, "ckpts", fn), map_location="cpu", weights_only=False)
    fv = int(ck.get("feat_version", 1)); hor = int(ck.get("horizon", 6)); nf = n_feat_for(fv)
    st = ck["stats"]
    la = ((np.log10(np.maximum(area, 1.0)) - st["la_mean"]) / max(st["la_std"], 1e-6)).astype("float32")
    stz = None
    if fv >= 3:
        mu, sd, med = (np.array(st[k]) for k in ("static_mean", "static_std", "static_median"))
        rows = []
        for g in gids:
            r = raw_statics.get(g, med.copy()); r = np.where(np.isfinite(r), r, med)
            rows.append((r - mu) / sd)
        stz = np.asarray(rows, "float32")
    net = DILSTM(n_feat=nf, horizon=hor).to(dev); net.load_state_dict(ck["state_dict"]); net.eval()
    nets[name] = (net, fv, hor, nf, la, stz)

# ---- accumulators -----------------------------------------------------------
nS, nM = len(SCEN), len(MODELS)
ACC = {k: np.zeros((nS, nM, nG, LEADS), "float64") for k in ("so", "so2", "se2", "sp")}
ACC["n"] = np.zeros((nS, nM, nG, LEADS), "int32")
ACC["maxp"] = np.full((nS, nM, nG, LEADS), -np.inf, "float32")
ACC["maxo"] = np.full((nS, nM, nG, LEADS), -np.inf, "float32")
n_fresh_t0 = np.zeros(nG, "int32")           # issues where obs age at t0 <= 1 h

t0_pos = np.arange(i0, nH0)
nI = len(t0_pos)
lead = np.arange(1, LEADS + 1)
SEL = np.where(np.isfinite(TR[:, i0 + 1:]).any(1))[0]      # gauges scorable this month
nSel = len(SEL)
print(f"{ym}: {nSel}/{nG} gauges have truth this month -> inference restricted to them", flush=True)
sub = {name: (net, fv, hor, nf, la[SEL], None if stz is None else stz[SEL]) for name, (net, fv, hor, nf, la, stz) in nets.items()}


def forward(net, f):
    out = []
    for c0 in range(0, f.shape[0], CHUNK):
        out.append(net(torch.from_numpy(f[c0:c0 + CHUNK]).to(dev)).cpu().numpy())
    return np.concatenate(out, 0)


def accumulate(si, mi, pred, Y):
    """pred (nSel, h), Y (nSel, LEADS): accumulate leads 1..h into the SEL rows."""
    h = pred.shape[1]
    y = Y[:, :h]
    m = np.isfinite(y) & np.isfinite(pred)
    a = ACC
    a["n"][si, mi, SEL, :h] += m
    a["so"][si, mi, SEL, :h] += np.where(m, y, 0)
    a["so2"][si, mi, SEL, :h] += np.where(m, y * y, 0)
    a["se2"][si, mi, SEL, :h] += np.where(m, (pred - y) ** 2, 0)
    a["sp"][si, mi, SEL, :h] += np.where(m, pred, 0)
    a["maxp"][si, mi, SEL, :h] = np.maximum(a["maxp"][si, mi, SEL, :h], np.where(m, pred, -np.inf))
    a["maxo"][si, mi, SEL, :h] = np.maximum(a["maxo"][si, mi, SEL, :h], np.where(m, y, -np.inf))


with torch.no_grad():
    for j, tp in enumerate(t0_pos):
        W = np.arange(tp - LOOKBACK + 1, tp + 1)
        Pw = P[SEL][:, W]
        Y = TR[SEL][:, tp + lead]                                # (nSel, 12)
        n_fresh_t0 += AGEH[:, tp] <= 1.0
        OB, AG = OBSH[SEL], AGEH[SEL]
        for si, sc in enumerate(SCEN):
            if sc == "fresh":
                obs, age = OB[:, W], AG[:, W]
                pers = np.where(AG[:, tp] < 999, OB[:, tp], np.nan)
            elif sc == "stale24":
                obs, age = OB[:, W].copy(), AG[:, W].copy()
                c = tp - 24
                obs[:, -24:] = OB[:, c][:, None]
                age[:, -24:] = np.where(AG[:, c][:, None] < 999,
                                        AG[:, c][:, None] + np.arange(1, 25)[None, :], 999.0)
                pers = np.where(AG[:, c] < 999, OB[:, c], np.nan)
            else:
                obs = np.zeros((nSel, LOOKBACK), "float32"); age = np.full((nSel, LOOKBACK), 999.0, "float32")
                pers = np.full(nSel, np.nan, "float32")
            for name, (net, fv, hor, nf, la, stz) in sub.items():
                f = np.zeros((nSel, LOOKBACK, nf), "float32")
                f[:, :, 0] = Pw
                f[:, :, 3] = la[:, None]
                if fv >= 2:
                    miss = (age > AGE_CAP_H) | (age >= 999.0)
                    f[:, :, 1] = np.where(miss, 0.0, np.log1p(np.maximum(obs, 0.0)))
                    f[:, :, 2] = np.minimum(age, AGE_CAP_H) / 24.0
                    f[:, :, 4] = miss
                    if fv >= 3:
                        f[:, :, 5:] = stz[:, None, :]
                else:
                    f[:, :, 1] = np.log1p(np.maximum(obs, 0.0))
                    f[:, :, 2] = age / 24.0
                pred = np.maximum(np.expm1(forward(net, f)), 0.0)
                accumulate(si, MODELS.index(name), pred, Y)
            accumulate(si, MODELS.index("persist"), np.repeat(pers[:, None], LEADS, 1), Y)
        if j % 100 == 0:
            print(f"  {ym} issue {j}/{nI} ({time.time() - T0:.0f}s)", flush=True)

np.savez_compressed(outp, gids=np.array(gids), scen=np.array(SCEN), models=np.array(MODELS),
                    n_issues=nI, n_fresh_t0=n_fresh_t0, n_precip_missing_hours=n_pmiss, **ACC)
print(f"{ym}: done {nI} issues x {nG} gauges x {nS} scenarios, {time.time() - T0:.0f}s", flush=True)
