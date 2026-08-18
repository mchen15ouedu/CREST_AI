"""DI-LSTM nowcaster — data-integration LSTM for hourly streamflow.

Clean-room implementation of the *method* of Feng, Fang & Shen (2020, WRR
10.1029/2019WR026793): the most recent discharge observation is simply
concatenated to the forcing inputs, merging assimilation and prediction into
one forward pass. Extensions over the published daily setup:
  - hourly timestep with MRMS radar precipitation (their diagnosed failure
    mode for flashy basins was exactly the lack of sub-daily rain intensity);
  - the observation's AGE (hours) is an input, so the model learns how much
    to trust stale gauges (adaptive-kernel idea from Fang & Shen 2020, JHM);
  - random staleness augmentation at training time.

Two feature layouts, selected by the checkpoint's feat_version:

feat_version 1 (legacy, N_FEAT=4), per hourly step (L=72 lookback):
  0 precip     log1p(basin-mean MRMS, mm/h)
  1 obs_lag    log1p(most recent observed Q at or before t, m3/s)
  2 obs_age    hours since that observation / 24  (999/24 if never)
  3 log_area   log10(drainage area km2), z-scored with stored stats

feat_version 2 (obs-robust, N_FEAT_V2=5) adds an explicit missingness flag
and caps the age channel so "no gauge" is a well-defined input state instead
of an out-of-distribution 999:
  0 precip     log1p(basin-mean MRMS, mm/h)
  1 obs_lag    log1p(last obs Q, m3/s); 0 when missing
  2 obs_age    min(hours since obs, AGE_CAP_H) / 24
  3 log_area   z-scored log10 area
  4 obs_missing  1.0 when no obs, or obs older than AGE_CAP_H, else 0.0

Output: next H hourly log1p(Q).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

L = 72          # lookback hours
H = 6           # forecast horizon hours (default; checkpoints carry their own)
N_FEAT = 4      # feat_version 1
N_FEAT_V2 = 5   # feat_version 2
N_STATIC = 16   # feat_version 3: HydroBASINS-L7 catchment attributes
N_FEAT_V3 = N_FEAT_V2 + N_STATIC
AGE_CAP_H = 240.0   # obs older than this counts as missing (feat_version 2)


def n_feat_for(feat_version: int) -> int:
    v = int(feat_version)
    return N_FEAT_V3 if v >= 3 else (N_FEAT_V2 if v == 2 else N_FEAT)


class DILSTM(nn.Module):
    def __init__(self, n_feat: int = N_FEAT, hidden: int = 128,
                 layers: int = 2, horizon: int = H):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                            batch_first=True, dropout=0.1 if layers > 1 else 0.0)
        # parameter-free, so feat_version-1 checkpoints still load; identity in
        # eval mode, active in train mode — which is what mc_predict exploits
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):                       # x: [B, L, n_feat]
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1]))  # [B, H] log1p(Q)


def tq(q):                                      # transform discharge
    return np.log1p(np.maximum(np.asarray(q, dtype="float64"), 0.0))


def itq(y):                                     # inverse transform
    return np.expm1(np.asarray(y, dtype="float64"))


def build_features(precip: np.ndarray, obs_q: np.ndarray, obs_age_h: np.ndarray,
                   area_km2: float, stats: dict, feat_version: int = 1,
                   statics: np.ndarray | None = None) -> np.ndarray:
    """Stack the per-step feature matrix [T, n_feat]. All inputs length T;
    obs_q is the LAST-KNOWN observation at each step (forward-filled),
    obs_age_h its age in hours (large, e.g. 999+, when there has never been
    one). stats = {'la_mean','la_std'} from the checkpoint."""
    la = (np.log10(max(area_km2, 1.0)) - stats["la_mean"]) / max(stats["la_std"], 1e-6)
    T = len(precip)
    age = np.asarray(obs_age_h, dtype="float64")
    f = np.zeros((T, n_feat_for(feat_version)), dtype="float32")
    f[:, 0] = np.log1p(np.maximum(precip, 0.0))
    f[:, 3] = la
    if int(feat_version) >= 2:
        missing = ~np.isfinite(age) | (age > AGE_CAP_H) | (age >= 999.0)
        f[:, 1] = np.where(missing, 0.0, tq(obs_q))
        f[:, 2] = np.where(missing, AGE_CAP_H,
                           np.minimum(np.nan_to_num(age, nan=AGE_CAP_H), AGE_CAP_H)) / 24.0
        f[:, 4] = missing.astype("float32")
        if int(feat_version) >= 3:
            if statics is None or len(statics) != N_STATIC:
                raise ValueError(f"feat_version 3 needs a {N_STATIC}-dim statics vector")
            f[:, N_FEAT_V2:] = np.asarray(statics, "float32")[None, :]
    else:
        f[:, 1] = tq(obs_q)
        f[:, 2] = np.nan_to_num(age, nan=999.0).astype("float32") / 24.0
    return f


def blank_obs(feat: np.ndarray) -> np.ndarray:
    """Return a copy with the obs channels set to the 'gauge missing' state.
    Used for obs-channel dropout in training and for the no-obs (hallucination)
    diagnostic in validation. Only meaningful for feat_version-2 layouts
    (last axis N_FEAT_V2); for v1 layouts it mimics a never-reporting gauge."""
    f = feat.copy()
    f[..., 1] = 0.0
    if f.shape[-1] >= N_FEAT_V2:
        f[..., 2] = AGE_CAP_H / 24.0
        f[..., 4] = 1.0
    else:
        f[..., 2] = 999.0 / 24.0
    return f


def make_windows(feat: np.ndarray, target_q: np.ndarray, stride: int = 3,
                 horizon: int = H):
    """Sliding (X[L,F], y[horizon]) pairs; windows containing NaN targets
    skipped. Returns (X, Y, t_idx) where t_idx is each window's issue step."""
    X, Y, TI = [], [], []
    ty = tq(target_q)
    for t in range(L, len(feat) - horizon, stride):
        y = ty[t:t + horizon]
        if np.isnan(y).any() or np.isnan(feat[t - L:t]).any():
            continue
        X.append(feat[t - L:t])
        Y.append(y)
        TI.append(t)
    if not X:
        nf = feat.shape[-1] if feat.ndim == 2 else N_FEAT
        return (np.zeros((0, L, nf), "float32"), np.zeros((0, horizon), "float32"),
                np.zeros(0, "int64"))
    return (np.stack(X).astype("float32"), np.stack(Y).astype("float32"),
            np.asarray(TI, "int64"))


def mc_predict(model: DILSTM, x: torch.Tensor, n: int = 16, batch: int = 2048):
    """MC-dropout: n stochastic passes (dropout active), no grad. Returns
    (mean, std) numpy arrays in log1p space, shape [B, horizon]. The model's
    prior train/eval mode is restored on return."""
    was_training = model.training
    model.train()                                # activates LSTM + head dropout
    outs = []
    with torch.no_grad():
        for _ in range(max(n, 2)):
            parts = []
            for i in range(0, len(x), batch):
                parts.append(model(x[i:i + batch]).cpu().numpy())
            outs.append(np.concatenate(parts))
    model.eval()
    if was_training:
        model.train()
    arr = np.stack(outs)                         # [n, B, horizon]
    return arr.mean(0), arr.std(0)


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < 3:
        return float("nan")
    s, o = sim[m], obs[m]
    den = ((o - o.mean()) ** 2).sum()
    return float(1 - ((s - o) ** 2).sum() / den) if den > 0 else float("nan")
