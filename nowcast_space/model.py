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

Feature vector per hourly step (L=72 lookback):
  0 precip     log1p(basin-mean MRMS, mm/h)
  1 obs_lag    log1p(most recent observed Q at or before t, m3/s)
  2 obs_age    hours since that observation / 24
  3 log_area   log10(drainage area km2), z-scored with stored stats
Output: next H=6 hourly log1p(Q).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

L = 72          # lookback hours
H = 6           # forecast horizon hours
N_FEAT = 4


class DILSTM(nn.Module):
    def __init__(self, n_feat: int = N_FEAT, hidden: int = 128,
                 layers: int = 2, horizon: int = H):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                            batch_first=True, dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):                       # x: [B, L, N_FEAT]
        out, _ = self.lstm(x)
        return self.head(out[:, -1])            # [B, H] log1p(Q)


def tq(q):                                      # transform discharge
    return np.log1p(np.maximum(np.asarray(q, dtype="float64"), 0.0))


def itq(y):                                     # inverse transform
    return np.expm1(np.asarray(y, dtype="float64"))


def build_features(precip: np.ndarray, obs_q: np.ndarray, obs_age_h: np.ndarray,
                   area_km2: float, stats: dict) -> np.ndarray:
    """Stack the per-step feature matrix [T, N_FEAT]. All inputs length T;
    obs_q is the LAST-KNOWN observation at each step (forward-filled),
    obs_age_h its age. stats = {'la_mean','la_std'} from the checkpoint."""
    la = (np.log10(max(area_km2, 1.0)) - stats["la_mean"]) / max(stats["la_std"], 1e-6)
    T = len(precip)
    f = np.zeros((T, N_FEAT), dtype="float32")
    f[:, 0] = np.log1p(np.maximum(precip, 0.0))
    f[:, 1] = tq(obs_q)
    f[:, 2] = np.asarray(obs_age_h, dtype="float32") / 24.0
    f[:, 3] = la
    return f


def make_windows(feat: np.ndarray, target_q: np.ndarray, stride: int = 3):
    """Sliding (X[L,F], y[H]) pairs; windows containing NaN targets skipped."""
    X, Y = [], []
    ty = tq(target_q)
    for t in range(L, len(feat) - H, stride):
        y = ty[t:t + H]
        if np.isnan(y).any() or np.isnan(feat[t - L:t]).any():
            continue
        X.append(feat[t - L:t])
        Y.append(y)
    if not X:
        return (np.zeros((0, L, N_FEAT), "float32"), np.zeros((0, H), "float32"))
    return np.stack(X).astype("float32"), np.stack(Y).astype("float32")


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < 3:
        return float("nan")
    s, o = sim[m], obs[m]
    den = ((o - o.mean()) ** 2).sum()
    return float(1 - ((s - o) ** 2).sum() / den) if den > 0 else float("nan")
