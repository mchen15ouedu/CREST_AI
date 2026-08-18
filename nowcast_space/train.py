"""Resumable DI-LSTM training. GPU bursts fit inside ZeroGPU's per-call window;
press Train again to continue — the checkpoint round-trips through the private
model repo, so progress survives Space restarts."""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from model import (DILSTM, H, L, N_FEAT, AGE_CAP_H, blank_obs, build_features,
                   make_windows, n_feat_for, nse, itq)

MODEL_REPO = os.environ.get("NOWCAST_MODEL_REPO", "vincewin/CREST_nowcast_model")
CKPT = os.environ.get("NOWCAST_CKPT", "dilstm.pt")


def _token():
    return os.environ.get("HF_TOKEN")


def load_ckpt(map_location="cpu", name: str = CKPT):
    """Checkpoint from the model repo, or None."""
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(MODEL_REPO, name, token=_token())
        return torch.load(p, map_location=map_location, weights_only=False)
    except Exception:
        return None


def save_ckpt(ck: dict, name: str = CKPT):
    from huggingface_hub import HfApi
    api = HfApi(token=_token())
    api.create_repo(MODEL_REPO, repo_type="model", private=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(ck, buf)
    api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=name,
                    repo_id=MODEL_REPO, repo_type="model",
                    commit_message=f"{name}: epoch {ck['epoch']} val_nse={ck.get('val_nse')}")


def build_dataset(gauges: list[dict], months: list[str], val_months: list[str],
                  log=print):
    """Assemble train/val tensors from the prepared per-month series."""
    from data import load_series_bulk
    la = [np.log10(max(g["area_km2"], 1.0)) for g in gauges]
    stats = {"la_mean": float(np.mean(la)), "la_std": float(np.std(la) or 1.0)}
    rng = np.random.default_rng(42)

    gids = [g["id"] for g in gauges]
    tr_frames = load_series_bulk(gids, months, log=log)
    va_frames = load_series_bulk(gids, val_months, log=log)

    def one(gid_g, df, stride):
        if df.empty or df["q"].notna().sum() < 500:
            return None
        q = df["q"].to_numpy()
        p = np.nan_to_num(df["p"].to_numpy(), nan=0.0)
        # random staleness augmentation: the "last known obs" the model sees
        # lags truth by 1..12 h (teaches the obs_age channel)
        lag = int(rng.integers(1, 13))
        n = len(q)
        # last_t[t] = index of most recent finite q at or before t-lag
        vidx = np.where(np.isfinite(q), np.arange(n, dtype="float64"), np.nan)
        shifted = np.full(n, np.nan)
        if lag < n:
            shifted[lag:] = vidx[:n - lag]
        last_t = pd.Series(shifted).ffill().to_numpy()
        fin = np.isfinite(last_t)
        obs_ff = np.where(fin, q[np.where(fin, last_t, 0).astype(np.int64)], np.nan)
        age = np.where(fin, np.arange(n) - last_t, 999.0)
        feat = build_features(p, np.nan_to_num(obs_ff, nan=0.0), age,
                              gid_g["area_km2"], stats)
        return make_windows(feat, q, stride=stride)

    Xtr, Ytr, Xva, Yva = [], [], [], []
    for g in gauges:
        tr = one(g, tr_frames[g["id"]], stride=3)
        va = one(g, va_frames[g["id"]], stride=6)
        if tr is not None and len(tr[0]):
            Xtr.append(tr[0]); Ytr.append(tr[1])
        if va is not None and len(va[0]):
            Xva.append(va[0]); Yva.append(va[1])
        log(f"  {g['id']}: train {0 if tr is None else len(tr[0])} / "
            f"val {0 if va is None else len(va[0])} windows")
    if not Xtr:
        return None
    return (np.concatenate(Xtr), np.concatenate(Ytr),
            np.concatenate(Xva) if Xva else np.zeros((0, L, N_FEAT), "float32"),
            np.concatenate(Yva) if Yva else np.zeros((0, H), "float32"), stats)


def _obs_process(q: np.ndarray, rng, augment: bool):
    """Simulate the observation stream the model sees at inference: a
    reporting lag, and (when augment=True) random multi-hour OUTAGES during
    which the last obs freezes and its age grows — the exact regime where the
    v1 model hallucinated. Returns (obs_ff, age_h); age is huge before the
    first obs (build_features maps that to the missing state)."""
    T = len(q)
    lag = int(rng.integers(1, 7)) if augment else 1
    reporting = np.ones(T, bool)
    if augment:
        for _ in range(int(rng.poisson(max(T / 720.0, 0.5)))):   # ~1 per 30 d
            s = int(rng.integers(0, T))
            reporting[s:s + int(rng.integers(12, 97))] = False   # 12-96 h out
    obs_ff = np.zeros(T)
    age = np.full(T, 1e9)
    last, last_t = 0.0, None
    for t in range(T):
        ts = t - lag
        if ts >= 0 and np.isfinite(q[ts]) and reporting[ts]:
            last, last_t = q[ts], ts
        if last_t is not None:
            obs_ff[t] = last
            age[t] = t - last_t
    return obs_ff, age


def build_dataset_v2(gauges: list[dict], months: list[str], val_months: list[str],
                     horizon: int = H, obs_dropout: float = 0.15, seed: int = 42,
                     statics: dict[str, np.ndarray] | None = None,
                     bake_statics: bool = True, log=print):
    """feat_version-2 dataset with per-window gauge indices.

    bake_statics=False (feat_version 3 only) keeps the window tensors at the
    5-channel v2 layout and returns the z-scored per-gauge static vectors as
    a separate "statics_z" array [n_gauge, N_STATIC] instead — the trainer
    concatenates them per batch, which cuts window memory ~4x at full scale.

    Training windows get (a) the outage-augmented obs stream from
    _obs_process and (b) window-level obs-channel dropout: a fraction
    obs_dropout of windows is rewritten to the dead-gauge state (obs=0,
    age=AGE_CAP, missing=1) with the TRUE discharge kept as target, teaching
    a rainfall-driven fallback instead of free extrapolation.

    Validation windows use a clean 1-h-lag obs stream (obs-fresh skill);
    the no-obs diagnostic is computed by the caller via model.blank_obs.

    Returns dict(Xtr, Ytr, Gtr, Xva, Yva, Gva, stats, gauge_ids) or None.
    """
    from data import load_series_bulk
    la = [np.log10(max(g["area_km2"], 1.0)) for g in gauges]
    stats = {"la_mean": float(np.mean(la)), "la_std": float(np.std(la) or 1.0)}
    rng = np.random.default_rng(seed)

    fv, zs = 2, {}
    if statics is not None:
        # z-score the raw static vectors over THIS gauge set; NaNs -> column
        # median. Everything stored in stats so serving can reproduce it.
        fv = 3
        raw = np.stack([np.asarray(statics[g["id"]], "float64") for g in gauges])
        med = np.nanmedian(raw, axis=0)
        raw = np.where(np.isfinite(raw), raw, med[None, :])
        mu, sd = raw.mean(0), np.maximum(raw.std(0), 1e-6)
        stats.update(static_mean=mu.tolist(), static_std=sd.tolist(),
                     static_median=med.tolist())
        zs = {g["id"]: ((raw[i] - mu) / sd).astype("float32")
              for i, g in enumerate(gauges)}
    fv_bake = fv if bake_statics else min(fv, 2)     # channel layout of windows

    gids = [g["id"] for g in gauges]
    tr_frames = load_series_bulk(gids, months, log=log)
    va_frames = load_series_bulk(gids, val_months, log=log)

    def one(g, df, stride, augment):
        if df.empty or df["q"].notna().sum() < 500:
            return None
        q = df["q"].to_numpy()
        p = np.nan_to_num(df["p"].to_numpy(), nan=0.0)
        obs_ff, age = _obs_process(q, rng, augment)
        feat = build_features(p, obs_ff, age, g["area_km2"], stats,
                              feat_version=fv_bake,
                              statics=zs.get(g["id"]) if fv_bake >= 3 else None)
        X, Y, _ = make_windows(feat, q, stride=stride, horizon=horizon)
        if augment and len(X) and obs_dropout > 0:
            k = rng.random(len(X)) < obs_dropout
            if k.any():
                X[k] = blank_obs(X[k])
        return X, Y

    Xtr, Ytr, Gtr, Xva, Yva, Gva = [], [], [], [], [], []
    gauge_ids = [g["id"] for g in gauges]
    for gi, g in enumerate(gauges):
        tr = one(g, tr_frames[g["id"]], stride=3, augment=True)
        va = one(g, va_frames[g["id"]], stride=6, augment=False)
        if tr is not None and len(tr[0]):
            Xtr.append(tr[0]); Ytr.append(tr[1])
            Gtr.append(np.full(len(tr[0]), gi, "int32"))
        if va is not None and len(va[0]):
            Xva.append(va[0]); Yva.append(va[1])
            Gva.append(np.full(len(va[0]), gi, "int32"))
        log(f"  {g['id']}: train {0 if tr is None else len(tr[0])} / "
            f"val {0 if va is None else len(va[0])} windows")
    if not Xtr:
        return None
    nf = n_feat_for(fv_bake)
    out = {"Xtr": np.concatenate(Xtr), "Ytr": np.concatenate(Ytr),
           "Gtr": np.concatenate(Gtr),
           "Xva": (np.concatenate(Xva) if Xva
                   else np.zeros((0, L, nf), "float32")),
           "Yva": (np.concatenate(Yva) if Yva
                   else np.zeros((0, horizon), "float32")),
           "Gva": np.concatenate(Gva) if Gva else np.zeros(0, "int32"),
           "stats": stats, "gauge_ids": gauge_ids}
    if fv >= 3 and not bake_statics:
        out["statics_z"] = np.stack([zs[g["id"]] for g in gauges])
    return out


def train_burst(dataset, ck: dict | None, seconds: float = 200.0,
                device: str = "cuda", log=print) -> dict:
    """One time-boxed training burst; returns an updated checkpoint dict."""
    Xtr, Ytr, Xva, Yva, stats = dataset
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = DILSTM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epoch = 0
    if ck:
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        epoch = ck["epoch"]
        stats = ck["stats"]
    Xt = torch.from_numpy(Xtr); Yt = torch.from_numpy(Ytr)
    n = len(Xt)
    bs = 512
    t_end = time.time() + seconds
    model.train()
    while time.time() < t_end:
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx].to(dev), Yt[idx].to(dev)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
            if time.time() > t_end:
                break
        epoch += 1
        log(f"  epoch {epoch}: train MSE {tot / n:.4f}")
    # validation NSE in real space, all horizons pooled
    val_nse = None
    if len(Xva):
        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xva).to(dev)).cpu().numpy()
        val_nse = round(nse(itq(pv).ravel(), itq(Yva).ravel()), 3)
        log(f"  val NSE (pooled, {len(Xva)} windows): {val_nse}")
    # EVERYTHING returned must be CPU tensors: ZeroGPU pickles the result back
    # into a parent process where CUDA is emulated (rebuild_cuda_tensor crashes)
    opt_sd = opt.state_dict()
    for st in opt_sd.get("state", {}).values():
        for k2, v2 in list(st.items()):
            if torch.is_tensor(v2):
                st[k2] = v2.cpu()
    return {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "opt": opt_sd, "epoch": epoch, "stats": stats,
            "val_nse": val_nse, "n_train": int(n), "horizon": H, "lookback": L,
            "feat_version": 1, "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
