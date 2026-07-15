"""Resumable DI-LSTM training. GPU bursts fit inside ZeroGPU's per-call window;
press Train again to continue — the checkpoint round-trips through the private
model repo, so progress survives Space restarts."""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import torch

from model import DILSTM, H, L, N_FEAT, build_features, make_windows, nse, itq

MODEL_REPO = os.environ.get("NOWCAST_MODEL_REPO", "vincewin/CREST_nowcast_model")
CKPT = "dilstm.pt"


def _token():
    return os.environ.get("HF_TOKEN")


def load_ckpt(map_location="cpu"):
    """Checkpoint from the model repo, or None."""
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(MODEL_REPO, CKPT, token=_token())
        return torch.load(p, map_location=map_location, weights_only=False)
    except Exception:
        return None


def save_ckpt(ck: dict):
    from huggingface_hub import HfApi
    api = HfApi(token=_token())
    api.create_repo(MODEL_REPO, repo_type="model", private=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(ck, buf)
    api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=CKPT,
                    repo_id=MODEL_REPO, repo_type="model",
                    commit_message=f"epoch {ck['epoch']} val_nse={ck.get('val_nse')}")


def build_dataset(gauges: list[dict], months: list[str], val_months: list[str],
                  log=print):
    """Assemble train/val tensors from the prepared per-month series."""
    from data import load_series
    la = [np.log10(max(g["area_km2"], 1.0)) for g in gauges]
    stats = {"la_mean": float(np.mean(la)), "la_std": float(np.std(la) or 1.0)}
    rng = np.random.default_rng(42)

    def one(gid_g, mlist, stride):
        df = load_series(gid_g["id"], mlist)
        if df.empty or df["q"].notna().sum() < 500:
            return None
        q = df["q"].to_numpy()
        p = np.nan_to_num(df["p"].to_numpy(), nan=0.0)
        # random staleness augmentation: the "last known obs" the model sees
        # lags truth by 1..12 h (teaches the obs_age channel)
        lag = int(rng.integers(1, 13))
        obs_ff = np.full_like(q, np.nan)
        age = np.zeros_like(q)
        last, last_t = np.nan, -1
        for t in range(len(q)):
            if t - lag >= 0 and np.isfinite(q[t - lag]):
                last, last_t = q[t - lag], t - lag
            obs_ff[t] = last
            age[t] = t - last_t if last_t >= 0 else 999
        feat = build_features(p, np.nan_to_num(obs_ff, nan=0.0), age,
                              gid_g["area_km2"], stats)
        return make_windows(feat, q, stride=stride)

    Xtr, Ytr, Xva, Yva = [], [], [], []
    for g in gauges:
        tr = one(g, months, stride=3)
        va = one(g, val_months, stride=6)
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
