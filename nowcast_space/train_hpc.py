"""Standalone DI-LSTM training for an HPC GPU node (no Gradio, no ZeroGPU).

The Space stays the inference server; this script trains on the same prepped
data (private dataset repo vincewin/CREST_nowcast_data) and uploads the SAME
checkpoint format to vincewin/CREST_nowcast_model — press "Reload model" on
the CREST_nowcast Space (or call api_name=reload) to serve the new weights.

Needs alongside it: model.py, train.py, data.py (this repo). See README_HPC.md
for environment setup and a SLURM example.

    export HF_TOKEN=hf_...          # read/write on the two private repos
    python train_hpc.py                                   # defaults below
    python train_hpc.py --gauges "08167000, 08144500" \
        --months 2023_01-2024_12 --max-epochs 5000 --patience 300
"""
from __future__ import annotations

import argparse
import os
import sys
import time

try:                               # OU-managed machines intercept TLS; no-op on HPC
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import numpy as np
import torch

from model import DILSTM, H, L, itq, nse
import train as T

DEFAULT_GAUGES = "01011000, 08166200, 08167000, 08144500"
DEFAULT_MONTHS = "2023_01-2024_12"
VAL_MONTHS = ["2025_01", "2025_02", "2025_03", "2025_04", "2025_05", "2025_06"]


def gauge_meta(gauge_ids: str) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("vincewin/CREST_data", "gauges/gagesII_9322.parquet",
                        repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    df = pd.read_parquet(p)
    df["STAID"] = df["STAID"].astype(str).str.zfill(8)
    df = df.set_index("STAID")
    out = []
    for s in gauge_ids.split(","):
        gid = s.strip().zfill(8)
        if not s.strip():
            continue
        if gid not in df.index:
            print(f"  !! {gid} not in the GAGES-II catalog — skipped")
            continue
        r = df.loc[gid]
        out.append({"id": gid, "lat": float(r["LAT_GAGE"]),
                    "lon": float(r["LNG_GAGE"]), "area_km2": float(r["DRAIN_SQKM"])})
    return out


def months_range(spec: str) -> list[str]:
    a, b = [s.strip() for s in spec.split("-")]
    y0, m0 = map(int, a.split("_")); y1, m1 = map(int, b.split("_"))
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}_{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def cpu_ckpt(model, opt, epoch, stats, val_nse_v, n_train) -> dict:
    """Checkpoint with deep-copied CPU tensors (does not disturb live training)."""
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    osd = opt.state_dict()
    osd = {"state": {k: {k2: (v2.detach().cpu().clone() if torch.is_tensor(v2) else v2)
                         for k2, v2 in st.items()}
                     for k, st in osd["state"].items()},
           "param_groups": [dict(pg) for pg in osd["param_groups"]]}
    return {"state_dict": sd, "opt": osd, "epoch": epoch, "stats": stats,
            "val_nse": round(float(val_nse_v), 3), "n_train": int(n_train),
            "horizon": H, "lookback": L, "feat_version": 1,
            "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gauges", default=DEFAULT_GAUGES)
    ap.add_argument("--months", default=DEFAULT_MONTHS)
    ap.add_argument("--max-epochs", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=300,
                    help="stop after this many epochs without val-NSE improvement")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fresh", action="store_true",
                    help="start from scratch instead of resuming the repo checkpoint")
    ap.add_argument("--no-upload", action="store_true",
                    help="write dilstm_best.pt locally instead of uploading")
    ap.add_argument("--upload-every-min", type=float, default=15.0,
                    help="push the current best checkpoint at most this often")
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN env var not set (needed for the private data/model repos)")

    gauges = gauge_meta(args.gauges)
    months = months_range(args.months)
    if not gauges:
        sys.exit("no valid gauges")
    print(f"gauges {[g['id'] for g in gauges]}  months {months[0]}..{months[-1]}"
          f"  val {VAL_MONTHS[0]}..{VAL_MONTHS[-1]}")

    ds = T.build_dataset(gauges, months, VAL_MONTHS)
    if ds is None:
        sys.exit("no training data — run Data prep on the Space first (prep_month "
                 "fills the dataset repo; this script only reads it)")
    Xtr, Ytr, Xva, Yva, stats = ds
    if not len(Xva):
        sys.exit("no validation windows — early stopping needs the 2025 val months prepped")
    print(f"train {len(Xtr)} / val {len(Xva)} windows")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev,
          torch.cuda.get_device_name(0) if dev.type == "cuda" else "(no GPU — slow)")

    model = DILSTM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    epoch = 0
    if not args.fresh:
        ck = T.load_ckpt()
        if ck:
            model.load_state_dict(ck["state_dict"])
            opt.load_state_dict(ck["opt"])
            epoch, stats = ck["epoch"], ck["stats"]
            print(f"resumed from epoch {epoch} (val NSE {ck.get('val_nse')})")
            for st in opt.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(dev)

    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xv = torch.from_numpy(Xva).to(dev)
    yva_real = itq(Yva).ravel()

    # persistence baseline: hold the last-known obs flat over the horizon —
    # the number the DI-LSTM has to beat for the skill to be real
    pers = np.repeat(itq(Xva[:, -1, 1])[:, None], H, axis=1).ravel()
    print(f"persistence baseline val NSE: {nse(pers, yva_real):.4f}")

    def val_nse() -> float:
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xv), 4096):
                outs.append(model(Xv[i:i + 4096]).cpu().numpy())
        model.train()
        return nse(itq(np.concatenate(outs)).ravel(), yva_real)

    def push(ckd, note):
        if args.no_upload:
            torch.save(ckd, "dilstm_best.pt")
            print(f"  >> saved dilstm_best.pt ({note})")
        else:
            T.save_ckpt(ckd)
            print(f"  >> uploaded checkpoint ({note})")

    n, bs = len(Xt), args.batch
    best, best_ck, since_best = -np.inf, None, 0
    last_push, pushed_best = time.time(), -np.inf
    t0 = time.time()
    model.train()
    for _ in range(args.max_epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx].to(dev, non_blocking=True), Yt[idx].to(dev, non_blocking=True)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        epoch += 1
        v = val_nse()
        if v > best + 1e-4:
            best, since_best = v, 0
            best_ck = cpu_ckpt(model, opt, epoch, stats, v, n)
            print(f"epoch {epoch}: train MSE {tot / n:.5f}, val NSE {v:.4f}  * new best")
        else:
            since_best += 1
            if epoch % 25 == 0:
                print(f"epoch {epoch}: train MSE {tot / n:.5f}, val NSE {v:.4f} "
                      f"(best {best:.4f}, {since_best} since)")
        if (best_ck is not None and best > pushed_best
                and time.time() - last_push > args.upload_every_min * 60):
            push(best_ck, f"periodic, epoch {best_ck['epoch']}")
            last_push, pushed_best = time.time(), best
        if since_best >= args.patience:
            print(f"early stop: no improvement in {args.patience} epochs")
            break

    if best_ck is not None and best > pushed_best:
        push(best_ck, f"final best, epoch {best_ck['epoch']}, val NSE {best:.4f}")
    print(f"done: {epoch} total epochs, best val NSE {best:.4f}, "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
