"""Standalone DI-LSTM training for an HPC GPU node (no Gradio, no ZeroGPU).

The Space stays the inference server; this script trains on the same prepped
data (private dataset repo vincewin/CREST_nowcast_data) and uploads the SAME
checkpoint format to vincewin/CREST_nowcast_model — press "Reload model" on
the CREST_nowcast Space (or call api_name=reload) to serve the new weights.

v2 (default): feat_version-2 model with an explicit obs-missing flag, obs
outage augmentation and obs-channel dropout — the fix for DI-LSTM
hallucination at data-dead gauges. v2 checkpoints upload under NEW names
(dilstm_v2.pt / dilstm_h12_v2.pt) so the live v1 model is untouched until
the serving side finds the v2 files; delete them from the model repo to
roll back. Alongside the final checkpoint a per-gauge validation skill
table (skill_<ckpt-stem>.parquet: gid, val_nse, val_nse_noobs, n_windows)
uploads, enabling per-gauge skill gating in the dashboard risk tiers.

Needs alongside it: model.py, train.py, data.py (this repo). See
RETRAIN_V2_BRIEF.md for the retraining plan and README_HPC.md for the
environment and SLURM setup.

    export HF_TOKEN=hf_...          # read/write on the two private repos
    python train_hpc.py                                   # v2, 6-h horizon
    python train_hpc.py --horizon 12                      # 12-h companion
    python train_hpc.py --gauges-file gauges_v2.txt --months 2023_01-2025_06
"""
from __future__ import annotations

import argparse
import io
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

from model import DILSTM, H, L, blank_obs, itq, mc_predict, n_feat_for, nse
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


def cpu_ckpt(model, opt, epoch, stats, val_nse_v, n_train,
             feat_version, horizon) -> dict:
    """Checkpoint with deep-copied CPU tensors (does not disturb live training)."""
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    osd = opt.state_dict()
    osd = {"state": {k: {k2: (v2.detach().cpu().clone() if torch.is_tensor(v2) else v2)
                         for k2, v2 in st.items()}
                     for k, st in osd["state"].items()},
           "param_groups": [dict(pg) for pg in osd["param_groups"]]}
    return {"state_dict": sd, "opt": osd, "epoch": epoch, "stats": stats,
            "val_nse": round(float(val_nse_v), 3), "n_train": int(n_train),
            "horizon": int(horizon), "lookback": L,
            "feat_version": int(feat_version),
            "n_feat": n_feat_for(feat_version),
            "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}


def per_gauge_table(model, Xva, Yva, Gva, gauge_ids, dev, batch=4096):
    """Per-gauge val NSE, obs-fresh AND no-obs, plus a hallucination score:
    the 95th-percentile ratio of predicted to observed window-max flow under
    no-obs (large = the model invents floods when the gauge goes dark)."""
    def infer(X):
        outs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch):
                outs.append(model(torch.from_numpy(X[i:i + batch]).to(dev))
                            .cpu().numpy())
        return itq(np.concatenate(outs))                      # real space [N, hor]
    y = itq(Yva)
    p_obs = infer(Xva)
    p_no = infer(blank_obs(Xva))
    rows = []
    for gi, gid in enumerate(gauge_ids):
        m = Gva == gi
        if not m.any():
            continue
        rows.append({"gid": gid,
                     "val_nse": nse(p_obs[m].ravel(), y[m].ravel()),
                     "val_nse_noobs": nse(p_no[m].ravel(), y[m].ravel()),
                     "n_windows": int(m.sum())})
    ratio = p_no.max(axis=1) / np.maximum(y.max(axis=1), 0.5)
    diag = {"pooled_nse": nse(p_obs.ravel(), y.ravel()),
            "pooled_nse_noobs": nse(p_no.ravel(), y.ravel()),
            "noobs_peak_ratio_p95": float(np.percentile(ratio, 95)),
            "noobs_peak_ratio_p99": float(np.percentile(ratio, 99))}
    return rows, diag


def upload_skill_table(rows: list[dict], name: str):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    t = pa.table({"gid": [r["gid"] for r in rows],
                  "val_nse": np.array([r["val_nse"] for r in rows], "float32"),
                  "val_nse_noobs": np.array([r["val_nse_noobs"] for r in rows],
                                            "float32"),
                  "n_windows": np.array([r["n_windows"] for r in rows], "int32")})
    buf = io.BytesIO()
    pq.write_table(t, buf, compression="zstd")
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj=buf.getvalue(), path_in_repo=name,
        repo_id=T.MODEL_REPO, repo_type="model",
        commit_message=f"{name}: per-gauge val skill, {len(rows)} gauges")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gauges", default=DEFAULT_GAUGES)
    ap.add_argument("--gauges-file", default="",
                    help="file with gauge ids (comma/newline separated); "
                         "overrides --gauges. See select_gauges.py")
    ap.add_argument("--months", default=DEFAULT_MONTHS)
    ap.add_argument("--horizon", type=int, default=6, choices=(6, 12))
    ap.add_argument("--feat-version", type=int, default=2, choices=(1, 2))
    ap.add_argument("--obs-dropout", type=float, default=0.15,
                    help="fraction of training windows rewritten to the "
                         "dead-gauge obs state (feat_version 2)")
    ap.add_argument("--ckpt-name", default="",
                    help="checkpoint filename in the model repo; default "
                         "dilstm_v2.pt (or dilstm_h12_v2.pt for --horizon 12); "
                         "v1: dilstm.pt / dilstm_h12.pt")
    ap.add_argument("--max-epochs", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=300,
                    help="stop after this many epochs without val-NSE improvement")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mc-eval", type=int, default=16,
                    help="MC-dropout passes for the final spread diagnostic "
                         "(0 = skip)")
    ap.add_argument("--fresh", action="store_true",
                    help="start from scratch instead of resuming the repo checkpoint")
    ap.add_argument("--no-upload", action="store_true",
                    help="write <ckpt-name> + skill table locally instead of uploading")
    ap.add_argument("--upload-every-min", type=float, default=15.0,
                    help="push the current best checkpoint at most this often")
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN env var not set (needed for the private data/model repos)")

    fv, hor = args.feat_version, args.horizon
    ckpt_name = args.ckpt_name or {
        (1, 6): "dilstm.pt", (1, 12): "dilstm_h12.pt",
        (2, 6): "dilstm_v2.pt", (2, 12): "dilstm_h12_v2.pt"}[(fv, hor)]
    skill_name = f"skill_{os.path.splitext(ckpt_name)[0]}.parquet"

    gauge_spec = args.gauges
    if args.gauges_file:
        raw = open(args.gauges_file).read()
        gauge_spec = ",".join(s for s in raw.replace("\n", ",").split(",") if s.strip())
    gauges = gauge_meta(gauge_spec)
    months = months_range(args.months)
    if not gauges:
        sys.exit("no valid gauges")
    print(f"feat_version {fv}  horizon {hor} h  ckpt {ckpt_name}\n"
          f"{len(gauges)} gauges  months {months[0]}..{months[-1]}"
          f"  val {VAL_MONTHS[0]}..{VAL_MONTHS[-1]}")

    if fv >= 2:
        ds = T.build_dataset_v2(gauges, months, VAL_MONTHS, horizon=hor,
                                obs_dropout=args.obs_dropout, seed=args.seed)
        if ds is None:
            sys.exit("no training data — run prep_hpc.py for these gauges/months first")
        Xtr, Ytr, Xva, Yva = ds["Xtr"], ds["Ytr"], ds["Xva"], ds["Yva"]
        Gva, gauge_ids, stats = ds["Gva"], ds["gauge_ids"], ds["stats"]
    else:
        legacy = T.build_dataset(gauges, months, VAL_MONTHS)
        if legacy is None:
            sys.exit("no training data — run prep_hpc.py for these gauges/months first")
        Xtr, Ytr, Xva, Yva, stats = legacy
        Gva, gauge_ids = np.zeros(len(Xva), "int32"), [g["id"] for g in gauges]
    if not len(Xva):
        sys.exit("no validation windows — early stopping needs the 2025 val months prepped")
    print(f"train {len(Xtr)} / val {len(Xva)} windows")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev,
          torch.cuda.get_device_name(0) if dev.type == "cuda" else "(no GPU — slow)")

    model = DILSTM(n_feat=n_feat_for(fv), horizon=hor).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    epoch = 0
    if not args.fresh:
        ck = T.load_ckpt(name=ckpt_name)
        if ck and (int(ck.get("feat_version", 1)) != fv
                   or int(ck.get("horizon", H)) != hor):
            sys.exit(f"repo checkpoint {ckpt_name} has feat_version="
                     f"{ck.get('feat_version', 1)} horizon={ck.get('horizon', H)} — "
                     "mismatch with the requested run; use --fresh or --ckpt-name")
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
    pers = np.repeat(itq(Xva[:, -1, 1])[:, None], hor, axis=1).ravel()
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
            torch.save(ckd, ckpt_name)
            print(f"  >> saved {ckpt_name} ({note})")
        else:
            T.save_ckpt(ckd, name=ckpt_name)
            print(f"  >> uploaded {ckpt_name} ({note})")

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
            best_ck = cpu_ckpt(model, opt, epoch, stats, v, n, fv, hor)
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

    # -- final diagnostics on the BEST weights ---------------------------------
    if best_ck is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_ck["state_dict"].items()})
    model.eval()
    rows, diag = per_gauge_table(model, Xva, Yva, Gva, gauge_ids, dev)
    print(f"\nval pooled NSE: obs-fresh {diag['pooled_nse']:.4f} | "
          f"no-obs {diag['pooled_nse_noobs']:.4f}")
    print(f"no-obs peak ratio (pred/true window max): "
          f"p95 {diag['noobs_peak_ratio_p95']:.2f}, "
          f"p99 {diag['noobs_peak_ratio_p99']:.2f}   "
          f"(v1 hallucination shows up here as >>1)")
    worst = sorted(rows, key=lambda r: (r["val_nse"] if r["val_nse"] == r["val_nse"]
                                        else -9e9))[:5]
    for r in worst:
        print(f"  low-skill gauge {r['gid']}: NSE {r['val_nse']:.3f} "
              f"(no-obs {r['val_nse_noobs']:.3f}, {r['n_windows']} windows)")
    if args.mc_eval > 0:
        mu, sd = mc_predict(model, Xv[:4096], n=args.mc_eval)
        cv = np.expm1(sd).mean() / max(np.expm1(mu).mean(), 1e-6)
        print(f"MC-dropout ({args.mc_eval} passes, first 4096 windows): "
              f"mean log-space std {sd.mean():.4f} (rough CV ~{cv:.3f})")
    if rows:
        if args.no_upload:
            import pyarrow as pa
            import pyarrow.parquet as pq
            pq.write_table(pa.table(
                {"gid": [r["gid"] for r in rows],
                 "val_nse": np.array([r["val_nse"] for r in rows], "float32"),
                 "val_nse_noobs": np.array([r["val_nse_noobs"] for r in rows], "float32"),
                 "n_windows": np.array([r["n_windows"] for r in rows], "int32")}),
                skill_name, compression="zstd")
            print(f">> saved {skill_name} locally")
        else:
            upload_skill_table(rows, skill_name)
            print(f">> uploaded {skill_name} ({len(rows)} gauges)")


if __name__ == "__main__":
    main()
