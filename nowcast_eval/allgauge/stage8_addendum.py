"""Stage 8 (all-gauge): (a) per-gauge results split by whether the gauge is
in the 6,036-gauge training set of BOTH models (v1 and v3-full share it);
(b) replay fidelity on all gauges — replayed v1 vs archived operational v1
on the v1-era issues (t0 <= 2026-08-11 00Z), log space."""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN = "/home/MengyuChen/CREST_AI/nowcast_space/gauges_all_6036.txt"
train = {x.strip().zfill(8) for x in open(TRAIN).read().replace(",", " ").split() if x.strip()}
rt = pd.read_csv(os.path.join(BASE, "route_pergauge.csv"), dtype={"gid": str})
rt["in_train"] = rt["gid"].isin(train)
lines = [f"training-set split: {int(rt['in_train'].sum())} served gauges in the 6,036 training set, "
         f"{int((~rt['in_train']).sum())} not (never seen by either model)"]
for fam in ("6", "12"):
    c1, c3, cp = f"nse_v1_{fam}", f"nse_v3f_{fam}", f"nse_agerule_{fam}"
    d = rt.dropna(subset=[c1, c3])
    for lab, m in (("in training set ", d["in_train"]), ("NOT in training", ~d["in_train"])):
        s = d[m]
        best = np.where(s[c1] > s[c3], s[c1], s[c3])
        lines.append(f"{fam:>2s}h {lab} n={len(s):>5d}: median v1 {s[c1].median():.3f} v3f {s[c3].median():.3f} "
                     f"age-rule {s[cp].median():.3f} winner {np.median(best):.3f}; v3f wins {int((s[c3] > s[c1]).sum())}/{len(s)}; "
                     f"frac>0 v1 {(s[c1] > 0).mean():.2f} v3f {(s[c3] > 0).mean():.2f}; frac>0.5 v1 {(s[c1] > 0.5).mean():.2f} v3f {(s[c3] > 0.5).mean():.2f}")

# fidelity: replayed v1 vs archived operational predictions (v1 era)
rp = np.load(os.path.join(BASE, "replay_preds.npz"), allow_pickle=True)
ae = np.load(os.path.join(BASE, "archive_eval.npz"), allow_pickle=True)
t0s = np.array([str(t) for t in rp["t0"]])
era = t0s <= "2026081100"
for fam, key in (("v1_6", "q6"), ("v1_12", "q12")):
    a = ae[key][era]; r = rp[fam][era]
    m = np.isfinite(a) & np.isfinite(r)
    la, lr = np.log1p(a[m]), np.log1p(r[m])
    cc = np.corrcoef(la, lr)[0, 1]
    nse = 1 - ((lr - la) ** 2).sum() / ((la - la.mean()) ** 2).sum()
    lines.append(f"fidelity {fam}: replayed vs archived operational, log space: r={cc:.4f} NSE={nse:.4f} n={int(m.sum())} "
                 f"(issues <= 2026-08-11 00Z: {int(era.sum())})")
rt.to_csv(os.path.join(BASE, "route_pergauge_with_split.csv"), index=False)
open(os.path.join(BASE, "addendum_stats.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
