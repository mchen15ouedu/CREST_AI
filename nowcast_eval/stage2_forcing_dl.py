"""Stage 2: download replay forcing + model checkpoints.

mrms_recent Pass1 hourly grids (what the operational runs actually used),
Pass2 month tars for hours that predate the recent store (Jul 18-22 window
tails), and all six DI-LSTM checkpoints + statics table.
"""
import os
from huggingface_hub import snapshot_download, hf_hub_download

BASE = os.path.dirname(os.path.abspath(__file__))
TOK = open(os.path.expanduser("~/huggingface.txt")).read().strip()

print("downloading mrms_recent Pass1 grids ...", flush=True)
snapshot_download("vincewin/CREST_data", repo_type="dataset", token=TOK,
                  allow_patterns=["mrms_recent/*"],
                  local_dir=os.path.join(BASE, "forcing"))

print("downloading Pass2 month tars (2026-07, 2026-08) ...", flush=True)
for m in ("07", "08"):
    p = hf_hub_download("vincewin/CREST_data", f"mrms/2026/mrms_2026_{m}.tar",
                        repo_type="dataset", token=TOK)
    print("  tar cached:", p, flush=True)

print("downloading checkpoints ...", flush=True)
os.makedirs(os.path.join(BASE, "ckpts"), exist_ok=True)
for f in ("dilstm.pt", "dilstm_h12.pt", "dilstm_v2.pt", "dilstm_h12_v2.pt",
          "dilstm_v3.pt", "dilstm_h12_v3.pt"):
    p = hf_hub_download("vincewin/CREST_nowcast_model", f, repo_type="model",
                        token=TOK, local_dir=os.path.join(BASE, "ckpts"))
    print("  ", f, flush=True)

hf_hub_download("vincewin/CREST_nowcast_data", "gauges/gauge_statics.parquet",
                repo_type="dataset", token=TOK,
                local_dir=os.path.join(BASE, "ckpts"))
print("done", flush=True)
