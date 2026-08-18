"""Recon: what's on HF for the v1/v2/v3 replay eval."""
import os, re, collections
from huggingface_hub import HfApi

tok = open(os.path.expanduser("~/huggingface.txt")).read().strip()
api = HfApi(token=tok)

files = api.list_repo_files("vincewin/CREST_data", repo_type="dataset")
arch = sorted(f for f in files if f.startswith("nowcast/archive/"))
print(f"archive files: {len(arch)}")
print("first:", arch[0], "| last:", arch[-1])
# per-day counts
days = collections.Counter(re.search(r"nc_(\d{8})", f).group(1) for f in arch)
print("per-day issue counts:", dict(sorted(days.items())))

recent = sorted(f for f in files if f.startswith("mrms_recent/"))
print(f"\nmrms_recent files: {len(recent)}")
if recent:
    print("first:", recent[0], "| last:", recent[-1])

tars = sorted(f for f in files if f.startswith("mrms/2026/"))
print("\nPass2 2026 tars:", tars)

state = api.list_repo_files("vincewin/CREST_state", repo_type="dataset")
print("\nCREST_state files (first 20):", state[:20])

model = api.list_repo_files("vincewin/CREST_nowcast_model", repo_type="model")
print("\nmodel repo:", sorted(model))
