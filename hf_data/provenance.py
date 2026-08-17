"""V30 reproducibility stamps for published event manifests.

Every engine (Space CPU window, HPC worker, ZeroGPU backup) publishes
through eventstore.publish_event on its own machine, so a stamp computed
HERE describes the machine that actually ran the solver. Recorded once per
process (versions don't change mid-run) and attached to the manifest as
manifest["provenance"]:

    engine        who solved it — EVENT_ENGINE_IDENT env (workers set it)
                  or "space-cpu:<SPACE_ID>"
    crest_ai      CREST_demo code revision: git SHA when hf_data lives in a
                  clone (HPC / ZeroGPU bootstrap), else the Space repo's
                  current sha via HfApi (the Docker image COPYies the code
                  without .git)
    crestimap     fork git SHA when the package is a clone, else __version__
    ef5           EF5 fork git SHA (/EF5 clone in the Space image; absent on
                  workers, which never run EF5)

Six months from now, a reviewer's "which code produced this map?" is
answered by the manifest alone.
"""
from __future__ import annotations

import os
import subprocess
import threading

_lock = threading.Lock()
_stamp: dict | None = None


def _git_sha(path: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        sha = r.stdout.strip()
        return sha[:12] if r.returncode == 0 and len(sha) >= 12 else None
    except Exception:
        return None


def _space_repo_sha() -> str | None:
    """The Space repo's head sha (Docker images carry no .git)."""
    sid = os.environ.get("SPACE_ID", "")
    if "/" not in sid:
        return None
    try:
        from huggingface_hub import HfApi
        info = HfApi(token=os.environ.get("HF_TOKEN")).repo_info(
            sid, repo_type="space")
        return (info.sha or "")[:12] or None
    except Exception:
        return None


def stamp() -> dict:
    global _stamp
    with _lock:
        if _stamp is not None:
            return dict(_stamp)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    crest_ai = _git_sha(here) or _space_repo_sha()
    imap_sha = None
    imap_ver = None
    try:
        import crestimap
        d = os.path.dirname(os.path.abspath(crestimap.__file__))
        imap_sha = _git_sha(d) or _git_sha(os.path.dirname(d))
        imap_ver = getattr(crestimap, "__version__", None)
    except Exception:
        pass
    ef5 = None
    for p in (os.environ.get("EF5_DIR", ""), "/EF5"):
        if p and os.path.isdir(p):
            ef5 = _git_sha(p)
            if ef5:
                break
    s = {"engine": os.environ.get("EVENT_ENGINE_IDENT")
                or f"space-cpu:{os.environ.get('SPACE_ID', 'local')}",
         "crest_ai": crest_ai, "crestimap": imap_sha or imap_ver,
         "ef5": ef5}
    with _lock:
        _stamp = s
    return dict(s)
