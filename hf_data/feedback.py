"""Test-user improvement comments: record + persist.

Two sinks per comment:
  1. LOCAL   CACHE_DIR/feedback/feedback.jsonl — instant, feeds GET /api/feedback.
  2. DURABLE one JSON file per comment in the vincewin/CREST_data dataset under
     feedback/ (needs the HF_TOKEN Space secret). The Space's _cache is wiped on
     every rebuild, so the dataset copy is what the daily review job reads —
     comments survive redeploys.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone

from hf_data.statecache import CACHE_DIR

HF_REPO = os.environ.get("CREST_FEEDBACK_REPO", "vincewin/CREST_data")
MAX_LEN = 4000


def _path() -> str:
    d = os.path.join(CACHE_DIR, "feedback")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "feedback.jsonl")


def _persist_hf(rec: dict):
    """Best-effort durable copy (one file per comment; no append races)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import HfApi
        name = f"feedback/{rec['when'].replace(':', '').replace(' ', '_')}_{rec['id']}.json"
        HfApi(token=token).upload_file(
            path_or_fileobj=json.dumps(rec, default=str).encode(),
            path_in_repo=name, repo_id=HF_REPO, repo_type="dataset",
            commit_message=f"user feedback {rec['id']}")
    except Exception:
        try:
            from hf_data import crashlog
            crashlog.capture("feedback:persist", message="HF upload failed",
                             fb_id=rec.get("id"))
        except Exception:
            pass


def record(text: str, user: str | None = None, contact: str | None = None,
           context: dict | None = None) -> dict:
    rec = {
        "id": uuid.uuid4().hex[:8],
        "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "user": user, "contact": (contact or "")[:200],
        "text": (text or "")[:MAX_LEN],
        "context": {k: str(v)[:200] for k, v in (context or {}).items()},
    }
    with open(_path(), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    threading.Thread(target=_persist_hf, args=(rec,), daemon=True).start()
    return rec


# ---- addressed-flags (daily review bookkeeping) ---------------------------
# Comments are NEVER deleted or edited; a flag file at feedback/status/<id>.json
# marks one as handled so the daily review does no repetitive work:
#   {"id","addressed":true,"when","by","note"}

def mark_addressed(fb_id: str, note: str = "", by: str = "daily-review",
                   token: str | None = None) -> bool:
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        return False
    from huggingface_hub import HfApi
    rec = {"id": fb_id, "addressed": True, "by": by, "note": (note or "")[:500],
           "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
    HfApi(token=token).upload_file(
        path_or_fileobj=json.dumps(rec).encode(),
        path_in_repo=f"feedback/status/{fb_id}.json", repo_id=HF_REPO,
        repo_type="dataset", commit_message=f"feedback {fb_id} addressed")
    return True


def unaddressed(token: str | None = None) -> list[dict]:
    """All durable comments that have no addressed-flag yet (newest last)."""
    token = token or os.environ.get("HF_TOKEN")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    files = api.list_repo_files(HF_REPO, repo_type="dataset")
    flagged = {os.path.basename(f)[:-5] for f in files
               if f.startswith("feedback/status/") and f.endswith(".json")}
    out = []
    for f in sorted(files):
        if not f.startswith("feedback/") or f.startswith("feedback/status/") \
                or not f.endswith(".json"):
            continue
        fb_id = f.rsplit("_", 1)[-1][:-5]
        if fb_id in flagged:
            continue
        try:
            p = hf_hub_download(HF_REPO, f, repo_type="dataset", token=token)
            out.append(json.load(open(p, encoding="utf-8")))
        except Exception:
            pass
    return out


def recent(n: int = 100) -> list[dict]:
    try:
        with open(_path(), encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        out = []
        for ln in lines[-n:]:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
        return out[::-1]
    except Exception:
        return []
