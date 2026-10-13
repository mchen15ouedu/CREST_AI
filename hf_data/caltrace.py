"""Calibration round traces for the human-in-the-loop review pilot.

The CREST_calreview pilot (a separate private Space) shows an expert the AI
calibration's reasoning round by round and collects ratings. This module only
RECORDS what run_calibration already does: what the LLM was shown, what it
proposed (raw reply kept), and how each candidate scored. Each calibration is
held in memory and, when it ends (finished, failed, or preempted), uploaded as
ONE JSON file to a separate private dataset from a daemon thread.

It never changes a calibration: it consumes no randomness, raises nothing,
waits for nothing, and a failed upload is dropped. Mock runs (synthetic NSE)
are not recorded. Disable with CREST_CALTRACE=0.
"""
from __future__ import annotations

import io
import json
import math
import os
import threading
import uuid
from datetime import datetime, timezone

REPO = os.environ.get("CREST_CALTRACE_REPO", "vincewin/CREST_calreview_data")
ENABLED = os.environ.get("CREST_CALTRACE", "1") == "1"
RAW_MAX = 8000          # chars of the LLM reply kept per round


def note(trace, **kw):
    """Append one proposal note to a propose() trace list (None = off)."""
    if trace is None:
        return
    try:
        if "raw" in kw and kw["raw"] is not None:
            kw["raw"] = str(kw["raw"])[:RAW_MAX]
        trace.append(kw)
    except Exception:
        pass


class Recorder:
    """Collects one calibration's rounds; finish() uploads them once."""

    def __init__(self, gauge, model, t_start, t_end, *, use_mock=False,
                 auto=False, rounds=None, k=None):
        self.on = False
        self.done = False
        try:
            if not (ENABLED and not use_mock and os.environ.get("HF_TOKEN")):
                # recording off: nothing below runs, nothing is uploaded
                return
            now = datetime.now(timezone.utc)
            self.uid = uuid.uuid4().hex[:8]
            self.rec = {
                "schema": "crest_ai.caltrace/1",
                "id": f"{now:%Y%m%dT%H%M%S}_{str(gauge).zfill(8)}_{self.uid}",
                "gauge": str(gauge).zfill(8), "model_request": model,
                "window": [_ts(t_start), _ts(t_end)],
                "mode": "autocal" if auto else "user",
                "space": os.environ.get("SPACE_ID", "local"),
                "budget": {"rounds": rounds, "k": k},
                "started": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "baseline": None, "rounds": [], "done": None, "status": "running",
            }
            self.on = True
        except Exception:
            self.on = False

    def setup(self, **kw):
        if self.on:
            _safe(lambda: self.rec.update(_jsonable(kw)))

    def baseline(self, params, nse, metrics):
        if self.on:
            _safe(lambda: self.rec.__setitem__("baseline", _jsonable(
                {"params": params, "nse": nse, "metrics": metrics})))

    def round(self, round_index, proposal, candidates, best_before, best_after,
              extended):
        if self.on:
            _safe(lambda: self.rec["rounds"].append(_jsonable({
                "round": round_index, "extended": extended,
                "proposal": proposal, "candidates": candidates,
                "best_nse_before": best_before, "best_nse_after": best_after})))

    def finish(self, done=None, status=None):
        """Upload once in the background. Safe to call more than once."""
        if not self.on or self.done:
            return
        self.done = True
        try:
            if done is not None:
                self.rec["done"] = _jsonable(done)
            self.rec["status"] = status or ("complete" if done is not None else "incomplete")
            self.rec["ended"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if not self.rec["rounds"] and self.rec["baseline"] is None:
                return                       # nothing happened worth reviewing
            body = json.dumps(self.rec, default=str).encode()
            path = (f"traces/crest_ai/{self.rec['started'][:4]}/"
                    f"{self.rec['started'][5:7]}/{self.rec['id']}.json")
            threading.Thread(target=_upload, args=(path, body, self.rec["gauge"],
                                                   self.rec["status"]),
                             daemon=True).start()
        except Exception:
            pass


def _upload(path, body, gauge, status):
    try:
        from huggingface_hub import HfApi
        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=io.BytesIO(body), path_in_repo=path,
            repo_id=REPO, repo_type="dataset",
            commit_message=f"caltrace {gauge} ({status})")
    except Exception:
        pass


def _ts(t):
    try:
        return t.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(t)


def _finite(x):
    """NaN/inf -> None, so the trace stays strict JSON (browsers reject NaN)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def _jsonable(x):
    """Deep copy through JSON so later mutation of the source can't leak in."""
    return _finite(json.loads(json.dumps(x, default=str)))


def _safe(fn):
    try:
        fn()
    except Exception:
        pass
