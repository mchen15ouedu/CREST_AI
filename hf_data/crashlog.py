"""System error recorder (the second watchdog): every error the app runs into
is captured with a traceback + context and kept queryable.

Two sinks, adapted from the standard open-source error-tracking stack:
  1. LOCAL (always on): rotating JSONL at CACHE_DIR/errors/errors.jsonl —
     zero infrastructure, works on the free Space; served by GET /api/errors.
  2. REMOTE (optional): the Sentry Python SDK (github.com/getsentry/sentry-python).
     Set SENTRY_DSN to any Sentry-protocol backend — sentry.io, or self-hosted
     GlitchTip / Bugsink (both are Sentry-SDK-compatible open-source trackers) —
     and every capture() is mirrored there with full stack traces.

Capture points: FastAPI middleware (all unhandled HTTP errors), simulation and
calibration workers, EF5 failures/watchdog kills, LLM/chat failures.
"""
from __future__ import annotations

import json
import os
import threading
import traceback
from datetime import datetime, timezone

from hf_data.statecache import CACHE_DIR

_LOCK = threading.Lock()
MAX_BYTES = int(os.environ.get("CREST_ERRORLOG_BYTES", str(2_000_000)))
KEEP_ON_ROTATE = 400              # newest entries kept when the log rotates
_sentry_on = False


def init():
    """Optional remote sink — Sentry-protocol DSN (Sentry/GlitchTip/Bugsink)."""
    global _sentry_on
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return False
    try:
        import sentry_sdk
        sentry_sdk.init(dsn=dsn, traces_sample_rate=0.0,
                        environment=os.environ.get("SPACE_HOST", "local"),
                        release=os.environ.get("SPACE_REPO_ID", "crest-demo"))
        _sentry_on = True
    except Exception:
        _sentry_on = False
    return _sentry_on


def install_thread_hook():
    """Backstop: a thread that dies OUTSIDE its own try/capture still gets
    recorded (all current workers are wrapped — this catches future ones)."""
    orig = threading.excepthook

    def hook(args):
        try:
            capture(f"thread:{args.thread.name if args.thread else '?'}",
                    args.exc_value)
        except Exception:
            pass
        orig(args)

    threading.excepthook = hook


def _path() -> str:
    d = os.path.join(CACHE_DIR, "errors")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "errors.jsonl")


def capture(where: str, exc: BaseException | None = None,
            message: str = "", **context) -> dict:
    """Record one error event. Safe to call anywhere — never raises."""
    rec = {
        "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "where": where,
        "error": repr(exc) if exc is not None else message,
        "traceback": traceback.format_exc()
        if exc is not None and exc.__traceback__ is not None else None,
        "context": {k: str(v)[:400] for k, v in context.items()},
    }
    try:
        with _LOCK:
            p = _path()
            if os.path.exists(p) and os.path.getsize(p) > MAX_BYTES:   # rotate
                with open(p, encoding="utf-8", errors="replace") as fh:
                    tail = fh.readlines()[-KEEP_ON_ROTATE:]
                with open(p, "w", encoding="utf-8") as fh:
                    fh.writelines(tail)
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass
    if _sentry_on and exc is not None:
        try:
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("where", where)
                for k, v in context.items():
                    scope.set_extra(k, str(v)[:400])
                sentry_sdk.capture_exception(exc)
        except Exception:
            pass
    return rec


def recent(n: int = 50) -> list[dict]:
    """Newest-first recent error events."""
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


def stats() -> dict:
    p = _path()
    n = 0
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            n = sum(1 for _ in fh)
    except Exception:
        pass
    return {"entries": n, "bytes": os.path.getsize(p) if os.path.exists(p) else 0,
            "sentry": _sentry_on}
