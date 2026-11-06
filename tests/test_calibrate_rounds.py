"""hf_data.calibrate.run_calibration bookkeeping the CREST_autocal loop reads:
each round's candidate source (LLM or heuristic) and the done payload's
rounds / llm_rounds / pushed / model; require_llm stops before a
heuristic-only search (nothing saved, pushed or traced) while a dashboard
calibration falls back to the heuristic as before; cand_errors collects the
candidate runs' exceptions. EF5, the LLM and the Hub are faked (no network,
no EF5)."""
from __future__ import annotations

import json
import math
import types
from datetime import datetime

import pytest

T0, T1 = datetime(2026, 4, 1), datetime(2026, 10, 1)
GID = "09371492"


@pytest.fixture
def cal(monkeypatch, tmp_path):
    """calibrate with a fake EF5 (NSE best at wm = 1.6), LLM, push and
    caltrace upload. Returns (calibrate, record of what happened)."""
    from hf_data import (basinextent, calibrate, caltrace, crashlog, llm, multipliers,
                         paramstore, pipeline)
    seen = {"runs": [], "llm": [], "pushes": [], "uploads": [], "llm_ok": [True] * 99,
            "push_ok": True, "fail_candidates": None}
    monkeypatch.setattr(paramstore, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(paramstore, "_remote", lambda g, m: None)

    def push_remote(g, m):
        seen["pushes"].append((g, m))
        return seen["push_ok"]
    monkeypatch.setattr(paramstore, "push_remote", push_remote)
    monkeypatch.setattr(crashlog, "capture", lambda *a, **k: None)
    monkeypatch.setattr(llm, "available", lambda: ["openai"])

    def chat(messages, **kw):
        ok = seen["llm_ok"][len(seen["llm"])]
        seen["llm"].append(ok)
        if not ok:
            raise RuntimeError("All LLM providers failed — openai: RateLimitError: "
                               "Error code: 429 - insufficient_quota")
        return json.dumps({"candidates": [{"id": f"c{i}", "goal": "wm up",
                                           "updates": {"wm": 1.0 + 0.3 * (i + 1)}}
                                          for i in range(3)]}), "openai"
    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(pipeline, "gauge_info", lambda g: {"id": GID, "lon": -109.0,
                                                           "lat": 37.0, "name": "x"})
    monkeypatch.setattr(basinextent, "ef5_extent", lambda g: {})
    monkeypatch.setattr(multipliers, "to_control_params", lambda g, model=None: (
        {"wm": 1.0, "b": 1.0, "im": 0.1, "ke": 1.0, "fc": 1.0},
        {"under": 1.0, "leaki": 1.0, "alpha": 1.0, "beta": 1.0, "alpha0": 1.0}))

    def run_gauge(gid, t_start, t_end, overrides=None, **kw):
        seen["runs"].append(dict(overrides or {}))
        if seen["fail_candidates"] is not None and len(seen["runs"]) > 1:
            raise seen["fail_candidates"]
        wm = (overrides or {}).get("wm", 1.0)
        rows = []
        for h in range(100):
            o = 10 + 5 * math.sin(h / 10)
            rows.append({"time": f"t{h}", "obs_q": o, "sim_q": o * (1 + abs(wm - 1.6)) + 3})
        yield ("hydro", {"rows": rows})
    monkeypatch.setattr(pipeline, "run_gauge", run_gauge)
    # caltrace on (as on the Spaces), its upload recorded synchronously
    monkeypatch.setenv("HF_TOKEN", "x-not-a-token")
    monkeypatch.setattr(caltrace, "ENABLED", True)
    monkeypatch.setattr(caltrace, "_upload", lambda path, body, g, status:
                        seen["uploads"].append(json.loads(body)["status"]))
    monkeypatch.setattr(caltrace, "threading", types.SimpleNamespace(Thread=lambda target, args,
                        daemon: types.SimpleNamespace(start=lambda: target(*args))))
    return calibrate, seen


def _run(calibrate, **kw):
    events = list(calibrate.run_calibration(GID, T0, T1, use_mock=False, rounds=2, k=3,
                                            ext_rounds=0, **kw))
    rounds = [p for k, p in events if k == "round"]
    (done,) = [p for k, p in events if k == "done"]
    return rounds, done


def test_llm_rounds_and_sources(cal):
    calibrate, seen = cal
    rounds, done = _run(calibrate, save_vs_baseline=True, require_llm=True)
    assert [r.get("source") for r in rounds] == [None, "llm", "llm"]   # round 0: the baseline
    assert (done["rounds"], done["llm_rounds"], done["model"]) == (2, 2, "crest")
    assert done["saved"] is True and done["pushed"] is True and "llm_stopped" not in done
    assert seen["pushes"] == [(GID, "crest")] and seen["uploads"] == ["complete"]
    json.dumps(done)                                             # the dashboard's SSE payload


def test_require_llm_stops_before_a_heuristic_only_search(cal):
    calibrate, seen = cal
    seen["llm_ok"] = [False] * 99
    rounds, done = _run(calibrate, save_vs_baseline=True, require_llm=True)
    # the baseline ran, no candidate did; nothing saved, pushed or traced
    assert len(seen["runs"]) == 1 and [r["round"] for r in rounds] == [0]
    assert (done["rounds"], done["llm_rounds"], done["saved"], done["pushed"]) == (0, 0, False,
                                                                                  None)
    assert done["llm_stopped"].startswith("LLM error: RuntimeError: All LLM providers failed")
    assert done["best_nse"] == done["baseline_nse"] and seen["pushes"] == []
    assert seen["uploads"] == []
    from hf_data import paramstore
    assert paramstore._local(GID, "crest") is None


def test_dashboard_calibration_falls_back_as_before(cal):
    """require_llm off (the dashboard's /api/calibrate): heuristic rounds run
    and their winner is saved; nothing is pushed (persist syncs it)."""
    calibrate, seen = cal
    seen["llm_ok"] = [False] * 99
    rounds, done = _run(calibrate)
    assert len(seen["runs"]) == 1 + 2 * 3
    assert [r.get("source") for r in rounds[1:]] == ["heuristic"] * 2
    assert (done["rounds"], done["llm_rounds"], done["pushed"]) == (2, 0, None)
    assert seen["pushes"] == [] and seen["uploads"] == ["complete"]


def test_llm_failing_later_still_counts(cal):
    calibrate, seen = cal
    seen["llm_ok"] = [True] + [False] * 98
    rounds, done = _run(calibrate, save_vs_baseline=True, require_llm=True)
    assert [r.get("source") for r in rounds[1:]] == ["llm", "heuristic"]
    assert (done["rounds"], done["llm_rounds"]) == (2, 1) and len(seen["runs"]) == 7
    assert done["saved"] is True and done["pushed"] is True


def test_failed_push_is_reported(cal):
    calibrate, seen = cal
    seen["push_ok"] = False
    _, done = _run(calibrate, save_vs_baseline=True, require_llm=True)
    assert done["saved"] is True and done["pushed"] is False
    assert seen["pushes"] == [(GID, "crest")]


def test_candidate_errors_are_collected(cal):
    calibrate, seen = cal
    seen["fail_candidates"] = OSError(28, "No space left on device")
    errs = []
    rounds, done = _run(calibrate, save_vs_baseline=True, require_llm=True, cand_errors=errs)
    assert len(errs) == 6 and all(e is seen["fail_candidates"] for e in errs)
    assert all(t["nse"] is None for r in rounds[1:] for t in r["tried"])
    assert done["best_nse"] == done["baseline_nse"] and done["saved"] is False
