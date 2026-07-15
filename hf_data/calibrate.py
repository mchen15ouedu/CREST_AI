"""AI-guided calibration (adapted from mchen15ouedu/CREST_GPT_cali AI_cali).

LLM-proposed multiplier updates within hydrologic bounds, evaluated by real
EF5 runs (1-D output only — no 2-D grids), keeping the best NSE. The physics
guide, parameter bounds, and frozen set mirror AI_cali's hydrocalib package
(hydrocalib/config.py + agents/physics_info.py); the two-stage propose/refine
is collapsed into one bounded proposal per round for the demo budget.

Candidate runs warm-start from the existing state at t_start (saved by the
first simulation), so each run is seconds instead of a full warm-up — a demo
approximation: initial states reflect the pre-calibration parameters.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from datetime import datetime

from hf_data import analysis, llm, paramstore

# ---- calibration rules (AI_cali hydrocalib/config.py) ----------------------
PARAM_BOUNDS = {
    # CREST/CRESTPHYS water balance (multipliers over the 1 km grids)
    "wm": (0.1, 10.0), "b": (0.0, 3.0), "im": (0.0, 1.0), "ke": (0.8, 1.2),
    "fc": (0.1, 2.0), "iwu": (25.0, 25.0),
    # KW routing
    "under": (0.1, 10.0), "leaki": (0.1, 10.0), "th": (10.0, 1000.0),
    "isu": (0.0, 0.01), "alpha": (0.1, 3.0), "beta": (0.1, 3.0), "alpha0": (0.0, 3.0),
    # SNOW17
    "uadj": (0.03, 0.19), "mbase": (0.0, 1.0), "mfmax": (0.8, 1.2),
    "mfmin": (0.25, 0.4), "tipm": (0.1, 1.0), "nmf": (0.04, 0.15),
    "plwhc": (0.02, 0.3), "scf": (0.7, 1.2),
}
FROZEN = {"th", "iwu", "isu"}

# extended search: if the standard rounds*k budget still ends below EXT_NSE,
# one extra stage of EXT_ROUNDS*EXT_K candidates is appended automatically
EXT_ROUNDS = int(os.environ.get("CREST_CAL_EXT_ROUNDS", "5"))
EXT_K = int(os.environ.get("CREST_CAL_EXT_K", "4"))
EXT_NSE = float(os.environ.get("CREST_CAL_EXT_NSE", "0.3"))

# AI_cali hydrocalib/agents/physics_info.py PHYSICS_PARAMETER_GUIDE (verbatim core)
PHYSICS_GUIDE = (
    "EF5 Parameter Overview for Calibration\n"
    "WM controls the total soil water storage capacity; higher WM increases infiltration and reduces runoff. "
    "B defines the shape of the variable infiltration curve; larger B yields more surface runoff for a given soil moisture. "
    "IM is the impervious area fraction—higher values reduce infiltration and increase runoff. "
    "KE scales potential evapotranspiration (PET); larger KE increases evaporation and decreases runoff. "
    "FC is the saturated hydraulic conductivity; higher FC allows faster infiltration, reducing surface flow. "
    "IWU sets the initial soil moisture; too high a value can exaggerate early runoff. "
    "TH determines the drainage threshold for channel initiation; a larger TH produces fewer, coarser channels. "
    "UNDER controls interflow velocity—higher values accelerate subsurface flow. "
    "LEAKI defines the leakage rate from the interflow layer; higher LEAKI speeds lateral drainage. "
    "ISU is the initial interflow storage; nonzero values may create unrealistic early peaks. "
    "ALPHA and BETA are routing parameters in the discharge equation Q = αA^β; increasing either slows wave "
    "propagation and broadens flood peaks. ALPHA0 applies the same relationship for non-channel cells. "
    "Adjust WM, B, IM, and FC to shape runoff volume; tune KE for evapotranspiration balance; modify ALPHA, "
    "BETA, UNDER, and LEAKI to match hydrograph timing."
)

PROPOSE_SYSTEM = (
    "You are a hydrologic calibration strategist (EF5/CREST)."
    "\nGiven current parameters, performance metrics, and the history of attempts, propose diverse parameter"
    " update strategies. Return STRICT JSON: {\"candidates\": [{\"id\": str, \"goal\": short str,"
    " \"updates\": {param: number}}]}. Each candidate must explore a clearly different direction within the"
    " allowed bounds; large steps are permitted. Do NOT modify th, iwu, or isu."
    "\n" + PHYSICS_GUIDE
)


def clamp(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        lo, hi = PARAM_BOUNDS.get(k, (None, None))
        if lo is None:
            out[k] = v
        else:
            out[k] = min(hi, max(lo, float(v)))
    return out


def _summarize(metrics: dict) -> dict:
    keep = ("nsce", "cc", "bias_pct", "rmse", "peak_sim", "peak_obs", "peak_time", "n_pairs")
    return {k: metrics.get(k) for k in keep}


def _heuristic_candidates(base: dict, k: int, rnd: random.Random) -> list[dict]:
    """LLM-free fallback: physically-motivated moves toward canonical values."""
    moves = [
        {"id": "runoff-up", "goal": "more runoff volume (lower storage, flashier VIC curve)",
         "updates": {"wm": base.get("wm", 1) * 0.5, "b": min(3.0, max(0.5, base.get("b", 1) * 0.5)),
                     "fc": base.get("fc", 1) * 0.6}},
        {"id": "physical-routing", "goal": "canonical KW routing (beta→0.6)",
         "updates": {"beta": 0.6, "alpha": min(3.0, max(0.3, base.get("alpha", 1)))}},
        {"id": "impervious-up", "goal": "raise impervious fraction + reduce ET",
         "updates": {"im": min(1.0, base.get("im", 0.01) + 0.1), "ke": 0.85}},
        {"id": "storage-up", "goal": "less runoff (raise storage) — opposite direction probe",
         "updates": {"wm": min(10.0, base.get("wm", 1) * 1.8), "b": max(0.2, base.get("b", 1) * 0.8)}},
        {"id": "timing", "goal": "faster interflow + drainage",
         "updates": {"under": min(10.0, base.get("under", 1) * 2.0),
                     "leaki": min(10.0, base.get("leaki", 0.5) * 1.5)}},
    ]
    rnd.shuffle(moves)
    return moves[:k]


def propose(base: dict, metrics: dict, history: list[dict], k: int,
            round_index: int, rnd: random.Random) -> list[dict]:
    """One round of candidates: LLM if available, heuristic ladder otherwise."""
    if llm.available():
        payload = {
            "round": round_index,
            "current_params": base,
            "bounds": {p: PARAM_BOUNDS[p] for p in base if p in PARAM_BOUNDS},
            "frozen": sorted(FROZEN),
            "current_metrics": _summarize(metrics),
            "history": history[-12:],
            "requested_candidates": k,
            "note": ("Parameters are MULTIPLIERS applied to gridded a-priori values, "
                     "except scalars th/isu/iwu/igw/hmaxaq/gwc/gwe. NSE (nsce) is the "
                     "objective; bias_pct shows volume error; peak_sim vs peak_obs shows "
                     "peak magnitude error."),
        }
        try:
            txt, _prov = llm.chat(
                [{"role": "system", "content": PROPOSE_SYSTEM},
                 {"role": "user", "content": json.dumps(payload)}],
                temperature=0.6, max_tokens=900, json_mode=True)
            cands = json.loads(txt).get("candidates", [])[:k]
            out = []
            for i, c in enumerate(cands):
                ups = {kk: float(vv) for kk, vv in (c.get("updates") or {}).items()
                       if kk in PARAM_BOUNDS and kk not in FROZEN}
                if ups:
                    out.append({"id": str(c.get("id", f"cand{i}")),
                                "goal": str(c.get("goal", ""))[:120], "updates": ups})
            if out:
                return out
        except Exception as e:                    # heuristic fallback degrades
            from hf_data import crashlog          # calibration quality — record it
            crashlog.capture("cal:llm-propose", e, round=rnd)
    return _heuristic_candidates(base, k, rnd)


def _mock_nse(params: dict) -> float:
    """Deterministic synthetic NSE for mock mode (UI testing only): rewards
    b→1.0 and beta→0.6, so the loop visibly 'improves'."""
    pb = abs(params.get("b", 1.0) - 1.0)
    pbeta = abs(params.get("beta", 0.6) - 0.6)
    return round(max(-1.0, 0.9 - 0.35 * pb - 0.6 * pbeta), 3)


def run_calibration(gauge_id: str, t_start: datetime, t_end: datetime,
                    model: str = "auto", snow: str = "auto", use_mock: bool = True,
                    rounds: int = 4, k: int = 3, timestep: str = "1h",
                    ext_rounds: int = EXT_ROUNDS, ext_k: int = EXT_K,
                    ext_nse: float = EXT_NSE):
    """Generator of calibration events for the SSE stream.

    Two-stage budget: rounds*k candidates, then — only if the best NSE is
    still below ext_nse — one extended stage of ext_rounds*ext_k more.
    Yields ("status", str) | ("round", dict) | ("hydro", {rows}) |
    ("done", {best_nse, baseline_nse, best_params, saved, improved, extended}).
    """
    from hf_data import multipliers
    from hf_data.pipeline import gauge_info, run_gauge

    g = gauge_info(gauge_id)
    if g is None:
        yield ("status", f"gauge {gauge_id} not found")
        yield ("done", {"error": "gauge not found"})
        return
    wb_model = "crest" if (model == "crest" or (model == "auto" and g["lon"] < -105)) else "crestphys"
    ef5_model = wb_model if model in ("auto", "crest", "crestphys") else model

    # baseline = stored best (if any) else donor multipliers
    stored = paramstore.get(g["id"], ef5_model)
    wbkw = multipliers.to_control_params(g["id"], model=wb_model)
    if wbkw is None:
        yield ("done", {"error": "no calibrated params for this gauge"})
        return
    wb0, kw0 = wbkw
    base = {**wb0, **kw0}
    if stored:
        base = {**base, **stored.get("wb", {}), **stored.get("kw", {})}
    base = clamp(base)

    def run_candidate(params: dict):
        """Run EF5 with the candidate params, 1-D only; return (nse, metrics, rows).
        Each run gets its own workdir, deleted the moment the run ends — a
        calibration leaves no intermediary data behind (the 1-D rows and the
        winning parameter set are all that survive)."""
        rows = []
        work = tempfile.mkdtemp(prefix=f"crest_cal_{g['id']}_")
        try:
            for kind, payload in run_gauge(
                    g["id"], t_start, t_end, model=model, use_mock=use_mock,
                    overrides=params, snow=snow, timestep=timestep,
                    grids=False, no_cache=True, workdir=work):
                if kind == "hydro":
                    rows += payload["rows"]
        finally:
            shutil.rmtree(work, ignore_errors=True)
        m = analysis.compute_metrics(rows)
        nse = _mock_nse(params) if use_mock else m.get("nsce")
        return nse, m, rows

    yield ("status", f"baseline run — current parameters ({'stored best' if stored else 'a-priori'})")
    base_nse, base_metrics, base_rows = run_candidate(base)
    yield ("round", {"round": 0, "tried": [{"id": "baseline", "goal": "current parameters",
                                            "nse": base_nse}],
                     "best_nse": base_nse})

    best_nse, best_params, best_rows = base_nse, dict(base), base_rows
    history = [{"id": "baseline", "updates": {}, "nse": base_nse}]
    rnd = random.Random(int(gauge_id) if str(gauge_id).isdigit() else 42)

    total = rounds * k
    done_runs = 0
    stages = [(rounds, k)]          # extended stage appended below if warranted
    extended = False
    r = 0                           # global round counter across stages
    si = 0
    while si < len(stages):
        n_rounds, kk = stages[si]
        for _ in range(n_rounds):
            r += 1
            cands = propose(best_params, base_metrics, history, kk, r, rnd)
            tried = []
            for c in cands:
                params = clamp({**best_params, **c["updates"]})
                yield ("status", f"round {r}: testing “{c['goal'] or c['id']}”")
                try:
                    nse, m, rows = run_candidate(params)
                except Exception as e:
                    nse, m, rows = None, {}, []
                    yield ("status", f"round {r}: candidate {c['id']} failed ({e})")
                done_runs += 1
                tried.append({"id": c["id"], "goal": c["goal"], "nse": nse,
                              "progress": round(done_runs / total, 3)})
                history.append({"id": c["id"], "updates": c["updates"], "nse": nse})
                if nse is not None and (best_nse is None or nse > best_nse):
                    best_nse, best_params, best_rows = nse, params, rows
                    yield ("hydro", {"rows": rows})
            yield ("round", {"round": r, "tried": tried, "best_nse": best_nse,
                             "best_params": best_params, "extended": extended})
        si += 1
        # still a poor fit after the standard budget -> widen the search once
        if si == 1 and not extended and ext_rounds > 0 and ext_k > 0 \
                and (best_nse is None or best_nse < ext_nse):
            extended = True
            stages.append((ext_rounds, ext_k))
            total += ext_rounds * ext_k
            shown = "n/a" if best_nse is None else f"{best_nse:.3f}"
            yield ("status", f"📉 best NSE {shown} still below {ext_nse:g} after "
                             f"{done_runs} trials — extended search: "
                             f"+{ext_rounds} rounds × {ext_k} candidates "
                             f"({ext_rounds * ext_k} more runs)")

    improved = (base_nse is None) or (best_nse is not None and best_nse > base_nse)
    saved = False
    if best_nse is not None:
        wb_best = {p: best_params[p] for p in wb0 if p in best_params}
        kw_best = {p: best_params[p] for p in kw0 if p in best_params}
        saved = paramstore.maybe_save(
            g["id"], ef5_model, wb_best, kw_best, best_nse, source="ai-cali",
            window=[t_start.strftime("%Y-%m-%d %H:%M"), t_end.strftime("%Y-%m-%d %H:%M")])
    yield ("done", {"best_nse": best_nse, "baseline_nse": base_nse,
                    "best_params": best_params, "saved": saved, "improved": improved,
                    "extended": extended})
