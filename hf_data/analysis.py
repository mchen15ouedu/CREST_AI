"""Post-simulation analysis: skill metrics + the ARW report text.

compute_metrics()  - NSCE / CC / %bias / RMSE + peak discharge & timing from the
                     streamed hydrograph rows (sim vs observed).
build_report()     - a concise hydrologic report; written by the LLM router
                     (vLLM -> OpenAI) when configured, else a template fallback.
"""
from __future__ import annotations

import math


def _num(x, nd=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def _peaks(rows):
    best_t, peak_sim, peak_obs = None, None, None
    for r in rows:
        s = r.get("sim_q")
        if s is not None and (peak_sim is None or s > peak_sim):
            peak_sim, best_t = s, r.get("time")
        o = r.get("obs_q")
        if o is not None and not (isinstance(o, float) and math.isnan(o)):
            peak_obs = o if peak_obs is None else max(peak_obs, o)
    return {"peak_sim": _num(peak_sim, 1), "peak_obs": _num(peak_obs, 1), "peak_time": best_t}


def compute_metrics(rows: list[dict]) -> dict:
    """Skill metrics from paired sim/obs discharge (drops missing obs)."""
    import numpy as np
    sim, obs = [], []
    for r in rows:
        s, o = r.get("sim_q"), r.get("obs_q")
        if s is None or o is None or (isinstance(o, float) and math.isnan(o)):
            continue
        sim.append(s); obs.append(o)
    out = {"n_pairs": len(sim), **_peaks(rows)}
    if len(sim) < 2:
        return out
    sim, obs = np.asarray(sim, float), np.asarray(obs, float)
    denom = float(np.sum((obs - obs.mean()) ** 2))
    out["nsce"] = _num(1 - np.sum((obs - sim) ** 2) / denom) if denom > 0 else None
    out["cc"] = _num(np.corrcoef(sim, obs)[0, 1]) if sim.std() > 0 and obs.std() > 0 else None
    out["bias_pct"] = _num(100 * (sim.sum() - obs.sum()) / obs.sum(), 1) if obs.sum() != 0 else None
    out["rmse"] = _num(np.sqrt(np.mean((sim - obs) ** 2)), 1)
    return out


def _facts(meta, metrics, t_start, t_end) -> str:
    m = meta or {}
    parts = [
        f"Gauge {m.get('id','?')} ({m.get('name','')}), drainage {m.get('area','?')} km².",
        f"Model {m.get('model','?').upper()} + kinematic-wave routing.",
        f"Event window {t_start:%Y-%m-%d %H:%M} to {t_end:%Y-%m-%d %H:%M}.",
        f"Peak simulated discharge {metrics.get('peak_sim')} m³/s at {metrics.get('peak_time')}.",
    ]
    if metrics.get("peak_obs") is not None:
        parts.append(f"Peak observed discharge {metrics.get('peak_obs')} m³/s.")
    for k, lbl in (("nsce", "NSCE"), ("cc", "CC"), ("bias_pct", "%bias"), ("rmse", "RMSE")):
        if metrics.get(k) is not None:
            parts.append(f"{lbl}={metrics[k]}.")
    return " ".join(parts)


def _template_report(meta, metrics, t_start, t_end) -> str:
    m = meta or {}
    peak = metrics.get("peak_sim")
    lines = [f"Flood simulation for USGS {m.get('id','?')} ({m.get('name','')}), "
             f"drainage {m.get('area','?')} km², using {m.get('model','?').upper()}+KW over "
             f"{t_start:%Y-%m-%d} to {t_end:%Y-%m-%d}."]
    if peak is not None:
        lines.append(f"Simulated discharge peaks at {peak} m³/s around {metrics.get('peak_time')}.")
    skill = [f"{lbl} {metrics[k]}" for k, lbl in
             (("nsce", "NSCE"), ("cc", "CC"), ("bias_pct", "%bias"), ("rmse", "RMSE"))
             if metrics.get(k) is not None]
    if skill:
        nsce = metrics.get("nsce")
        verdict = ("good" if (nsce or -9) >= 0.5 else "fair" if (nsce or -9) >= 0 else "poor")
        lines.append(f"Skill vs. observed: {', '.join(skill)} ({verdict} agreement).")
    else:
        lines.append("No overlapping observed discharge was available for skill scoring.")
    return " ".join(lines)


def build_report(meta, metrics, t_start, t_end, use_llm: bool = True) -> str:
    """ARW-style report: LLM (vLLM->OpenAI) when configured, else template."""
    if use_llm:
        try:
            from hf_data import llm
            if llm.available():
                sys_p = ("You are a hydrologist writing a brief flood-simulation report "
                         "(~110 words, plain prose, no markdown headers).")
                user_p = ("Write the report from these facts, covering peak discharge and "
                          "timing, model skill, and a short flood assessment:\n"
                          + _facts(meta, metrics, t_start, t_end))
                txt, _ = llm.chat([{"role": "system", "content": sys_p},
                                   {"role": "user", "content": user_p}],
                                  temperature=0.3, max_tokens=320)
                return txt.strip()
        except Exception:
            pass
    return _template_report(meta, metrics, t_start, t_end)
