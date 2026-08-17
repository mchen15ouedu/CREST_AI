"""V30 episode-scoped nowcast verification (user directive 2026-08-14).

Skill is evaluated ONLY inside flood episodes: on flat days any model that
predicts "tomorrow like today" scores near-perfect, so an all-days NSE is
persistence-inflated and says nothing about the hours that matter. Scoring
the episode window alone — rise, crest, recession — keeps the number pure.

Data: every hourly nowcast issue is archived immutably at
nowcast/archive/<YYYYMM>/nc_<YYYYMMDDHH>.parquet (run_nowcast_all.py wrote
them "so forecasts can later be scored against the obs that actually
arrived" — this module is that later). Observations come from hf_data.obs.

score_episode() is called once, at episode finalization, and its result
rides the final product (events/<id>/final.json + index summary).
"""
from __future__ import annotations

import datetime
import math
import os

REPO = os.environ.get("CREST_FEEDBACK_REPO", "vincewin/CREST_data")
LEADS = (1, 3, 6, 12)          # forecast leads [h] reported per episode
MAX_ISSUES = 120               # 96-h cap + backset; bounds downloads


def _archived_issue(t0: datetime.datetime):
    """{lead_h: predicted_cms} for one archived issue, or None if missing."""
    try:
        import numpy as np
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(
            REPO, f"nowcast/archive/{t0:%Y%m}/nc_{t0:%Y%m%d%H}.parquet",
            repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        t = pq.read_table(p)
        return t, np
    except Exception:
        return None


def _gauge_row(table, np, gid: str):
    """{lead_h: cms} from one archive table for one gauge; {} if absent.
    Prefers the 12-h model columns (q12_*), falls back to q1..qN."""
    gids = table.column("gid").to_numpy(zero_copy_only=False)
    idx = np.nonzero(gids == str(gid).zfill(8))[0]
    if len(idx) == 0:
        return {}
    i = int(idx[0])
    names = table.schema.names
    q12 = sorted((n for n in names if n.startswith("q12_")
                  and n[4:].isdigit()), key=lambda n: int(n[4:]))
    q6 = sorted((n for n in names if n[0] == "q" and n[1:].isdigit()),
                key=lambda n: int(n[1:]))
    cols = q12 or q6
    out = {}
    for k, name in enumerate(cols):
        v = float(table.column(name)[i].as_py())
        if v == v and abs(v) != float("inf"):
            out[k + 1] = max(0.0, v)
    return out


def _hourly_obs(gid: str, start: datetime.datetime,
                end: datetime.datetime) -> dict:
    """{hour_dt: observed_cms} nearest-to-the-hour (±30 min)."""
    from . import obs as _obs
    rows = _obs.get_series(gid, start, end)
    out = {}
    for t, q in rows:
        h = t.replace(minute=0, second=0, microsecond=0)
        if t.minute >= 30:
            h += datetime.timedelta(hours=1)
        d = abs((t - h).total_seconds())
        if h not in out or d < out[h][1]:
            out[h] = (float(q), d)
    return {h: v[0] for h, v in out.items()}


def _nse(pairs) -> float | None:
    """Nash-Sutcliffe over (pred, obs) pairs; None below 4 pairs or when
    obs variance is ~0 (an episode window should never be flat, but guard)."""
    if len(pairs) < 4:
        return None
    mo = sum(o for _, o in pairs) / len(pairs)
    ss_tot = sum((o - mo) ** 2 for _, o in pairs)
    if ss_tot < 1e-9:
        return None
    ss_err = sum((p - o) ** 2 for p, o in pairs)
    return round(1.0 - ss_err / ss_tot, 3)


def score_episode(gid: str, start: datetime.datetime,
                  end: datetime.datetime, log=print) -> dict | None:
    """Verification of the AI nowcast at `gid` over one episode window.

    Per lead in LEADS: NSE over all (prediction, later-observed) pairs whose
    valid time falls in [start, end]; predicted-peak magnitude ratio and
    timing error vs the observed peak. None when there's nothing to score
    (no archived issues, or the gauge went silent)."""
    start = start.replace(minute=0, second=0, microsecond=0)
    end = end.replace(minute=0, second=0, microsecond=0)
    n_hours = int((end - start).total_seconds() // 3600)
    if n_hours <= 0:
        return None
    issues = [start + datetime.timedelta(hours=h)
              for h in range(min(n_hours + 1, MAX_ISSUES))]
    obs = _hourly_obs(gid, start,
                      end + datetime.timedelta(hours=max(LEADS)))
    if len(obs) < 4:
        log(f"score {gid}: too few observations — skipped")
        return None
    obs_in = {h: q for h, q in obs.items() if start <= h <= end}
    if not obs_in:
        return None
    peak_t = max(obs_in, key=obs_in.get)
    peak_q = obs_in[peak_t]

    pairs = {k: [] for k in LEADS}          # lead -> [(pred, obs)]
    series = {k: {} for k in LEADS}         # lead -> {valid_t: pred}
    n_found = 0
    for t0 in issues:
        got = _archived_issue(t0)
        if got is None:
            continue
        table, np = got
        preds = _gauge_row(table, np, gid)
        if not preds:
            continue
        n_found += 1
        for k in LEADS:
            if k not in preds:
                continue
            vt = t0 + datetime.timedelta(hours=k)
            if start <= vt <= end:
                series[k][vt] = preds[k]
                if vt in obs:
                    pairs[k].append((preds[k], obs[vt]))
    if n_found == 0:
        log(f"score {gid}: no archived nowcast issues in the window")
        return None

    leads = {}
    for k in LEADS:
        d = {"n": len(pairs[k]), "nse": _nse(pairs[k])}
        if series[k] and peak_q > 0:
            pt = max(series[k], key=series[k].get)
            d["peak_ratio"] = round(series[k][pt] / peak_q, 3)
            d["peak_timing_err_h"] = round(
                (pt - peak_t).total_seconds() / 3600.0, 1)
        leads[str(k)] = d
    out = {"gauge": gid,
           "window": [start.strftime("%Y-%m-%dT%H:%MZ"),
                      end.strftime("%Y-%m-%dT%H:%MZ")],
           "n_issues": n_found,
           "obs_peak_m3s": round(peak_q, 1),
           "obs_peak_t": peak_t.strftime("%Y-%m-%dT%H:%MZ"),
           "leads": leads,
           "nse_h1": leads.get("1", {}).get("nse"),
           "nse_h6": leads.get("6", {}).get("nse")}
    log(f"score {gid}: {n_found} issues, obs peak {peak_q:.1f} m3/s @ "
        f"{peak_t:%m-%d %H:%M}Z, NSE(1h)={out['nse_h1']} "
        f"NSE(6h)={out['nse_h6']}")
    return out
