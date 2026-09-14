"""National flood situation brief for Nowcast mode (the chat's opening card).

Two layers, deliberately separate:

  snapshot()   — STRUCTURED, no LLM: the live CONUS picture assembled from the
                 hourly nowcast issue (nowcaststore): every flagged gauge with
                 its name/state, predicted peak and flood-threshold ratio, the
                 state-by-state tally, the ranked hotspot clusters, the 72-h
                 rainfall leaders and the 2-D inundation events in progress.
  brief(force) — the LLM turns that snapshot into a short nationwide summary
                 and CROSS-CHECKS it against current news: with an OpenAI key
                 the Responses API's web-search tool looks for flood / flash
                 flood / heavy-rain reports from the last 48 h in the flagged
                 states and marks each region corroborated / unverified. Any
                 other configured LLM (vLLM etc.) writes the summary from the
                 snapshot alone with an explicit "no news check" caveat.

The brief is cached per nowcast issue time (t0 — hourly), so a busy day costs
at most one web-grounded call per hour, and only when someone actually opens
Nowcast mode. It is built in a background thread: callers get the structured
snapshot immediately (`building: true`) and poll until `text` is present.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timedelta

import numpy as np

from hf_data import llm, nowcaststore

TOP_GAUGES = int(os.environ.get("BRIEF_TOP_GAUGES", "14"))
TOP_STATES = int(os.environ.get("BRIEF_TOP_STATES", "8"))
LLM_TIMEOUT_S = float(os.environ.get("BRIEF_LLM_TIMEOUT_S", "120"))
TIER_NAME = {3: "flood", 2: "minor", 1: "elevated"}

_lock = threading.Lock()
_cache: dict = {"t0": None, "text": None, "provider": None, "generated": None,
                "building": False, "error": None, "verified": False, "sources": []}


# ---- gauge metadata -----------------------------------------------------------
_meta: dict = {"at": 0.0, "map": None}


def _gauge_meta() -> dict:
    """{gid: (name, state)} from the GAGES-II catalog (cached ~6 h)."""
    with _lock:
        if _meta["map"] is not None and time.time() - _meta["at"] < 6 * 3600:
            return _meta["map"]
    m = {}
    try:
        from hf_data import gauges
        cat = gauges.load_catalog()
        for sid, name, st in zip(cat.STAID, cat.STANAME, cat.STATE):
            m[str(sid).zfill(8)] = (str(name), str(st) if st is not None else "")
    except Exception:
        pass
    with _lock:
        _meta.update(at=time.time(), map=m)
    return m


def _f(v):
    try:
        v = float(v)
        return round(v, 2) if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ---- structured snapshot -----------------------------------------------------
def snapshot() -> dict:
    """The live CONUS flood picture, no LLM involved. {"ok": False} when the
    hourly nowcast issue is not available."""
    risk = nowcaststore.all_risk()
    if not risk.get("ok"):
        return {"ok": False, "reason": risk.get("reason")}
    meta, cols = nowcaststore._fresh()
    thr = nowcaststore._thresholds()
    names = _gauge_meta()
    t0 = str(meta.get("t0") or "")
    try:
        t0_dt = datetime.strptime(t0[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        t0_dt = None

    idx = {str(g): i for i, g in enumerate(cols["gid"])}
    qn = nowcaststore._qcols(cols)[:nowcaststore.RISK_H]
    qmat = np.stack([cols[n] for n in qn], 1) if qn else None
    has_obs = "obs_last_q" in cols

    flagged = []
    by_state: dict = {}
    for lat, lon, tier, gid in risk["flagged"]:
        name, st = names.get(gid, ("", ""))
        s = by_state.setdefault(st or "??", {"state": st or "??", "flood": 0,
                                             "minor": 0, "elevated": 0, "score": 0.0})
        s[TIER_NAME[tier]] += 1
        s["score"] += {3: 9.0, 2: 3.0, 1: 1.0}[tier]
        i = idx.get(gid)
        row = {"id": gid, "name": name, "state": st, "tier": tier,
               "tier_name": TIER_NAME[tier], "lat": lat, "lon": lon}
        if i is not None and qmat is not None:
            q = qmat[i]
            k = int(np.nanargmax(q)) if np.isfinite(q).any() else 0
            row["peak_q_m3s"] = _f(q[k])
            row["peak_in_h"] = k + 1
            if t0_dt is not None:
                row["peak_time_utc"] = (t0_dt + timedelta(hours=k + 1)).strftime("%Y-%m-%d %H:%M")
            th = thr.get(gid)
            if th:
                qb, q2, q5, q10 = th
                row["x_bankfull_q2"] = _f(q[k] / q2) if q2 and q2 > 0 else None
                row["x_5yr_q5"] = _f(q[k] / q5) if q5 and q5 > 0 else None
                if tier == 1 and (row["x_bankfull_q2"] or 0) >= 1:
                    # all_risk demoted it: observed flow far below the predicted
                    # rise, or a low-skill gauge — the ratio alone overstates it
                    row["note"] = "kept at 'elevated' by the observation/skill gate"
            if has_obs:
                row["obs_last_q_m3s"] = _f(cols["obs_last_q"][i])
                row["obs_age_h"] = _f(cols["obs_age_h"][i])
        flagged.append(row)
    flagged.sort(key=lambda r: (-r["tier"], -(r.get("x_bankfull_q2") or 0)))
    # red/orange gauges are always listed; yellow ones only pad a quiet day
    serious = [r for r in flagged if r["tier"] >= 2]
    listed = serious[:TOP_GAUGES]
    if len(listed) < 5:
        listed += [r for r in flagged if r["tier"] == 1][:5 - len(listed)]
    states = sorted(by_state.values(), key=lambda s: -s["score"])[:TOP_STATES]

    # ranked clusters, labelled with the states of their member gauges
    hot = []
    try:
        hs = nowcaststore.hotspots()
        for h in (hs.get("hotspots") or [])[:6]:
            sts = sorted({names.get(g["id"], ("", ""))[1] for g in h.get("top_gauges", [])} - {""})
            hot.append({"center": h["center"], "states": sts, "score": h["score"],
                        "n_flood": h["n_flood"], "n_minor": h["n_minor"],
                        "n_elevated": h["n_elevated"],
                        "top_gauges": [f"{g['id']} {names.get(g['id'], ('', ''))[0]}"
                                       for g in h.get("top_gauges", [])[:3]]})
    except Exception:
        pass

    # rainfall drivers: 24-h / 72-h basin-mean MRMS leaders
    rain = []
    try:
        times, pidx, arr = nowcaststore._precip_cache()
        if arr is not None and len(times):
            p72 = np.nansum(arr, 1)
            p24 = np.nansum(arr[:, -24:], 1)
            order = np.argsort(-p24)[:6]
            inv = {i: g for g, i in pidx.items()}
            for i in order:
                if p24[i] <= 0:
                    break
                g = str(inv.get(int(i), ""))
                nm, st = names.get(g, ("", ""))
                rain.append({"id": g, "name": nm, "state": st,
                             "rain_24h_mm": _f(p24[i]), "rain_72h_mm": _f(p72[i])})
    except Exception:
        pass

    # 2-D inundation events in progress or ended in the last day
    events = []
    try:
        from hf_data import eventstore
        cutoff = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%MZ")
        for eid, s in eventstore.load_index().items():
            if s.get("status") == "active" or str(s.get("t_end") or "") >= cutoff:
                gid = str(s.get("gauge") or (s.get("trigger") or {}).get("gauge") or "")
                nm, st = names.get(gid, ("", ""))
                fin = s.get("final") or {}
                events.append({"id": eid, "gauge": gid, "name": nm, "state": st,
                               "status": s.get("status"), "t0": s.get("t0"),
                               "t_end": s.get("t_end"),
                               "peak_depth_m": _f(fin.get("peak_depth_m")),
                               "engine": s.get("engine")})
    except Exception:
        pass

    return {"ok": True, "t0": t0, "generated": meta.get("generated"),
            "n_flood": risk["n_flood"], "n_minor": risk["n_minor"],
            "n_elevated": risk["n_elevated"], "n_rated": risk["n_rated"],
            "states": states, "gauges": listed,
            "n_serious_gauges": len(serious), "hotspots": hot, "rain": rain,
            "events": events}


# ---- LLM brief with news cross-check --------------------------------------------
_SYSTEM = """You write the opening card of the Nowcast mode chat in CREST-AI, a US flood
dashboard. You receive a JSON SNAPSHOT of the live AI nowcast: for every USGS gauge the
DI-LSTM model's predicted next-6-hour peak flow was compared with the gauge's flood
thresholds — tier "flood" = at or above the 5-year return flow, "minor" = above the
2-year (bankfull) flow, "elevated" = above 5x baseflow — issued at t0 (UTC). The snapshot
also carries the state tally, ranked hotspot clusters, 24/72-h basin rainfall leaders and
the 2-D inundation simulations in progress.

Write for a duty forecaster in <= 230 words of markdown:
1. **Headline** — one sentence on the nationwide picture (quiet / localized / widespread),
   with the red/orange/yellow counts.
2. **Where** — 2-5 bullets, most serious first: state or river basin, the named gauges
   (river + town), predicted peak timing (UTC) and how far above bankfull, and the
   rainfall driving it when the rain leaders coincide. Mention active 2-D inundation
   simulations by river/town.
3. **News check** — search the web for reports from the LAST 48 HOURS about flooding,
   flash-flood warnings, heavy rain or storms in the flagged states/regions, AND for any
   major US flooding the nowcast does NOT flag. One bullet per top region:
   "✅ corroborated — <what the report says>" with a markdown link,
   "⚠️ unverified — no recent reports found" (the AI flag stands unconfirmed), or
   "ℹ️ related weather — <warning / forecast context>" with a link.
   At most 4 links total, only URLs you actually retrieved. Never invent gauges,
   numbers, places or links. If nothing is flagged, say the nowcast is quiet and still
   report any major flood news you find.
Finish with the line: _Data as of <t0> UTC · news check: <web search | not available>_
"""


def _compose(snap: dict) -> str:
    return "SNAPSHOT:\n" + json.dumps(snap, default=str)


def _citations(resp) -> list[dict]:
    """URL citations the web-search tool attached to the response text
    (Responses API annotations of type url_citation) — the only links we
    trust. [{"url", "title"}], de-duplicated, in order of appearance."""
    out, seen = [], set()
    for item in getattr(resp, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            for a in getattr(part, "annotations", None) or []:
                url = getattr(a, "url", None)
                if url and url not in seen and getattr(a, "type", "") == "url_citation":
                    seen.add(url)
                    out.append({"url": url, "title": (getattr(a, "title", "") or url)[:120]})
    return out


_MD_LINK = __import__("re").compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _enforce_links(text: str, sources: list[dict]) -> str:
    """Keep markdown links only when their URL is one of the tool citations;
    any other link is reduced to its label + a visible note."""
    ok = {s["url"] for s in sources}
    ok |= {u.rstrip("/") for u in ok}

    def fix(m):
        label, url = m.group(1), m.group(2)
        if url in ok or url.rstrip("/") in ok:
            return m.group(0)
        return f"{label} _(link removed — not among the search results)_"
    return _MD_LINK.sub(fix, text)


_STATE_NAMES = {
    "AL": "Alabama", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "DC"}
NEWS_REGIONS = int(os.environ.get("BRIEF_NEWS_REGIONS", "3"))


def _search_news(client, model: str, query: str) -> tuple[str, list[dict]]:
    """One FORCED web search (tool_choice) — the model may not skip it. Returns
    (findings markdown with the tool's own citations, citations)."""
    r = client.responses.create(
        model=model,
        tools=[{"type": "web_search_preview", "search_context_size": "medium"}],
        tool_choice={"type": "web_search_preview"},
        instructions=("You are a news researcher. Use the web search tool, then report "
                      "ONLY what the retrieved pages say, as 2-4 short bullets: outlet, "
                      "date, what happened / what is warned, with a markdown link to each "
                      "page you cite. Prefer the last 48 hours; say so if reports are "
                      "older. If nothing relevant was found, answer exactly: "
                      "'No relevant reports found.'"),
        input=query)
    return (getattr(r, "output_text", "") or "").strip(), _citations(r)


def _news_queries(snap: dict) -> list[tuple[str, str]]:
    """(label, query) per top region + one national sweep."""
    now = datetime.utcnow().strftime("%B %d, %Y")
    out = []
    for s in (snap.get("states") or [])[:NEWS_REGIONS]:
        st = _STATE_NAMES.get(s["state"], s["state"])
        rivers = [g["name"] for g in snap.get("gauges") or [] if g.get("state") == s["state"]][:2]
        hint = f" ({'; '.join(rivers)})" if rivers else ""
        out.append((st, f"Flooding, flash flood warning or heavy rain in {st}{hint} — "
                        f"news from the last 48 hours as of {now}"))
    out.append(("United States", f"Major flooding or flash flood emergencies anywhere in the "
                                 f"United States — news from the last 48 hours as of {now}"))
    return out


def _build(snap: dict) -> None:
    t0 = snap.get("t0")
    text, provider, verified, sources = None, None, False, []
    try:
        key = os.environ.get("OPENAI_API_KEY")
        news_md = None
        if key:
            # Step 1 — DETERMINISTIC news research: one forced web search per top
            # region + a national sweep. A single "you may search" call was
            # skipped by the model (12-s run, zero citations) and it then
            # invented links; forcing the tool and collecting the tool's own
            # url_citation annotations is the only reliable grounding.
            try:
                from openai import OpenAI
                client = OpenAI(api_key=key, timeout=LLM_TIMEOUT_S)
                model = os.environ.get("OPENAI_MODEL", "gpt-4o")
                parts = []
                for label, q in _news_queries(snap):
                    try:
                        found, cites = _search_news(client, model, q)
                        sources += [c for c in cites if c["url"] not in {s["url"] for s in sources}]
                        parts.append(f"### {label}\n{found or 'No relevant reports found.'}")
                    except Exception as e:
                        parts.append(f"### {label}\n_search failed: {type(e).__name__}_")
                news_md = "\n\n".join(parts)
                provider = "openai+web"
            except Exception as e:
                _cache["error"] = f"web search unavailable: {type(e).__name__}"
        # Step 2 — compose from SNAPSHOT + NEWS FINDINGS (any provider); the
        # writer may only link URLs that appear in the findings.
        sys_p = _SYSTEM
        if news_md is not None:
            user = (_compose(snap) + "\n\nNEWS FINDINGS (from web searches just run; the ONLY "
                    "links you may use, copied verbatim):\n" + news_md)
            sys_p += ("\nThe News check must be written from the NEWS FINDINGS block only "
                      "(do not search again, do not add links that are not in it). Put "
                      "'web search' in the footer.")
        else:
            user = _compose(snap)
            sys_p += ("\nNOTE: you have NO web access in this run — skip the search, write "
                      "the News check section as a single line '⚠️ news check not "
                      "available in this deployment', and put 'not available' in the footer.")
        txt, prov = llm.chat([{"role": "system", "content": sys_p},
                              {"role": "user", "content": user}],
                             temperature=0.3, max_tokens=800)
        text = _enforce_links((txt or "").strip(), sources)
        provider = provider or prov
        verified = bool(sources)
        if news_md is not None and not sources:
            text += ("\n\n⚠️ _The web searches returned no citable sources — treat the "
                     "news check as unverified._")
    except Exception as e:
        with _lock:
            _cache.update(building=False, error=f"{type(e).__name__}: {str(e)[:160]}")
        return
    with _lock:
        _cache.update(t0=t0, text=text, provider=provider, verified=verified,
                      sources=sources,
                      generated=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                      building=False, error=None)


def brief(force: bool = False) -> dict:
    """Snapshot + (cached or in-progress) LLM brief for the current nowcast
    issue. `building: true` means poll again in a few seconds."""
    snap = snapshot()
    if not snap.get("ok"):
        return snap
    with _lock:
        fresh = (_cache["text"] is not None and _cache["t0"] == snap["t0"] and not force)
        if not fresh and not _cache["building"] and llm.available():
            _cache.update(building=True, error=None)
            if force:
                _cache.update(text=None)
            threading.Thread(target=_build, args=(snap,), daemon=True).start()
        out = {"ok": True, "snapshot": snap, "building": _cache["building"],
               "llm": bool(llm.available()),
               "text": _cache["text"] if _cache["t0"] == snap["t0"] else None,
               "provider": _cache["provider"], "verified": _cache["verified"],
               "sources": _cache["sources"] if _cache["t0"] == snap["t0"] else [],
               "generated": _cache["generated"], "error": _cache["error"]}
    if not out["llm"]:
        out["error"] = "no LLM configured — structured snapshot only"
    return out
