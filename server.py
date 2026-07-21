"""CREST_demo — FastAPI backend for the map-centric dashboard.

Serves the Leaflet frontend + JSON/SSE APIs, reusing the hf_data layer + agents.
  POST /api/query      NL query -> region + event pins + gauge pins (agent-driven)
  GET  /api/gauges     gauge pins for a map viewport (map-first, no AI needed)
  POST /api/simulate   selected gauges -> sim job (cap 10; ~4 concurrent)
  GET  /api/stream     SSE live status + hydrograph + 2-D overlays
  POST /api/eventinfo  LLM event brief (damage/fatalities/links) for the AI info feed
  POST /api/calibrate  AI calibration job (NSE<0.3 flow);  GET /api/calstream SSE
  /login /auth/callback /logout /api/me /api/profile   HF OAuth + profile store
  GET/POST /api/favorites   registered-user focused basins (up to 5 gauges)
Run: python server.py   (uvicorn on :7860, Docker Space entrypoint)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
import math
import secrets
import time
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.sessions import SessionMiddleware

from hf_data import caljobs, crashlog, datamgr, gauges, persist, simjobs
from hf_data.pipeline import parse_query
from hf_data.statecache import CACHE_DIR

crashlog.init()                    # optional SENTRY_DSN mirror (Sentry/GlitchTip/Bugsink)
crashlog.install_thread_hook()     # record any thread that dies unwrapped
datamgr.start_janitor()            # hourly cache cleanup + result compaction
persist.start()                    # restore states/results/params/users from the
                                   # private dataset, then keep them synced

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(HERE, "frontend")
MAX_SIMS = 10                      # demo cap (selection)

app = FastAPI(title="CREST_demo")
app.add_middleware(SessionMiddleware,
                   secret_key=os.environ.get("SESSION_SECRET", secrets.token_hex(32)),
                   max_age=14 * 24 * 3600, same_site="lax", https_only=False)

# ---- HF OAuth (Space sets OAUTH_* env when README has hf_oauth: true) ------
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
_oauth = None
if OAUTH_CLIENT_ID:
    from authlib.integrations.starlette_client import OAuth

    _oauth = OAuth()
    _oauth.register(
        name="huggingface",
        client_id=OAUTH_CLIENT_ID,
        client_secret=os.environ.get("OAUTH_CLIENT_SECRET"),
        server_metadata_url=(os.environ.get("OPENID_PROVIDER_URL", "https://huggingface.co")
                             + "/.well-known/openid-configuration"),
        client_kwargs={"scope": os.environ.get("OAUTH_SCOPES", "openid profile")},
    )


def _profile_path(username: str) -> str:
    d = os.path.join(CACHE_DIR, "users")
    os.makedirs(d, exist_ok=True)
    safe = "".join(c for c in username if c.isalnum() or c in "-_")
    return os.path.join(d, f"{safe}.json")


def _load_profile(username: str) -> dict:
    try:
        with open(_profile_path(username), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


@app.middleware("http")
async def _error_recorder(request: Request, call_next):
    """Error watchdog: every unhandled server error is recorded with context."""
    try:
        return await call_next(request)
    except Exception as e:
        crashlog.capture(f"http:{request.url.path}", e, method=request.method)
        raise


@app.get("/api/errors")
def api_errors(n: int = 50):
    """Recent recorded errors (the error-watchdog log)."""
    return {"errors": crashlog.recent(min(n, 200)), **crashlog.stats()}


class ClientError(BaseModel):
    message: str = ""
    source: str | None = None
    line: int | None = None
    stack: str | None = None


_client_err_ts: list[float] = []   # flood guard for the browser error beacon


@app.post("/api/clienterror")
def api_clienterror(err: ClientError):
    """Browser-side error beacon (app.js window.onerror/unhandledrejection) —
    frontend crashes land in the same watchdog log as server errors."""
    now = time.time()
    _client_err_ts[:] = [t for t in _client_err_ts if now - t < 60]
    if len(_client_err_ts) >= 30:                    # global: max 30/min
        return {"ok": False, "throttled": True}
    _client_err_ts.append(now)
    crashlog.capture("client:js", message=(err.message or "")[:400],
                     source=err.source, line=err.line, stack=err.stack)
    return {"ok": True}


@app.get("/api/datastats")
def api_datastats():
    """Data-manager view: per-category disk usage + caps."""
    return datamgr.stats()


@app.post("/api/datacleanup")
def api_datacleanup():
    """Run a janitor pass now (cleanup + result compaction)."""
    rep = datamgr.cleanup()
    rep["compact"] = datamgr.compact_results()
    return rep


@app.get("/api/report/{sim_id}/{gauge_id}")
def api_report(sim_id: str, gauge_id: str, request: Request):
    """Downloadable simulation report (AQUAH report-writer agent -> PDF).
    Generated on first request (LLM + pandoc, ~30-90 s), then cached. Signed-in
    users get a persistent copy in their report library; anonymous reports stay
    in the ephemeral cache and are discarded when they close the app."""
    job = simjobs.get_job(sim_id)
    if job is None:
        return JSONResponse({"error": "simulation not found (it may have been "
                                      "restarted away) — re-run it, then download "
                                      "the report"}, status_code=404)
    if not (job.hydro.get(gauge_id) or []):
        return JSONResponse({"error": "no results for this gauge yet"}, status_code=409)
    from hf_data import report
    try:
        path = report.generate(job, gauge_id)
    except Exception as e:
        crashlog.capture("report", e, sim_id=sim_id, gauge=gauge_id)
        return JSONResponse({"error": f"report generation failed: {e}"}, status_code=500)
    saved = False
    u = request.session.get("user")
    if u:                              # registered-user benefit: keep the report
        try:
            report.save_for_user(u["username"], gauge_id, job, path)
            persist.poke()
            saved = True
        except Exception as e:
            crashlog.capture("report:save", e, user=u.get("username"))
    ext = os.path.splitext(path)[1]
    return FileResponse(
        path,
        media_type="application/pdf" if ext == ".pdf" else "text/markdown",
        filename=f"CREST_report_{gauge_id}_{job.t_start:%Y%m%d}{ext}",
        headers={"X-Report-Saved": "1" if saved else "0"})


@app.get("/api/myreports")
def api_myreports(request: Request):
    """Signed-in user's saved report library."""
    u = request.session.get("user")
    if not u:
        return JSONResponse({"error": "not signed in"}, status_code=401)
    from hf_data import report
    return {"reports": report.list_saved(u["username"]),
            "max": report.MAX_SAVED_PER_USER}


@app.get("/api/myreports/{name}")
def api_myreports_get(name: str, request: Request):
    u = request.session.get("user")
    if not u:
        return JSONResponse({"error": "not signed in"}, status_code=401)
    from hf_data import report
    p = report.saved_path(u["username"], name)
    if not p:
        return JSONResponse({"error": "report not found"}, status_code=404)
    return FileResponse(p, media_type="application/pdf" if p.endswith(".pdf")
                        else "text/markdown", filename=os.path.basename(p))


@app.delete("/api/myreports/{name}")
def api_myreports_delete(name: str, request: Request):
    u = request.session.get("user")
    if not u:
        return JSONResponse({"error": "not signed in"}, status_code=401)
    from hf_data import report
    p = report.saved_path(u["username"], name)
    if p:
        os.remove(p)
        persist.poke()                 # mirror the deletion to the private dataset
    return {"ok": True, "reports": report.list_saved(u["username"])}


@app.post("/api/report_discard/{sim_id}")
def api_report_discard(sim_id: str):
    """App-close beacon from anonymous users: drop their ephemeral reports.
    Saved (registered) copies are untouched; everything here is regenerable."""
    from hf_data import report
    return {"ok": True, "removed": report.discard_sim(sim_id)}


@app.get("/api/persist")
def api_persist_status():
    """Durable-cache sync status (private-dataset mirror)."""
    return persist.status()


@app.post("/api/persist")
def api_persist_now():
    """Push a sync pass right now (normally every 10 min / after each run)."""
    try:
        return {"ok": True, **persist.backup()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


APP_VERSION = str(int(time.time()))    # changes every deploy/restart


@app.get("/")
def index():
    """index.html is never browser-cached, and its asset URLs carry a version
    that changes on every deploy — users always get the latest frontend
    without needing a hard refresh."""
    with open(os.path.join(FRONTEND, "index.html"), encoding="utf-8") as fh:
        html = fh.read().replace("__V__", APP_VERSION)
    return Response(html, media_type="text/html",
                    headers={"Cache-Control": "no-cache"})


@app.get("/login")
async def login(request: Request):
    if _oauth is None:            # local dev fallback: fake session, no HF round-trip
        request.session["user"] = {"username": "dev-user", "name": "Dev User",
                                   "picture": None, "dev": True}
        return RedirectResponse("/")
    redirect_uri = str(request.url_for("auth_callback")).replace("http://", "https://") \
        if os.environ.get("SPACE_HOST") else request.url_for("auth_callback")
    return await _oauth.huggingface.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback")
async def auth_callback(request: Request):
    if _oauth is None:
        return RedirectResponse("/")
    try:
        token = await _oauth.huggingface.authorize_access_token(request)
        info = token.get("userinfo") or {}
        request.session["user"] = {
            "username": info.get("preferred_username") or info.get("sub", "user"),
            "name": info.get("name") or info.get("preferred_username", "user"),
            "picture": info.get("picture"),
        }
    except Exception as e:
        crashlog.capture("auth:callback", e)   # user lands signed-out; know why
    return RedirectResponse("/")


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")


@app.get("/api/me")
def api_me(request: Request):
    u = request.session.get("user")
    if not u:
        return {"user": None, "oauth": _oauth is not None}
    return {"user": u, "profile": _load_profile(u["username"]), "oauth": _oauth is not None}


class ProfileUpdate(BaseModel):
    display_name: str | None = None
    affiliation: str | None = None
    email: str | None = None
    bio: str | None = None


@app.post("/api/profile")
def api_profile(request: Request, p: ProfileUpdate):
    u = request.session.get("user")
    if not u:
        return JSONResponse({"error": "not signed in"}, status_code=401)
    prof = _load_profile(u["username"])
    prof.update({k: v for k, v in p.model_dump().items() if v is not None})
    with open(_profile_path(u["username"]), "w", encoding="utf-8") as fh:
        json.dump(prof, fh, indent=1)
    persist.poke()
    return {"ok": True, "profile": prof}


# ---- registered-user benefit: focused basins (favorite gauges) -------------
MAX_FAVORITES = 5


def _fav_pins(ids: list) -> list:
    """Favorite gauge ids -> map pins with catalog metadata (unknown ids skipped)."""
    cat = gauges.load_catalog()
    out = []
    for gid in ids:
        row = cat.loc[cat.STAID == gid]
        if row.empty:
            continue
        r = row.iloc[0]
        out.append({"id": gid, "name": str(r.STANAME), "lat": float(r.LAT_GAGE),
                    "lon": float(r.LNG_GAGE), "area_km2": float(r.DRAIN_SQKM)})
    return out


@app.get("/api/favorites")
def api_favorites(request: Request):
    u = request.session.get("user")
    if not u:
        return JSONResponse({"error": "not signed in"}, status_code=401)
    ids = _load_profile(u["username"]).get("favorites", [])
    return {"favorites": _fav_pins(ids), "max": MAX_FAVORITES}


class FavoriteReq(BaseModel):
    gauge_id: str
    action: str = "add"           # add | remove


@app.post("/api/favorites")
def api_favorites_post(request: Request, f: FavoriteReq):
    u = request.session.get("user")
    if not u:
        return JSONResponse({"error": "Sign in to save favorite gauges."}, status_code=401)
    digits = "".join(c for c in f.gauge_id if c.isdigit())
    if not digits:
        return JSONResponse({"error": "Enter a USGS gauge number, e.g. 08167000."},
                            status_code=422)
    gid = digits.zfill(8)          # catalog STAIDs are zero-padded to >= 8 digits
    prof = _load_profile(u["username"])
    favs = [g for g in prof.get("favorites", []) if g]
    if f.action == "remove":
        favs = [g for g in favs if g != gid]
    elif gid not in favs:
        if gauges.get_gauge_coordinates(gid) is None:
            return JSONResponse(
                {"error": f"Gauge {gid} isn't in the GAGES-II catalog — "
                          "double-check the USGS station number."}, status_code=404)
        if len(favs) >= MAX_FAVORITES:
            return JSONResponse(
                {"error": f"You already have {MAX_FAVORITES} focused basins — "
                          "remove one first."}, status_code=409)
        favs.append(gid)
    prof["favorites"] = favs
    with open(_profile_path(u["username"]), "w", encoding="utf-8") as fh:
        json.dump(prof, fh, indent=1)
    persist.poke()                 # favorites are account data — sync soon
    return {"ok": True, "favorites": _fav_pins(favs), "max": MAX_FAVORITES}


# ---- map-first gauge pins ---------------------------------------------------
def _pins_for_bbox(w: float, s: float, e: float, n: float, limit: int = 300):
    cat = gauges.load_catalog((w, s, e, n))
    if len(cat) > limit:                            # keep the map responsive
        cx, cy = (w + e) / 2, (s + n) / 2
        cat = cat.assign(_d=((cat.LAT_GAGE - cy) ** 2 + (cat.LNG_GAGE - cx) ** 2)).nsmallest(limit, "_d")
    from hf_data import gaugetz
    return [{
        "id": str(r.STAID).zfill(8), "name": str(r.STANAME),
        "lat": float(r.LAT_GAGE), "lon": float(r.LNG_GAGE),
        "area_km2": float(r.DRAIN_SQKM),
        "tz": gaugetz.tz_of(str(r.STAID).zfill(8)),
    } for r in cat.itertuples()]


@app.get("/api/gauges")
def api_gauges(w: float, s: float, e: float, n: float):
    """USGS gauge pins for the current viewport — works with zero AI interaction."""
    pins = _pins_for_bbox(w, s, e, n)
    return {"gauge_pins": pins, "n_gauges": len(pins), "max_sims": MAX_SIMS}


class Query(BaseModel):
    query: str
    radius_deg: float = 1.0


@app.post("/api/query")
def api_query(q: Query):
    """Parse the query into a map view + event pins + nearby USGS gauge pins."""
    try:
        ctx = parse_query(q.query)
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=422)
    lat, lon = ctx.anchor
    r = q.radius_deg
    bbox = [lon - r, lat - r, lon + r, lat + r]
    gauge_pins = _pins_for_bbox(*bbox)
    event_pins = [{"lat": lat, "lon": lon, "label": ctx.label}]

    return {
        "label": ctx.label, "center": [lat, lon], "bbox": bbox,
        "t_start": ctx.t_start.isoformat(), "t_end": ctx.t_end.isoformat(),
        "time_known": ctx.time_known,
        "event_pins": event_pins, "gauge_pins": gauge_pins,
        "n_gauges": len(gauge_pins), "max_sims": MAX_SIMS,
    }


class FeedbackReq(BaseModel):
    text: str
    contact: str | None = None


@app.post("/api/feedback")
def api_feedback(req: FeedbackReq, request: Request):
    """Test-user improvement comments — recorded locally + persisted to the HF
    dataset for the daily review job."""
    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "empty comment"}, status_code=422)
    from hf_data import feedback
    u = request.session.get("user")
    rec = feedback.record(text, user=u["username"] if u else None,
                          contact=req.contact,
                          context={"ua": request.headers.get("user-agent", "")[:120]})
    return {"ok": True, "id": rec["id"]}


@app.get("/api/feedback")
def api_feedback_list(n: int = 100):
    from hf_data import feedback
    return {"feedback": feedback.recent(min(n, 500))}


class ChatReq(BaseModel):
    message: str
    history: list = []
    context: dict = {}


@app.post("/api/chat")
def api_chat(req: ChatReq):
    """Conversational agent turn: guides users to an event, answers questions
    about the current simulation, and routes locate/set_time actions.
    {"action": "fallback"} tells the client to use its rule-based path."""
    from hf_data import chatagent
    try:
        d = chatagent.respond(req.message, req.history, req.context)
    except Exception as e:
        crashlog.capture("chat", e, message_text=req.message[:200])
        return {"action": "fallback", "error": str(e)}
    return d if d else {"action": "fallback"}


class ExplainReq(BaseModel):
    error: str
    where: str | None = None       # e.g. "simulation" | "calibration"
    context: dict = {}


@app.post("/api/explain")
def api_explain(req: ExplainReq):
    """Translate a backend error into a human-readable chat explanation."""
    from hf_data import llm
    if not llm.available():
        return {"text": None}
    try:
        sys_p = ("You are the assistant in a flood-simulation dashboard (CREST/EF5 "
                 "hydrologic model, data streamed from Hugging Face). A backend step "
                 "failed. Explain to a NON-developer user in 2-4 short sentences: "
                 "(1) what went wrong in plain words, (2) the most likely cause, "
                 "(3) what to do — e.g. simply try again (transient network reads are "
                 "common), pick a different time period/gauge, or use the 💡 Feedback "
                 "button to report it. No stack traces, no jargon, no blame.")
        user_p = (f"Failed step: {req.where or 'simulation'}\n"
                  f"Raw error: {req.error[:1500]}\n"
                  f"Context: {str(req.context)[:500]}")
        text, provider = llm.chat([{"role": "system", "content": sys_p},
                                   {"role": "user", "content": user_p}],
                                  temperature=0.2)
        return {"text": text, "provider": provider}
    except Exception as e:
        crashlog.capture("explain", e)
        return {"text": None}


class EventInfoReq(BaseModel):
    label: str
    t_start: str
    t_end: str


@app.post("/api/eventinfo")
def api_eventinfo(req: EventInfoReq):
    """LLM brief about the event (damage, fatalities, links) for the AI info feed."""
    from hf_data import llm
    if not llm.available():
        return {"text": "_(no LLM configured — event background unavailable)_", "provider": None}
    try:
        text, provider = llm.event_brief(req.label, req.t_start[:10], req.t_end[:10])
        return {"text": text, "provider": provider}
    except Exception as e:
        return {"text": f"_(event lookup failed: {e})_", "provider": None}


class SimRequest(BaseModel):
    gauge_ids: list[str]
    t_start: str | None = None
    t_end: str | None = None
    hours: int = 48
    model: str = "auto"
    snow: str = "auto"
    timestep: str = "1h"
    warmup_days: int = 90
    overrides: dict | None = None
    label: str | None = None          # event label for the history entry
    prev_sim_id: str | None = None    # caller's previous job — superseded (cancelled)
    scheme: str = "full"              # "full" = whole basin; "speed" = domain
                                      # truncated at boundary gauges (obs = inflow)


MAX_HISTORY = 20


def _save_history(username: str, entry: dict):
    """Registered-user benefit: keep their simulation history server-side."""
    prof = _load_profile(username)
    hist = [h for h in prof.get("history", []) if h.get("sim_id") != entry["sim_id"]]
    hist.insert(0, entry)
    prof["history"] = hist[:MAX_HISTORY]
    with open(_profile_path(username), "w", encoding="utf-8") as fh:
        json.dump(prof, fh, indent=1)


# generous ceiling: a full water year fits; only guards the shared demo CPU
# against extreme multi-year requests (0 disables the cap entirely)
MAX_SIM_HOURS = int(os.environ.get("CREST_MAX_SIM_HOURS", "8784"))   # 366 days


@app.post("/api/simulate")
def api_simulate(req: SimRequest, request: Request):
    warnings = []
    if len(req.gauge_ids) > MAX_SIMS:
        warnings.append(f"This demo simulates at most {MAX_SIMS} gauges at once; "
                        f"running the first {MAX_SIMS} of {len(req.gauge_ids)} selected.")
    t0 = datetime.fromisoformat(req.t_start) if req.t_start else datetime(2025, 7, 3)
    t1 = datetime.fromisoformat(req.t_end) if req.t_end else t0 + timedelta(hours=req.hours)
    if t1 <= t0:                                       # nonsensical window
        t1 = t0 + timedelta(hours=req.hours)
        warnings.append(f"End time was not after the start — using {req.hours} h instead.")
    if MAX_SIM_HOURS > 0 and (t1 - t0) > timedelta(hours=MAX_SIM_HOURS):
        t1 = t0 + timedelta(hours=MAX_SIM_HOURS)       # shared-CPU sanity ceiling
        warnings.append(f"Simulation window capped at {MAX_SIM_HOURS // 24} days "
                        f"(now ends {t1:%Y-%m-%d %H:%M}).")
    # supersede the caller's previous run: a stale in-flight job would otherwise
    # hold the per-gauge lock and the new run would queue behind it indefinitely
    if req.prev_sim_id:
        prev = simjobs.get_job(req.prev_sim_id)
        if prev and not prev.done.is_set():
            prev.cancel.set()
            warnings.append("Your previous simulation was still running — "
                            "it was stopped and replaced by this one.")
    opts = {"model": req.model, "hours": req.hours, "snow": req.snow,
            "timestep": req.timestep, "warmup_days": req.warmup_days,
            "overrides": req.overrides,
            "scheme": req.scheme if req.scheme in ("full", "speed") else "full"}
    job = simjobs.start_job(req.gauge_ids, t0, t1, opts)
    u = request.session.get("user")
    if u:                                              # signed-in -> history entry
        try:
            _save_history(u["username"], {
                "sim_id": job.id, "gauge_ids": job.gauge_ids,
                "t_start": t0.isoformat(), "t_end": t1.isoformat(),
                "label": req.label, "model": req.model, "snow": req.snow,
                "when": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")})
        except Exception as e:
            crashlog.capture("history:save", e, user=u.get("username"))
    return {"sim_id": job.id, "gauge_ids": job.gauge_ids,
            "t_start": t0.isoformat(), "t_end": t1.isoformat(),
            "warning": " ".join(warnings) or None,
            "max_concurrent": simjobs.MAX_CONCURRENT}


@app.get("/api/nowcast/{sim_id}/{gauge_id}")
def api_nowcast(sim_id: str, gauge_id: str):
    """AI nowcast tail (+6 h) from the CREST_nowcast Space for a finished
    gauge. Always answers; {"ok": false} just means no tail is shown."""
    job = simjobs.get_job(sim_id)
    if not job:
        return JSONResponse({"ok": False, "reason": "unknown job"}, status_code=404)
    from hf_data import nowcast as _nc
    return _nc.for_job(job, gauge_id)


@app.get("/api/nowcast_risk")
def api_nowcast_risk():
    """CONUS-wide flood-risk snapshot for Nowcast mode: which gauges' AI
    next-6-h peak exceeds their 10-yr return flood (red density layer /
    red pins)."""
    from hf_data import nowcaststore
    return nowcaststore.all_risk()


@app.get("/api/nowcast_now")
def api_nowcast_now(w: float, s: float, e: float, n: float, limit: int = 100,
                    obs_hours: int = 0, ids: str = ""):
    """Precomputed fleet nowcasts for every gauge in the viewport/rectangle
    (or an explicit `ids` comma list) — instant (the updater Space refreshes
    them hourly), no simulation needed. obs_hours>0 attaches recent observed
    series (<=25 gauges) for plotting."""
    from hf_data import nowcaststore
    return nowcaststore.for_bbox(w, s, e, n, limit, obs_hours, ids)


@app.get("/api/upstream")
def api_upstream(gid: str):
    """River network upstream of a gauge (HydroRIVERS topology walk) — drawn
    on the Nowcast map when the gauge is focused. Static data, LRU-cached.
    JSONResponse directly: jsonable_encoder walking ~50k coord floats costs
    ~2 s; plain json.dumps is ~50 ms."""
    from hf_data import rivernet
    return JSONResponse(rivernet.upstream(gid))


@app.post("/api/cancel/{sim_id}")
def api_cancel(sim_id: str):
    """Stop a running simulation job: the EF5 processes are killed and the
    per-gauge run locks released so a new run can start immediately."""
    job = simjobs.get_job(sim_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    if job.done.is_set():
        return {"ok": True, "already_done": True}
    job.cancel.set()
    return {"ok": True}


@app.get("/api/history")
def api_history(request: Request):
    """Signed-in users: their saved simulations, with live job status."""
    u = request.session.get("user")
    if not u:
        return JSONResponse({"error": "not signed in"}, status_code=401)
    hist = _load_profile(u["username"]).get("history", [])
    for h in hist:
        job = simjobs.get_job(h.get("sim_id", ""))
        h["status"] = ("done" if job and job.done.is_set() else
                       "running" if job else "expired")   # expired -> cache re-run
    return {"history": hist}


def _sse_json(ev) -> str:
    """json.dumps emits bare NaN/Infinity, which JSON.parse rejects — one such
    value breaks the whole event stream in the browser. Missing values travel as
    null, matching the convention the rest of the pipeline already uses."""
    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v
    return json.dumps(clean(ev), allow_nan=False)


async def _drain(job, cursor: int = 0):
    """Replay the job's event log from `cursor`, then follow it live. Multiple
    clients (or a reopened browser) can each attach with their own cursor."""
    try:
        while True:
            if cursor < len(job.events):
                ev = job.events[cursor]
                cursor += 1
                yield {"data": _sse_json(ev)}
                if ev.get("kind") in ("all_done", "cal_done"):
                    break
            elif job.done.is_set():
                break
            else:
                await asyncio.sleep(0.15)
    except asyncio.CancelledError:     # client closed the tab — not an error
        raise
    except Exception as e:             # SSE runs outside the HTTP middleware
        crashlog.capture("sse:drain", e, job=getattr(job, "id", "?"), cursor=cursor)


@app.get("/api/stream/{sim_id}")
async def api_stream(sim_id: str, cursor: int = 0):
    job = simjobs.get_job(sim_id)
    if not job:
        return Response(status_code=404)
    return EventSourceResponse(_drain(job, cursor))


@app.get("/api/job/{sim_id}")
def api_job(sim_id: str):
    """Job descriptor for reattaching after the browser was closed. The run
    keeps going server-side; the client replays the event log via /api/stream."""
    job = simjobs.get_job(sim_id)
    if not job:
        return JSONResponse({"error": "unknown or expired job"}, status_code=404)
    return {"sim_id": job.id, "gauge_ids": job.gauge_ids,
            "t_start": job.t_start.isoformat(), "t_end": job.t_end.isoformat(),
            "done": job.done.is_set(), "n_events": len(job.events),
            "age_s": int(time.time() - job.created)}


class CalRequest(BaseModel):
    gauge_id: str
    t_start: str
    t_end: str
    model: str = "auto"
    snow: str = "auto"
    # budget = rounds*k candidate runs + 1 baseline (defaults: 4*3+1 = 13 EF5
    # runs). If the best NSE is still < 0.3 afterwards, calibrate.py appends
    # one extended stage of 5 rounds x 4 candidates (+20 runs) automatically.
    # Space variables: CREST_CAL_ROUNDS / CREST_CAL_K (standard stage) and
    # CREST_CAL_EXT_ROUNDS / CREST_CAL_EXT_K / CREST_CAL_EXT_NSE (extension) —
    # no redeploy needed (candidates warm-start from the saved state, so each
    # extra run costs seconds-to-minutes, not a full warm-up).
    rounds: int = int(os.environ.get("CREST_CAL_ROUNDS", "4"))
    k: int = int(os.environ.get("CREST_CAL_K", "3"))


@app.post("/api/calibrate")
def api_calibrate(req: CalRequest):
    job = caljobs.start_job(req.gauge_id,
                            datetime.fromisoformat(req.t_start),
                            datetime.fromisoformat(req.t_end),
                            {"model": req.model, "snow": req.snow,
                             "rounds": req.rounds, "k": req.k})
    return {"cal_id": job.id, "gauge_id": req.gauge_id}


@app.get("/api/calstream/{cal_id}")
async def api_calstream(cal_id: str, cursor: int = 0):
    job = caljobs.get_job(cal_id)
    if not job:
        return Response(status_code=404)
    return EventSourceResponse(_drain(job, cursor))


@app.get("/api/overlay/{sim_id}/{gauge_id}.png")
def api_overlay(sim_id: str, gauge_id: str):
    job = simjobs.get_job(sim_id)
    png = job.overlay_png(gauge_id) if job else None
    if png is None:
        return Response(status_code=404)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/frame/{sim_id}/{gauge_id}/{idx}.png")
def api_frame(sim_id: str, gauge_id: str, idx: int):
    """A single pre-rendered animation frame (fixed color scale)."""
    job = simjobs.get_job(sim_id)
    png = job.frame_png(gauge_id, idx) if job else None
    if png is None:
        return Response(status_code=404)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "max-age=3600"})


if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
