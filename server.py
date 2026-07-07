"""CREST_demo — FastAPI backend for the map-centric dashboard.

Serves the Leaflet frontend + JSON/SSE APIs, reusing the hf_data layer + agents.
  POST /api/query      NL query -> region + event pins + gauge pins (agent-driven)
  GET  /api/gauges     gauge pins for a map viewport (map-first, no AI needed)
  POST /api/simulate   selected gauges -> sim job (cap 10; ~4 concurrent)
  GET  /api/stream     SSE live status + hydrograph + 2-D overlays
  POST /api/eventinfo  LLM event brief (damage/fatalities/links) for the AI info feed
  POST /api/calibrate  AI calibration job (NSE<0.3 flow);  GET /api/calstream SSE
  /login /auth/callback /logout /api/me /api/profile   HF OAuth + profile store
Run: python server.py   (uvicorn on :7860, Docker Space entrypoint)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
import queue
import secrets
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.sessions import SessionMiddleware

from hf_data import caljobs, gauges, simjobs
from hf_data.pipeline import parse_query
from hf_data.statecache import CACHE_DIR

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


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND, "index.html"))


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
    except Exception:
        pass
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
    return {"ok": True, "profile": prof}


# ---- map-first gauge pins ---------------------------------------------------
def _pins_for_bbox(w: float, s: float, e: float, n: float, limit: int = 300):
    cat = gauges.load_catalog((w, s, e, n))
    if len(cat) > limit:                            # keep the map responsive
        cx, cy = (w + e) / 2, (s + n) / 2
        cat = cat.assign(_d=((cat.LAT_GAGE - cy) ** 2 + (cat.LNG_GAGE - cx) ** 2)).nsmallest(limit, "_d")
    return [{
        "id": str(r.STAID).zfill(8), "name": str(r.STANAME),
        "lat": float(r.LAT_GAGE), "lon": float(r.LNG_GAGE),
        "area_km2": float(r.DRAIN_SQKM),
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


MAX_SIM_HOURS = int(os.environ.get("CREST_MAX_SIM_HOURS", "2160"))   # 90 days


@app.post("/api/simulate")
def api_simulate(req: SimRequest):
    warnings = []
    if len(req.gauge_ids) > MAX_SIMS:
        warnings.append(f"This demo simulates at most {MAX_SIMS} gauges at once; "
                        f"running the first {MAX_SIMS} of {len(req.gauge_ids)} selected.")
    t0 = datetime.fromisoformat(req.t_start) if req.t_start else datetime(2025, 7, 3)
    t1 = datetime.fromisoformat(req.t_end) if req.t_end else t0 + timedelta(hours=req.hours)
    if t1 <= t0:                                       # nonsensical window
        t1 = t0 + timedelta(hours=req.hours)
        warnings.append(f"End time was not after the start — using {req.hours} h instead.")
    if (t1 - t0) > timedelta(hours=MAX_SIM_HOURS):     # demo-hardware sanity cap
        t1 = t0 + timedelta(hours=MAX_SIM_HOURS)
        warnings.append(f"Window capped at {MAX_SIM_HOURS // 24} days for this demo "
                        f"(now ends {t1:%Y-%m-%d %H:%M}).")
    opts = {"model": req.model, "hours": req.hours, "snow": req.snow,
            "timestep": req.timestep, "warmup_days": req.warmup_days,
            "overrides": req.overrides}
    job = simjobs.start_job(req.gauge_ids, t0, t1, opts)
    return {"sim_id": job.id, "gauge_ids": job.gauge_ids,
            "t_start": t0.isoformat(), "t_end": t1.isoformat(),
            "warning": " ".join(warnings) or None,
            "max_concurrent": simjobs.MAX_CONCURRENT}


async def _drain(job):
    while True:
        try:
            ev = job.q.get_nowait()
        except queue.Empty:
            if job.done.is_set():
                break
            await asyncio.sleep(0.15)
            continue
        yield {"data": json.dumps(ev)}
        if ev.get("kind") in ("all_done", "cal_done"):
            break


@app.get("/api/stream/{sim_id}")
async def api_stream(sim_id: str):
    job = simjobs.get_job(sim_id)
    if not job:
        return Response(status_code=404)
    return EventSourceResponse(_drain(job))


class CalRequest(BaseModel):
    gauge_id: str
    t_start: str
    t_end: str
    model: str = "auto"
    snow: str = "auto"
    rounds: int = 4
    k: int = 3


@app.post("/api/calibrate")
def api_calibrate(req: CalRequest):
    job = caljobs.start_job(req.gauge_id,
                            datetime.fromisoformat(req.t_start),
                            datetime.fromisoformat(req.t_end),
                            {"model": req.model, "snow": req.snow,
                             "rounds": req.rounds, "k": req.k})
    return {"cal_id": job.id, "gauge_id": req.gauge_id}


@app.get("/api/calstream/{cal_id}")
async def api_calstream(cal_id: str):
    job = caljobs.get_job(cal_id)
    if not job:
        return Response(status_code=404)
    return EventSourceResponse(_drain(job))


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
