"""Parallel simulation manager for the map dashboard.

A SimJob runs the selected gauges through hf_data.pipeline.run_gauge in a worker
pool (MAX_CONCURRENT, sized for free cpu-basic ~4), queuing the rest up to the
10-selection cap. Per-gauge events (status / hydrograph / 2-D overlay / done) are
pushed to a thread-safe queue that the SSE endpoint drains. 2-D frames are
rendered to PNG overlays served by the backend.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from hf_data import analysis, viz
from hf_data.pipeline import run_gauge

MAX_CONCURRENT = int(os.environ.get("CREST_MAX_CONCURRENT", "4"))
MAX_SIMS = int(os.environ.get("CREST_MAX_SIMS", "10"))
USE_MOCK = os.environ.get("CREST_DEMO_MOCK", "1") == "1"

_JOBS: dict[str, "SimJob"] = {}


class SimJob:
    def __init__(self, gauge_ids, t_start, t_end, opts):
        self.id = uuid.uuid4().hex[:8]
        self.gauge_ids = gauge_ids[:MAX_SIMS]
        self.t_start, self.t_end = t_start, t_end
        self.opts = opts or {}
        self.created = time.time()
        # append-only event log — SSE clients replay from any cursor, so a
        # closed/reopened browser reattaches without losing the run
        self.events: list[dict] = []
        self.overlays: dict[str, tuple] = {}     # gid -> (png_bytes, bounds, frame)  (live latest)
        self.q_paths: dict[str, list] = {}       # gid -> [q.*.tif paths] (for the animation)
        self.frames: dict[str, list] = {}        # gid -> [(png_bytes, bounds, time_label)]
        self.hydro: dict[str, list] = {}         # gid -> accumulated rows
        self.meta: dict[str, dict] = {}          # gid -> {id,name,area,lat,lon,model}
        self.params: dict[str, dict] = {}        # gid -> {wb,kw,model,source}
        self.done = threading.Event()
        self.cancel = threading.Event()          # user stop / superseded by a new run
        self._q2d_err: set = set()               # gauges with a reported render error
        self._q2d_min: dict = {}                 # gid -> running per-cell min (live baseline)

    def _emit(self, ev: dict):
        self.events.append(ev)               # append-only; readers poll by index

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
            list(ex.map(self._run_one, self.gauge_ids))
        self._emit({"kind": "all_done"})
        self.done.set()

    def _run_one(self, gid):
        try:
            for kind, payload in run_gauge(
                    gid, self.t_start, self.t_end,
                    model=self.opts.get("model", "auto"), use_mock=USE_MOCK,
                    hours=int(self.opts.get("hours", 48)),
                    overrides=self.opts.get("overrides"),
                    snow=self.opts.get("snow", "auto"),
                    timestep=self.opts.get("timestep", "1h"),
                    warmup_days=int(self.opts.get("warmup_days", 90)),
                    scheme=self.opts.get("scheme", "full"),
                    cancel=self.cancel):
                if kind == "meta":
                    self.meta[gid] = payload
                elif kind == "params":
                    self.params[gid] = payload      # effective wb/kw for paramstore
                elif kind == "status":
                    self._emit({"kind": "status", "gauge_id": gid, "msg": payload})
                elif kind == "hydro":
                    self.hydro.setdefault(gid, []).extend(payload["rows"])
                    self._emit({"kind": "hydro", "gauge_id": gid, "rows": payload["rows"]})
                elif kind == "q2d":
                    self.q_paths.setdefault(gid, []).append(payload["path"])
                    try:
                        png, bounds, self._q2d_min[gid] = viz.q2d_live(
                            payload["path"], self._q2d_min.get(gid))
                        frame = self.overlays.get(gid, (None, None, 0))[2] + 1
                        self.overlays[gid] = (png, bounds, frame)
                        self._emit({"kind": "q2d", "gauge_id": gid, "bounds": bounds, "frame": frame})
                    except Exception as e:
                        if gid not in self._q2d_err:      # report once, don't spam
                            self._q2d_err.add(gid)
                            self._emit({"kind": "status", "gauge_id": gid,
                                        "msg": f"⚠️ 2-D frame render failed: {e}"})
                elif kind == "done":
                    rc = payload.get("returncode")
                    if payload.get("cancelled"):
                        # user stop / superseded — not an error, nothing to render
                        self._emit({"kind": "gauge_done", "gauge_id": gid,
                                    "returncode": -9,
                                    "n": len(self.hydro.get(gid, []))})
                        continue
                    if payload.get("error") or rc not in (0, None):
                        from hf_data import crashlog
                        crashlog.capture(f"ef5:{gid}", message=payload.get("error")
                                         or f"EF5 exited rc={rc}",
                                         sim_id=self.id, returncode=rc)
                    if payload.get("cached"):
                        self._cached_timeline(gid)        # replay frames from disk
                    else:
                        self._build_timeline(gid)
                    self._emit({"kind": "gauge_done", "gauge_id": gid,
                                "returncode": rc,
                                "n": len(self.hydro.get(gid, []))})
                    self._build_result(gid)
        except Exception as e:
            from hf_data import crashlog
            crashlog.capture(f"sim:{gid}", e, sim_id=self.id)
            self._emit({"kind": "status", "gauge_id": gid, "msg": f"⚠️ {e}"})
            self._emit({"kind": "gauge_done", "gauge_id": gid, "returncode": -1})

    def _build_timeline(self, gid):
        """After a gauge finishes, pre-render all frames with a fixed color scale
        and emit a 'timeline' event so the client can animate/scrub."""
        paths = self.q_paths.get(gid, [])
        if not paths:
            return
        try:
            # anchor the color scale to the gauge's real (USGS-observed) baseflow
            baseflow = viz.obs_baseflow(self.hydro.get(gid, []))
            frames, vmax = viz.q2d_frames(paths, baseflow_cms=baseflow)
            self.frames[gid] = frames
            self._emit({"kind": "timeline", "gauge_id": gid, "n": len(frames),
                        "times": [f[2] for f in frames],
                        "bounds": frames[0][1] if frames else None,
                        "vmax": vmax})
            # persist for future cache-hit runs (only when the frames cover the
            # whole requested window — a tail-only extension must not masquerade
            # as the full animation). Key = scheme-aware cache model, so speed-
            # and full-run animations never mix.
            model = ((self.params.get(gid) or {}).get("cache_model")
                     or (self.meta.get(gid) or {}).get("model", "crestphys"))
            expected = (self.t_end - self.t_start).total_seconds() / 3600 + 1
            if len(frames) >= 0.9 * expected:
                viz.save_frames_cache(gid, model, self.t_start, self.t_end, frames, vmax)
        except Exception as e:
            self._emit({"kind": "status", "gauge_id": gid,
                        "msg": f"⚠️ animation build failed: {e}"})

    def _cached_timeline(self, gid):
        """Cache-served run: replay the previously rendered frames from disk."""
        model = ((self.params.get(gid) or {}).get("cache_model")
                 or (self.meta.get(gid) or {}).get("model", "crestphys"))
        got = viz.load_frames_cache(gid, model, self.t_start, self.t_end)
        if not got:
            return
        frames, vmax = got
        self.frames[gid] = frames
        if frames:                                    # latest frame as live overlay
            self.overlays[gid] = (frames[-1][0], frames[-1][1], len(frames))
            self._emit({"kind": "q2d", "gauge_id": gid,
                        "bounds": frames[-1][1], "frame": len(frames)})
        self._emit({"kind": "timeline", "gauge_id": gid, "n": len(frames),
                    "times": [f[2] for f in frames],
                    "bounds": frames[0][1] if frames else None, "vmax": vmax})

    def _build_result(self, gid):
        """Compute skill metrics + the ARW report, emit as a 'result' event."""
        try:
            rows = self.hydro.get(gid, [])
            metrics = analysis.compute_metrics(rows)
            # any completed run that beats the stored NSE becomes the new best
            # parameter set for this basin (manual overrides included)
            p = self.params.get(gid)
            if p and metrics.get("nsce") is not None and not USE_MOCK:
                try:
                    from hf_data import paramstore
                    paramstore.maybe_save(gid, p["model"], p["wb"], p["kw"],
                                          metrics["nsce"], source=p.get("source", "run"),
                                          window=[str(self.t_start), str(self.t_end)])
                except Exception:
                    pass
            report = analysis.build_report(self.meta.get(gid), metrics,
                                           self.t_start, self.t_end)
            self._emit({"kind": "result", "gauge_id": gid,
                        "meta": self.meta.get(gid), "metrics": metrics, "report": report})
        except Exception as e:
            self._emit({"kind": "result", "gauge_id": gid, "metrics": {},
                        "report": f"(report unavailable: {e})"})

    def overlay_png(self, gid):
        o = self.overlays.get(gid)
        return o[0] if o else None

    def frame_png(self, gid, idx):
        fr = self.frames.get(gid)
        if fr and 0 <= idx < len(fr):
            return fr[idx][0]
        return None


MAX_KEPT_JOBS = 30            # finished jobs kept in RAM for reattach/replay


def start_job(gauge_ids, t_start: datetime, t_end: datetime, opts) -> SimJob:
    job = SimJob(gauge_ids, t_start, t_end, opts)
    _JOBS[job.id] = job
    if len(_JOBS) > MAX_KEPT_JOBS:            # prune the oldest FINISHED jobs
        for jid in sorted(_JOBS, key=lambda j: _JOBS[j].created):
            if len(_JOBS) <= MAX_KEPT_JOBS:
                break
            if _JOBS[jid].done.is_set():
                del _JOBS[jid]
    job.start()
    return job


def get_job(sim_id) -> "SimJob | None":
    return _JOBS.get(sim_id)
