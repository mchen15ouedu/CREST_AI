"""Parallel simulation manager for the map dashboard.

A SimJob runs the selected gauges through hf_data.pipeline.run_gauge in a worker
pool (MAX_CONCURRENT, sized for free cpu-basic ~4), queuing the rest up to the
10-selection cap. Per-gauge events (status / hydrograph / 2-D overlay / done) are
pushed to a thread-safe queue that the SSE endpoint drains. 2-D frames are
rendered to PNG overlays served by the backend.
"""
from __future__ import annotations

import os
import queue
import threading
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
        self.q: queue.Queue = queue.Queue()
        self.overlays: dict[str, tuple] = {}     # gid -> (png_bytes, bounds, frame)  (live latest)
        self.q_paths: dict[str, list] = {}       # gid -> [q.*.tif paths] (for the animation)
        self.frames: dict[str, list] = {}        # gid -> [(png_bytes, bounds, time_label)]
        self.hydro: dict[str, list] = {}         # gid -> accumulated rows
        self.meta: dict[str, dict] = {}          # gid -> {id,name,area,lat,lon,model}
        self.done = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
            list(ex.map(self._run_one, self.gauge_ids))
        self.q.put({"kind": "all_done"})
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
                    warmup_days=int(self.opts.get("warmup_days", 90))):
                if kind == "meta":
                    self.meta[gid] = payload
                elif kind == "status":
                    self.q.put({"kind": "status", "gauge_id": gid, "msg": payload})
                elif kind == "hydro":
                    self.hydro.setdefault(gid, []).extend(payload["rows"])
                    self.q.put({"kind": "hydro", "gauge_id": gid, "rows": payload["rows"]})
                elif kind == "q2d":
                    self.q_paths.setdefault(gid, []).append(payload["path"])
                    try:
                        png, bounds, _ = viz.q2d_png(payload["path"])
                        frame = self.overlays.get(gid, (None, None, 0))[2] + 1
                        self.overlays[gid] = (png, bounds, frame)
                        self.q.put({"kind": "q2d", "gauge_id": gid, "bounds": bounds, "frame": frame})
                    except Exception:
                        pass
                elif kind == "done":
                    self._build_timeline(gid)
                    self.q.put({"kind": "gauge_done", "gauge_id": gid,
                                "returncode": payload.get("returncode"),
                                "n": len(self.hydro.get(gid, []))})
                    self._build_result(gid)
        except Exception as e:
            self.q.put({"kind": "status", "gauge_id": gid, "msg": f"⚠️ {e}"})
            self.q.put({"kind": "gauge_done", "gauge_id": gid, "returncode": -1})

    def _build_timeline(self, gid):
        """After a gauge finishes, pre-render all frames with a fixed color scale
        and emit a 'timeline' event so the client can animate/scrub."""
        paths = self.q_paths.get(gid, [])
        if not paths:
            return
        try:
            frames, vmax = viz.q2d_frames(paths)
            self.frames[gid] = frames
            self.q.put({"kind": "timeline", "gauge_id": gid, "n": len(frames),
                        "times": [f[2] for f in frames],
                        "bounds": frames[0][1] if frames else None,
                        "vmax": vmax})
        except Exception:
            pass

    def _build_result(self, gid):
        """Compute skill metrics + the ARW report, emit as a 'result' event."""
        try:
            rows = self.hydro.get(gid, [])
            metrics = analysis.compute_metrics(rows)
            report = analysis.build_report(self.meta.get(gid), metrics,
                                           self.t_start, self.t_end)
            self.q.put({"kind": "result", "gauge_id": gid,
                        "meta": self.meta.get(gid), "metrics": metrics, "report": report})
        except Exception as e:
            self.q.put({"kind": "result", "gauge_id": gid, "metrics": {},
                        "report": f"(report unavailable: {e})"})

    def overlay_png(self, gid):
        o = self.overlays.get(gid)
        return o[0] if o else None

    def frame_png(self, gid, idx):
        fr = self.frames.get(gid)
        if fr and 0 <= idx < len(fr):
            return fr[idx][0]
        return None


def start_job(gauge_ids, t_start: datetime, t_end: datetime, opts) -> SimJob:
    job = SimJob(gauge_ids, t_start, t_end, opts)
    _JOBS[job.id] = job
    job.start()
    return job


def get_job(sim_id) -> "SimJob | None":
    return _JOBS.get(sim_id)
