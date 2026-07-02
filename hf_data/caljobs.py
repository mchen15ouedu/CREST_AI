"""Background AI-calibration jobs (one gauge each) with an SSE event queue."""
from __future__ import annotations

import os
import queue
import threading
import uuid
from datetime import datetime

from hf_data.calibrate import run_calibration

USE_MOCK = os.environ.get("CREST_DEMO_MOCK", "1") == "1"

_JOBS: dict[str, "CalJob"] = {}


class CalJob:
    def __init__(self, gauge_id: str, t_start: datetime, t_end: datetime, opts: dict):
        self.id = uuid.uuid4().hex[:8]
        self.gauge_id = gauge_id
        self.t_start, self.t_end = t_start, t_end
        self.opts = opts or {}
        self.q: queue.Queue = queue.Queue()
        self.done = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            for kind, payload in run_calibration(
                    self.gauge_id, self.t_start, self.t_end,
                    model=self.opts.get("model", "auto"),
                    snow=self.opts.get("snow", "auto"),
                    use_mock=USE_MOCK,
                    rounds=int(self.opts.get("rounds", 4)),
                    k=int(self.opts.get("k", 3))):
                if kind == "status":
                    self.q.put({"kind": "cal_status", "gauge_id": self.gauge_id, "msg": payload})
                elif kind == "round":
                    self.q.put({"kind": "cal_round", "gauge_id": self.gauge_id, **payload})
                elif kind == "hydro":
                    self.q.put({"kind": "cal_hydro", "gauge_id": self.gauge_id,
                                "rows": payload["rows"]})
                elif kind == "done":
                    self.q.put({"kind": "cal_done", "gauge_id": self.gauge_id, **payload})
        except Exception as e:
            self.q.put({"kind": "cal_done", "gauge_id": self.gauge_id, "error": str(e)})
        finally:
            self.done.set()


def start_job(gauge_id: str, t_start: datetime, t_end: datetime, opts: dict) -> CalJob:
    job = CalJob(gauge_id, t_start, t_end, opts)
    _JOBS[job.id] = job
    job.start()
    return job


def get_job(cal_id: str) -> "CalJob | None":
    return _JOBS.get(cal_id)
