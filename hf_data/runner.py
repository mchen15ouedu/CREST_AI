"""Streaming EF5 runner for the live dashboard.

Runs the fork's ef5 non-blocking and streams outputs as they are written:
  - ts.<gauge>.<model>.csv  -> hydrograph rows (fork flushes every timestep)
  - q.<datetime>.<model>.tif -> 2-D streamflow frames (OUTPUT_GRIDS=STREAMFLOW)

`stream_run()` yields event dicts the Gradio app consumes. `MockEF5` writes the
same two output streams so the UI + reader can be tested without the binary.
"""
from __future__ import annotations

import os
import re
import glob
import time
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

TS_TIME_FMT = "%Y-%m-%d %H:%M"

# Watchdog policy: a run that KEEPS PRODUCING OUTPUT is never killed for
# running long. Killed only when it goes silent (no new ts rows / q grids)
# for STALL_TIMEOUT_S, or exceeds the RUN_TIMEOUT_S hard cap.
RUN_TIMEOUT_S = float(os.environ.get("CREST_RUN_TIMEOUT_S", str(10 * 3600)))    # 10 h
STALL_TIMEOUT_S = float(os.environ.get("CREST_STALL_TIMEOUT_S", str(45 * 60)))  # 45 min


# --------------------------------------------------------------------------- #
# output readers
# --------------------------------------------------------------------------- #
# extra ts.csv columns streamed for the click-a-timestep readout (header token
# -> row key); real EF5 appends Temp/SWE when SNOW17 is on
_EXTRA_COLS = (("pet", "pet"), ("sm", "sm"), ("groundwater", "gw"),
               ("fast flow", "fast"), ("slow flow", "slow"), ("base flow", "base"),
               ("temp", "temp"), ("swe", "swe"))


def _col_map(header: list[str]) -> dict[str, int]:
    m: dict[str, int] = {}
    for i, h in enumerate(header):
        hl = h.strip().lower()
        for tok, key in _EXTRA_COLS:
            if hl.startswith(tok) and key not in m:
                m[key] = i
    return m


def _parse_ts_line(header: list[str], line: str, colmap: dict | None = None) -> dict | None:
    parts = line.rstrip("\n").split(",")
    if len(parts) < 4 or parts[0] == "Time":
        return None
    def f(x):
        try:
            return float(x)
        except ValueError:
            return None
    row = {"time": parts[0], "sim_q": f(parts[1]), "obs_q": f(parts[2]), "precip": f(parts[3])}
    for key, i in (colmap or {}).items():
        if i < len(parts):
            v = f(parts[i])
            if v is not None:
                row[key] = v
    return row


class _TsTail:
    """Incrementally read new rows from a ts.csv as it grows."""
    def __init__(self, path: str):
        self.path = path
        self.header: list[str] | None = None
        self.colmap: dict[str, int] = {}
        self.pos = 0

    def read_new(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "rb") as fh:          # binary: exact byte position
            fh.seek(self.pos)
            data = fh.read()
        nl = data.rfind(b"\n")                       # consume complete lines only
        if nl == -1:
            return []
        self.pos += nl + 1
        rows = []
        for raw in data[:nl + 1].decode("utf-8", "replace").splitlines():
            if self.header is None and raw.startswith("Time"):
                self.header = raw.split(",")
                self.colmap = _col_map(self.header)
                continue
            r = _parse_ts_line(self.header or [], raw, self.colmap)
            if r:
                rows.append(r)
        return rows


def _new_q_grids(output_dir: str, model: str, seen: set) -> list[str]:
    found = sorted(glob.glob(os.path.join(output_dir, f"q.*.{model}.tif")),
                   key=lambda p: os.path.getmtime(p))
    fresh = [p for p in found if p not in seen]
    seen.update(fresh)
    return fresh


# --------------------------------------------------------------------------- #
# streaming driver
# --------------------------------------------------------------------------- #
@dataclass
class RunHandle:
    output_dir: str
    ts_path: str
    model: str = "crestphys"
    proc: subprocess.Popen | None = None
    stop_flag: threading.Event = field(default_factory=threading.Event)

    def alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        return not self.stop_flag.is_set()

    def kill(self):
        """Hard-stop a stuck run (watchdog)."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=10)
            except Exception:
                pass
        self.stop_flag.set()


def run_ef5(control_path: str, output_dir: str, gauge_id: str, model: str = "crestphys",
            ef5_bin: str = "./EF5/bin/ef5") -> RunHandle:
    os.makedirs(output_dir, exist_ok=True)
    log = open(os.path.join(output_dir, "ef5_run.log"), "w")
    cmd = [ef5_bin, control_path]
    if os.name != "nt":            # line-buffer stdout so crashes keep the trail
        cmd = ["stdbuf", "-oL", "-eL"] + cmd
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    ts = os.path.join(output_dir, f"ts.{gauge_id}.{model}.csv")
    return RunHandle(output_dir=output_dir, ts_path=ts, model=model, proc=proc)


def stream_run(handle: RunHandle, poll: float = 0.4, timeout: float = RUN_TIMEOUT_S,
               stall_timeout: float = STALL_TIMEOUT_S, cancel=None):
    """Yield {'kind': 'hydro'|'q2d'|'done', ...} as EF5 writes output.

    A run that keeps producing output is left alone no matter how long it
    takes. The process is KILLED only when it goes silent for `stall_timeout`
    (stuck), passes the `timeout` hard cap, or `cancel` (a threading.Event)
    is set — the user stopped or superseded the job."""
    tail = _TsTail(handle.ts_path)
    seen: set = set()
    t0 = time.time()
    last_progress = t0                       # last time ANY output appeared
    while True:
        if cancel is not None and cancel.is_set():
            handle.kill()
            yield {"kind": "done", "returncode": -9, "cancelled": True}
            return
        rows = tail.read_new()
        if rows:
            yield {"kind": "hydro", "rows": rows}
        fresh = _new_q_grids(handle.output_dir, handle.model, seen)
        for p in fresh:
            yield {"kind": "q2d", "path": p}
        if rows or fresh:
            last_progress = time.time()
        done = not handle.alive()
        if done:
            # final drain
            rows = tail.read_new()
            if rows:
                yield {"kind": "hydro", "rows": rows}
            for p in _new_q_grids(handle.output_dir, handle.model, seen):
                yield {"kind": "q2d", "path": p}
            rc = handle.proc.returncode if handle.proc else 0
            yield {"kind": "done", "returncode": rc}
            return
        now = time.time()
        if now - last_progress > stall_timeout:
            handle.kill()                    # silent too long -> stuck
            yield {"kind": "done", "returncode": -1,
                   "error": f"no model output for {stall_timeout / 60:.0f} min — "
                            "killed (stall watchdog)"}
            return
        if now - t0 > timeout:
            handle.kill()                    # absolute cap
            yield {"kind": "done", "returncode": -1,
                   "error": f"killed after {timeout / 3600:.1f} h (run-time cap)"}
            return
        time.sleep(poll)


# --------------------------------------------------------------------------- #
# mock EF5 (for local testing without the binary)
# --------------------------------------------------------------------------- #
class MockEF5:
    """Writes ts.csv rows + q.*.tif grids incrementally, like a live EF5 run."""
    HEADER = ("Time,Discharge(m^3 s^-1),Observed(m^3 s^-1),Precip(mm h^-1),PET(mm h^-1),"
              "SM(%),Groundwater (mm),Fast Flow(mm*1000),Slow Flow(mm*1000),Base Flow(mm*1000)")

    def __init__(self, output_dir, gauge_id="01011000", model="crestphys",
                 bounds=(-69.83, 46.32, -68.33, 47.82), n_steps=48, t0=None,
                 dt_hours=1, delay=0.15, write_grids=True, facc_path=None):
        self.output_dir = output_dir
        self.gauge_id, self.model = gauge_id, model
        self.bounds, self.n_steps = bounds, n_steps
        self.t0 = t0 or datetime(2025, 7, 3, 0, 0)
        self.dt = timedelta(hours=dt_hours)
        self.delay, self.write_grids = delay, write_grids
        self.facc_path = facc_path            # clipped flow-acc -> realistic network
        self.ts_path = os.path.join(output_dir, f"ts.{gauge_id}.{model}.csv")

    def handle(self) -> RunHandle:
        return RunHandle(output_dir=self.output_dir, ts_path=self.ts_path, model=self.model)

    def start(self) -> RunHandle:
        os.makedirs(self.output_dir, exist_ok=True)
        h = self.handle()
        threading.Thread(target=self._run, args=(h,), daemon=True).start()
        return h

    def _run(self, h: RunHandle):
        import math
        import numpy as np
        try:
            with open(self.ts_path, "w") as ts:
                ts.write(self.HEADER + "\n"); ts.flush()
                for k in range(self.n_steps):
                    t = self.t0 + k * self.dt
                    # a plausible flood pulse
                    q = 5 + 60 * math.exp(-((k - self.n_steps * 0.4) ** 2) / (2 * (self.n_steps * 0.12) ** 2))
                    obs = q * (0.9 + 0.05 * math.sin(k / 3.0))
                    p = max(0.0, 8 * math.exp(-((k - self.n_steps * 0.3) ** 2) / (2 * 3.0 ** 2)))
                    sm = 55 + 30 * (q - 5) / 60.0            # wets up with the pulse
                    gw = 7.0 + 0.02 * k
                    ts.write(f"{t:%Y-%m-%d %H:%M},{q:.2f},{obs:.2f},{p:.2f},0.10,"
                             f"{sm:.2f},{gw:.2f},0.0000,0.0000,0.6960\n")
                    ts.flush()
                    if self.write_grids:
                        self._grid(t, q, np)
                    time.sleep(self.delay)
        finally:
            h.stop_flag.set()          # signal completion to stream_run()

    def _grid(self, t: datetime, qpeak: float, np):
        """Write a per-timestep discharge grid. If a clipped flow-accumulation
        raster is available, put discharge on the actual stream network
        (discharge ~ upstream area), like a real EF5 q grid; else a fallback."""
        import rasterio
        from rasterio.transform import from_bounds as tr_from_bounds
        name = os.path.join(self.output_dir, f"q.{t:%Y%m%d%H%M}.{self.model}.tif")
        if self.facc_path and os.path.exists(self.facc_path):
            with rasterio.open(self.facc_path) as ds:
                scale = max(1, max(ds.width, ds.height) // 500)     # decimate
                oh, ow = max(1, ds.height // scale), max(1, ds.width // scale)
                facc = ds.read(1, out_shape=(oh, ow)).astype("float32")
                b, nod = ds.bounds, ds.nodata
            facc = np.where((facc == nod) | (facc <= 0), np.nan, facc)
            fmax = np.nanmax(facc)
            fmax = fmax if np.isfinite(fmax) and fmax > 0 else 1.0
            thr = np.nanpercentile(facc, 96)                        # top ~4% = channels
            onnet = np.isfinite(facc) & (facc >= thr)
            disch = qpeak * (facc / fmax)                          # ~ drainage area
            grid = np.where(onnet, disch, -9999.0).astype("float32")
            transform = tr_from_bounds(b.left, b.bottom, b.right, b.top, grid.shape[1], grid.shape[0])
        else:
            W, S, E, N = self.bounds
            nx = ny = 60
            yy, xx = np.mgrid[0:ny, 0:nx]
            chan = np.exp(-((xx - nx / 2) ** 2) / (2 * 3.0 ** 2))
            grid = np.where(chan * (yy / ny) > 0.05, qpeak * chan * (yy / ny), -9999.0).astype("float32")
            transform = tr_from_bounds(W, S, E, N, nx, ny)
        with rasterio.open(name, "w", driver="GTiff", height=grid.shape[0], width=grid.shape[1],
                           count=1, dtype="float32", crs="EPSG:4326", transform=transform,
                           nodata=-9999.0) as dst:
            dst.write(grid, 1)


if __name__ == "__main__":
    import sys, tempfile
    out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    print("mock run ->", out)
    mock = MockEF5(out, n_steps=12, delay=0.1)
    h = mock.start()
    n_hydro = n_q = 0
    for ev in stream_run(h, poll=0.1):
        if ev["kind"] == "hydro":
            n_hydro += len(ev["rows"])
            last = ev["rows"][-1]
            print(f"  hydro row {last['time']} simQ={last['sim_q']:.1f} obsQ={last['obs_q']:.1f} P={last['precip']:.1f}")
        elif ev["kind"] == "q2d":
            n_q += 1
            print(f"  q2d frame -> {os.path.basename(ev['path'])}")
        elif ev["kind"] == "done":
            print(f"  DONE rc={ev['returncode']} | {n_hydro} hydro rows, {n_q} q-grids")
