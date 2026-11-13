"""The one typed result a gauge run ends with (pipeline.run_gauge).

A run streams ("status" | "meta" | "params" | "hydro" | "q2d", payload)
progress events and then exactly one ("done", GaugeRunResult). Callers branch
on `outcome`, never on EF5 return codes, and read the record / state
locations, keys and provenance from the result instead of globbing the cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Outcome(str, Enum):
    OK = "ok"                    # simulated (rows may be partly served from cache)
    CACHED = "cached"            # served entirely from the row cache, no EF5 run
    NO_BASIN = "no_basin"        # no basin outline - deliberately not simulated
    NO_STATE = "no_state"        # nowcast forecast without a usable saved state
    CUT_FAILED = "cut_failed"    # the run's fixed speed cut set could not be rebuilt
    CANCELLED = "cancelled"      # user stop / superseded
    FAILED = "failed"            # EF5 or setup failure; `reason` says why


@dataclass(frozen=True)
class GaugeRunResult:
    outcome: Outcome
    gauge_id: str
    rows: list = field(default_factory=list)     # the window's rows (cached + new)
    reason: str | None = None
    cache_model: str | None = None   # rows/frames key  (record <gauge>_<cache_model>)
    state_model: str | None = None   # EF5 state-grid key
    record_path: str | None = None   # the row record this run wrote/served (hourly runs)
    state_dir: str | None = None     # shared state folder of state_model
    provenance: dict | None = None   # {"extent", "cut_ids", "lakes"}

    @property
    def ok(self) -> bool:
        return self.outcome in (Outcome.OK, Outcome.CACHED)

    @property
    def record_key(self) -> str | None:
        return f"{self.gauge_id}_{self.cache_model}" if self.cache_model else None

    @property
    def state_key(self) -> str | None:
        return f"{self.gauge_id}_{self.state_model}" if self.state_model else None
