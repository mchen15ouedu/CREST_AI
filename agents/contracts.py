"""Shared data contracts for the CREST_demo agent pipeline.

Every agent consumes upstream artifacts and produces one typed artifact. The
pipeline threads them through a single `SessionState`:

    ACP  -> EventContext        (parse free-form query -> metadata + anchor)
    ADR  -> DatasetManifest     (retrieve DEM/climate parquet; cache 90 days)
    AP   -> BasinPerception     (vision-LLM reads maps -> morphology, outlets)
    AOS  -> OutletSelection     (USGS gauges near basin -> recommend; USER GATE)
    API  -> ParameterSet        (RAG + Chen 2023 analogs + priors -> params)
    AO   -> RunSpec, RunResult  (emit control file; run; metrics + FI/F-IDF)
    ARW  -> Report              (compile viz + diagnostics -> structured report)

Parameter names (CREST wm/b/im/ke/fc/iwu, kinematic-wave under/leaki/th/isu/
alpha/beta/alpha0, lake klake/b) mirror EF5's Models.tbl exactly so the AO
control-file renderer is a direct mapping.

Pydantic v2.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Geo / time primitives
# --------------------------------------------------------------------------- #
class Coordinate(BaseModel):
    lat: float
    lon: float


class BBox(BaseModel):
    """Axis-aligned extent in EPSG:4326 (degrees)."""

    west: float
    south: float
    east: float
    north: float


class TimeWindow(BaseModel):
    start: datetime
    end: datetime
    tz: str = "UTC"


class AdminUnit(BaseModel):
    county: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None


class SourceRef(BaseModel):
    """Provenance for a claim (news verification, RAG citation, analog study)."""

    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    kind: str = "web"  # web | news | literature | manual | dataset


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    flash_flood = "flash_flood"
    riverine_flood = "riverine_flood"
    urban_flood = "urban_flood"
    dam_break = "dam_break"
    unknown = "unknown"


class DataKind(str, Enum):
    dem = "dem"           # elevation
    ddm = "ddm"           # drainage direction (flow direction)
    fam = "fam"           # flow accumulation
    precip = "precip"     # precipitation forcing
    pet = "pet"           # potential evapotranspiration forcing
    temp = "temp"         # temperature (snow module, optional)
    gauge_obs = "gauge_obs"   # observed discharge timeseries
    param_grid = "param_grid"  # spatially distributed parameter grid


class PipelineStage(str, Enum):
    init = "init"
    parsed = "parsed"          # ACP done
    retrieved = "retrieved"    # ADR done
    perceived = "perceived"    # AP done
    outlet_pending = "outlet_pending"   # AOS awaiting user confirmation
    outlet_set = "outlet_set"           # AOS confirmed
    parameterized = "parameterized"     # API done
    running = "running"        # AO executing
    complete = "complete"      # AO + ARW done
    failed = "failed"


# --------------------------------------------------------------------------- #
# ACP  -> EventContext
# --------------------------------------------------------------------------- #
class EventContext(BaseModel):
    raw_query: str
    detected_language: str = "en"
    event_type: EventType = EventType.unknown
    location_name: str                    # normalized, e.g. "Kerr County, TX"
    admin: AdminUnit = Field(default_factory=AdminUnit)
    anchor: Coordinate                    # LLM-guided anchor for basin retrieval
    bbox: Optional[BBox] = None           # coarse extent, if inferable
    time_window: TimeWindow
    news_refs: list[SourceRef] = Field(default_factory=list)
    confidence: float = 0.0               # 0..1
    clarifications_needed: list[str] = Field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------- #
# ADR  -> DatasetManifest
# --------------------------------------------------------------------------- #
class DataAsset(BaseModel):
    kind: DataKind
    source_uri: str                       # hf://datasets/vincewin/CREST_data/...
    local_path: str
    fmt: str = "parquet"                  # parquet | tif | bif | asc | csv
    spatial_extent: Optional[BBox] = None
    temporal_extent: Optional[TimeWindow] = None
    resolution_deg: Optional[float] = None
    crs: str = "EPSG:4326"
    variables: list[str] = Field(default_factory=list)
    n_bytes: Optional[int] = None


class CacheEntry(BaseModel):
    """90-day cache with reuse. Evicted when expired AND ref_count == 0."""

    key: str
    stored_at: datetime
    expires_at: datetime                  # stored_at + 90 days
    last_used: datetime
    reused: bool = False
    ref_count: int = 0                    # active simulations needing this asset


class DatasetManifest(BaseModel):
    request_bbox: BBox
    request_window: TimeWindow
    assets: list[DataAsset] = Field(default_factory=list)
    cache: list[CacheEntry] = Field(default_factory=list)
    preview_maps: list[str] = Field(default_factory=list)  # rendered DEM/FAM maps for AP
    status: str = "complete"              # complete | partial | failed
    missing: list[DataKind] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# AP  -> BasinPerception
# --------------------------------------------------------------------------- #
class OutletCandidate(BaseModel):
    coord: Coordinate
    flow_accum: Optional[float] = None
    rank: int = 0
    rationale: str = ""


class BasinMorphology(BaseModel):
    area_km2: Optional[float] = None
    drainage_density: Optional[float] = None
    main_channel_length_km: Optional[float] = None
    relief_m: Optional[float] = None
    mean_slope: Optional[float] = None
    shape_descriptor: Optional[str] = None


class BasinPerception(BaseModel):
    map_refs: list[str] = Field(default_factory=list)   # maps read by the vision LLM
    morphology: BasinMorphology = Field(default_factory=BasinMorphology)
    drainage_structure: str = ""
    candidate_outlets: list[OutletCandidate] = Field(default_factory=list)
    vision_model: str = ""
    confidence: float = 0.0


# --------------------------------------------------------------------------- #
# AOS  -> OutletSelection   (USER-CONFIRMATION GATE)
# --------------------------------------------------------------------------- #
class USGSGauge(BaseModel):
    site_no: str
    name: str
    coord: Coordinate
    drainage_area_km2: Optional[float] = None
    inside_basin: bool = False
    distance_to_outlet_km: Optional[float] = None


class OutletSelection(BaseModel):
    basin_polygon_geojson: dict[str, Any] = Field(default_factory=dict)  # buffered basin
    candidate_gauges: list[USGSGauge] = Field(default_factory=list)
    recommended: Optional[USGSGauge] = None
    rationale: str = ""
    user_confirmed: bool = False
    confirmed_gauge: Optional[USGSGauge] = None


# --------------------------------------------------------------------------- #
# API  -> ParameterSet   (names mirror EF5 Models.tbl)
# --------------------------------------------------------------------------- #
class ParamValue(BaseModel):
    value: float
    min: Optional[float] = None           # calibration lower bound
    max: Optional[float] = None           # calibration upper bound
    source: str = ""                      # RAG citation / analog / prior
    grid_path: Optional[str] = None       # set if spatially distributed


class CrestParams(BaseModel):
    wm: ParamValue        # max soil water capacity (mm)
    b: ParamValue         # infiltration curve exponent
    im: ParamValue        # impervious area ratio
    ke: ParamValue        # PET adjustment factor
    fc: ParamValue        # soil saturated hydraulic conductivity
    iwu: ParamValue       # initial soil water (fraction)


class KinematicParams(BaseModel):
    under: ParamValue
    leaki: ParamValue
    th: ParamValue
    isu: ParamValue
    alpha: ParamValue
    beta: ParamValue
    alpha0: ParamValue


class LakeParams(BaseModel):
    klake: ParamValue     # linear-reservoir retention constant (h)
    b: ParamValue


class ParameterSet(BaseModel):
    model: str = "crest"
    routing: str = "kw"                   # kw (kinematic) | lr (linear)
    water_balance: CrestParams
    kinematic: KinematicParams
    lake: Optional[LakeParams] = None
    citations: list[SourceRef] = Field(default_factory=list)  # RAG sources
    analog_study: str = "Chen et al. 2023"
    soil_priors: dict[str, Any] = Field(default_factory=dict)
    calibratable: bool = True


# --------------------------------------------------------------------------- #
# AO  -> RunSpec + RunResult + streaming events
# --------------------------------------------------------------------------- #
class RunSpec(BaseModel):
    control_file_text: str                # fully rendered EF5 control file
    control_file_path: str
    task_name: str
    model: str = "crest"
    routing: str = "kw"
    time_step: str = "1H"
    time_window: TimeWindow
    output_dir: str
    lake_module: bool = False


class SkillMetrics(BaseModel):
    nse: Optional[float] = None           # Nash-Sutcliffe efficiency
    kge: Optional[float] = None           # Kling-Gupta efficiency
    cc: Optional[float] = None            # correlation coefficient
    bias_pct: Optional[float] = None
    rmse: Optional[float] = None


class HydroAssessment(BaseModel):
    peak_q: Optional[float] = None
    peak_time: Optional[datetime] = None
    return_period_yr: Optional[float] = None
    flashiness_intensity: Optional[float] = None   # FI
    f_idf: dict[str, Any] = Field(default_factory=dict)  # F-IDF (Li et al. 2023)


class RunResult(BaseModel):
    ts_csv_paths: list[str] = Field(default_factory=list)   # ts.<gauge>.<model>.csv
    grid_outputs: list[str] = Field(default_factory=list)   # GeoTIFFs
    metrics: SkillMetrics = Field(default_factory=SkillMetrics)
    assessment: HydroAssessment = Field(default_factory=HydroAssessment)
    status: str = "complete"              # complete | failed


# ---- streaming (live dashboard while CREST runs) -------------------------- #
class HydrographPoint(BaseModel):
    """One row of the CREST ts.*.csv, streamed as it is written."""

    time: datetime
    sim_q: float                          # Discharge (m^3 s^-1)
    obs_q: Optional[float] = None         # Observed (m^3 s^-1)
    precip: Optional[float] = None        # Precip (mm h^-1)
    lake_vol: Optional[float] = None      # Lake_Vol (m^3), lake module only


class StreamflowFrame(BaseModel):
    """2-D streamflow snapshot at one timestep."""

    time: datetime
    grid_path: str                        # raster written for this timestep


class RunProgress(BaseModel):
    """Event streamed to the dashboard during a run."""

    kind: str                             # log | hydrograph | streamflow_2d | metric | done
    message: Optional[str] = None
    point: Optional[HydrographPoint] = None
    frame: Optional[StreamflowFrame] = None
    pct: Optional[float] = None


# --------------------------------------------------------------------------- #
# ARW  -> Report
# --------------------------------------------------------------------------- #
class ReportSection(BaseModel):
    title: str
    body_markdown: str = ""
    figures: list[str] = Field(default_factory=list)


class Report(BaseModel):
    title: str
    event_summary: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)
    generated_at: Optional[datetime] = None


# --------------------------------------------------------------------------- #
# Top-level threaded state
# --------------------------------------------------------------------------- #
class SessionState(BaseModel):
    session_id: str
    stage: PipelineStage = PipelineStage.init
    event: Optional[EventContext] = None
    manifest: Optional[DatasetManifest] = None
    perception: Optional[BasinPerception] = None
    outlet: Optional[OutletSelection] = None
    params: Optional[ParameterSet] = None
    run_spec: Optional[RunSpec] = None
    result: Optional[RunResult] = None
    report: Optional[Report] = None
    errors: list[str] = Field(default_factory=list)
