"""Fork+PQF control-file generator for the AQUAH->CREST_demo integration.

Adapts AQUAH's generate_control_file() to the mchen15ouedu/EF5 fork:
  - forcing blocks are TYPE=PQF (native parquet reader; configure --with-arrow)
  - OUTPUT_GRIDS=STREAMFLOW so EF5 writes per-timestep 2-D Q rasters (live map)
  - [Gauge] LON/LAT kept (EF5 locates the cell from the clipped DEM/DDM/FAM)
  - optional <param>_grid= lines (grid x scalar multiplier) from hf_data.params
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

CREST_KEYS = ("wm", "b", "im", "ke", "fc", "iwu")
KW_KEYS = ("under", "leaki", "th", "isu", "alpha", "beta", "alpha0")


@dataclass
class Gauge:
    id: str
    lon: float
    lat: float
    area: float


@dataclass
class ControlSpec:
    control_path: str
    time_begin: datetime
    time_end: datetime
    timestep: str
    basic_dir: str
    precip_dir: str
    pet_dir: str
    output_dir: str
    gauges: list[Gauge]
    crest: dict = field(default_factory=lambda: dict(wm=100.0, b=1.0, im=0.01, ke=1.0, fc=20.0, iwu=25.0))
    kw: dict = field(default_factory=lambda: dict(under=1.0, leaki=0.1, th=10.0, isu=0.0, alpha=1.0, beta=0.6, alpha0=0.8))
    param_grids: dict | None = None                 # {wm_grid: path, ...}
    usgs_dir: str = ""
    precip_name: str = "mrms_YYYYMMDDHH.pqf"
    precip_freq: str = "1h"
    precip_unit: str = "mm/h"
    pet_name: str = "etYYYYMMDD.pqf"
    pet_freq: str = "d"
    pet_unit: str = "mm/100d"
    model: str = "CREST"
    routing: str = "KW"
    output_grids: bool = True                        # 2-D streamflow for live map
    proj: str = "geographic"
    esriddm: bool = True
    selffam: bool = True
    # --- warm-up + state (task #6) ---
    state_dir: str | None = None                     # STATES= dir (load at start / save)
    warmup_start: datetime | None = None             # if set -> [Task warmup] before Simu
    save_state_end: bool = True                      # Simu saves state at time_end
    # --- snow / SNOW17 (task #7) ---
    snow_on: bool = False
    snow_scalars: dict | None = None                 # uadj,mbase,mfmax,...,scf
    snow_grids: dict | None = None                   # {mfmax_grid: path, ...}
    temp_dir: str | None = None                      # TEMPForcing LOC
    temp_name: str = "temp_YYYYMMDDHH.pqf"


def _param_block(header: str, gauges, scalar_keys, scalars, grids: dict | None) -> str:
    lines = [f"[{header}]"]
    for g in gauges:
        lines.append(f"gauge={g.id}")
        if grids:
            for gkey, path in grids.items():
                lines.append(f"{gkey}={os.path.abspath(path)}")
        for k in scalar_keys:
            lines.append(f"{k}={scalars[k]}")
    return "\n".join(lines) + "\n\n"


def build_control(spec: ControlSpec) -> str:
    basic = os.path.abspath(spec.basic_dir)
    precip = os.path.abspath(spec.precip_dir)
    pet = os.path.abspath(spec.pet_dir)
    out = os.path.abspath(spec.output_dir)
    usgs = os.path.abspath(spec.usgs_dir) if spec.usgs_dir else ""

    basic_sec = (
        f"[Basic]\nDEM={basic}/dem_clip.tif\nDDM={basic}/fdir_clip.tif\nFAM={basic}/facc_clip.tif\n\n"
        f"PROJ={spec.proj}\nESRIDDM={'true' if spec.esriddm else 'false'}\n"
        f"SelfFAM={'true' if spec.selffam else 'false'}\n\n"
    )
    precip_sec = (
        f"[PrecipForcing MRMS]\nTYPE=PQF\nUNIT={spec.precip_unit}\nFREQ={spec.precip_freq}\n"
        f"LOC={precip}\nNAME={spec.precip_name}\n\n"
    )
    pet_sec = (
        f"[PETForcing PET]\nTYPE=PQF\nUNIT={spec.pet_unit}\nFREQ={spec.pet_freq}\n"
        f"LOC={pet}\nNAME={spec.pet_name}\n\n"
    )
    temp_sec = (
        f"[TEMPForcing TEMP]\nTYPE=PQF\nUNIT=C\nFREQ=1h\n"
        f"LOC={os.path.abspath(spec.temp_dir) if spec.temp_dir else ''}\nNAME={spec.temp_name}\n\n"
        if spec.snow_on else ""
    )
    gauges_sec = "".join(
        f"[Gauge {g.id}]\nLON={g.lon}\nLAT={g.lat}\n"
        + (f"OBS={usgs}/USGS_{g.id}_UTC_m3s.csv\n" if usgs else "")
        + f"OUTPUTTS=TRUE\nWANTCO=TRUE\nBASINAREA={g.area}\n\n"
        for g in spec.gauges
    )
    basin_sec = "[Basin 0]\n" + "".join(f"GAUGE={g.id}\n" for g in spec.gauges) + "\n"

    # param grids: split the {..._grid} map into crest vs kw sets
    crest_grids = {k: v for k, v in (spec.param_grids or {}).items()
                   if k in ("wm_grid", "im_grid", "fc_grid", "b_grid")}
    kw_grids = {k: v for k, v in (spec.param_grids or {}).items()
                if k in ("leaki_grid", "alpha_grid", "beta_grid", "alpha0_grid")}
    # EF5 names the water-balance section after the model: [<model>paramset]
    # (crestparamset / crestphysparamset / hpparamset) — Model.cpp modelParamSetStrings
    model_l = spec.model.lower()
    if model_l == "hp":
        crest_grids = None                           # hp has no gridded params
    crest_sec = _param_block(f"{model_l}paramset CrestParam", spec.gauges,
                             list(spec.crest), spec.crest, crest_grids or None)
    kw_sec = _param_block("kwparamset KWParam", spec.gauges, list(spec.kw), spec.kw, kw_grids or None)
    snow_sec = (_param_block("snow17paramset SnowParam", spec.gauges,
                             list(spec.snow_scalars or {}), spec.snow_scalars or {}, spec.snow_grids or None)
                if spec.snow_on else "")

    sd = os.path.abspath(spec.state_dir) if spec.state_dir else None

    def _task(name, tbegin, tend, task_out, grids, save_state_at):
        b = (f"[Task {name}]\nSTYLE=SIMU\nMODEL={spec.model}\nROUTING={spec.routing}\nBASIN=0\n"
             f"PRECIP=MRMS\nPET=PET\n")
        if spec.snow_on:
            b += "TEMP=TEMP\nSNOW=SNOW17\nSNOW_PARAM_SET=SnowParam\n"
        b += (f"OUTPUT={task_out}\nPARAM_SET=CrestParam\n"
              f"ROUTING_PARAM_Set=KWParam\nTIMESTEP={spec.timestep}\n")
        if grids:
            b += "output_grids=streamflow\n"          # per EF5 manual; writes q.<time>.<model>.tif
        if sd:                                         # STATES= -> load state dated TIME_BEGIN
            b += f"STATES={sd}\n"
            if save_state_at is not None:              # STATES+TIME_STATE => save at this time
                b += f"TIME_STATE={save_state_at:%Y%m%d%H%M}\n"
        b += f"TIME_BEGIN={tbegin:%Y%m%d%H%M}\nTIME_END={tend:%Y%m%d%H%M}\n\n"
        return b

    header = (basic_sec + precip_sec + pet_sec + temp_sec + gauges_sec + basin_sec
              + crest_sec + kw_sec + snow_sec)
    cdir = os.path.dirname(os.path.abspath(spec.control_path)) or "."
    os.makedirs(cdir, exist_ok=True)

    # The warm-up runs as a SEPARATE ef5 process (own control file): running
    # two [Task]s in one process segfaults between tasks; the state files on
    # disk connect the two runs.
    if spec.warmup_start and sd:
        os.makedirs(os.path.join(out, "warmup"), exist_ok=True)
        wu = (header
              + _task("warmup", spec.warmup_start, spec.time_begin,
                      os.path.join(out, "warmup"), False, spec.time_begin)
              + "[Execute]\nTASK=warmup\n")
        with open(os.path.join(cdir, "control_warmup.txt"), "w", encoding="utf-8") as fp:
            fp.write(wu)

    content = (header
               + _task("Simu", spec.time_begin, spec.time_end, out, spec.output_grids,
                       spec.time_end if (sd and spec.save_state_end) else None)
               + "[Execute]\nTASK=Simu\n")
    with open(spec.control_path, "w", encoding="utf-8") as fp:
        fp.write(content)
    return os.path.abspath(spec.control_path)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "._control_test"
    os.makedirs(out, exist_ok=True)
    spec = ControlSpec(
        control_path=os.path.join(out, "control.txt"),
        time_begin=datetime(2025, 7, 3, 0, 0), time_end=datetime(2025, 7, 6, 0, 0),
        timestep="1h",
        basic_dir=os.path.join(out, "BasicData_Clip"),
        precip_dir=os.path.join(out, "CREST_input/MRMS"),
        pet_dir=os.path.join(out, "CREST_input/PET"),
        output_dir=os.path.join(out, "CREST_output"),
        usgs_dir=os.path.join(out, "USGS_gauge"),
        gauges=[Gauge(id="01011000", lon=-69.0788, lat=47.0696, area=3186.844)],
        param_grids={"wm_grid": "wm.tif", "im_grid": "im.tif", "fc_grid": "fc.tif",
                     "b_grid": "b.tif", "leaki_grid": "leaki.tif", "alpha_grid": "alpha.tif",
                     "beta_grid": "beta.tif", "alpha0_grid": "alpha0.tif"},
    )
    path = build_control(spec)
    print("wrote", path, "\n" + "=" * 60)
    with open(path) as fp:
        print(fp.read())
