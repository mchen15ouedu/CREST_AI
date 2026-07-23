"""Calibrated parameter-multiplier lookup for the AQUAH->CREST_demo integration.

Every GAGES-II gauge maps to a calibrated CRESTPHYS+KW multiplier set (from
`vincewin/CREST_data` params/multipliers.parquet, consolidated from the
conus-crest-hydro per-gauge sim controls). Gauges without their own calibration
borrow the nearest calibrated donor (source_station / source_dist_km).

These SCALARS multiply the 1 km param grids from hf_data.params (grid x scalar),
reproducing the production conus-crest-hydro parameterization instead of guessing.
The physics keys (igw, hmaxaq, gwc, gwe) require MODEL=CRESTPHYS.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
import truststore
truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download

HF_REPO = "vincewin/CREST_data"

CREST_KEYS = ("wm", "b", "im", "ke", "fc", "iwu")
CRESTPHYS_EXTRA = ("igw", "hmaxaq", "gwc", "gwe")
KW_KEYS = ("under", "leaki", "th", "isu", "alpha", "beta", "alpha0")


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    path = hf_hub_download(HF_REPO, "params/multipliers.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df["station"] = df["station"].astype(str).str.zfill(8)
    df["source_station"] = df["source_station"].astype(str).str.zfill(8)
    return df.set_index("station")


def get_multipliers(station_id: str) -> dict | None:
    """Calibrated multipliers + provenance for a gauge, or None if absent."""
    sid = str(station_id).zfill(8)
    tbl = _table()
    if sid not in tbl.index:
        return None
    r = tbl.loc[sid]
    # ksoil (soil->groundwater drain rate) postdates the calibration table;
    # EF5 requires every crestphys param, so fill the docs default when absent
    ksoil = float(r["ksoil"]) if "ksoil" in r.index else 0.1
    return {
        "crest": {k: float(r[k]) for k in CREST_KEYS},
        "crestphys": {**{k: float(r[k]) for k in CREST_KEYS + CRESTPHYS_EXTRA},
                      "ksoil": ksoil},
        "kw": {k: float(r[k]) for k in KW_KEYS},
        "source_station": str(r["source_station"]),
        "source_dist_km": float(r["source_dist_km"]),
        "own_calibration": str(r["source_station"]) == sid,
    }


def to_control_params(station_id: str, model: str = "crestphys"):
    """Return (crest_or_crestphys_dict, kw_dict) ready for hf_data.control.ControlSpec."""
    m = get_multipliers(station_id)
    if m is None:
        return None
    wb = m["crestphys"] if model.lower() == "crestphys" else m["crest"]
    return wb, m["kw"]


def nearest_station(lat: float, lon: float) -> tuple[str, float] | None:
    """(station_id, dist_km) of the calibrated gauge closest to a coordinate.

    Regionalization for ungauged virtual points: with no observations to
    calibrate against, the point borrows the full multiplier set of its
    nearest calibrated neighbor — the same spatial-proximity transfer the
    table itself uses for uncalibrated gauges (source_station).
    """
    from hf_data import gauges
    cat = gauges.load_catalog()
    tbl = _table()
    cat = cat[cat.STAID.isin(tbl.index)]
    if cat.empty:
        return None
    d2 = (cat.LAT_GAGE - lat) ** 2 + ((cat.LNG_GAGE - lon) ** 2)
    i = d2.idxmin()
    r = cat.loc[i]
    km = float(d2.loc[i]) ** 0.5 * 111.0
    return str(r.STAID).zfill(8), km


if __name__ == "__main__":
    tbl = _table()
    print(f"multiplier table: {len(tbl)} gauges")
    for sid in ("01011000", "01013500"):
        m = get_multipliers(sid)
        prov = "own" if m["own_calibration"] else f"donor {m['source_station']} @ {m['source_dist_km']:.1f} km"
        print(f"\n{sid} ({prov})")
        print("  crestphys:", {k: round(v, 3) for k, v in m["crestphys"].items()})
        print("  kw:       ", {k: round(v, 3) for k, v in m["kw"].items()})
