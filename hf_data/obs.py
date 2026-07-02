"""USGS observed streamflow — for real skill scores (task #6 bundle).

Fetches observed discharge from the USGS waterservices instantaneous-values API
(parameter 00060, cfs -> m³/s) and writes the EF5 OBS file
`USGS_<site>_UTC_m3s.csv` (time,value per line — matches EF5's `%[^,],%f`
reader and AQUAH's format). Real runs point the [Gauge] OBS= at this file so
EF5 populates the 'Observed' column of ts.csv, giving meaningful NSCE/CC/bias.
"""
from __future__ import annotations

import os
from datetime import datetime

import truststore
truststore.inject_into_ssl()
import requests

CFS_TO_CMS = 0.0283168
_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"


def fetch_usgs_discharge(site: str, t_start: datetime, t_end: datetime) -> list[tuple[datetime, float]]:
    """Observed discharge (m³/s) time series, or [] if unavailable."""
    params = {"sites": str(site).zfill(8), "parameterCd": "00060", "format": "json",
              "startDT": t_start.strftime("%Y-%m-%d"), "endDT": t_end.strftime("%Y-%m-%d"),
              "siteStatus": "all"}
    r = requests.get(_IV_URL, params=params, timeout=30)
    r.raise_for_status()
    ts = r.json().get("value", {}).get("timeSeries", [])
    if not ts:
        return []
    out = []
    for v in ts[0]["values"][0]["value"]:
        try:
            cfs = float(v["value"])
        except (TypeError, ValueError):
            continue
        if cfs < 0:                                  # USGS missing sentinel (-999999)
            continue
        dt = datetime.fromisoformat(v["dateTime"].replace("Z", "+00:00")).replace(tzinfo=None)
        out.append((dt, cfs * CFS_TO_CMS))
    return out


def write_ef5_obs(site: str, series: list[tuple[datetime, float]], out_dir: str) -> str | None:
    """Write USGS_<site>_UTC_m3s.csv (EF5 OBS). Returns the path, or None if empty."""
    if not series:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"USGS_{str(site).zfill(8)}_UTC_m3s.csv")
    with open(path, "w") as f:
        f.write("datetime,discharge\n")
        for dt, cms in series:
            f.write(f"{dt:%Y-%m-%d %H:%M:%S},{cms:.6f}\n")
    return path


if __name__ == "__main__":
    import sys
    site = sys.argv[1] if len(sys.argv) > 1 else "08144500"
    s = fetch_usgs_discharge(site, datetime(2024, 6, 1), datetime(2024, 6, 4))
    print(f"USGS {site}: {len(s)} obs points")
    if s:
        vals = [v for _, v in s]
        print(f"  {s[0][0]} .. {s[-1][0]}  range [{min(vals):.2f}, {max(vals):.2f}] m³/s")
