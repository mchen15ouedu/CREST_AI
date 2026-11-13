"""Snow detection reads basin temperature through the forcing archive.

Tars are built in memory and served through the archive's `fetch` seam: no
Hub, no network."""
from __future__ import annotations

from datetime import datetime

import numpy as np

from test_forcing_archive import (BBOX, TEMP_MONTH_TAR, FakeStore, pqf_bytes,
                                  tar_bytes)

H00 = "NLDAS_FORA0125_H.A20240115.0000.002.grb.SUB_T.pqf"
H06 = "NLDAS_FORA0125_H.A20240115.0600.002.grb.SUB_T.pqf"
WARM_C = np.full((4, 5), 5.0, dtype="float32")


def _detect(members):
    from hf_data.forcing_archive import ForcingArchive
    from hf_data.snow import detect_snow
    arc = ForcingArchive(fetch=FakeStore({TEMP_MONTH_TAR: tar_bytes(members)}))
    return detect_snow(BBOX, datetime(2024, 1, 15, 0), datetime(2024, 1, 15, 6),
                       archive=arc)


def test_a_kelvin_hour_is_converted_and_counted_not_dropped():
    # 00Z is a mild degC hour; 06Z was stored in Kelvin at 268.15 K (-5 degC).
    kelvin = np.full((4, 5), 268.15, dtype="float32")
    res = _detect({H00: pqf_bytes(WARM_C), H06: pqf_bytes(kelvin)})
    assert res["snow"] is True
    assert res["min_temp_c"] == -5.0


def test_degc_hours_above_threshold_turn_snow_off():
    res = _detect({H00: pqf_bytes(WARM_C), H06: pqf_bytes(WARM_C + 1.0)})
    assert res["snow"] is False
    assert res["min_temp_c"] == 5.0


def test_snow_parameter_grids_come_from_the_configured_data_repo(tmp_path, monkeypatch):
    import rasterio
    from hf_data import crashlog, snow
    opened = []

    def no_network(path, *a, **kw):
        opened.append(path)
        raise OSError("offline")
    monkeypatch.setattr(rasterio, "open", no_network)
    monkeypatch.setattr(crashlog, "capture", lambda *a, **kw: None)
    monkeypatch.setenv("CREST_DATA_REPO", "me/data")
    assert snow.clip_snow_grids(BBOX, str(tmp_path), unsafe_ssl=False) == {}
    assert opened and all(p.startswith(
        "/vsicurl/https://huggingface.co/datasets/me/data/resolve/main/param/snow17/v1/")
        for p in opened)
