"""Forcing archive: the one reader of the CONUS forcing tars.

Tars are built in memory and served through the archive's `fetch` seam — no
Hub, no network."""
from __future__ import annotations

import io
import tarfile
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

NODATA = -9999.0
# 4x5 grid, 1-degree cells, lower-left (-100, 40) -> top edge at 44N
XLL, YLL, CELL = -100.0, 40.0, 1.0
BBOX = (-99.0, 41.0, -97.0, 43.0)      # -> rows 1:3, cols 1:3
T = datetime(2024, 1, 15, 6)
TEMP_MEMBER = "NLDAS_FORA0125_H.A20240115.0600.002.grb.SUB_T.pqf"
TEMP_MONTH_TAR = "temp/2024/temp_2024_01.tar"
TEMP_YEAR_TAR = "temp/temp_2024.tar"


def pqf_bytes(a, xll=XLL, yll=YLL, cell=CELL, nodata=NODATA) -> bytes:
    """Independent PQF encoder: float32 column 'v', geometry in metadata."""
    a = np.asarray(a, dtype="float32")
    nr, nc = a.shape
    meta = {b"ncols": str(nc).encode(), b"nrows": str(nr).encode(),
            b"xllcorner": repr(xll).encode(), b"yllcorner": repr(yll).encode(),
            b"cellsize": repr(cell).encode(), b"nodata": repr(nodata).encode()}
    schema = pa.schema([pa.field("v", pa.float32())]).with_metadata(meta)
    buf = io.BytesIO()
    pq.write_table(pa.table({"v": a.reshape(-1)}, schema=schema), buf)
    return buf.getvalue()


def tar_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeStore:
    """`fetch` seam: repo path -> tar bytes; absent paths raise FileNotFoundError."""
    def __init__(self, files: dict[str, bytes]):
        self.files, self.asked = files, []

    def __call__(self, path):
        self.asked.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


DEGC = np.arange(20, dtype="float32").reshape(4, 5) - 5.0     # -5 .. 14 degC


def test_degc_member_comes_back_clipped_to_the_box():
    from hf_data.forcing_archive import ForcingArchive
    arc = ForcingArchive(fetch=FakeStore({TEMP_MONTH_TAR: tar_bytes({TEMP_MEMBER: pqf_bytes(DEGC)})}))
    g = arc.read("temp", T, BBOX)
    np.testing.assert_array_equal(g.values, [[1.0, 2.0], [6.0, 7.0]])
    assert (g.xll, g.yll, g.cell, g.nodata) == (-99.0, 41.0, 1.0, NODATA)
    assert g.was_kelvin is False


def test_kelvin_temperature_member_comes_back_in_degc():
    from hf_data.forcing_archive import ForcingArchive
    kelvin = DEGC + 273.15
    kelvin[1, 1] = NODATA                        # nodata stays nodata
    arc = ForcingArchive(fetch=FakeStore({TEMP_MONTH_TAR: tar_bytes({TEMP_MEMBER: pqf_bytes(kelvin)})}))
    g = arc.read("temp", T, BBOX)
    np.testing.assert_allclose(g.values, [[NODATA, 2.0], [6.0, 7.0]], atol=1e-4)
    assert g.was_kelvin is True


def test_missing_member_reads_as_none():
    from hf_data.forcing_archive import ForcingArchive
    other = "NLDAS_FORA0125_H.A20240115.0700.002.grb.SUB_T.pqf"
    arc = ForcingArchive(fetch=FakeStore({TEMP_MONTH_TAR: tar_bytes({other: pqf_bytes(DEGC)})}))
    assert arc.read("temp", T, BBOX) is None


def test_generic_member_name_is_the_fallback():
    """NARR-derived temp members are stored under the generic output name."""
    from hf_data.forcing_archive import ForcingArchive
    arc = ForcingArchive(fetch=FakeStore({TEMP_MONTH_TAR: tar_bytes({"temp_2024011506.pqf": pqf_bytes(DEGC)})}))
    g = arc.read("temp", T, BBOX)
    np.testing.assert_array_equal(g.values, [[1.0, 2.0], [6.0, 7.0]])
    assert g.member == "temp_2024011506.pqf"


def test_implausible_temperature_grid_is_rejected_whole():
    """Neither plausible degC nor Kelvin: rejected, even if the box looks fine."""
    from hf_data.forcing_archive import ForcingArchive, ForcingRangeError
    bad = DEGC.copy()
    bad[3, 4] = 95.0                              # outside the box, still rejects
    arc = ForcingArchive(fetch=FakeStore({TEMP_MONTH_TAR: tar_bytes({TEMP_MEMBER: pqf_bytes(bad)})}))
    with pytest.raises(ForcingRangeError):
        arc.read("temp", T, BBOX)


def test_year_tar_serves_a_month_with_no_month_tar():
    from hf_data.forcing_archive import ForcingArchive
    store = FakeStore({TEMP_YEAR_TAR: tar_bytes({TEMP_MEMBER: pqf_bytes(DEGC)})})
    g = ForcingArchive(fetch=store).read("temp", T, BBOX)
    np.testing.assert_array_equal(g.values, [[1.0, 2.0], [6.0, 7.0]])
    assert store.asked == [TEMP_MONTH_TAR, TEMP_YEAR_TAR]


def test_month_with_neither_tar_is_an_archive_gap():
    from hf_data.forcing_archive import ArchiveGap, ForcingArchive
    with pytest.raises(ArchiveGap):
        ForcingArchive(fetch=FakeStore({})).read("temp", T, BBOX)


def test_network_failure_on_the_year_tar_is_not_a_gap():
    """Only a genuine 'not there' is a gap; a transient error propagates."""
    from hf_data.forcing_archive import ArchiveGap, ForcingArchive

    def flaky(path):
        raise ConnectionError(path)
    with pytest.raises(ConnectionError):
        ForcingArchive(fetch=flaky).read("temp", T, BBOX)


# --- EF5 run preparation reads through the archive -------------------------- #
def read_back(path):
    pf = pq.ParquetFile(path)
    m = {k.decode(): v.decode() for k, v in pf.schema_arrow.metadata.items()
         if not k.startswith(b"ARROW")}
    a = pf.read().column("v").to_numpy().reshape(int(m["nrows"]), int(m["ncols"]))
    return a, float(m["xllcorner"]), float(m["yllcorner"]), float(m["cellsize"])


def test_prepare_forcing_writes_clipped_degc_hours(tmp_path):
    from hf_data import forcing
    from hf_data.forcing_archive import ForcingArchive
    h7 = "NLDAS_FORA0125_H.A20240115.0700.002.grb.SUB_T.pqf"
    tar = tar_bytes({TEMP_MEMBER: pqf_bytes(DEGC), h7: pqf_bytes(DEGC + 1.0)})
    out = tmp_path / "TEMP"
    res = forcing.prepare_forcing("temp", BBOX, T, datetime(2024, 1, 15, 8), str(out),
                                  archive=ForcingArchive(fetch=FakeStore({TEMP_MONTH_TAR: tar})))
    assert res.written == ["temp_2024011506.pqf", "temp_2024011507.pqf"]
    assert res.missing == ["NLDAS_FORA0125_H.A20240115.0800.002.grb.SUB_T.pqf"]
    assert res.converted == []
    a, xll, yll, cell = read_back(out / "temp_2024011507.pqf")
    np.testing.assert_array_equal(a, [[2.0, 3.0], [7.0, 8.0]])
    assert (xll, yll, cell) == (-99.0, 41.0, 1.0)
    assert res.control_block == (f"[TEMPForcing TEMP]\nTYPE=PQF\nUNIT=C\nFREQ=1h\n"
                                 f"LOC={out}\nNAME=temp_YYYYMMDDHH.pqf\n\n")


def test_prepare_forcing_reports_kelvin_hours_it_converted(tmp_path):
    from hf_data import forcing
    from hf_data.forcing_archive import ForcingArchive
    tar = tar_bytes({TEMP_MEMBER: pqf_bytes(DEGC + 273.15)})
    out = tmp_path / "TEMP"
    res = forcing.prepare_forcing("temp", BBOX, T, T, str(out),
                                  archive=ForcingArchive(fetch=FakeStore({TEMP_MONTH_TAR: tar})))
    assert res.converted == ["temp_2024011506.pqf"]
    a, *_ = read_back(out / "temp_2024011506.pqf")
    np.testing.assert_allclose(a, [[1.0, 2.0], [6.0, 7.0]], atol=1e-4)


def test_prepare_forcing_fails_readably_on_an_archive_gap(tmp_path):
    from hf_data import forcing
    from hf_data.forcing_archive import ForcingArchive
    with pytest.raises(RuntimeError, match="not available for 2024-01"):
        forcing.prepare_forcing("temp", BBOX, T, T, str(tmp_path / "TEMP"),
                                archive=ForcingArchive(fetch=FakeStore({})))


# --- write side: update scripts store grids through the archive ------------ #
WHOLE = (-100.0, 40.0, -95.0, 44.0)             # the full 4x5 domain


def test_stored_grid_reads_back_identical():
    from hf_data.forcing_archive import ForcingArchive, store_bytes
    member = store_bytes("temp", DEGC, XLL, YLL, CELL, NODATA, name=TEMP_MEMBER)
    arc = ForcingArchive(fetch=FakeStore({TEMP_MONTH_TAR: tar_bytes({TEMP_MEMBER: member})}))
    g = arc.read("temp", T, WHOLE)
    np.testing.assert_array_equal(g.values, DEGC)
    assert (g.xll, g.yll, g.cell, g.nodata) == (XLL, YLL, CELL, NODATA)
    assert g.was_kelvin is False


def test_storing_a_kelvin_temperature_grid_is_refused(tmp_path):
    from hf_data.forcing_archive import TempUnitsError, store_bytes, store_file
    kelvin = DEGC + 273.15
    with pytest.raises(TempUnitsError):
        store_bytes("temp", kelvin, XLL, YLL, CELL, NODATA, name=TEMP_MEMBER)
    out = tmp_path / "temp_2024011506.pqf"
    with pytest.raises(TempUnitsError):
        store_file(str(out), "temp", kelvin, XLL, YLL, CELL, NODATA)
    assert list(tmp_path.iterdir()) == []


def test_store_file_reads_back_identical(tmp_path):
    from hf_data.forcing_archive import ForcingArchive, store_file
    out = tmp_path / TEMP_MEMBER
    store_file(str(out), "temp", DEGC, XLL, YLL, CELL, NODATA)
    arc = ForcingArchive(fetch=FakeStore({TEMP_MONTH_TAR: tar_bytes({TEMP_MEMBER: out.read_bytes()})}))
    np.testing.assert_array_equal(arc.read("temp", T, WHOLE).values, DEGC)


def test_kelvin_to_degc_is_the_archive_conversion():
    """Writers holding a Kelvin source (NARR) convert with the archive's own offset."""
    from hf_data.forcing_archive import kelvin_to_degc
    np.testing.assert_allclose(kelvin_to_degc(np.array([273.15, 300.0])), [0.0, 26.85])


def test_update_scripts_import_no_private_forcing_helpers():
    """Forcing writers go through the archive's public store operation."""
    import ast
    import pathlib
    scripts = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    offenders = []
    for p in sorted(scripts.glob("*.py")):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.ImportFrom) and node.module
                    and node.module.startswith("hf_data.forcing")):
                offenders += [f"{p.name}: {a.name}" for a in node.names
                              if a.name.startswith("_")]
    assert offenders == []


def test_no_module_reaches_into_private_forcing_helpers():
    """Ticket 04 contract: outside forcing_archive nobody imports or touches a
    private forcing helper (scripts, tests, Spaces, nowcast_eval included)."""
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    mods = {"forcing", "forcing_archive", "hf_data.forcing", "hf_data.forcing_archive"}
    offenders = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel == "hf_data/forcing_archive.py" or rel.startswith(".") or "/." in rel:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in mods:
                offenders += [f"{rel}: {a.name}" for a in node.names
                              if a.name.startswith("_")]
            elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                  and node.value.id in mods and node.attr.startswith("_")
                  and not node.attr.startswith("__")):
                offenders.append(f"{rel}: {node.value.id}.{node.attr}")
    assert offenders == []


def test_archive_reads_the_data_repo_named_by_the_environment(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError
    from hf_data.forcing_archive import ArchiveGap, ForcingArchive
    asked = []

    def dl(repo, path, **kw):
        asked.append(repo)
        raise EntryNotFoundError(path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", dl)
    monkeypatch.setenv("CREST_DATA_REPO", "me/data")
    with pytest.raises(ArchiveGap):
        ForcingArchive().read("temp", T, BBOX)
    assert asked and set(asked) == {"me/data"}


def test_transient_error_on_the_month_tar_propagates_without_the_year_fallback():
    """Only a month tar that is not there falls back to the year tar."""
    from hf_data.forcing_archive import ForcingArchive
    store = FakeStore({TEMP_YEAR_TAR: tar_bytes({TEMP_MEMBER: pqf_bytes(DEGC)})})

    def flaky_month(path):
        if path == TEMP_MONTH_TAR:
            store.asked.append(path)
            raise ConnectionError(path)
        return store(path)
    with pytest.raises(ConnectionError):
        ForcingArchive(fetch=flaky_month).read("temp", T, BBOX)
    assert store.asked == [TEMP_MONTH_TAR]


def _recent_hour():
    from datetime import timedelta
    return (datetime.utcnow() - timedelta(hours=3)).replace(minute=0, second=0,
                                                           microsecond=0)


def test_prepare_forcing_fills_tar_missing_hours_from_the_recent_store(tmp_path,
                                                                       monkeypatch):
    """The near-real-time fallback reads through the archive's fetch seam."""
    import huggingface_hub
    from hf_data import forcing
    from hf_data.forcing_archive import ForcingArchive

    def no_hub(*a, **kw):
        raise AssertionError("the recent store must be read through the archive")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_hub)
    monkeypatch.setattr(forcing, "hf_hub_download", no_hub, raising=False)
    t = _recent_hour()
    recent = t.strftime("mrms_recent/mrms1h_pass1_%Y%m%d%H.pqf")
    rain = np.arange(20, dtype="float32").reshape(4, 5)
    store = FakeStore({recent: pqf_bytes(rain)})              # no month/year tar yet
    res = forcing.prepare_forcing("mrms", BBOX, t, t, str(tmp_path / "MRMS"),
                                  archive=ForcingArchive(fetch=store))
    assert res.recent == 1 and res.missing == []
    assert recent in store.asked
    a, xll, yll, cell = read_back(tmp_path / "MRMS" / t.strftime("mrms_%Y%m%d%H.pqf"))
    np.testing.assert_array_equal(a, [[6.0, 7.0], [11.0, 12.0]])
    assert (xll, yll, cell) == (-99.0, 41.0, 1.0)


def test_archive_reads_a_loose_recent_member():
    from hf_data.forcing_archive import ForcingArchive
    t = _recent_hour()
    recent = t.strftime("mrms_recent/mrms1h_pass1_%Y%m%d%H.pqf")
    arc = ForcingArchive(fetch=FakeStore({recent: pqf_bytes(DEGC)}))
    g = arc.read_recent("mrms", t, BBOX)
    np.testing.assert_array_equal(g.values, [[1.0, 2.0], [6.0, 7.0]])
    assert arc.read_recent("mrms", t.replace(year=t.year - 1), BBOX) is None
    assert arc.read_recent("temp", t, BBOX) is None           # no recent store
