"""evtpng: max-pooled 2-D inundation overlays for basin zoom.

Pure numpy for the pooling/colouring; the endpoint test serves a synthetic
event from tmp_path (no network)."""
import json
import os

import numpy as np
import pytest

from conftest import RES, write_grid


def _channels():
    """1-cell-wide channels on a dry grid (uint16 cm): N-S at col 13 (1.5 m),
    E-W at row 25 (0.4 m) — neither on the stride-8 sampling lattice."""
    d = np.zeros((40, 37), dtype=np.uint16)
    d[:, 13] = 150
    d[25, :] = 40
    return d


def test_pool_keeps_one_cell_channel():
    from hf_data import evtpng
    d = _channels()
    p = evtpng.pool_max(d, 8)
    assert p.shape == (5, 5)                       # ceil(40/8) x ceil(37/8): padded
    assert (p[:, 1] == 150).all()                  # col 13 -> block column 1
    assert (p[3, [0, 2, 3, 4]] == 40).all()        # row 25 -> block row 3
    assert p[3, 1] == 150                          # crossing keeps the deeper
    assert (np.delete(np.delete(p, 1, axis=1), 3, axis=0) == 0).all()
    # nearest-neighbour at the same size samples rows/cols 0, 8, ..., 32 only:
    # both channels vanish — the empty basin-zoom map this module fixes
    assert (d[::8, ::8] == 0).all()
    # every factor keeps the channel; the padded edge block stays dry
    for f in evtpng.FACTORS:
        q = evtpng.pool_max(d, f)
        assert q.shape == (-(-40 // f), -(-37 // f))
        assert (q[:, 13 // f] == 150).all()
    assert evtpng.pool_max(d, 1).tolist() == d.tolist()


def test_dry_stays_transparent_and_ramp_at_cap():
    from hf_data import eventsim, evtpng
    cap = 1.5
    rgba = evtpng.pooled_rgba(_channels(), 8, cap)
    assert rgba.shape == (5, 5, 4) and rgba.dtype == np.uint8
    dry = rgba[0, 0]
    assert dry[3] == 0                             # dry block: transparent
    assert (rgba[..., 3] > 0).sum() == 9           # 5 channel + 4 crossing-row blocks
    # (the shallow row's blocks are faded, never dropped: no alpha floor)
    # 1.5 m == cap: the ramp's last stop, fully opaque
    top = (eventsim.DEPTH_STOPS_R[-1], eventsim.DEPTH_STOPS_G[-1],
           eventsim.DEPTH_STOPS_B[-1], 255)
    assert tuple(rgba[0, 1]) == top
    # deeper than the cap saturates at the same colour
    assert tuple(evtpng.colorize(np.array([[9.0]]), cap)[0, 0]) == top
    # 0.3 x cap sits exactly on the second stop (yellow)
    mid = evtpng.colorize(np.array([[0.30 * 2.0]]), 2.0)[0, 0]
    assert tuple(mid[:3]) == (eventsim.DEPTH_STOPS_R[1], eventsim.DEPTH_STOPS_G[1],
                              eventsim.DEPTH_STOPS_B[1])
    # below MIN_SHOW_M (2 cm) is dry on the map, wet in the raster
    assert evtpng.pooled_rgba(np.full((4, 4), 1, np.uint16), 2, cap)[..., 3].max() == 0
    assert evtpng.pooled_rgba(np.zeros((9, 9), np.uint16), 4, cap)[..., 3].max() == 0
    # declared nodata (outside the domain) is dry, not a 655 m flood
    nd = _channels()
    nd[0, 0] = 65535
    assert evtpng.pooled_rgba(nd, 8, cap, nodata=65535.0)[0, 0, 3] == 0


def test_shallow_blocks_fade_by_wet_fraction():
    """Max-pooling keeps every wet cell but inflates the wet EXTENT. A block
    keeps the colour of its deepest cell; if that stays in the green band
    (< 0.30 x cap) its alpha fades by sqrt(wet share), no floor, while a
    block reaching >= 0.30 x cap keeps the native alpha whatever its share."""
    from hf_data import evtpng
    cap = 1.5
    d = _channels()                     # col 13: 1.5 m; row 25: 0.4 m < 0.45 m
    # the east edge block has 5 real columns: its share is of 40 cells, not 64
    np.testing.assert_allclose(evtpng.pool_wet_frac(d, 8)[3],
                               [8 / 64, 15 / 64, 8 / 64, 8 / 64, 5 / 40])
    assert evtpng.pool_wet_frac(d, 8)[0, 4] == 0
    assert (evtpng.pool_wet_frac(d, 1) == (d >= 2)).all()
    rgba = evtpng.pooled_rgba(d, 8, cap)
    native = evtpng.colorize(np.array([[1.5, 0.4]]), cap)[0]
    # the channel: 8 of 64 cells wet, still the native colour AND alpha
    assert tuple(rgba[0, 1]) == tuple(native[0])
    # the shallow row: the native colour, alpha x sqrt(1/8)
    a04 = 90 + 165 * np.sqrt(0.4 / cap)
    for j in (0, 2, 3, 4):
        assert tuple(rgba[3, j, :3]) == tuple(native[1, :3])
        assert rgba[3, j, 3] == int(a04 * np.sqrt(1 / 8)) < native[1, 3]
    # the band edge, one wet cell of 64: 0.30 x cap stays opaque, just under fades
    lone = np.zeros((8, 8), np.uint16)
    lone[3, 3] = 60                     # 0.60 m = 0.30 x 2.0
    assert evtpng.pooled_rgba(lone, 8, 2.0)[0, 0, 3] == \
        evtpng.colorize(np.array([[0.6]]), 2.0)[0, 0, 3]
    lone[3, 3] = 59
    assert evtpng.pooled_rgba(lone, 8, 2.0)[0, 0, 3] == \
        int((90 + 165 * np.sqrt(0.59 / 2.0)) * 0.125)
    # sub-2-cm cells are dry for the share too (the native png's threshold)
    lone[0, :] = 1
    assert evtpng.pool_wet_frac(lone, 8)[0, 0] == 1 / 64


def test_colours_match_the_cdn_overlay(tmp_path):
    """f=1 pooling is the identity, so the module must reproduce the native
    png eventsim._make_pngs writes for the same tif, byte for byte in RGBA."""
    from PIL import Image

    from hf_data import eventsim, evtpng
    rng = np.random.default_rng(7)
    d = rng.integers(0, 400, size=(30, 41)).astype(np.uint16)
    d[rng.random(d.shape) < 0.5] = 0
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(tmp_path / "maxdepth.tif", "w", driver="GTiff", height=30,
                       width=41, count=1, dtype="uint16", crs="EPSG:4269",
                       transform=from_origin(-111.0, 36.0, RES, RES)) as ds:
        ds.write(d, 1)
    man = {"grid": {"nx": 41, "ny": 30,
                    "transform": [RES, 0.0, -111.0, 0.0, -RES, 36.0]}, "frames": []}
    eventsim._make_pngs(str(tmp_path), man)
    native = np.asarray(Image.open(tmp_path / "maxdepth.png").convert("RGBA"))
    ours = evtpng.pooled_rgba(d, 1, man["depth_cap_m"])
    assert ours.shape == native.shape
    assert (ours == native).all()


@pytest.mark.parametrize("eid,ok", [
    ("2026102122_09401260", True),
    ("2026103020_09353800", True),
    ("2026102122_V70520944", True),
    ("2026102122_", False),
    ("202610212_09401260", False),
    ("2026102122-09401260", False),
    ("2026102122_09401260\n", False),
    ("2026102122_0940/1260", False),
    ("../2026102122_09401260", False),
    ("2026102122_0940126.", False),
    ("２026102122_09401260", False),           # fullwidth digit
    (None, False),
])
def test_event_id_validation(eid, ok):
    from hf_data import evtpng
    assert evtpng.valid_event_id(eid) is ok


@pytest.mark.parametrize("name,ok", [
    ("maxdepth", True),
    ("depth_202610211000", True),
    ("depth_2026102110", False),
    ("depth_2026102110000", False),
    ("maxdepth.png", False),
    ("maxdepth.tif", False),
    ("maxspeed", False),
    ("dem", False),
    ("../maxdepth", False),
    ("depth_202610211000\n", False),
    ("Maxdepth", False),
])
def test_name_validation(name, ok):
    from hf_data import evtpng
    assert evtpng.valid_name(name) is ok


def test_factor_validation():
    from hf_data import evtpng
    assert all(evtpng.valid_factor(f) for f in (2, 4, 8, 16))
    assert not any(evtpng.valid_factor(f) for f in (0, 1, 3, 32, -8, True, 8.0, "8"))


def _fake_store(tmp_path, monkeypatch, generated="2026-10-22T19:16:27Z",
                stems=("depth_202610211000",)):
    """A synthetic event served from tmp_path instead of the HF dataset."""
    from hf_data import evtpng
    eid = "2026102122_09401260"
    src = tmp_path / "store" / "events" / eid
    src.mkdir(parents=True)
    write_grid(str(src / "maxdepth.tif"), _channels(), -111.0, 36.0, nodata=None)
    (src / "manifest.json").write_text(json.dumps({
        "generated": generated, "depth_cap_m": 1.5,
        "frames": [{"t": "2026-10-21T10:00Z", "file": f"{s}.tif", "png": f"{s}.png"}
                   for s in stems]}))
    calls = []

    def fake_download(rel, dest):
        calls.append(rel)
        p = tmp_path / "store" / "events" / rel
        if not p.exists():
            raise LookupError(rel)
        return str(p)

    monkeypatch.setattr(evtpng, "_download", fake_download)
    monkeypatch.setattr(evtpng, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(evtpng, "_man_memo", {})
    return eid, calls


def test_endpoint_serves_pooled_png_and_caches(tmp_path, monkeypatch):
    import io

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from PIL import Image

    from hf_data import evtpng
    eid, calls = _fake_store(tmp_path, monkeypatch)
    app = FastAPI()
    app.include_router(evtpng.router)
    c = TestClient(app)
    r = c.get(f"/api/evt_png/{eid}/maxdepth.png?f=8&s={evtpng.STYLE}"
              "&g=2026-10-22T19:16:27Z")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert "immutable" in r.headers["cache-control"]
    im = np.asarray(Image.open(io.BytesIO(r.content)))
    assert im.shape == (5, 5, 4) and (im[:, 1, 3] == 255).all()
    n_dl = len(calls)
    r2 = c.get(f"/api/evt_png/{eid}/maxdepth.png?f=8")   # disk-cache hit
    assert r2.content == r.content and len(calls) == n_dl
    assert "immutable" not in r2.headers["cache-control"]   # no generation named
    # an old app.js (another style, or none): today's colours, never pinned
    for s in (f"&s={evtpng.STYLE - 1}", "&s=x", ""):
        r3 = c.get(f"/api/evt_png/{eid}/maxdepth.png?f=8&g=2026-10-22T19:16:27Z{s}")
        assert r3.status_code == 200 and r3.content == r.content, s
        assert r3.headers["cache-control"] == evtpng.CC_SHORT, s
    assert len(calls) == n_dl
    assert os.listdir(tmp_path / "cache")
    # strict validation before anything touches the store
    for bad in (f"/api/evt_png/{eid}/maxdepth.png?f=3",
                f"/api/evt_png/{eid}/maxdepth.png",
                f"/api/evt_png/{eid}/dem.png?f=8",
                f"/api/evt_png/{eid}/depth_2026.png?f=8",
                "/api/evt_png/x_09401260/maxdepth.png?f=8"):
        assert c.get(bad).status_code == 400, bad
    # a frame the manifest doesn't list, or an unknown event: 404
    assert c.get(f"/api/evt_png/{eid}/depth_202610211015.png?f=8").status_code == 404
    assert c.get("/api/evt_png/2026102122_00000000/maxdepth.png?f=8").status_code == 404


def test_style_in_the_url_is_in_step_with_app_js():
    """frontend/app.js puts evtpng.STYLE in every pooled URL (&s=), so a
    colour change is a URL no browser has cached for good."""
    import re

    from conftest import ROOT
    from hf_data import evtpng
    with open(os.path.join(ROOT, "frontend", "app.js"), encoding="utf-8") as fp:
        js = fp.read()
    m = re.search(r"^const EVT_PNG_STYLE = (\d+);", js, re.M)
    assert m and int(m.group(1)) == evtpng.STYLE
    assert re.search(r"/api/evt_png/.*\?f=\$\{f\}&s=\$\{EVT_PNG_STYLE\}", js)


def test_manifest_reread_only_for_a_newer_generation(tmp_path, monkeypatch):
    """A client naming a NEWER generation than the memo forces a manifest
    re-read (the event re-published); an older one (a page loaded before
    that) is answered from the memo, under the short cache header."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from hf_data import evtpng
    g0, g1, g2, g3 = (f"2026-10-22T{h}:16:27Z" for h in (18, 19, 20, 21))
    eid, calls = _fake_store(tmp_path, monkeypatch, generated=g1)
    monkeypatch.setattr(evtpng, "MAN_TTL_S", 60.0)
    monkeypatch.setattr(evtpng, "_limiters", {})
    app = FastAPI()
    app.include_router(evtpng.router)
    c = TestClient(app)
    url = f"/api/evt_png/{eid}/maxdepth.png?f=8&s={evtpng.STYLE}&g="

    def reads():
        return calls.count(f"{eid}/manifest.json")

    def age_memo():                     # past the 5 s re-read limit, inside the TTL
        t, slim = evtpng._man_memo[eid]
        evtpng._man_memo[eid] = (t - 10, slim)

    assert "immutable" in c.get(url + g1).headers["cache-control"] and reads() == 1
    age_memo()
    r = c.get(url + g0)                                     # older: the memo
    assert r.status_code == 200 and reads() == 1
    assert r.headers["cache-control"] == evtpng.CC_SHORT
    man = tmp_path / "store" / "events" / eid / "manifest.json"
    man.write_text(json.dumps({**json.loads(man.read_text()), "generated": g2}))
    r = c.get(url + g2)                                     # newer: re-read
    assert reads() == 2 and evtpng._man_memo[eid][1]["generated"] == g2
    assert "immutable" in r.headers["cache-control"]
    age_memo()
    r = c.get(url + g1)                                     # older than the memo now
    assert reads() == 2 and r.headers["cache-control"] == evtpng.CC_SHORT
    assert "immutable" in c.get(url + g2).headers["cache-control"] and reads() == 2
    # a newer one right after a read waits out the 5 s limit on the memo
    evtpng._man_memo.clear()
    c.get(url + g2)
    r = c.get(url + g3)
    assert reads() == 3 and r.headers["cache-control"] == evtpng.CC_SHORT


def test_unexpected_failures_reach_crashlog_once_per_type(tmp_path, monkeypatch):
    """The generic 502 path records the exception (event, overlay, factor),
    once per exception type per CRASH_EVERY_S; a missing overlay (404) or a
    full queue (503) is an answer, not a crash, and is never recorded."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from hf_data import crashlog, evtpng
    eid, _ = _fake_store(tmp_path, monkeypatch)
    monkeypatch.setattr(evtpng, "_limiters", {})
    monkeypatch.setattr(evtpng, "_crashed", {})
    got = []
    monkeypatch.setattr(crashlog, "capture", lambda where, exc=None, message="", **ctx:
                        got.append((where, type(exc).__name__, ctx)))
    boom = [OSError("network down")]

    def failing_compute(event_id, tif, f, cap):
        raise boom[0]

    monkeypatch.setattr(evtpng, "_compute", failing_compute)
    app = FastAPI()
    app.include_router(evtpng.router)
    c = TestClient(app)
    url = f"/api/evt_png/{eid}/maxdepth.png?f={{}}"
    r = c.get(url.format(8))
    assert r.status_code == 502 and "OSError" in r.text
    assert got == [("evt_png", "OSError", {"event_id": eid, "name": "maxdepth", "f": 8})]
    assert c.get(url.format(4)).status_code == 502 and len(got) == 1   # same type
    boom[0] = ValueError("bad tif")
    assert c.get(url.format(2)).status_code == 502
    assert [k for _, k, _ in got] == ["OSError", "ValueError"]
    monkeypatch.setattr(evtpng, "CRASH_EVERY_S", 0.0)      # the window passed
    boom[0] = OSError("still down")
    assert c.get(url.format(16)).status_code == 502 and len(got) == 3
    # answers, not crashes
    assert c.get(f"/api/evt_png/{eid}/depth_202610211015.png?f=8").status_code == 404
    assert c.get("/api/evt_png/2026102122_00000000/maxdepth.png?f=8").status_code == 404
    boom[0] = LookupError("the tif went away")
    assert c.get(url.format(8)).status_code == 404
    boom[0] = evtpng.Busy("queue full")
    assert c.get(url.format(8)).status_code == 503
    assert len(got) == 3


def test_cold_render_burst_leaves_other_routes_responsive(tmp_path, monkeypatch):
    """Frame scrubbing at basin zoom fires a burst of cold renders. They wait
    on the module's own limiter, never on AnyIO's shared threadpool, so an
    unrelated sync route answers while every render slot is busy; a cached
    tile is served without a slot; past WORKERS + QUEUE_MAX distinct renders
    the surplus gets 503 + Retry-After, and identical tiles render once."""
    import threading
    import time

    import anyio
    import anyio.to_thread
    import httpx
    from fastapi import FastAPI

    from hf_data import evtpng
    stems = [f"depth_2026102110{m:02d}" for m in range(0, 60, 3)]     # 20 frames
    eid, _ = _fake_store(tmp_path, monkeypatch, stems=stems)
    monkeypatch.setattr(evtpng, "_limiters", {})
    monkeypatch.setattr(evtpng, "_inflight", {})
    gate = threading.Event()
    lock = threading.Lock()
    renders, live = [], [0, 0]                  # [now, peak] concurrent renders

    def slow_compute(event_id, tif, f, cap):    # a render that holds its slot
        with lock:
            renders.append((tif, f))
            live[0] += 1
            live[1] = max(live)
        try:
            gate.wait(10)
        finally:
            with lock:
                live[0] -= 1
        return b"png:" + f"{tif}|{f}".encode()

    monkeypatch.setattr(evtpng, "_compute", slow_compute)
    app = FastAPI()
    app.include_router(evtpng.router)

    @app.get("/ping")
    def ping():                                 # sync: a token from the shared pool
        return {"ok": True}

    url = f"/api/evt_png/{eid}/{{}}.png?f={{}}"
    out = {}

    async def main():
        # a small shared pool: a render holding (or waiting with) its tokens
        # would starve the ping outright
        anyio.to_thread.current_default_thread_limiter().total_tokens = 4
        tr = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=tr, base_url="http://t") as c:
            gate.set()                          # warm one tile into the disk cache
            assert (await c.get(url.format(stems[-1], 16))).status_code == 200
            gate.clear()

            async def get(key, u):
                out[key] = await c.get(u)

            try:
                async with anyio.create_task_group() as tg:
                    for k in range(3):          # one tile asked thrice
                        tg.start_soon(get, ("same", k), url.format("maxdepth", 4))
                    await anyio.sleep(0.1)
                    for i, s in enumerate(stems[:16]):
                        tg.start_soon(get, ("frame", i), url.format(s, 8))
                    await anyio.sleep(0.3)      # queued, rendering or refused
                    with anyio.fail_after(3):
                        t0 = time.perf_counter()
                        assert (await c.get("/ping")).status_code == 200
                        out["ping_s"] = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        hit = await c.get(url.format(stems[-1], 16))
                        out["hit_s"] = time.perf_counter() - t0
                    assert hit.status_code == 200
                    out["refused_early"] = sum(
                        1 for v in list(out.values())
                        if isinstance(v, httpx.Response) and v.status_code == 503)
                    gate.set()
            finally:
                gate.set()

    anyio.run(main)
    assert out["ping_s"] < 0.5 and out["hit_s"] < 0.5
    codes = [v.status_code for k, v in out.items() if isinstance(k, tuple)]
    cap = evtpng.WORKERS + evtpng.QUEUE_MAX     # distinct renders in flight
    assert codes.count(200) == 3 + (cap - 1) and codes.count(503) == 16 - (cap - 1)
    assert out["refused_early"] == codes.count(503)   # refused at once, not queued
    for v in out.values():
        if isinstance(v, httpx.Response) and v.status_code == 503:
            assert v.headers["retry-after"] == "2"
    assert renders.count(("maxdepth.tif", 4)) == 1      # the trio shared one render
    assert len({out[("same", k)].content for k in range(3)}) == 1
    assert len(renders) == 1 + cap and live[1] == evtpng.WORKERS
