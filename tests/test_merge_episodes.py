"""scripts/merge_episodes.py: one flood's fragmented episodes merged into one
on the store. Plan helpers (frame union, eligibility, index rewrite,
idempotence, cap-change re-render) are pure; the products and the commit run
against a local stand-in for the dataset (tiny rasters, a fake HfApi). No
network."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil

import numpy as np
import pytest

from hf_data import eventlist, eventsim
from scripts import merge_episodes as me

G = "08353000"
T, B, A = f"2026102800_{G}", f"2026102722_{G}", f"2026102721_{G}"   # target newest


def _h(d):
    return d.strftime("%Y-%m-%dT%H:%MZ")


def _entry(eid, status="ended", **kw):
    start = dt.datetime.strptime(eid[:10], "%Y%m%d%H")
    s = {"t0": _h(start), "sim_start": _h(start - dt.timedelta(hours=12)),
         "t_end": _h(start + dt.timedelta(hours=12)), "episode_started": _h(start),
         "status": status, "gauge": G, "trigger": {"gauge": G},
         "generated": (start + dt.timedelta(minutes=50)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if status == "ended":
        s["final"] = {"ended": _h(start + dt.timedelta(hours=2)), "peak_depth_m": 0.0}
    s.update(kw)
    return s


def _index(*ids, **status):
    other = _entry("2026102500_07000000")          # another gauge, untouched
    idx = {e: _entry(e, status.get(e, "ended")) for e in ids}
    idx["2026102500_07000000"] = other
    return dict(sorted(idx.items(), reverse=True))


def _group(idx, eid=T):
    return next(g for g in eventlist.groups(idx, 12) if eid in g["members"] or eid in g["ids"])


def _fr(t):
    s = t.replace("-", "").replace("T", "").replace(":", "").replace("Z", "")
    return {"t": t, "file": f"depth_{s}.tif", "png": f"depth_{s}.png"}


T0, T1, T2, T3 = ("2026-10-27T09:00Z", "2026-10-27T10:00Z", "2026-10-27T11:00Z",
                  "2026-10-27T12:00Z")
TX = "2026-10-27T08:00Z"                          # a time only archives hold


# ---- pure helpers -----------------------------------------------------------

def test_frame_plan_target_wins_then_newest_member():
    plan = me.frame_plan(T, {A: [_fr(T0), _fr(T1)], B: [_fr(T1), _fr(T2)],
                             T: [_fr(T2), _fr(T3)]})
    assert [(t, src) for t, src, _ in plan] == [(T0, A), (T1, B), (T2, T), (T3, T)]
    # a member frame with no file is not a frame
    plan = me.frame_plan(T, {T: [{"t": T1}], B: [_fr(T1)]})
    assert [(t, src) for t, src, _ in plan] == [(T1, B)]


def test_eligibility_active_pinned_queued():
    idx = _index(T, B, A)
    g = _group(idx)
    assert set(g["members"]) == {T, B, A}
    assert me.classify(g, idx, frozenset()) == ("merge", None)
    act = _index(T, B, A, **{T: "active"})
    kind, why = me.classify(_group(act), act, frozenset())
    assert kind == "skip" and T in why and "re-run" in why
    kind, why = me.classify(g, idx, frozenset({B}))
    assert kind == "skip" and why.startswith("pinned") and B in why
    assert me.classify(g, idx, None)[0] == "skip"             # pinned set unreadable
    kind, why = me.classify(g, idx, frozenset(), queued={A})
    assert kind == "skip" and A in why
    # status missing counts as active (eventstore's default)
    idx2 = _index(T, B)
    idx2[B].pop("status")
    assert me.classify(_group(idx2), idx2, frozenset())[0] == "skip"
    one = _index(T)
    assert me.classify(_group(one), one, frozenset()) == ("single", None)


def test_final_pending_waits_for_the_runners_finalize():
    idx = _index(T, B, A)
    g = _group(idx)
    pub = me._epoch(idx[T]["generated"])
    del idx[T]["final"]                     # T just ended: the runner's finalize is running
    kind, why = me.classify(g, idx, frozenset(), now=pub + 3600)
    assert kind == "skip" and why.startswith(f"final pending: {T}") and me.waits(why)
    assert "1.0 h ago" in why
    assert me.classify(g, idx, frozenset(), now=pub + me.FINAL_WAIT_H * 3600) == ("merge", None)
    idx[T]["reopened"] = me._iso(pub + 3 * 3600)      # reopened and ended again: from then
    assert me.classify(g, idx, frozenset(), now=pub + 4 * 3600)[0] == "skip"
    idx[T]["final"] = {"ended": "2026-10-28T02:00Z", "n_frames": 4}   # the final landed
    assert me.classify(g, idx, frozenset(), now=pub + 600) == ("merge", None)
    del idx[B]["final"]                     # an older member without a final holds nothing
    assert me.classify(g, idx, frozenset(), now=pub + 600) == ("merge", None)


def test_grid_check():
    gr = {"nx": 6, "ny": 5, "transform": [0.1, 0.0, -100.0, 0.0, -0.1, 40.0],
          "crs": "EPSG:4269"}
    same = {T: {"grid": dict(gr)}, B: {"grid": json.loads(json.dumps(gr))}}
    assert me.grid_problem(same) is None
    moved = {T: {"grid": gr}, B: {"grid": {**gr, "transform": [0.1, 0.0, -100.1, 0.0, -0.1, 40.0]}}}
    assert "mismatch" in me.grid_problem(moved)
    assert "mismatch" in me.grid_problem({T: {"grid": gr}, B: {"grid": {**gr, "crs": "EPSG:4326"}}})
    assert "mismatch" in me.grid_problem({T: {"grid": gr}, B: {"grid": {**gr, "nx": 7}}})
    assert "no grid" in me.grid_problem({T: {"grid": gr}, B: {}})


def test_cap_change_rerender_selection():
    plan = [(T0, A, _fr(T0)), (T1, B, _fr(T1)), (T2, T, _fr(T2)), (T3, T, _fr(T3))]
    pngs = {A: {_fr(T0)["png"]}, B: {_fr(T1)["png"]}, T: {_fr(T2)["png"], _fr(T3)["png"]}}
    caps = {A: 3.0, B: 1.2, T: 1.2}
    assert me.rerender(plan, caps, 1.2, pngs) == [_fr(T0)["file"]]
    assert me.rerender(plan, caps, 3.0, pngs) == [_fr(t)["file"] for t in (T1, T2, T3)]
    # equal to adaptive_cap's 2 decimals = same scale
    assert me.rerender(plan, {A: 3.0, B: 1.2, T: 1.2000001}, 1.2, pngs) == [_fr(T0)["file"]]
    # a source frame without its PNG is rendered whatever the cap
    pngs[T].discard(_fr(T3)["png"])
    assert me.rerender(plan, caps, 1.2, pngs) == [_fr(T0)["file"], _fr(T3)["file"]]
    # caps fall back like the legend does
    assert me.cap_of({}) == eventsim.DEPTH_CAP_M == me.cap_of({"depth_cap_m": "nan"})
    assert me.cap_of({"depth_cap_m": 1.37}) == 1.37


def test_hydro_merged_by_time_with_preference():
    rows = {T: [{"time": "2026-10-27 11:00", "sim_q": 3.0}, {"time": "2026-10-27 12:00", "sim_q": 4.0}],
            B: [{"time": "2026-10-27 10:00", "sim_q": 20.0}, {"time": "2026-10-27 11:00", "sim_q": 30.0}],
            A: [{"time": "2026-10-27T09:00Z", "sim_q": 100.0}, {"time": "2026-10-27T10:00Z", "sim_q": 200.0},
                {"time": "garbage", "sim_q": -1.0}]}
    out = me.merge_hydro(rows, me.preference(T, rows))
    assert [r["sim_q"] for r in out] == [100.0, 20.0, 3.0, 4.0, -1.0]


def test_archive_times_follow_the_frame_sources():
    frame_src = {T0: A, T1: B, T2: T, T3: T}
    times = {T: {T2, T3}, B: {T1, T2, TX}, A: {T0, T1, TX}}
    take = me.archive_takes(frame_src, times, [T, B, A])
    assert take == {T: {T2, T3}, B: {T1, TX}, A: {T0}}
    # the frame's source has no rows there (dry frame): no member fills in
    assert me.archive_takes({T1: T}, {T: set(), B: {T1}}, [T, B]) == {}


def test_merged_manifest_fields():
    idx = _index(T, B, A)
    mans = {m: {"event_id": m, "t0": idx[m]["t0"], "sim_start": idx[m]["sim_start"],
                "t_end": idx[m]["t_end"], "status": "ended", "dry": True,
                "provenance": {"engine": "gpu"}, "maxspeed": "maxspeed.tif",
                "frames": []} for m in (T, B, A)}
    plan = [(T0, A, _fr(T0)), (T2, T, _fr(T2))]
    man = me.merged_manifest(T, mans, idx, plan, [{"time": "x"}], 2.5,
                             "2026-11-03T21:00:00Z", archive_frames=2)
    assert [f["t"] for f in man["frames"]] == [T0, T2]
    assert man["sim_start"] == idx[A]["sim_start"] and man["t_end"] == idx[T]["t_end"]
    assert man["episode_started"] == idx[A]["episode_started"]
    assert man["episode_start"] == T0
    assert man["status"] == "ended" and man["generated"] == "2026-11-03T21:00:00Z"
    assert man["merged_from"] == [B, A] and man["depth_cap_m"] == 2.5
    assert man["archive_frames"] == 2 and "dry" not in man
    m = man["provenance"]["merged"]
    assert m["members"] == [T, B, A] and m["script"] == me.SCRIPT and m["rule"]
    assert "maxspeed" in m and man["provenance"]["engine"] == "gpu"
    assert mans[T]["frames"] == [] and mans[T]["status"] == "ended"   # input untouched


def test_index_rewrite():
    idx = _index(T, B, A)
    before = json.dumps(idx)
    g = _group(idx)
    man = {"frames": [_fr(T0), _fr(T1)], "t0": idx[T]["t0"], "sim_start": idx[A]["sim_start"],
           "t_end": idx[T]["t_end"], "generated": "2026-11-03T21:00:00Z", "gauge": G,
           "trigger": {"gauge": G}, "episode_started": idx[A]["episode_started"],
           "archive_frames": 2, "provenance": {"engine": "gpu"}}
    entry = me.index_entry(idx[T], man, eventlist.merged_record(g))
    new = me.rewrite_index(idx, T, [B, A], entry)
    assert json.dumps(idx) == before                       # input untouched
    assert list(new) == list(idx)                          # list position kept
    t = new[T]
    assert "final" not in t and t["status"] == "ended" and t["n_frames"] == 2
    assert t["episode_started"] == idx[A]["episode_started"] and t["engine"] == "gpu"
    assert t["merged"] == {"n": 3, "ids": [T, B, A], "first_start": idx[A]["episode_started"],
                           "last_end": idx[T]["final"]["ended"]}
    for m in (B, A):
        assert new[m]["merged_into"] == T
        assert new[m]["status"] == "ended" and new[m]["final"] == idx[m]["final"]
    assert new["2026102500_07000000"] == idx["2026102500_07000000"]


def test_merged_group_is_skipped_on_rerun():
    idx = _index(T, B, A)
    g = _group(idx)
    entry = me.index_entry(idx[T], {"frames": [_fr(T0)], "episode_started": idx[A]["t0"],
                                    "generated": idx[T]["generated"]},
                           eventlist.merged_record(g))
    new = me.rewrite_index(idx, T, [B, A], entry)
    g2 = _group(new)
    assert g2["members"] == [T] and set(g2["ids"]) == {T, B, A}
    assert me.classify(g2, new, frozenset()) == ("merged", None)
    # no other group of the index turns mergeable either
    assert all(me.classify(x, new, frozenset())[0] != "merge" for x in eventlist.groups(new, 12))
    # and a plan made before someone else's merge aborts at commit time
    why, _ = me.recheck({m: idx[m]["generated"] for m in (T, B, A)}, new)
    assert why and "merged into" in why
    # the list shows the one merged entry
    events, info = eventlist.list_view(new, dt.datetime(2026, 10, 29), max_age_d=0, merge_gap_h=12)
    assert T in events and B not in events and events[T]["merged"]["n"] == 3
    assert info["absorbed"] == 2


def test_recheck_aborts_when_a_member_changed():
    idx = _index(T, B, A)
    planned = {m: idx[m]["generated"] for m in (T, B, A)}
    why, g = me.recheck(planned, idx)
    assert why is None and set(g["members"]) == {T, B, A}
    reopened = json.loads(json.dumps(idx))
    reopened[T]["status"] = "active"
    assert "active" in me.recheck(planned, reopened)[0]
    repub = json.loads(json.dumps(idx))
    repub[B]["generated"] = "2026-11-03T20:00:00Z"
    assert "re-published" in me.recheck(planned, repub)[0]
    joined = dict(idx)                       # a new fragment joined the group
    joined[f"2026102801_{G}"] = _entry(f"2026102801_{G}")
    assert "group" in me.recheck(planned, joined)[0]
    gone = {k: v for k, v in idx.items() if k != A}
    assert "left the index" in me.recheck(planned, gone)[0]


# ---- products + commit against a local store ---------------------------------

GRID = {"ny": 5, "nx": 6, "dx_m": 25.0, "dy_m": 30.0,
        "transform": [1 / 3600, 0.0, -100.0, 0.0, -1 / 3600, 40.0], "crs": "EPSG:4269"}


class FakeStore:
    """events/<id>/<files> under a local root; .png/.parquet count as LFS."""

    def __init__(self, root):
        self.root = root
        self.api = None

    def _dir(self, ev):
        return os.path.join(self.root, "events", ev)

    def tree(self, ev, recursive=False):
        out, base = {}, self._dir(ev)
        for dp, dns, fns in os.walk(base):
            if not recursive:
                dns[:] = []
            for fn in fns:
                p = os.path.join(dp, fn)
                rel = os.path.relpath(p, base).replace(os.sep, "/")
                with open(p, "rb") as fp:
                    ident = hashlib.sha256(fp.read()).hexdigest()
                out[rel] = me.FileInfo(os.path.getsize(p), ident,
                                       fn.endswith((".png", ".parquet")))
        return out

    def manifest(self, ev):
        try:
            with open(os.path.join(self._dir(ev), "manifest.json"), encoding="utf-8") as fp:
                return json.load(fp)
        except OSError:
            return None

    def fetch(self, ev, name, dest):
        out = os.path.join(dest, "events", ev, name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(os.path.join(self._dir(ev), name), out)
        return out


def _tif(path, a, dtype="uint16", nodata=None):
    import rasterio
    from rasterio.transform import Affine
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1],
                       count=1, dtype=dtype, crs=GRID["crs"], nodata=nodata,
                       transform=Affine(*GRID["transform"]), compress="deflate") as ds:
        ds.write(np.asarray(a, dtype=dtype), 1)


def _archive(path, ev, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    ts = [dt.datetime.strptime(t, "%Y-%m-%dT%H:%MZ") for t, _, _, _ in rows]
    tb = pa.table({"time": pa.array(ts, pa.timestamp("s")),
                   "row": pa.array([r for _, r, _, _ in rows], pa.uint16()),
                   "col": pa.array([c for _, _, c, _ in rows], pa.uint16()),
                   "depth_cm": pa.array([d for _, _, _, d in rows], pa.uint16())})
    tb = tb.replace_schema_metadata({b"grid": json.dumps(GRID).encode(),
                                     b"depth_unit": b"centimeters", b"event_id": ev.encode()})
    pq.write_table(tb, path)


def _grid(**cells):
    a = np.zeros((GRID["ny"], GRID["nx"]), np.uint16)
    for k, v in cells.items():
        r, c = map(int, k[1:].split("_"))
        a[r, c] = v
    return a


@pytest.fixture
def store(tmp_path):
    """Three ended episodes of one flood. T (target, cap 0.5) has T2, T3; B has
    T1, T2; A has T0, T1. B's T2 and A's T1 carry deep cells that the merge
    must NOT see (newer frames win those times)."""
    deep = _grid(**{f"c{r}_{c}": 300 for r in range(2) for c in range(6)})  # 12 wet cells
    frames = {T: {T2: deep, T3: deep},
              B: {T1: _grid(c3_3=250), T2: _grid(c2_2=600)},
              A: {T0: _grid(c4_4=100), T1: _grid(c4_0=900)}}
    caps = {T: 0.5, B: 3.0, A: 3.0}
    speed = {T: _grid(c0_0=100), B: _grid(c0_0=200, c1_1=50), A: _grid(c0_0=200, c2_2=70)}
    sector = {T: 1, B: 2, A: 3}
    archives = {T: [(T2, 0, 0, 300), (T3, 0, 1, 300)],
                B: [(TX, 1, 1, 7), (T1, 3, 3, 250), (T2, 2, 2, 600)],
                A: [(TX, 4, 4, 9), (T0, 4, 4, 100), (T1, 4, 0, 900)]}
    hydro = {T: [{"time": "2026-10-27 11:00", "sim_q": 1.0}],
             B: [{"time": "2026-10-27 10:00", "sim_q": 2.0}, {"time": "2026-10-27 11:00", "sim_q": 9.0}],
             A: [{"time": "2026-10-27 09:00", "sim_q": 3.0}]}
    for ev in (T, B, A):
        d = tmp_path / "events" / ev
        d.mkdir(parents=True)
        fl = []
        for t, a in sorted(frames[ev].items()):
            fr = _fr(t)
            _tif(str(d / fr["file"]), a)
            (d / fr["png"]).write_bytes(f"png {ev} {t}".encode())
            fl.append(fr)
        md = np.max(np.stack(list(frames[ev].values())), axis=0)
        _tif(str(d / "maxdepth.tif"), md)
        (d / "maxdepth.png").write_bytes(f"maxpng {ev}".encode())
        _tif(str(d / "maxspeed.tif"), speed[ev])
        _tif(str(d / "flowdir.tif"), np.where(speed[ev] > 0, sector[ev], 255), "uint8", 255)
        _archive(str(d / "archive.parquet"), ev, archives[ev])
        (d / "final.json").write_text(json.dumps({"event_id": ev}))
        man = {"event_id": ev, "grid": GRID, "frames": fl, "depth_cap_m": caps[ev],
               "maxdepth": "maxdepth.tif", "maxspeed": "maxspeed.tif",
               "flowdir": "flowdir.tif", "hydro": hydro[ev], "status": "ended",
               "provenance": {"engine": "gpu"}, **{k: v for k, v in _entry(ev).items()
                                                   if k in ("t0", "sim_start", "t_end")}}
        (d / "manifest.json").write_text(json.dumps(man))
    return FakeStore(str(tmp_path))


def _products(store, tmp_path, idx=None):
    idx = idx or _index(T, B, A)
    plan = me.plan_group(store, _group(idx), idx)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return idx, plan, me.build_products(store, plan, str(work), idx, log=lambda s: None)


def _read(p):
    import rasterio
    with rasterio.open(p) as ds:
        return ds.read(1), ds.profile


def test_products_end_to_end(store, tmp_path):
    idx, plan, prod = _products(store, tmp_path)
    assert plan["target"] == T and plan["others"] == [B, A]
    assert [(t, s) for t, s, _ in plan["frames"]] == [(T0, A), (T1, B), (T2, T), (T3, T)]
    render = os.path.join(prod["work"], "render")
    md, prof = _read(os.path.join(render, "maxdepth.tif"))
    want = np.maximum.reduce([_grid(**{f"c{r}_{c}": 300 for r in range(2) for c in range(6)}),
                              _grid(c3_3=250), _grid(c4_4=100)])
    assert (md == want).all()                 # B's T2 (600) and A's T1 (900) lost to newer frames
    assert prof["dtype"] == "uint16" and prof["compress"] == "deflate"
    assert prod["cap"] == eventsim.adaptive_cap(os.path.join(render, "maxdepth.tif")) == 3.0
    # T's frames were coloured at 0.5 m: re-rendered at 3.0; A/B's (3.0) are copied
    assert prod["rerender"] == [_fr(T2)["file"], _fr(T3)["file"]]
    sp, _ = _read(os.path.join(render, "maxspeed.tif"))
    fd, fprof = _read(os.path.join(render, "flowdir.tif"))
    assert sp[0, 0] == 200 and fd[0, 0] == 2          # B beats T; A ties B: no win
    assert sp[1, 1] == 50 and fd[1, 1] == 2
    assert sp[2, 2] == 70 and fd[2, 2] == 3
    assert fd[4, 4] == 255 and fprof["nodata"] == 255 and fprof["dtype"] == "uint8"
    import pyarrow.parquet as pq
    arch = pq.read_table(os.path.join(render, "archive.parquet")).to_pylist()
    got = sorted((_h(r["time"]), r["row"], r["col"], r["depth_cm"]) for r in arch)
    assert got == sorted([(TX, 1, 1, 7), (T0, 4, 4, 100), (T1, 3, 3, 250),
                          (T2, 0, 0, 300), (T3, 0, 1, 300)])
    assert prod["archive"]["frames"] == 5
    meta = pq.read_schema(os.path.join(render, "archive.parquet")).metadata
    assert json.loads(meta[b"grid"]) == GRID and meta[b"event_id"] == T.encode()
    man = prod["manifest"]
    assert [f["t"] for f in man["frames"]] == [T0, T1, T2, T3]
    assert [r["sim_q"] for r in man["hydro"]] == [3.0, 2.0, 1.0]
    assert man["depth_cap_m"] == 3.0 and man["merged_from"] == [B, A]
    assert man["episode_started"] == idx[A]["episode_started"]
    with open(os.path.join(prod["work"], "manifest.json")) as fp:
        assert json.load(fp) == man

    ops = {o[2]: o for o in prod["ops"]}
    tdir = f"events/{T}"
    # every operation stays in the target folder; members' folders never touched
    assert all(d.startswith(tdir + "/") for d in ops)
    assert {d for d in ops if "/premerge/" in d} == {f"{tdir}/premerge/{n}" for n in me.BACKUP}
    assert all(ops[f"{tdir}/premerge/{n}"][:2] == ("copy", f"{tdir}/{n}") for n in me.BACKUP)
    # added frames: tif re-uploaded from the bytes read, png copied server side
    for t, src in ((T0, A), (T1, B)):
        fr = _fr(t)
        assert ops[f"{tdir}/{fr['file']}"][0] == "add"
        assert ops[f"{tdir}/{fr['png']}"][:2] == ("copy", f"events/{src}/{fr['png']}")
        assert ops[f"{tdir}/{fr['png']}"][4] is True               # LFS: server side
    for t in (T2, T3):                                             # re-rendered own PNGs
        assert ops[f"{tdir}/{_fr(t)['png']}"][0] == "add"
        assert f"{tdir}/{_fr(t)['file']}" not in ops               # tifs never rewritten
    # existing target files written: only the recomputed products + re-rendered PNGs
    existing = set(store.tree(T, recursive=True))
    overwritten = {d[len(tdir) + 1:] for d in ops if d[len(tdir) + 1:] in existing}
    assert overwritten == {"manifest.json", "maxdepth.tif", "maxdepth.png", "maxspeed.tif",
                           "flowdir.tif", "archive.parquet", _fr(T2)["png"], _fr(T3)["png"]}
    lines = me.describe(plan, prod, eventlist.merged_record(_group(idx)))
    assert any("2 own + 1 from " + B in ln for ln in lines)


def test_existing_backup_and_identical_frame_are_kept(store, tmp_path):
    d = os.path.join(store.root, "events", T)
    os.makedirs(os.path.join(d, "premerge"))
    shutil.copyfile(os.path.join(d, "manifest.json"), os.path.join(d, "premerge", "manifest.json"))
    fr = _fr(T0)                     # A's T0 frame already sits in the target, same bytes
    shutil.copyfile(os.path.join(store.root, "events", A, fr["file"]), os.path.join(d, fr["file"]))
    _, plan, prod = _products(store, tmp_path)
    dests = [o[2] for o in prod["ops"]]
    assert f"events/{T}/premerge/manifest.json" not in dests     # first merge's backup kept
    assert f"events/{T}/premerge/final.json" in dests
    assert f"events/{T}/{fr['file']}" not in dests               # referenced as is
    assert fr["file"] in plan["reuse"]


def test_different_file_in_target_skips_the_group(store, tmp_path):
    d = os.path.join(store.root, "events", T)
    _tif(os.path.join(d, _fr(T0)["file"]), _grid(c0_0=1))     # stale, different
    idx = _index(T, B, A)
    with pytest.raises(me.Skip, match="different"):
        me.plan_group(store, _group(idx), idx)


def test_grid_mismatch_and_no_frames_skip(store, tmp_path):
    p = os.path.join(store.root, "events", B, "manifest.json")
    with open(p, encoding="utf-8") as fp:
        man = json.load(fp)
    with open(p, "w", encoding="utf-8") as fp:
        json.dump({**man, "grid": {**GRID, "nx": 7}}, fp)
    idx = _index(T, B, A)
    with pytest.raises(me.Skip, match="grid mismatch"):
        me.plan_group(store, _group(idx), idx)
    with open(p, "w", encoding="utf-8") as fp:
        json.dump(man, fp)
    for ev in (T, B, A):                                   # animation stripped everywhere
        for fn in os.listdir(os.path.join(store.root, "events", ev)):
            if fn.startswith("depth_"):
                os.remove(os.path.join(store.root, "events", ev, fn))
    with pytest.raises(me.Skip, match="no member has stored depth frames"):
        me.plan_group(store, _group(idx), idx)


class _Info:
    oid = "abc123def4567890"


class FakeApi:
    """create_commit fails once with a concurrent-commit 412, then lands."""

    def __init__(self, store, idx):
        self.store, self.idx, self.commits, self.heads, self.msgs = store, idx, [], 0, []

    def preupload_lfs_files(self, repo_id, additions, repo_type=None):
        self.preuploaded = [a.path_in_repo for a in additions]

    def create_commit(self, repo_id, operations, commit_message, repo_type=None,
                      parent_commit=None):
        import httpx
        from huggingface_hub.utils import HfHubHTTPError
        self.commits.append((parent_commit, list(operations)))
        self.msgs.append(commit_message)
        if len(self.commits) == 1:
            r = httpx.Response(412, request=httpx.Request("POST", "https://hub.invalid"))
            raise HfHubHTTPError("A commit has happened since", response=r)
        return _Info()


def test_commit_is_one_atomic_commit_and_retries_on_a_race(store, tmp_path, monkeypatch):
    idx, plan, prod = _products(store, tmp_path)
    api = FakeApi(store, idx)
    store.api = api
    heads = iter(["sha1", "sha2"])
    store.head = lambda: next(heads)
    store.index_at = lambda sha: json.loads(json.dumps(idx))
    monkeypatch.setattr(me.time, "sleep", lambda s: None)
    assert me.commit_group(store, plan, prod, 12, log=lambda s: None)
    assert [c[0] for c in api.commits] == ["sha1", "sha2"]      # re-read after the race
    ops = api.commits[-1][1]
    paths = [getattr(o, "path_in_repo") for o in ops]
    assert paths[-1] == "events/index.json" and len(paths) == len(prod["ops"]) + 1
    with open(ops[-1].path_or_fileobj, encoding="utf-8") as fp:
        new = json.load(fp)
    assert new[T]["merged"]["ids"] == [T, B, A] and "final" not in new[T]
    assert new[B]["merged_into"] == T and new[A]["merged_into"] == T
    assert new[T]["n_frames"] == 4 and new[T]["episode_started"] == idx[A]["episode_started"]
    kinds = {type(o).__name__ for o in ops}
    assert kinds == {"CommitOperationAdd", "CommitOperationCopy"}


def _land(store, prod):
    """What the merge commit does to the local store: its copies and uploads."""
    for kind, src, dest, _, _ in prod["ops"]:
        out = os.path.join(store.root, *dest.split("/"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(os.path.join(store.root, *src.split("/")) if kind == "copy" else src, out)


def test_reapply_after_a_revert_reuses_the_merged_folder(store, tmp_path, monkeypatch):
    """A stale index write put the pre-merge entries back (folders keep the
    merge): the re-apply copies nothing, backs nothing up again (premerge/
    holds the first merge's state; archive.parquet came from the merge,
    final.json from the post-merge finalize) and rebuilds the same products."""
    os.remove(os.path.join(store.root, "events", T, "archive.parquet"))   # the merge writes one
    idx, plan, prod = _products(store, tmp_path)
    assert "archive.parquet" not in plan["backup"] and "final.json" in plan["backup"]
    _land(store, prod)
    md1, _ = _read(os.path.join(prod["work"], "render", "maxdepth.tif"))
    with open(os.path.join(store.root, "events", T, "final.json"), "w") as fp:
        json.dump({"event_id": T, "n_frames": 4}, fp)                     # post-merge finalize
    shutil.rmtree(tmp_path / "work")
    _, plan2, prod2 = _products(store, tmp_path, idx)                      # index: pre-merge
    assert [(t, s) for t, s, _ in plan2["frames"]] == [(T0, T), (T1, T), (T2, T), (T3, T)]
    assert plan2["backup"] == [] and plan2["reuse"] == set() and prod2["rerender"] == []
    assert {o[2].split("/", 2)[2] for o in prod2["ops"]} == {
        "maxdepth.tif", "maxdepth.png", "maxspeed.tif", "flowdir.tif", "archive.parquet",
        "manifest.json"}                                                   # no copy, no premerge/
    md2, _ = _read(os.path.join(prod2["work"], "render", "maxdepth.tif"))
    assert (md1 == md2).all() and prod2["cap"] == prod["cap"]
    import pyarrow.parquet as pq

    def rows(p):
        return sorted(map(str, pq.read_table(p).to_pylist()))

    assert rows(os.path.join(prod2["work"], "render", "archive.parquet")) == rows(
        os.path.join(store.root, "events", T, "archive.parquet"))   # the merged one, unchanged
    assert prod2["archive"]["frames"] == prod["archive"]["frames"] == 3
    assert [f["t"] for f in prod2["manifest"]["frames"]] == [T0, T1, T2, T3]
    assert any("nothing new" in ln for ln in me.describe(plan2, prod2, eventlist.merged_record(
        _group(idx))))
    # the commit says what it is
    plan2["reapply"] = True
    store.api = api = FakeApi(store, idx)
    heads = iter(["sha1", "sha2"])
    store.head = lambda: next(heads)
    store.index_at = lambda sha: json.loads(json.dumps(idx))
    monkeypatch.setattr(me.time, "sleep", lambda s: None)
    assert me.commit_group(store, plan2, prod2, 12, log=lambda s: None)
    assert api.msgs[-1].startswith(f"re-apply merge episodes {T} <- {B}, {A}")


def test_commit_aborts_when_a_member_changed(store, tmp_path):
    idx, plan, prod = _products(store, tmp_path)
    api = FakeApi(store, idx)
    store.api = api
    store.head = lambda: "sha1"
    changed = json.loads(json.dumps(idx))
    changed[B]["status"] = "active"
    store.index_at = lambda sha: changed
    msgs = []
    assert not me.commit_group(store, plan, prod, 12, log=msgs.append)
    assert api.commits == [] and "ABORT" in msgs[-1]
