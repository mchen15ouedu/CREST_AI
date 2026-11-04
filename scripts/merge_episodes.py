"""Merge of a fragmented flood's episode RESULTS on the CREST_data store
(PI 2026-11-03: "we need a script that does this result merging on HF", then
"has to run automatically all the time"). By hand: read the dry-run plan,
--apply. Always on: --watch (below), the child process the runner Space
supervises (hf_data/eventmerger.py, EPISODE_MERGER=1); not a tick step.

Why: until the hourly tick reopened episodes (eventsim EVENT_REOPEN_H) a
gauge hovering at the continuation bar was ended one hour and re-triggered
the next, so one flood became many episodes (09353800 x12, 08353000 x4,
2026-10-27..31). The public list folds them as a VIEW (hf_data/eventlist.py);
this script folds the STORE, so each flood is one episode with one record.

Groups: eventlist.groups() over the index (merged_into entries are never
members), gap --gap-h (default EVENT_LIST_MERGE_GAP_H): exactly the list's
folding. A group is merged only when it has >= 2 members, every member is
'ended' (an active one: re-run after it ends), none is pinned, none has a
spec in the worker queue (its publish would re-list the member), the newest
member has a final or was last published FINAL_WAIT_H ago ("final pending":
the runner's finalize of the episode it just ended lands first), and every
member's manifest grid (nx, ny, transform, crs) is identical. Each skip is
reported with its reason.

Per group, target = the newest member (largest id):
  frames    the target's own stored frames win at every time it has; other
            times come from the newest other member that has them, copied
            into the target folder under their own names (LFS files with
            CommitOperationCopy, server side; regular files re-uploaded from
            the bytes already read for the max). An existing target file is
            never overwritten: same content is referenced as is, different
            content skips the group.
  backups   manifest.json, final.json, maxdepth.tif/.png, maxspeed.tif,
            flowdir.tif, archive.parquet -> events/<target>/premerge/<name>
            (written only when absent: premerge/ keeps the state before the
            FIRST merge). Frame PNGs are not backed up: they are renders of
            the frame tifs (never modified) at premerge/manifest.json's
            depth_cap_m, and a depth_* name under premerge/ would pass for a
            frame in eventstore's basename-keyed publish/reconcile passes.
  maxdepth  cellwise max of the merged frame tifs, streamed (one running
            uint16 grid), written with the target maxdepth.tif's profile;
            depth_cap_m = eventsim.adaptive_cap of it (the episode rule), and
            eventsim._make_pngs renders maxdepth.png plus every merged frame
            PNG whose source member's cap differs (or that is not stored), so
            the animation has one colour scale.
  maxspeed  cellwise max of the members' maxspeed.tif (each covers its
            member's last solve window); flowdir.tif follows the member that
            holds each cell's max (target first on ties). Recorded in
            provenance.merged.maxspeed.
  archive   archive.parquet merged by time: a frame time's rows come from the
            member that supplies the frame (none there = dry then); times
            only archives hold, from the target, else the newest member.
  hydro     manifest.hydro rows merged by time with the same preference.
  manifest  frames by time, sim_start earliest, episode_started the earliest
            member's, t_end latest, status ended, generated now, merged_from,
            provenance.merged = {script, at, members, rule, maxspeed}.
  index     ONE commit per group with the copies, uploads and the index
            (256 commits/h per account). The index is read-modify-committed
            under eventstore._lock against the head revision (parent_commit):
            a concurrent commit (live runner, GPU worker) retries on a fresh
            read; a member whose status, generated or merged_into changed
            abandons the group. Target entry refreshed from the new manifest,
            'final' dropped, merged = eventlist.merged_record; the other
            members get merged_into = target (status unchanged). Their
            folders are never modified.
Then eventstore.finalize_episodes([target]) scores the merged episode
window (--no-finalize skips it; the runner's heal pass takes ended episodes
without a final for 3 days). A merged target whose final counts other than
its merged frames (a finalize that read the pre-merge manifest landed after
the merge) has that final dropped (index commit) and is finalized again.

Idempotent and resumable: the commit is atomic, so a failed run leaves
nothing behind and is simply re-run; a merged group's members carry
merged_into, so it has one member left and is reported "already merged"
(its finalize is resumed when the target still has no final).

--watch (with --apply) loops forever: every --interval s (300) it reads
index.json's blob id (one paths-info call, no download) and runs a pass when
it changed, or when groups were left waiting (an episode not ended, a worker
spec queued, the pinned set or a manifest unreadable, the rate cap, a failed
merge) and --retry-min (60) minutes passed. At most --max-merges-per-hour (6)
merges in any hour, restarts included (the status file keeps them): HF allows
256 commits/h per account, shared with the runner, the worker and the fleet,
and a merge costs 2 (merge + finalize). A group a plan skipped for good (grid
mismatch, no frames, a conflicting target file) is re-planned only when a
member is re-published or after SKIP_RECHECK_H. A failing pass is logged and
retried with backoff (1 min doubling to 60 min, reset on success);
SIGTERM/SIGINT stop it between groups, never mid-commit. The live runner and
worker commit index.json without parent_commit, so a write from a copy read
before a merge silently reverts it: each pass checks the recent merges and
re-applies a reverted one (outside the hourly cap, REAPPLY_FREE times per
target; the files already in the target folder are reused, premerge/ is not
touched again). Every Hub call has finite timeouts (bound_hub_timeouts) and
the status file's 'updated' is refreshed at least every --interval, per
group and per download: the supervisor's watchdog restarts a child whose
heartbeat stops. stdout: one line per pass (UTC). --status-file (default
<CREST_CACHE_DIR or tempdir>/merger_status.json, replaced atomically):
started, pid, state, updated, last_pass, waiting, last_error, next_pass_at,
recent_merges (20); while it can't be written, <tempdir>/
merger_status.fallback.json (+ status_fallback), and when neither can,
STATUS_FAILS_MAX times in a row, --watch stops and exits STATUS_EXIT.

    python scripts/merge_episodes.py [--dry-run | --apply] [--gauge G]
        [--max-groups N] [--gap-h H] [--no-finalize]
    python scripts/merge_episodes.py --watch --apply [--interval S]
        [--retry-min M] [--max-merges-per-hour N] [--status-file F]
HF_TOKEN (env, never printed) is needed for --apply; the dry run reads the
public dataset and downloads what the merge would read (frames for the max,
archives) into a temp dir it removes, so its plan is the real one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hf_data import eventlist, eventsim, eventstore  # noqa: E402

P = eventstore.PREFIX
SCRIPT = "scripts/merge_episodes.py"
RULE = ("target = newest member; its own frames win at every time it has, "
        "other times from the newest member that has them; hydro rows and "
        "archive times by the same preference; maxdepth = cellwise max of "
        "the merged frames, depth_cap_m by eventsim.adaptive_cap")
MAXSPEED_NOTE = ("cellwise max of the members' maxspeed.tif (each member's "
                 "last solve window); flowdir.tif from the member holding "
                 "each cell's max, target first on ties")
BACKUP = ("manifest.json", "final.json", "maxdepth.tif", "maxdepth.png",
          "maxspeed.tif", "flowdir.tif", "archive.parquet")
COMMIT_TRIES = 5
DL_WORKERS = 4
FINAL_WAIT_H = 2.0           # an ended newest member without a final waits this long
_TS = "%Y-%m-%dT%H:%M:%SZ"


class Skip(Exception):
    """The group cannot be merged as it stands; the message says why."""


# ---- pure plan helpers (tests/test_merge_episodes.py) -----------------------

def preference(target: str, members) -> list:
    """The target first, then the other members newest first."""
    return [target] + sorted((m for m in members if m != target), reverse=True)


def classify(g: dict, idx: dict, pins, queued=frozenset(), now=None):
    """(kind, why) for one group of eventlist.groups(): "merge" (eligible as
    far as the index tells), "merged" (one member left that stands for
    absorbed episodes: merged before), "single" (nothing to merge) or
    "skip" with the reason. pins None = the pinned set could not be read.
    now (epoch s, default the clock): the newest member without a final
    waits ("final pending") until FINAL_WAIT_H after its last publish
    (generated, reopened; the index keeps no end time before the final and
    the end is no earlier), so the runner's finalize of the episode it just
    ended lands before the merge, not over it."""
    members = list(g["members"])
    if len(members) < 2:
        return ("merged", None) if len(g["ids"]) > 1 else ("single", None)
    live = [m for m in members if (idx.get(m) or {}).get("status") != "ended"]
    if live:
        return "skip", (f"not ended: {', '.join(live)}; re-run after the "
                        f"episode ends")
    if pins is None:
        return "skip", "pinned set unreadable; demo events can't be told apart"
    pin = [m for m in members if m in pins]
    if pin:
        return "skip", f"pinned: {', '.join(pin)}"
    q = [m for m in members if m in queued]
    if q:
        return "skip", (f"worker queue holds a spec for {', '.join(q)} "
                        f"(its publish would re-list it); re-run later")
    newest = max(members)
    s = idx.get(newest) or {}
    pub = [t for t in (_epoch(s.get("generated")), _epoch(s.get("reopened"))) if t is not None]
    now = time.time() if now is None else now
    if not s.get("final") and pub and now - max(pub) < FINAL_WAIT_H * 3600:
        return "skip", (f"final pending: {newest} ended without a final (last published "
                        f"{max(0.0, now - max(pub)) / 3600:.1f} h ago); its finalize "
                        f"lands first")
    return "merge", None


def stale_final(s) -> str | None:
    """Why a merged target's final is stale, or None: its n_frames is not the
    entry's (the merged frame count; a finalize that read the pre-merge
    manifest landed after the merge)."""
    fin = s.get("final") if isinstance(s, dict) else None
    if not isinstance(fin, dict) or not isinstance(s.get("merged"), dict):
        return None
    n, want = fin.get("n_frames"), s.get("n_frames")
    if n is None or want is None or n == want:
        return None
    return f"final counts {n} frames, merged {want}"


def reverted(recent, idx: dict, news=None) -> dict:
    """{target: record} of the merges in `recent` (the --watch status
    records, oldest first; the last one of a target decides) that the index
    no longer holds: the live runner and worker commit index.json without
    parent_commit, so a write from a copy read before the merge puts the
    target back without its merged record and the members without
    merged_into. A revert restores what the merge read: the target's
    generated is its pre-merge or the merge's one AND every member still
    without merged_into has its pre-merge generated. Not a revert: the merge
    in place, the target or a member re-published since (another generated:
    new content, merged again as an ordinary merge, within the hourly cap;
    `news`, a dict when given, gets {target: why} for those), absorbed into
    a later merge or gone, no member left, a record without the pre-merge
    generated."""
    out, news = {}, ({} if news is None else news)
    for r in recent or ():
        t = r.get("target") if isinstance(r, dict) else None
        s = idx.get(t)
        gens = r.get("generated") if isinstance(r, dict) else None
        others = [m for m in (r.get("members") or []) if m != t
                  and isinstance(idx.get(m), dict)] if t else []
        out.pop(t, None)
        news.pop(t, None)
        if not isinstance(s, dict) or not isinstance(gens, dict) or not others \
                or s.get("merged_into"):
            continue
        rec = s.get("merged") if isinstance(s.get("merged"), dict) else {}
        ids = set(rec.get("ids") or [])
        if rec and all(m in ids and eventlist._into(idx[m]) == t for m in others):
            continue                         # in place
        gen = s.get("generated")
        if gen is None:
            continue
        if gen not in (gens.get(t), r.get("merged_gen"))                 and all(eventlist._into(idx[m]) == t for m in others):
            continue                         # held: t re-published since, members still in
        new = [m for m in others if not eventlist._into(idx[m])
               and idx[m].get("generated") != gens.get(m)]
        if gen not in (gens.get(t), r.get("merged_gen")):
            new = [t] + new
        if new:
            news[t] = "; ".join(f"{m} re-published since (generated {idx[m].get('generated')}, "
                                f"{gens.get(m)} when merged)" for m in new)
        else:
            out[t] = r
    return out


def grid_key(man: dict | None) -> tuple:
    gr = (man or {}).get("grid") or {}
    tr = gr.get("transform")
    return (gr.get("nx"), gr.get("ny"),
            tuple(tr) if isinstance(tr, (list, tuple)) else tr, gr.get("crs"))


def grid_problem(mans: dict) -> str | None:
    """None when every manifest's grid (nx, ny, transform, crs) is the same;
    else what differs."""
    keys = {m: grid_key(man) for m, man in mans.items()}
    bad = [m for m, k in keys.items() if None in k[:3]]
    if bad:
        return f"no grid in manifest: {', '.join(bad)}"
    if len(set(keys.values())) > 1:
        return "grid mismatch: " + "; ".join(
            f"{m} {k[0]}x{k[1]} {k[3]} {list(k[2])[:6]}" for m, k in keys.items())
    return None


def png_name(fr: dict) -> str:
    return fr.get("png") or fr["file"][:-4] + ".png"


def frame_plan(target: str, frames_by_member: dict) -> list:
    """[(t, source member, frame)] in time order: the target's own frames at
    every time it has, every other time from the newest other member that
    has it. frames_by_member: {id: [manifest frame dicts]} (stored only)."""
    out = {}
    for m in preference(target, frames_by_member):
        for fr in frames_by_member.get(m) or []:
            t = fr.get("t")
            if t and fr.get("file") and t not in out:
                out[t] = (t, m, fr)
    return [out[t] for t in sorted(out)]       # 'YYYY-MM-DDTHH:MMZ' sorts by time


def cap_of(man: dict | None) -> float:
    """A manifest's colour-scale cap (the 3 m fallback the legend shows)."""
    try:
        c = float((man or {}).get("depth_cap_m"))
    except (TypeError, ValueError):
        return eventsim.DEPTH_CAP_M
    return c if math.isfinite(c) and c > 0 else eventsim.DEPTH_CAP_M


def rerender(plan: list, caps: dict, cap: float, pngs: dict) -> list:
    """Frame tifs whose PNG is rendered at the merged cap: the source
    member's cap differs (to the 2 decimals adaptive_cap rounds to) or the
    source's PNG is not stored. pngs: {member: stored file names}."""
    return [fr["file"] for _, src, fr in plan
            if round(caps[src], 2) != round(cap, 2)
            or png_name(fr) not in pngs.get(src, ())]


def _hydro_t(r):
    return eventstore._parse_hydro_t(r.get("time")) if isinstance(r, dict) else None


def merge_hydro(rows_by_member: dict, pref: list) -> list:
    """manifest.hydro rows merged by time: the first member of `pref` with a
    row at a time supplies it ("2026-09-29 13:00" and "...T13:00Z" are one
    time); time order, unparsable times last."""
    seen = {}
    for m in pref:
        for r in rows_by_member.get(m) or []:
            if not isinstance(r, dict):
                continue
            t = _hydro_t(r)
            seen.setdefault(t.strftime("%Y-%m-%d %H:%M") if t else str(r.get("time")), r)
    return sorted(seen.values(),
                  key=lambda r: (_hydro_t(r) or dt.datetime.max, str(r.get("time"))))


def archive_takes(frame_src: dict, arch_times: dict, pref: list) -> dict:
    """{member: archive times whose rows it supplies}. A frame time's rows
    come from the member supplying that frame (its archive may hold none:
    the frame is dry); a time only archives hold comes from the first member
    of `pref` whose archive has it. frame_src: {t: member}."""
    take, claimed = {}, set()
    for m in pref:
        for t in sorted(arch_times.get(m) or ()):
            ok = frame_src[t] == m if t in frame_src else t not in claimed
            if ok:
                take.setdefault(m, set()).add(t)
                claimed.add(t)
    return take


def _iso_min(vals, latest=False):
    """Earliest (latest) of time strings, the original string kept."""
    ok = [(eventlist._utc(v), v) for v in vals if v]
    ok = [(t, v) for t, v in ok if t is not None]
    if not ok:
        return None
    return (max if latest else min)(ok)[1]


def merged_manifest(target: str, mans: dict, idx: dict, plan: list, hydro: list,
                    cap: float, now: str, archive_frames=None) -> dict:
    """The target's manifest rewritten for the merged episode."""
    others = [m for m in preference(target, mans) if m != target]
    man = json.loads(json.dumps(mans[target]))
    man["frames"] = [{**fr, "png": png_name(fr)} for _, _, fr in plan]
    if man["frames"]:
        man["episode_start"] = man["frames"][0]["t"]
    man["sim_start"] = _iso_min(m.get("sim_start") for m in mans.values()) \
        or man.get("sim_start")
    man["t_end"] = _iso_min((m.get("t_end") for m in mans.values()), latest=True) \
        or man.get("t_end")
    man["episode_started"] = _iso_min(
        (idx.get(m) or {}).get("episode_started") or (idx.get(m) or {}).get("t0")
        or mans[m].get("t0") for m in mans) or man.get("t0")
    man["status"] = "ended"
    man["generated"] = now
    man["merged_from"] = others
    man["hydro"] = hydro
    man["depth_cap_m"] = cap
    man["maxdepth_png"] = "maxdepth.png"
    if archive_frames is not None:
        man["archive"] = "archive.parquet"
        man["archive_frames"] = archive_frames
        if archive_frames:
            man.pop("dry", None)
    prov = dict(man.get("provenance") or {})
    prov["merged"] = {"script": SCRIPT, "at": now, "members": [target] + others,
                      "rule": RULE}
    if man.get("maxspeed"):
        prov["merged"]["maxspeed"] = MAXSPEED_NOTE
    man["provenance"] = prov
    return man


def index_entry(prev: dict, man: dict, merged: dict) -> dict:
    """The target's index summary refreshed from the merged manifest (the
    fields publish_event writes), 'final' dropped, `merged` set."""
    s = dict(prev or {})
    s.update({k: man.get(k) for k in ("bbox", "t0", "sim_start", "t_end", "model",
                                      "trigger", "generated", "gauge", "domain")})
    s["engine"] = (man.get("provenance") or {}).get("engine") or s.get("engine")
    s["n_frames"] = len(man.get("frames") or [])
    s["status"] = "ended"
    s["episode_started"] = man.get("episode_started")
    s["archive_frames"] = man.get("archive_frames")
    if s["n_frames"]:
        s.pop("demoted", None)          # the merged frames are all stored
    s.pop("final", None)
    s["merged"] = merged
    return s


def rewrite_index(idx: dict, target: str, others, entry: dict) -> dict:
    """New index (`idx` untouched, order kept): the target's entry replaced,
    each other member stamped merged_into = target, status unchanged."""
    others = set(others)
    return {e: (entry if e == target
                else {**s, "merged_into": target} if e in others else s)
            for e, s in idx.items()}


def recheck(planned: dict, idx: dict, gap_h=None):
    """(why, group) on the FRESH index right before committing: why = None
    when every planned member still is ended, unabsorbed and unpublished
    since the plan (planned: {id: generated}) and the group around the
    target is still exactly the planned members."""
    for m, gen in planned.items():
        s = idx.get(m)
        if s is None:
            return f"{m} left the index", None
        if s.get("status") != "ended":
            return f"{m} is {s.get('status') or 'active'} now", None
        if s.get("merged_into"):
            return f"{m} was merged into {s['merged_into']} meanwhile", None
        if s.get("generated") != gen:
            return f"{m} was re-published meanwhile", None
    target = max(planned)
    g = next((g for g in eventlist.groups(idx, gap_h) if target in g["members"]), None)
    if g is None or set(g["members"]) != set(planned):
        return "the group around the target changed", None
    return None, g


# ---- raster/table merges (local files) --------------------------------------

def _read(path, dtype=None):
    import numpy as np
    import rasterio
    with rasterio.open(path) as ds:
        a = ds.read(1)
        nd = ds.nodata
    if dtype == "depth" and nd is not None and nd != 0 and not (
            isinstance(nd, float) and math.isnan(nd)):
        a = np.where(a == nd, 0, a)      # outside the domain: dry
    return a


def running_max(paths, shape):
    """Cellwise max of uint16 grids, one in memory at a time."""
    import numpy as np
    out = np.zeros(shape, dtype=np.uint16)
    for p in paths:
        a = _read(p, "depth")
        if a.shape != tuple(shape):
            raise Skip(f"{os.path.basename(p)} is {a.shape[1]}x{a.shape[0]}, "
                       f"grid {shape[1]}x{shape[0]}")
        np.maximum(out, a.astype(np.uint16, copy=False), out=out)
    return out


def merge_speed(speed: list, flowdir: dict, shape):
    """(max speed, flow direction): `speed` = [(member, maxspeed path)] in
    preference order; a later member wins a cell only where it is strictly
    faster, and its flowdir.tif sector comes with it (255 = no flowdir)."""
    import numpy as np
    sp = fd = None
    for m, p in speed:
        a = _read(p)
        if a.shape != tuple(shape):
            raise Skip(f"{m} maxspeed.tif is {a.shape[1]}x{a.shape[0]}")
        d = _read(flowdir[m]) if flowdir.get(m) else np.full(shape, 255, np.uint8)
        if sp is None:
            sp, fd = a.copy(), d.copy()
            continue
        win = a > sp
        sp = np.where(win, a, sp)
        fd = np.where(win, d, fd)
    return sp, fd


def write_like(path, a, like: str | None, grid: dict, nodata=None):
    """Write `a` with the profile of the tif at `like` (driver, dtype,
    nodata, tiling, compression + predictor); crestimap.io's layout when
    there is none."""
    import rasterio
    from rasterio.transform import Affine
    if like:
        with rasterio.open(like) as ds:
            prof = dict(ds.profile)
            pred = ds.tags(ns="IMAGE_STRUCTURE").get("PREDICTOR")
        if pred:
            prof["predictor"] = int(pred)
    else:
        prof = {"driver": "GTiff", "dtype": str(a.dtype), "nodata": nodata,
                "compress": "deflate", "tiled": True, "blockxsize": 256,
                "blockysize": 256, "crs": grid.get("crs"),
                "transform": Affine(*grid["transform"][:6])}
        if a.dtype.kind == "u" and a.dtype.itemsize == 2:
            prof["predictor"] = 2
    prof.update(count=1, height=a.shape[0], width=a.shape[1])
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(a.astype(prof["dtype"], copy=False), 1)
    return path


ARCH_SCHEMA_COLS = ("time", "row", "col", "depth_cm")


def _arch_schema():
    import pyarrow as pa
    return pa.schema([("time", pa.timestamp("s")), ("row", pa.uint16()),
                      ("col", pa.uint16()), ("depth_cm", pa.uint16())])


def _t_iso(t) -> str:
    return t.strftime("%Y-%m-%dT%H:%MZ")


def archive_times(path, grid: dict):
    """Times (frame-style strings) in one archive.parquet, or None when its
    grid metadata isn't this grid (eventsim restarts such an archive too)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    meta = pq.read_schema(path).metadata or {}
    g = json.loads(meta.get(b"grid", b"{}") or b"{}")
    if g.get("ny") != grid.get("ny") or g.get("nx") != grid.get("nx"):
        return None
    t = pq.read_table(path, columns=["time"])
    return {_t_iso(v) for v in pc.unique(t.column("time")).to_pylist() if v is not None}


def write_archive(out, takes: dict, paths: dict, grid: dict, event_id: str) -> tuple:
    """Merged archive.parquet from each member's rows at its `takes` times,
    in eventsim._update_archive's format. (wet-cell rows, frames)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    schema = _arch_schema()
    parts = []
    for m, ts in takes.items():
        t = pq.read_table(paths[m]).select(list(ARCH_SCHEMA_COLS)).cast(schema)
        want = pa.array([dt.datetime.strptime(x, "%Y-%m-%dT%H:%MZ") for x in sorted(ts)],
                        pa.timestamp("s"))
        parts.append(t.filter(pc.is_in(t.column("time"), value_set=want)))
    merged = pa.concat_tables(parts) if parts else schema.empty_table()
    merged = merged.sort_by([("time", "ascending")])
    meta = dict(merged.schema.metadata or {})
    meta[b"grid"] = json.dumps(grid).encode()
    meta[b"depth_unit"] = b"centimeters"
    meta[b"event_id"] = str(event_id).encode()
    merged = merged.replace_schema_metadata(meta)
    pq.write_table(merged, out, compression="zstd")
    return merged.num_rows, len(pc.unique(merged.column("time")))


# ---- store access ------------------------------------------------------------

class FileInfo:
    __slots__ = ("size", "ident", "lfs")

    def __init__(self, size, ident, lfs):
        self.size, self.ident, self.lfs = int(size or 0), ident, bool(lfs)


class Store:
    """Reads of events/ on the dataset (the tests use a local stand-in)."""

    def __init__(self, api, token=None):
        self.api, self.token = api, token

    def tree(self, ev: str, recursive: bool = False) -> dict:
        """{path under events/<ev>/: FileInfo}; content ident = LFS sha256,
        else the git blob id."""
        from huggingface_hub.hf_api import RepoFile
        pre = f"{P}/{ev}/"
        out = {}
        for f in self.api.list_repo_tree(eventstore.REPO, path_in_repo=f"{P}/{ev}",
                                         repo_type="dataset", recursive=recursive):
            if isinstance(f, RepoFile) and f.path.startswith(pre):
                lfs = getattr(f, "lfs", None)
                out[f.path[len(pre):]] = FileInfo(
                    f.size, (lfs.sha256 if lfs else f.blob_id), lfs is not None)
        return out

    def manifest(self, ev: str) -> dict | None:
        return eventstore._prior_manifest(ev)

    def fetch(self, ev: str, name: str, dest: str) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(eventstore.REPO, f"{P}/{ev}/{name}", repo_type="dataset",
                               local_dir=dest, token=self.token)

    def queued(self) -> set:
        """Event ids with a job spec in events/queue/ (claimed or not)."""
        from huggingface_hub.utils import EntryNotFoundError
        try:
            fs = [f.path for f in self.api.list_repo_tree(
                eventstore.REPO, path_in_repo=f"{P}/queue", repo_type="dataset")]
        except EntryNotFoundError:
            return set()
        return {os.path.basename(f)[:-5] for f in fs
                if f.endswith(".json") and not f.endswith(".failed.json")}

    def head(self) -> str:
        return self.api.repo_info(eventstore.REPO, repo_type="dataset").sha

    def index_at(self, sha: str) -> dict:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(eventstore.REPO, f"{P}/index.json", repo_type="dataset",
                            revision=sha, token=self.token)
        with open(p, encoding="utf-8") as fp:
            return json.load(fp)


# ---- plan + products -----------------------------------------------------------

def plan_group(store, g: dict, idx: dict) -> dict:
    """Everything the merge reads from the index, manifests and folder
    listings (no frame downloads). Raises Skip."""
    members = sorted(g["members"], reverse=True)
    target = members[0]                      # YYYYMMDDHH_<gauge>: newest start
    mans = {}
    for m in members:
        man = store.manifest(m)
        if not man:
            raise Skip(f"manifest unreadable: {m}")
        mans[m] = man
    why = grid_problem(mans)
    if why:
        raise Skip(why)
    trees = {m: store.tree(m, recursive=(m == target)) for m in members}
    stored = {m: [fr for fr in (mans[m].get("frames") or [])
                  if isinstance(fr, dict) and fr.get("file") in trees[m]]
              for m in members}
    plan = frame_plan(target, stored)
    if not plan:
        # maxdepth is rebuilt from the frames: none stored would zero it
        raise Skip("no member has stored depth frames (animation stripped)")
    tt = trees[target]
    reuse, conflicts = set(), []
    for _, src, fr in plan:
        if src == target:
            continue
        for name in (fr["file"], png_name(fr)):
            if name in tt:
                same = name in trees[src] and trees[src][name].ident == tt[name].ident
                (reuse.add(name) if same else conflicts.append(name))
    if conflicts:
        raise Skip(f"target folder already holds different {', '.join(conflicts[:4])}"
                   f"{' ...' if len(conflicts) > 4 else ''}; resolve by hand")
    # premerge/ keeps the state before the FIRST merge: once the target's
    # manifest is a merge's, a name premerge/ lacks came from that merge or
    # after it (a merged archive.parquet, the post-merge final.json)
    prior = (mans[target].get("provenance") or {}).get("merged")
    backup = [] if prior else [n for n in BACKUP if n in tt and f"premerge/{n}" not in tt]
    return {"gauge": eventlist._gauge(target, idx.get(target) or {}), "group": g,
            "target": target, "others": members[1:], "members": members,
            "mans": mans, "trees": trees, "stored": stored, "frames": plan,
            "reuse": reuse, "backup": backup,
            "generated": {m: (idx.get(m) or {}).get("generated") for m in members}}


def _fetch_all(store, jobs, dest, beat=None):
    """[(ev, name)] -> {(ev, name): local path}, a few at a time; beat() (the
    --watch heartbeat) after each file."""
    paths = []
    with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
        for p in ex.map(lambda j: store.fetch(j[0], j[1], dest), jobs):
            paths.append(p)
            if beat is not None:
                beat()
    return dict(zip(jobs, paths))


def build_products(store, plan: dict, work: str, idx: dict, log=print, beat=None) -> dict:
    """Download what the merge reads into `work`, write the merged files and
    the list of commit operations. Nothing is sent anywhere. beat: see
    _fetch_all."""
    target, mans, trees = plan["target"], plan["mans"], plan["trees"]
    pref = preference(target, plan["members"])
    grid = mans[target]["grid"]
    shape = (int(grid["ny"]), int(grid["nx"]))
    dl, render = os.path.join(work, "dl"), os.path.join(work, "render")
    os.makedirs(render, exist_ok=True)
    frames = plan["frames"]

    # 1. merged frame tifs -> render/ (flat: one source per time and name)
    got = _fetch_all(store, [(src, fr["file"]) for _, src, fr in frames], dl, beat)
    for _, src, fr in frames:
        shutil.move(got[(src, fr["file"])], os.path.join(render, fr["file"]))
    md = running_max([os.path.join(render, fr["file"]) for _, _, fr in frames], shape)
    like = (store.fetch(target, "maxdepth.tif", dl)
            if "maxdepth.tif" in trees[target] else None)
    write_like(os.path.join(render, "maxdepth.tif"), md, like, grid)
    peak_m = float(md.max()) / 100.0
    del md
    cap = eventsim.adaptive_cap(os.path.join(render, "maxdepth.tif"))

    # 2. PNGs: maxdepth + frames whose colour scale changes (native renderer)
    caps = {m: cap_of(mans[m]) for m in plan["members"]}
    pngs = {m: set(trees[m]) for m in plan["members"]}
    redo = rerender(frames, caps, cap, pngs)
    tt = trees[target]
    clash = [png_name(fr) for _, src, fr in frames
             if src != target and fr["file"] in redo and png_name(fr) in tt]
    if clash:
        raise Skip(f"re-rendering would overwrite {', '.join(clash[:4])} that the "
                   f"target folder holds for a time it doesn't list")
    mini = {"grid": grid, "frames": [{"file": f} for f in redo]}
    eventsim._make_pngs(render, mini)        # writes render/manifest.json: unused

    # 3. maxspeed (+ flowdir) over the members that have it
    speed_out = None
    if mans[target].get("maxspeed") and "maxspeed.tif" in tt:
        sp_jobs = [(m, "maxspeed.tif") for m in pref if "maxspeed.tif" in trees[m]]
        fd_jobs = [(m, "flowdir.tif") for m in pref if "flowdir.tif" in trees[m]]
        sp_like = _fetch_all(store, sp_jobs + fd_jobs, dl, beat)   # sp_jobs[0]: the target
        sp, fd = merge_speed([(m, sp_like[(m, n)]) for m, n in sp_jobs],
                             {m: sp_like[(m, n)] for m, n in fd_jobs}, shape)
        speed_out = {"n": len(sp_jobs), "flowdir": (target, "flowdir.tif") in sp_like}
        write_like(os.path.join(render, "maxspeed.tif"), sp,
                   sp_like[(target, "maxspeed.tif")], grid)
        if speed_out["flowdir"]:
            write_like(os.path.join(render, "flowdir.tif"), fd,
                       sp_like[(target, "flowdir.tif")], grid, nodata=255)
        del sp, fd

    # 4. archive.parquet merged by time
    arch = None
    a_jobs = [(m, "archive.parquet") for m in pref if "archive.parquet" in trees[m]]
    if a_jobs:
        a_paths = {m: p for (m, _), p in _fetch_all(store, a_jobs, dl, beat).items()}
        times, ignored = {}, []
        for m, p in a_paths.items():
            ts = archive_times(p, grid)
            if ts is None:
                ignored.append(m)
            else:
                times[m] = ts
        takes = archive_takes({t: src for t, src, _ in frames}, times, pref)
        rows, n_t = write_archive(os.path.join(render, "archive.parquet"), takes,
                                  a_paths, grid, target)
        arch = {"rows": rows, "frames": n_t, "ignored": ignored,
                "from": {m: len(ts) for m, ts in takes.items()}}

    # 5. hydro + manifest
    hydro = merge_hydro({m: mans[m].get("hydro") for m in plan["members"]}, pref)
    now = dt.datetime.now(dt.timezone.utc).strftime(_TS)
    man = merged_manifest(target, mans, idx, frames, hydro, cap, now,
                          archive_frames=arch["frames"] if arch else None)
    mpath = os.path.join(work, "manifest.json")
    with open(mpath, "w") as fp:
        json.dump(man, fp)

    # 6. commit operations (no index yet: that is read fresh at commit time)
    ops = []                                 # (kind, src, dest rel, bytes, server-side)
    tdir = f"{P}/{target}"
    for n in plan["backup"]:
        f = tt[n]
        ops.append(("copy", f"{tdir}/{n}", f"{tdir}/premerge/{n}", f.size, f.lfs))
    for _, src, fr in frames:
        if src == target:
            continue
        name = fr["file"]
        if name not in plan["reuse"]:
            f = trees[src][name]
            ops.append(("copy", f"{P}/{src}/{name}", f"{tdir}/{name}", f.size, True)
                       if f.lfs else
                       ("add", os.path.join(render, name), f"{tdir}/{name}", f.size, False))
        png = png_name(fr)
        if name not in redo and png not in plan["reuse"]:
            f = trees[src][png]
            ops.append(("copy", f"{P}/{src}/{png}", f"{tdir}/{png}", f.size, f.lfs))
    for name in redo:
        p = os.path.join(render, name[:-4] + ".png")
        ops.append(("add", p, f"{tdir}/{name[:-4]}.png", os.path.getsize(p), False))
    outs = ["maxdepth.tif", "maxdepth.png"]
    if speed_out:
        outs += ["maxspeed.tif"] + (["flowdir.tif"] if speed_out["flowdir"] else [])
    if arch:
        outs.append("archive.parquet")
    for n in outs:
        p = os.path.join(render, n)
        ops.append(("add", p, f"{tdir}/{n}", os.path.getsize(p), False))
    ops.append(("add", mpath, f"{tdir}/manifest.json", os.path.getsize(mpath), False))
    return {"manifest": man, "cap": cap, "peak_m": peak_m, "caps": caps,
            "rerender": redo, "speed": speed_out, "archive": arch,
            "hydro": {"n": len(hydro)}, "ops": ops, "outs": outs, "work": work}


def _mb(n) -> str:
    return f"{n / 1e6:.2f} MB" if n >= 1e5 else f"{n / 1e3:.1f} kB"


def describe(plan: dict, prod: dict, merged: dict) -> list:
    """The printed plan of one group."""
    t, mans = plan["target"], plan["mans"]
    gr = mans[t]["grid"]
    frames = plan["frames"]
    src_n = {m: sum(1 for _, s, _ in frames if s == m) for m in plan["members"]}
    man = prod["manifest"]
    L = [f"== {plan['gauge']}: {len(plan['members'])} episodes -> target {t}"]
    L.append("   members  " + " | ".join(
        f"{m} {len(plan['stored'][m])}/{len(mans[m].get('frames') or [])} frames "
        f"stored, cap {prod['caps'][m]:g} m" for m in plan["members"]))
    L.append(f"   grid     identical: {gr['nx']} x {gr['ny']}, {gr.get('crs')}, "
             f"origin ({gr['transform'][2]}, {gr['transform'][5]})")
    adds = ", ".join(f"{src_n[m]} from {m}" for m in plan["others"])
    L.append(f"   frames   {len(frames)} merged = {src_n[t]} own + {adds}"
             + (f"; {len(plan['reuse'])} already in the target folder" if plan["reuse"] else ""))
    if frames:
        L.append(f"            {frames[0][0]} .. {frames[-1][0]}")
    L.append(f"   span     sim_start {man['sim_start']}, episode_started "
             f"{man['episode_started']}, t_end {man['t_end']} (target: "
             f"{mans[t].get('sim_start')} / {mans[t].get('t_end')})")
    n_redo = len(prod["rerender"])
    L.append(f"   maxdepth cellwise max of the {len(frames)} frames: peak "
             f"{prod['peak_m']:.2f} m, depth_cap_m {prod['cap']:g} (target had "
             f"{prod['caps'][t]:g}); PNGs rendered: maxdepth.png + {n_redo} frame(s)"
             + (f" ({sum(1 for _, s, fr in frames if s == t and fr['file'] in prod['rerender'])}"
                f" of them the target's own)" if n_redo else ""))
    sp = prod["speed"]
    L.append("   maxspeed " + (f"cellwise max of {sp['n']} member(s)"
                               f"{' + flowdir.tif from the winner' if sp['flowdir'] else ''}"
                               if sp else "target has none: not merged"))
    a = prod["archive"]
    if a:
        L.append(f"   archive  {a['rows']} wet-cell rows over {a['frames']} frame(s); times from "
                 + (", ".join(f"{m} {n}" for m, n in a["from"].items()) or "none")
                 + (f"; grid differs, ignored: {', '.join(a['ignored'])}" if a["ignored"] else ""))
    else:
        L.append("   archive  no member has one")
    own = len(mans[t].get("hydro") or [])
    L.append(f"   hydro    {prod['hydro']['n']} rows ({own} own + "
             f"{prod['hydro']['n'] - own} from the others)")
    L.append(f"   backup   premerge/: {', '.join(plan['backup']) or 'nothing new (kept from an earlier merge)'}")
    L.append(f"   write    {', '.join(prod['outs'])}, manifest.json"
             + (f", {n_redo} frame PNG(s)" if n_redo else ""))
    copies = [o for o in prod["ops"] if o[0] == "copy" and "/premerge/" not in o[2]]
    readds = [o for o in prod["ops"] if o[0] == "add" and o[2].split("/")[-1].startswith("depth_")
              and o[2].endswith(".tif")]
    L.append(f"   copy     {len(copies)} frame file(s) server-side "
             f"({_mb(sum(o[3] for o in copies if o[4]))} LFS"
             + (f", {sum(1 for o in copies if not o[4])} regular re-uploaded" if any(not o[4] for o in copies) else "")
             + f"), {len(readds)} frame tif(s) re-uploaded from the bytes read "
               f"({_mb(sum(o[3] for o in readds))})")
    L.append(f"   index    {t}: final dropped, merged n={merged['n']} "
             f"{merged['first_start']}..{merged['last_end']}; merged_into={t} on "
             f"{', '.join(plan['others'])}")
    up = sum(o[3] for o in prod["ops"] if not o[4])
    srv = sum(o[3] for o in prod["ops"] if o[4])
    L.append(f"   commit   1 commit, {len(prod['ops']) + 1} operations: upload {_mb(up)} "
             f"+ index.json, server-side copy {_mb(srv)}")
    return L


# ---- commit ------------------------------------------------------------------

def _hf_ops(ops):
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy
    out = []
    for kind, src, dest, _, _ in ops:
        out.append(CommitOperationCopy(src_path_in_repo=src, path_in_repo=dest)
                   if kind == "copy" else CommitOperationAdd(dest, src))
    return out


def _conflict(e) -> bool:
    r = getattr(e, "response", None)
    return getattr(r, "status_code", None) in (409, 412)


def _commit_index(store, target, edit, ops, msg, ipath, log=print) -> tuple:
    """`ops` + events/index.json = edit(fresh index) in one commit against the
    head revision, under eventstore._lock; a concurrent commit (409/412)
    re-reads the index and retries, COMMIT_TRIES in all. edit(idx) -> (new
    index, None), or (None, why) to commit nothing. (commit id or None, why
    edit declined or None)."""
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.utils import HfHubHTTPError
    for attempt in range(1, COMMIT_TRIES + 1):
        with eventstore._lock:
            try:
                sha = store.head()
                idx = store.index_at(sha)
            except Exception as e:
                log(f"   INDEX READ FAILED for {target} ({type(e).__name__}); "
                    f"nothing committed")
                return None, None
            new, why = edit(idx)
            if new is None:
                return None, why
            with open(ipath, "w") as fp:
                json.dump(new, fp)
            try:
                info = store.api.create_commit(
                    repo_id=eventstore.REPO, repo_type="dataset", parent_commit=sha,
                    operations=list(ops) + [CommitOperationAdd(f"{P}/index.json", ipath)],
                    commit_message=msg)
                oid = str(getattr(info, "oid", None) or "?")
                log(f"   committed {oid[:12]} ({msg})")
                return oid, None
            except HfHubHTTPError as e:
                if not _conflict(e) or attempt == COMMIT_TRIES:
                    log(f"   COMMIT FAILED for {target} ({type(e).__name__}: "
                        f"{getattr(getattr(e, 'response', None), 'status_code', '')})")
                    return None, None
                log(f"   the store moved under us (attempt {attempt}): re-reading the index")
            except Exception as e:
                log(f"   COMMIT FAILED for {target} ({type(e).__name__})")
                return None, None
        time.sleep(3 * attempt)
    return None, None


def commit_group(store, plan: dict, prod: dict, gap_h, log=print):
    """One commit: copies + uploads + the index, against the head revision.
    The commit id when it landed, else False."""
    from huggingface_hub import CommitOperationAdd
    api, target, others = store.api, plan["target"], plan["others"]
    base = _hf_ops(prod["ops"])
    # LFS bytes go up first, so the index read -> commit window stays short
    try:
        api.preupload_lfs_files(eventstore.REPO, repo_type="dataset",
                                additions=[o for o in base if isinstance(o, CommitOperationAdd)])
    except Exception as e:
        log(f"   UPLOAD FAILED for {target} ({type(e).__name__}); nothing committed")
        return False
    msg = (f"{'re-apply ' if plan.get('reapply') else ''}merge episodes {target} <- "
           f"{', '.join(others)} ({len(plan['frames'])} frames, "
           f"{len(prod['rerender'])} PNGs re-rendered)")

    def edit(idx):
        why, g = recheck(plan["generated"], idx, gap_h)
        if why:
            return None, why
        entry = index_entry(idx[target], prod["manifest"], eventlist.merged_record(g))
        return rewrite_index(idx, target, others, entry), None

    oid, why = _commit_index(store, target, edit, base, msg,
                             os.path.join(prod["work"], "index.json"), log)
    if why:
        log(f"   ABORT {target}: {why}; nothing committed; re-run to re-plan")
    return oid or False


def drop_final(store, target: str, log=print):
    """Drop a merged target's stale final (stale_final) from the index in one
    commit, so finalize_episodes, which skips an entry with a final, scores
    the merged episode again. True: dropped, or no final any more; None: no
    stale final (re-finalized meanwhile, or the target left); False: failed."""
    def edit(idx):
        s = idx.get(target)
        if not isinstance(s, dict) or not s.get("final"):
            return None, ("no final" if isinstance(s, dict) else "gone")
        if not stale_final(s):
            return None, "current"
        return {e: ({k: v for k, v in x.items() if k != "final"} if e == target else x)
                for e, x in idx.items()}, None

    fd, ipath = tempfile.mkstemp(prefix="crest_merge_idx_", suffix=".json")
    os.close(fd)
    try:
        oid, why = _commit_index(store, target, edit, [],
                                 f"merged episode {target}: stale final dropped", ipath, log)
    finally:
        try:
            os.remove(ipath)
        except OSError:
            pass
    if oid:
        return True
    return {"no final": True, "current": None, "gone": None}.get(why, False)


# ---- one pass ----------------------------------------------------------------

# skip reasons that clear by themselves: --watch retries them after --retry-min
WAIT = ("not ended", "pinned set unreadable", "worker queue", "manifest unreadable",
        "rate cap", "final pending")
SKIP_RECHECK_H = 24          # --watch: a lasting plan skip is re-planned after this
FINAL_TRIES = 3              # --watch: finalize attempts per merged target, at most
REAPPLY_FREE = 3             # --watch: re-applies of one target outside the hourly cap


class Unreadable(Exception):
    """events/index.json could not be read (or is empty)."""


def waits(why) -> bool:
    """A skip --watch retries (the episode ends, the spec goes, a read works)."""
    return str(why or "").startswith(WAIT)


def _sig(members, idx: dict) -> tuple:
    """A group as the index has it (ids + generated): a lasting skip holds
    until this changes."""
    return tuple((m, (idx.get(m) or {}).get("generated")) for m in sorted(members))


def _finalize(target, log=print) -> bool:
    try:
        n = eventstore.finalize_episodes([target], log=log)
    except Exception as e:
        log(f"   finalize {target}: FAILED ({type(e).__name__}: {e})")
        return False
    log(f"   finalize {target}: {'done' if n else 'NOT done (re-run, or the runner heal pass)'}")
    return bool(n)


def run_pass(store, *, apply=False, gauge=None, max_groups=0, gap_h=None,
             finalize=True, budget=None, stop=None, memo=None, now=None,
             retry_s=0.0, on_merge=None, recent=(), beat=None, log=print) -> dict:
    """One pass over the index: every multi-episode group classified, the
    eligible ones planned and built (dry run: described) and, with apply,
    committed and finalized. A group's errors make it a failure, not an
    exception; raises Unreadable. Report: {index, groups, planned, merged
    [{gauge, target, members, commit, n_frames, generated, merged_gen,
    reapply}], failed [{gauge, target, members, why}], skipped [{gauge,
    members, why, wait}], already, finalized, unfinalized [{gauge, members,
    why}], reverted [targets], stopped}.
    --watch: budget = merges still allowed (the rest skip "rate cap"), stop
    (.is_set()) is checked between groups and before each commit, memo
    (kept across passes) holds lasting plan skips, finalize attempts
    (retried after retry_s, FINAL_TRIES in all; then the runner's heal pass
    or a hand run) and re-applies, now = its clock, on_merge(rec) right after
    a merge lands, recent = its merge records (reverted() ones are
    re-applied: reapply, outside the budget, REAPPLY_FREE times per target),
    beat() = its heartbeat (each group, each download)."""
    gap_h = eventlist.MERGE_GAP_H if gap_h is None else gap_h
    now = time.time() if now is None else now
    memo = {} if memo is None else memo
    beat = beat or (lambda: None)
    idx = eventstore.load_index()
    if not idx:
        raise Unreadable(f"{P}/index.json unreadable (or empty)")
    pins = eventstore.pinned_ids(force=True)
    queued = store.queued()
    groups = eventlist.groups(idx, gap_h)
    news = {}
    back = reverted(recent, idx, news)
    rep = {"index": len(idx), "groups": sum(1 for g in groups if len(g["members"]) > 1),
           "planned": 0, "merged": [], "failed": [], "skipped": [], "already": 0,
           "finalized": [], "unfinalized": [], "reverted": sorted(back), "stopped": False}
    log(f"merge_episodes: {eventstore.REPO} index {len(idx)} entries, "
        f"{rep['groups']} multi-episode group(s), gap {gap_h:g} h, "
        f"{'APPLY' if apply else 'DRY RUN'}{f', gauge {gauge}' if gauge else ''}")
    for t, r in sorted(back.items()):
        log(f"== {eventlist._gauge(t, idx.get(t) or {})}: merge into {t} reverted (merged "
            f"{r.get('at') or '?'}; an index write from an older copy dropped it); "
            f"re-applied as its group allows, outside the hourly cap")
    for t, why in sorted(news.items()):
        log(f"== {eventlist._gauge(t, idx.get(t) or {})}: merge into {t} undone but not "
            f"a revert: {why}; new content, merged again as an ordinary merge "
            f"(within the hourly cap)")

    def skip(gg, members, why):
        rep["skipped"].append({"gauge": gg, "members": list(members), "why": why,
                               "wait": waits(why)})
        log(f"== {gg}: SKIP {len(members)} episodes ({', '.join(members)}): {why}")

    def fail(gg, members, why):
        rep["failed"].append({"gauge": gg, "target": max(members),
                              "members": list(members), "why": why})

    def stopping():
        if stop is not None and stop.is_set():
            rep["stopped"] = True
        return rep["stopped"]

    def try_finalize(gg, t, members, stale=None):
        at, n = memo.get(("final", t), (-math.inf, 0))
        if n >= FINAL_TRIES:
            return
        if now - at < retry_s:               # still waiting: the watcher retries it
            rep["unfinalized"].append({"gauge": gg, "members": list(members),
                                       "why": f"finalize of {t} not done (try {n}); "
                                              f"retried after --retry-min"})
            return
        memo[("final", t)] = (now, n + 1)
        ok = drop_final(store, t, log) if stale else True
        if ok is None:                       # re-finalized meanwhile: nothing to do
            return
        if ok and _finalize(t, log):
            rep["finalized"].append(t)
        elif n + 1 < FINAL_TRIES:
            rep["unfinalized"].append({"gauge": gg, "members": list(members),
                                       "why": f"finalize of {t} not done (try {n + 1})"})

    def capped():
        return sum(1 for r in rep["merged"] if not r.get("reapply"))

    for g in groups:
        if stopping():
            log("stopping: the remaining groups wait for the next pass")
            break
        beat()
        rep_id = g["rep"]
        gg = eventlist._gauge(rep_id, idx.get(rep_id) or {})
        if gauge and gg != gauge:
            continue
        kind, why = classify(g, idx, pins, queued, now)
        if kind == "single":
            continue
        if kind == "merged":
            t = g["members"][0]
            s = idx.get(t) or {}
            stale = stale_final(s)
            fin = s.get("status") == "ended" and (not s.get("final") or bool(stale))
            log(f"== {gg}: already merged into {t} ({len(g['ids'])} episodes)"
                + ("; target has no final yet" if fin and not stale else "")
                + (f"; its final is stale ({stale})" if fin and stale else ""))
            rep["already"] += 1
            if fin and apply and finalize:
                try_finalize(gg, t, g["ids"], stale)
            continue
        if kind == "skip":
            skip(gg, g["members"], why)
            continue
        if max_groups and rep["planned"] >= max_groups:
            log(f"== {gg}: not processed (--max-groups {max_groups})")
            continue
        target = max(g["members"])
        free = target in back and memo.get(("reapply", target), 0) < REAPPLY_FREE
        if budget is not None and not free and capped() >= budget:
            skip(gg, g["members"], "rate cap (--max-merges-per-hour) reached; "
                                   "merged when the hour frees a slot")
            continue
        sig = ("skip", _sig(g["members"], idx))
        if sig in memo and now - memo[sig][1] < SKIP_RECHECK_H * 3600:
            skip(gg, g["members"], memo[sig][0])
            continue
        work = tempfile.mkdtemp(prefix="crest_merge_ep_")   # datamgr's janitor sweeps crest_*
        try:
            try:
                plan = plan_group(store, g, idx)
                plan["reapply"] = target in back
                prod = build_products(store, plan, work, idx, beat=beat)
            except Skip as e:
                if not waits(str(e)):
                    memo[sig] = (str(e), now)
                skip(gg, g["members"], str(e))
                continue
            rep["planned"] += 1
            for line in describe(plan, prod, eventlist.merged_record(g)):
                log(line)
            if not apply:
                continue
            if stopping():
                log(f"   stopping before the commit of {plan['target']}: nothing committed")
                break
            said = []

            def clog(s):
                said.append(s)
                log(s)

            oid = commit_group(store, plan, prod, gap_h, log=clog)
            if not oid:
                fail(gg, plan["members"], said[-1].strip() if said else "commit failed")
                continue
            # what reverted() checks it by later: the pre-merge generated of
            # every member (a stale index write restores those) and the merge's
            rec = {"gauge": gg, "target": plan["target"], "members": list(plan["members"]),
                   "commit": oid, "n_frames": len(plan.get("frames") or []),
                   "generated": plan.get("generated"),
                   "merged_gen": ((prod or {}).get("manifest") or {}).get("generated")}
            if free:
                memo[("reapply", target)] = memo.get(("reapply", target), 0) + 1
                rec["reapply"] = True
            rep["merged"].append(rec)
            if on_merge is not None:
                try:
                    on_merge(rec)
                except Exception as e:   # the merge landed: never a failure
                    log(f"   on_merge {plan['target']}: {type(e).__name__}: {e}")
            if finalize:
                memo.pop(("final", plan["target"]), None)   # a new merge: its own tries
                try_finalize(gg, plan["target"], plan["members"])
        except Exception as e:
            why = f"{type(e).__name__}: {e}"
            log(f"== {gg}: FAILED {', '.join(g['members'])}: {why}")
            fail(gg, g["members"], why)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return rep


# ---- --watch: the always-on merger ---------------------------------------------

BACKOFF_MIN_S, BACKOFF_MAX_S = 60.0, 3600.0
RECENT_KEEP = 20             # merges listed in the status file
LOG_KEEP = 80                # last pass's log lines kept in the status file
BEAT_S = 30.0                # heartbeat inside a pass: status written at most this often
# status writes failing in a row (the --status-file and its fallback): the
# child exits non-zero, well before the supervisor's watchdog (no heartbeat
# for 6 x interval) would kill it as hung; the supervisor restarts it
STATUS_FAILS_MAX = 5
STATUS_EXIT = 3              # the exit code then
# the Hub client's own defaults: none (1.x: its shared httpx client; 0.x: a
# requests session, and requests waits forever without one). read/write are
# per chunk (a 20 MB transfer only needs bytes to keep flowing); 0.x's
# requests takes (connect, read) only
HUB_TIMEOUT_S = {"connect": 30.0, "read": 300.0, "write": 300.0, "pool": 60.0}
HUB_FILE_TIMEOUT_S = {"HF_HUB_DOWNLOAD_TIMEOUT": 60, "HF_HUB_ETAG_TIMEOUT": 30}


def requests_timeout(t):
    """huggingface_hub 0.x: the requests timeout an adapter sends with; None,
    or a None part of a (connect, read) pair, gets HUB_TIMEOUT_S's."""
    want = (HUB_TIMEOUT_S["connect"], HUB_TIMEOUT_S["read"])
    if t is None:
        return want
    if isinstance(t, tuple) and len(t) == 2:
        return tuple(w if v is None else v for v, w in zip(t, want))
    return t


def bound_hub_timeouts() -> str:
    """Finite timeouts on every Hub call of this process: a hung paths-info,
    tree listing or commit (huggingface_hub defaults to none, and some calls
    pass timeout=None outright, e.g. repo_info) held the merger forever
    while its supervisor only waited for an exit. huggingface_hub 1.x: the
    library's client (set_client_factory) gets HUB_TIMEOUT_S, also where a
    call passes None (a request hook fills those in). 0.x (the Spaces:
    tokenizers==0.22.0 needs huggingface-hub<1.0): configure_http_backend's
    sessions mount a UniqueRequestIdAdapter whose send() fills a None
    timeout the same way (requests_timeout; with HF_HUB_OFFLINE the
    library's OfflineAdapter, as its own factory does). File downloads and
    their metadata HEAD keep their own HF_HUB_DOWNLOAD_TIMEOUT /
    HF_HUB_ETAG_TIMEOUT (set here unless the environment has them). Xet
    transfers time out on their own; the supervisor's watchdog is the
    backstop. One-line summary naming the hook installed (the --watch
    startup line)."""
    for k, v in HUB_FILE_TIMEOUT_S.items():
        os.environ.setdefault(k, str(v))
    import huggingface_hub
    from huggingface_hub import constants
    for k in HUB_FILE_TIMEOUT_S:             # constants read the env at import
        try:
            setattr(constants, k, int(float(os.environ[k])) or getattr(constants, k))
        except ValueError:
            pass
    files = ", ".join(f"{k}={getattr(constants, k, '?')}" for k in HUB_FILE_TIMEOUT_S)
    ver = getattr(huggingface_hub, "__version__", "?")
    try:
        from huggingface_hub import set_client_factory
    except ImportError:
        set_client_factory = None
    if set_client_factory is None:
        try:
            import requests
            from huggingface_hub import configure_http_backend
            from huggingface_hub.utils._http import OfflineAdapter, UniqueRequestIdAdapter
        except ImportError:
            return (f"Hub timeouts: {files} only (huggingface_hub {ver} has neither "
                    f"set_client_factory nor configure_http_backend)")

        class Bounded(UniqueRequestIdAdapter):
            def send(self, request, *args, **kwargs):
                if len(args) > 1:            # send(request, stream, timeout, ...)
                    args = (args[0], requests_timeout(args[1])) + tuple(args[2:])
                else:
                    kwargs["timeout"] = requests_timeout(kwargs.get("timeout"))
                return super().send(request, *args, **kwargs)

        def session():
            s = requests.Session()
            adapter = OfflineAdapter if constants.HF_HUB_OFFLINE else Bounded
            s.mount("http://", adapter())
            s.mount("https://", adapter())
            return s

        configure_http_backend(backend_factory=session)
        con, read = requests_timeout(None)
        return (f"Hub timeouts via configure_http_backend (huggingface_hub {ver}): "
                f"connect {con:g} s, read {read:g} s; {files}")
    import httpx
    from huggingface_hub.utils._http import default_client_factory

    def fill(request):
        t = request.extensions.get("timeout") or {}
        request.extensions["timeout"] = {k: (v if t.get(k) is None else t[k])
                                         for k, v in HUB_TIMEOUT_S.items()}

    def factory():
        c = default_client_factory()
        c.timeout = httpx.Timeout(**HUB_TIMEOUT_S)
        hooks = c.event_hooks
        c.event_hooks = {**hooks, "request": list(hooks.get("request") or []) + [fill]}
        return c

    set_client_factory(factory)
    return (f"Hub timeouts via set_client_factory (huggingface_hub {ver}): "
            f"{', '.join(f'{k} {v:g} s' for k, v in HUB_TIMEOUT_S.items())}; {files}")


def _iso(t) -> str | None:
    return None if t is None else dt.datetime.fromtimestamp(
        t, dt.timezone.utc).strftime(_TS)


def _epoch(s) -> float | None:
    t = eventlist._utc(s)
    return t.replace(tzinfo=dt.timezone.utc).timestamp() if t else None


def default_status_file() -> str:
    return os.path.join(os.environ.get("CREST_CACHE_DIR") or tempfile.gettempdir(),
                        "merger_status.json")


def fallback_status_file() -> str:
    """--watch writes its status here while --status-file can't be written
    (the supervisor's watchdog and /api/merger_status read it too:
    hf_data/eventmerger.fallback_status_file)."""
    return os.path.join(tempfile.gettempdir(), "merger_status.fallback.json")


def write_status(path: str, st: dict) -> None:
    """Replace `path` atomically: a reader sees the old file or the new one."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".merger_status.", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(st, fp, indent=1)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def read_status(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fp:
            st = json.load(fp)
    except (OSError, ValueError):
        return {}
    return st if isinstance(st, dict) else {}


def freshest(sts) -> dict:
    """Of status dicts, the one 'updated' last (the first non-empty one when
    none says); {} when all are empty."""
    best, at = {}, None
    for st in sts:
        if not st:
            continue
        t = _epoch(st.get("updated"))
        t = -math.inf if t is None else t
        if at is None or t > at:
            best, at = st, t
    return best


def index_rev(api) -> str:
    """events/index.json's git blob id: one paths-info call, no download; it
    changes exactly when the index content does."""
    info = api.get_paths_info(eventstore.REPO, [f"{P}/index.json"], repo_type="dataset")
    if not info:
        raise Unreadable(f"{P}/index.json not found on {eventstore.REPO}")
    return info[0].blob_id


class Stop:
    """Stop flag a signal handler sets: a plain attribute (a lock taken in a
    handler could deadlock the main thread it interrupted)."""

    def __init__(self):
        self.flag = False

    def set(self, *_):
        self.flag = True

    def is_set(self) -> bool:
        return self.flag


def nap(seconds: float, stop, sleep=time.sleep) -> None:
    """Sleep up to `seconds`, waking within 1 s of `stop`."""
    end = time.monotonic() + seconds
    while not stop.is_set():
        left = end - time.monotonic()
        if left <= 0:
            return
        sleep(min(1.0, left))


class Watcher:
    """The --watch loop over injected parts (the tests fake them): rev() ->
    the index revision; run(budget=, stop=, memo=, now=, on_merge=, recent=,
    beat=) -> a run_pass report; clock() -> epoch s; sleep(s). The status
    file's 'updated' is the heartbeat the supervisor's watchdog reads: it is
    written at least every interval while sleeping and on beat() inside a
    pass. While status_file can't be written the status goes to `fallback`
    (status_fallback in it says why; the watchdog reads both); when neither
    can, STATUS_FAILS_MAX times in a row, the loop stops as on SIGTERM and
    run_forever() returns STATUS_EXIT: an exit the supervisor restarts, not a
    healthy child its watchdog kills for a missing heartbeat."""

    def __init__(self, rev, run, status_file: str, *, interval=300.0, retry_min=60.0,
                 max_per_hour=6, clock=time.time, sleep=None, stop=None, log=None,
                 config=None, fallback=None):
        self.rev, self.run, self.path, self.clock = rev, run, status_file, clock
        self.fallback = fallback if fallback and fallback != status_file else None
        self.interval = max(1.0, float(interval))
        self.retry_s = max(0.0, 60.0 * float(retry_min))
        self.max_per_hour = max(0, int(max_per_hour))
        self.stop = stop if stop is not None else Stop()
        self.sleep = sleep or (lambda s: nap(s, self.stop))
        self.log = log or (lambda s: print(s, flush=True))
        self.last_rev = self.last_pass = None
        self.waiting, self.memo, self.backoff = [], {}, 0.0
        self.beat_at = -math.inf
        self.status_fails, self.on_fallback, self.status_dead = 0, False, False
        # merges of an earlier run count toward the cap: a crash loop can't reset it
        prev = freshest(read_status(p) for p in (status_file, self.fallback) if p
                        ).get("recent_merges")
        self.recent = [r for r in (prev if isinstance(prev, list) else [])
                       if isinstance(r, dict)][-self._keep():]
        now = clock()
        self.st = {"started": _iso(now), "pid": os.getpid(), "state": "starting",
                   "config": {**(config or {}), "interval_s": self.interval,
                              "retry_min": self.retry_s / 60.0,
                              "max_merges_per_hour": self.max_per_hour},
                   "passes": 0, "errors": 0, "last_pass": None, "waiting": [],
                   "last_error": None, "next_pass_at": None, "retry_at": None}

    def _keep(self) -> int:
        return max(RECENT_KEEP, self.max_per_hour)

    def in_hour(self, now: float) -> int:
        """Merges in the hour before `now` (re-applies of reverted ones not
        counted)."""
        return sum(1 for r in self.recent if not r.get("reapply")
                   and (_epoch(r.get("at")) or -math.inf) > now - 3600)

    def save(self, now=None) -> None:
        now = self.clock() if now is None else now
        self.beat_at = now
        self.st.update(updated=_iso(now), merges_last_hour=self.in_hour(now),
                       recent_merges=self.recent[-RECENT_KEEP:])
        fb = self.st.pop("status_fallback", None)
        try:
            write_status(self.path, self.st)
        except Exception as e:       # a full disk: the fallback, else a restart
            err = f"{type(e).__name__}: {e}"
        else:
            if self.on_fallback:
                self.log(f"{_iso(now)} merge status file {self.path} written again")
            self.status_fails, self.on_fallback = 0, False
            return
        if self.fallback:
            self.st["status_fallback"] = {"path": self.fallback, "status_file": self.path,
                                          "error": err[:300],
                                          "since": (fb or {}).get("since") or _iso(now)}
            try:
                write_status(self.fallback, self.st)
            except Exception as e:
                err += f"; fallback {self.fallback}: {type(e).__name__}: {e}"
            else:
                if not self.on_fallback:
                    self.log(f"{_iso(now)} merge status file {self.path} not written ({err}): "
                             f"written to {self.fallback} meanwhile")
                self.status_fails, self.on_fallback = 0, True
                return
        self.status_fails += 1
        self.log(f"{_iso(now)} merge status file not written ({err}; "
                 f"{self.status_fails} in a row)")
        if self.status_fails >= STATUS_FAILS_MAX and not self.stop.is_set():
            self.log(f"{_iso(now)} merger stopping: no status written {self.status_fails} "
                     f"times in a row (the supervisor's watchdog reads its heartbeat "
                     f"there); exit {STATUS_EXIT}, the supervisor restarts it")
            self.status_dead = True
            self.stop.set()

    def beat(self) -> None:
        """Heartbeat from inside a pass (each group, each download): the
        status 'updated' refreshed, at most every BEAT_S."""
        now = self.clock()
        if now - self.beat_at >= BEAT_S:
            self.save(now)

    def due(self, now: float, rev) -> str | None:
        """Why a pass runs now, or None."""
        if rev != self.last_rev:
            return "start" if self.last_rev is None else "index changed"
        if self.waiting and now - self.last_pass >= self.retry_s:
            return "retry"
        return None

    def _merged(self, rec: dict) -> None:
        r = {"target": rec.get("target"), "members": rec.get("members"),
             "at": _iso(self.clock()), "commit": rec.get("commit")}
        # what reverted() checks the merge by in later passes, and its cap exemption
        r.update({k: rec[k] for k in ("n_frames", "generated", "merged_gen", "reapply")
                  if rec.get(k) is not None})
        self.recent = (self.recent + [r])[-self._keep():]
        self.save()                  # recorded before anything else can fail

    def _pass(self, now: float, rev, why: str) -> None:
        self.st["state"] = "pass"
        self.save(now)
        budget = max(0, self.max_per_hour - self.in_hour(now))
        rep = self.run(budget=budget, stop=self.stop, memo=self.memo, now=now,
                       on_merge=self._merged, recent=list(self.recent), beat=self.beat)
        self.last_rev, self.last_pass = rev, now
        since = {tuple(w["members"]): w["since"] for w in self.waiting}
        held = ([s for s in rep["skipped"] if s.get("wait")] + rep["failed"]
                + list(rep.get("unfinalized") or []))
        self.waiting = [{"gauge": s.get("gauge"), "members": s["members"], "why": s["why"],
                         "since": since.get(tuple(s["members"]), _iso(now))} for s in held]
        merged = [r["target"] for r in rep["merged"]]
        self.st["passes"] += 1
        self.st["last_pass"] = {
            "at": _iso(now), "trigger": why, "index_rev": rev,
            "seconds": round(self.clock() - now, 1), "index": rep["index"],
            "groups": rep["groups"], "planned": rep["planned"], "merged": len(merged),
            "merged_targets": merged,
            "skipped": [{k: s[k] for k in ("gauge", "members", "why", "wait")}
                        for s in rep["skipped"]],
            "failed": rep["failed"], "already": rep["already"],
            "finalized": rep["finalized"], "unfinalized": rep.get("unfinalized") or [],
            "reverted": rep.get("reverted") or [], "stopped": rep["stopped"],
            "log": list(rep.get("log") or [])[-LOG_KEEP:]}
        self.st["waiting"] = self.waiting
        n_re = sum(1 for r in rep["merged"] if r.get("reapply"))
        self.log(f"{_iso(now)} merge pass ({why}) index {str(rev)[:12]}: "
                 f"{rep['groups']} group(s), planned {rep['planned']}, merged {len(merged)}"
                 + (f" [{', '.join(merged)}]" if merged else "")
                 + (f" ({n_re} re-applied after a revert)" if n_re else "")
                 + f", skipped {len(rep['skipped'])}, failed {len(rep['failed'])}, "
                   f"waiting {len(self.waiting)}; {self.in_hour(self.clock())}/"
                   f"{self.max_per_hour} merges in the last hour"
                 + ("; STOPPED" if rep["stopped"] else ""))

    def step(self) -> float:
        """One cycle: read the revision, pass when due. Seconds to sleep next."""
        now = self.clock()
        try:
            rev = self.rev()
            why = self.due(now, rev)
            if why:
                self._pass(now, rev, why)
        except Exception as e:
            self.backoff = min(BACKOFF_MAX_S, max(BACKOFF_MIN_S, 2 * self.backoff))
            err = f"{type(e).__name__}: {e}"[:500]
            self.st["errors"] += 1
            self.st["last_error"] = {"at": _iso(now), "error": err, "retry_in_s": self.backoff}
            self.st["state"] = "backoff"
            self.log(f"{_iso(now)} merge ERROR {err} -> retry in {self.backoff / 60:g} min")
            wait = self.backoff
        else:
            self.backoff = 0.0
            self.st["state"] = "idle"
            wait = self.interval
            if self.waiting:
                wait = min(wait, max(1.0, self.last_pass + self.retry_s - now))
        self.st["retry_at"] = (_iso(self.last_pass + self.retry_s)
                               if self.waiting and self.last_pass is not None else None)
        self.st["next_pass_at"] = _iso(now + wait)
        self.save(now)
        return wait

    def run_forever(self) -> int:
        self.log(f"{_iso(self.clock())} merger watching {eventstore.REPO} {P}/index.json: "
                 f"every {self.interval:g} s, waiting groups retried after "
                 f"{self.retry_s / 60:g} min, <= {self.max_per_hour} merges/h, "
                 f"status {self.path}"
                 + (f" (fallback {self.fallback})" if self.fallback else ""))
        self.save()                  # this pid's heartbeat starts now
        while not self.stop.is_set():
            wait = self.step()
            # naps of at most one interval, the heartbeat between them (a
            # backoff runs to 60 min; the watchdog must not take it for a hang)
            while wait > 0 and not self.stop.is_set():
                nap_s = min(wait, self.interval)
                self.sleep(nap_s)
                wait -= nap_s
                if wait > 0 and not self.stop.is_set():
                    self.save()
        self.st.update(state="stopped", next_pass_at=None, retry_at=None)
        self.save()
        if self.status_dead:
            self.log(f"{_iso(self.clock())} merger stopped: status file unwritable "
                     f"(exit {STATUS_EXIT})")
            return STATUS_EXIT
        self.log(f"{_iso(self.clock())} merger stopped")
        return 0


def watch(store, a, gap_h) -> int:
    """--watch: SIGTERM/SIGINT stop the Watcher between groups."""
    stop = Stop()
    for name in ("SIGTERM", "SIGINT"):
        try:
            signal.signal(getattr(signal, name), stop.set)
        except (AttributeError, ValueError, OSError):
            pass
    try:
        os.nice(10)                  # the runner's solves keep the CPU
    except (AttributeError, OSError):
        pass

    def run(**kw):
        lines, beat = [], kw.get("beat") or (lambda: None)

        def log(s):                  # a line logged (finalize's scoring too) is a sign of life
            lines.append(s)
            beat()

        rep = run_pass(store, apply=True, gauge=a.gauge, max_groups=a.max_groups,
                       gap_h=gap_h, finalize=not a.no_finalize,
                       retry_s=60.0 * a.retry_min, log=log, **kw)
        rep["log"] = lines
        return rep

    w = Watcher(lambda: index_rev(store.api), run, a.status_file or default_status_file(),
                interval=a.interval, retry_min=a.retry_min,
                max_per_hour=a.max_merges_per_hour, stop=stop,
                fallback=fallback_status_file(),
                config={"repo": eventstore.REPO, "gap_h": gap_h, "gauge": a.gauge,
                        "max_groups": a.max_groups, "finalize": not a.no_finalize})
    return w.run_forever()


# ---- main ----------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=None,
                      help="print the per-group plan, write nothing (default)")
    mode.add_argument("--apply", action="store_true", help="perform the merges")
    ap.add_argument("--gauge", help="only the group(s) of this trigger gauge")
    ap.add_argument("--max-groups", type=int, default=0,
                    help="merge at most N groups (0: all eligible)")
    ap.add_argument("--gap-h", type=float, default=None,
                    help=f"grouping gap in hours (default {eventlist.MERGE_GAP_H:g}, "
                         f"EVENT_LIST_MERGE_GAP_H)")
    ap.add_argument("--no-finalize", action="store_true",
                    help="skip eventstore.finalize_episodes on the merged target")
    ap.add_argument("--watch", action="store_true",
                    help="run forever: a pass whenever events/index.json changes "
                         "(needs --apply and HF_TOKEN)")
    ap.add_argument("--interval", type=float, default=300.0,
                    help="--watch: seconds between index revision checks (300)")
    ap.add_argument("--retry-min", type=float, default=60.0,
                    help="--watch: minutes before waiting groups are retried (60)")
    ap.add_argument("--max-merges-per-hour", type=int, default=6,
                    help="--watch: merge commits in any hour, at most (6)")
    ap.add_argument("--status-file", default=None,
                    help="--watch: status JSON (default <CREST_CACHE_DIR or "
                         "tempdir>/merger_status.json)")
    a = ap.parse_args(argv)
    if a.watch and not a.apply:
        print("--watch merges: pass --apply with it")
        return 2
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    hub = bound_hub_timeouts()
    if a.watch:
        print(f"{_iso(time.time())} merger {hub}", flush=True)
    from huggingface_hub import HfApi

    tok = os.environ.get("HF_TOKEN") or None
    if a.apply and not tok:
        print("--apply needs HF_TOKEN in the environment")
        return 2
    store = Store(HfApi(token=tok), tok)
    gap_h = eventlist.MERGE_GAP_H if a.gap_h is None else a.gap_h
    if a.watch:
        return watch(store, a, gap_h)
    try:
        rep = run_pass(store, apply=a.apply, gauge=a.gauge, max_groups=a.max_groups,
                       gap_h=gap_h, finalize=not a.no_finalize)
    except Unreadable as e:
        print(f"{e}; nothing done")
        return 1
    print(f"done: {rep['planned']} group(s) planned, {len(rep['merged'])} merged, "
          f"{len(rep['failed'])} failed, {len(rep['skipped'])} skipped, "
          f"{rep['already']} already merged"
          + ("" if a.apply else "; DRY RUN, nothing written (--apply performs it)"))
    return 1 if rep["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
