"""Inundation-event results on the CREST_data dataset (V25).

Layout:  events/<event_id>/manifest.json + depth_*.tif + maxdepth.tif + dem.tif
         events/index.json   (id -> summary; the app lists events from this)

Storage discipline (HF limits):
  * ONE create_commit per event — all frames + manifest + index update +
    retention deletions batched together (account budget: 256 commits/h).
  * Frames are uint16-cm DEFLATE GeoTIFFs (~0.1-0.5 MB each at 1"); a whole
    event is ~10-30 MB. KEEP_EVENTS caps steady-state usage (< ~1 GB).
  * No DEM/forcing archive: DEM comes from public 3DEP on demand; runoff
    grids are reproducible from the MRMS archive already in CREST_data.
  * Animation cleanup (monthly): while an event is on the list it stays fully
    animatable; when it rolls off (KEEP_EVENTS / MAX_AGE_D) only its animation
    files (depth_* frames + archive.parquet) are cleared, and the simulation
    RESULTS (manifest + max-depth map + final product + DEM) stay in the store
    as a lightweight archive until RESULTS_MAX_AGE_D.
"""
from __future__ import annotations

import datetime
import io
import json
import os
import re
import threading
import time

REPO = os.environ.get("CREST_DATA_REPO", "vincewin/CREST_data")
PREFIX = "events"
# tiered retention: newest KEEP_FULL events keep their full frame stacks;
# older ones are demoted to manifest + maxdepth (tif+png) only; beyond
# KEEP_EVENTS the folder is deleted. Independently of the count caps,
# MAX_AGE_D bounds how long the event LIST grows: events older than this are
# removed entirely (folder + index entry) by the hourly retention_sweep, so
# the panel stays a rolling ~month.
# KEEP_FULL defaults to KEEP_EVENTS (user directive 2026-08-21): every event
# still on the list keeps its depth frames, so the time scrubber works for
# ALL of them. Demotion at 8 was silently deleting the frames of ended
# episodes within hours (active events re-publish each tick and bounce to the
# front, pushing ended re-run episodes past position 8), leaving them
# max-depth-only with no scrubber. ~40 full events ~= 1-1.4 GB on the HF
# dataset, well within limits. Set EVENT_KEEP_FULL below EVENT_KEEP to
# re-enable demotion of the oldest listed events if storage ever needs it.
KEEP_EVENTS = int(os.environ.get("EVENT_KEEP", "40"))
# Demotion is CLAMPED OFF unless EVENT_DEMOTE=1 is set explicitly: a worker
# environment still carrying EVENT_KEEP_FULL=8 (the HPC launch env, found
# 2026-09-11) had silently stripped the animations of 25 of 33 listed events
# — the scrubber showed nothing but the manifests still listed every frame.
_kf = os.environ.get("EVENT_KEEP_FULL")
KEEP_FULL = (int(_kf) if _kf and os.environ.get("EVENT_DEMOTE") == "1"
             else KEEP_EVENTS)
MAX_AGE_D = float(os.environ.get("EVENT_MAX_AGE_D", "30"))
# calendar-month list (user directive 2026-09-11): the public list is cleared
# monthly — an ended event whose t_end falls before the first day of the
# current UTC month rolls off (results kept, exactly like the age roll-off).
# MAX_AGE_D stays as the outer bound. EVENT_LIST_MONTHLY=0 disables.
LIST_MONTHLY = os.environ.get("EVENT_LIST_MONTHLY", "1") != "0"
# when an event rolls OFF the list (count cap or MAX_AGE_D age), we clear only
# its ANIMATION files and KEEP the simulation results (manifest + max-depth map
# + final product + DEM) in the store — a monthly animation cleanup that leaves
# a permanent lightweight result archive (user directive 2026-08-21). The kept
# results are ~1.5 MB/event; RESULTS_MAX_AGE_D is the far horizon past which
# even the result is removed entirely, so the archive can't grow without bound.
RESULTS_MAX_AGE_D = float(os.environ.get("EVENT_RESULTS_MAX_AGE_D", "365"))
# animation = the scrubbable per-frame depth grids + the compact frame archive;
# everything else in an event folder is the "simulation result" we keep
_ANIM_RE = re.compile(r"^(?:depth_\d+\.(?:png|tif)|archive\.parquet)$")
_FRAME_RE = re.compile(r"^depth_\d+\.(?:png|tif)$")


def _anim_delete_ops(allfiles, ev):
    """Delete ops for just the animation files of event `ev` (keeps manifest,
    maxdepth.*, final.json, dem.tif, domain.geojson)."""
    from huggingface_hub import CommitOperationDelete
    pre = f"{PREFIX}/{ev}/"
    return [CommitOperationDelete(f) for f in allfiles
            if f.startswith(pre) and _ANIM_RE.match(os.path.basename(f))]

_lock = threading.Lock()


def _age_days(event_id: str) -> float | None:
    """Event age from the id's YYYYMMDDHH prefix (e.g. 2026081015_07233650)."""
    try:
        t = datetime.datetime.strptime(str(event_id)[:10], "%Y%m%d%H")
        return (datetime.datetime.utcnow() - t).total_seconds() / 86400.0
    except ValueError:
        return None


def _month_start() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-01T00:00Z")


def _ended_before_month(event_id: str, summary: dict | None) -> bool:
    """True when the event's t_end (index summary; id hour as fallback) is
    before the first day of the current UTC month. Both sides are
    'YYYY-MM-DDTHH:MMZ' strings, so a plain comparison orders them."""
    t_end = str((summary or {}).get("t_end") or "")
    if not t_end:
        try:
            t_end = datetime.datetime.strptime(
                str(event_id)[:10], "%Y%m%d%H").strftime("%Y-%m-%dT%H:%MZ")
        except ValueError:
            return False
    return t_end[:16] < _month_start()[:16]


def _aged_out(idx: dict, keep: str | None = None) -> list[str]:
    return [e for e in idx
            if e != keep
            and idx[e].get("status") != "active"
            and ((_age_days(e) or 0.0) > MAX_AGE_D
                 or (LIST_MONTHLY and _ended_before_month(e, idx[e])))]


def _serviceable(frames: list, local: set, stored: set) -> list:
    """Frames whose depth file AND overlay png exist (or are about to be
    uploaded) — the only ones a manifest may list. Anything else is a 404
    blank in the player."""
    have = local | stored
    out = []
    for f in frames or []:
        fn, png = f.get("file"), f.get("png")
        if fn in have and (not png or png in have):
            out.append(f)
    return out


def _api():
    from huggingface_hub import HfApi
    tok = os.environ.get("HF_TOKEN")
    return HfApi(token=tok) if tok else None


def load_index() -> dict:
    """{event_id: summary} newest-first insertion order; {} on any failure."""
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO, f"{PREFIX}/index.json", repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"))
        with open(p, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def publish_event(local_dir: str, manifest: dict) -> bool:
    """Upload one finished event dir + updated index + prune old events,
    all in a single commit. Returns True on success."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    api = _api()
    if api is None:
        return False
    ev = manifest["event_id"]
    # V30 provenance: stamped HERE so it describes the machine that actually
    # solved (every engine publishes through its own copy of this function),
    # and rewritten into the local manifest.json before upload below
    if "provenance" not in manifest:
        try:
            from . import provenance
            manifest["provenance"] = provenance.stamp()
        except Exception:
            pass
    try:
        with open(os.path.join(local_dir, "manifest.json"), "w") as fp:
            json.dump(manifest, fp)
    except Exception:
        pass
    with _lock:
        idx = load_index()
        prior = idx.pop(ev, None)
        summary = {k: manifest.get(k) for k in
                   ("bbox", "t0", "sim_start", "t_end", "model", "trigger",
                    "generated", "gauge", "domain")}
        summary["engine"] = (manifest.get("provenance") or {}).get("engine")
        summary["n_frames"] = len(manifest.get("frames", []))
        summary["status"] = manifest.get("status", "active")
        # the episode keeps its first trigger time across hourly re-publishes
        summary["episode_started"] = ((prior or {}).get("episode_started")
                                      or manifest.get("t0"))
        summary["archive_frames"] = manifest.get("archive_frames")
        # newest first
        idx = {ev: summary, **idx}
        drop = list(idx.keys())[KEEP_EVENTS:]
        drop += [d for d in _aged_out(idx, keep=ev) if d not in drop]
        for d in drop:
            idx.pop(d, None)

        ops = []
        local = {fn for fn in os.listdir(local_dir)
                 if os.path.isfile(os.path.join(local_dir, fn))}
        if prior is not None:
            # FIXED EPISODE WINDOW (user directive 2026-08-14): a re-publish
            # replaces only what this run re-simulated — frames OLDER than
            # its sim_start are carried forward, so the episode record always
            # starts at trigger-minus-backset and never slides (03190000's
            # crest was lost to the old replace-the-folder behavior).
            new_start = str(manifest.get("sim_start") or "")
            stored = None                     # this event's files on the store
            try:
                allfiles = api.list_repo_files(REPO, repo_type="dataset")
                stored = {os.path.basename(f) for f in allfiles
                          if f.startswith(f"{PREFIX}/{ev}/")}
            except Exception:
                stored = None
            pm = _prior_manifest(ev) if new_start else None
            carried = [f for f in (pm or {}).get("frames", [])
                       if f.get("t", "") < new_start]
            if carried:
                manifest["frames"] = carried + manifest.get("frames", [])
                manifest["episode_start"] = carried[0].get("t")
            if stored is not None:
                # list only frames the store will actually serve after this
                # commit (a prior publish may have lost files — see below)
                n0 = len(manifest.get("frames") or [])
                manifest["frames"] = _serviceable(manifest.get("frames"),
                                                  local, stored)
                if len(manifest["frames"]) < n0:
                    manifest["n_frames_dropped"] = n0 - len(manifest["frames"])
                referenced = set()
                for f in manifest["frames"]:
                    referenced.add(f.get("file"))
                    referenced.add(f.get("png"))
                for base in stored:
                    if base in local:
                        # replaced IN PLACE by the Add below. NEVER pair it
                        # with a Delete: huggingface_hub silently drops an Add
                        # whose bytes already match the store, so Delete+Add
                        # of an unchanged file nets to a DELETE — that is how
                        # every same-hour re-solve and the first (identical)
                        # hour of each catch-up visit lost its frames while
                        # the manifest still listed them (found 2026-09-11).
                        continue
                    if _FRAME_RE.match(base) and base not in referenced:
                        ops.append(CommitOperationDelete(
                            f"{PREFIX}/{ev}/{base}"))     # re-simulated span
            summary["n_frames"] = len(manifest.get("frames") or [])
            with open(os.path.join(local_dir, "manifest.json"),
                      "w") as fp:
                json.dump(manifest, fp)
        for fn in sorted(os.listdir(local_dir)):
            p = os.path.join(local_dir, fn)
            if os.path.isfile(p):
                ops.append(CommitOperationAdd(f"{PREFIX}/{ev}/{fn}", p))
        # demote events beyond KEEP_FULL (default = KEEP_EVENTS, so normally
        # none): strip the animation, keep the results, stay listed.
        demote = [d for d in list(idx.keys())[KEEP_FULL:]
                  if not idx[d].get("demoted")]
        # events leaving the list (count cap or aged out): keep their
        # simulation RESULTS, clear only the animation files — unless they are
        # older than RESULTS_MAX_AGE_D, then remove the result too.
        strip = [d for d in drop if (_age_days(d) or 0.0) <= RESULTS_MAX_AGE_D]
        purge = [d for d in drop if d not in strip]
        allfiles = []
        if demote or strip:
            try:
                allfiles = api.list_repo_files(REPO, repo_type="dataset")
            except Exception:
                allfiles, demote, strip = [], [], []   # leave folders for the sweep
        for d in demote:
            ops += _anim_delete_ops(allfiles, d)
            idx[d]["demoted"] = True
        for d in strip:
            ops += _anim_delete_ops(allfiles, d)        # keep results, drop anim
        ops.append(CommitOperationAdd(
            f"{PREFIX}/index.json",
            io.BytesIO(json.dumps(idx).encode())))
        for d in purge:
            ops.append(CommitOperationDelete(f"{PREFIX}/{d}/", is_folder=True))
        try:
            api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                              commit_message=f"event {ev} "
                                             f"(+{len(ops) - 1 - len(drop)} files, "
                                             f"-{len(drop)} old)")
            return True
        except Exception:
            return False


def _prior_manifest(ev: str) -> dict | None:
    """The event's currently-published manifest (None if never published)."""
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO, f"{PREFIX}/{ev}/manifest.json",
                            repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"),
                            force_download=True)
        with open(p, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def mark_ended(event_ids) -> bool:
    """Flip episodes to 'ended' in the index (single small commit). Their
    folders — including archive.parquet — stay until retention removes them."""
    from huggingface_hub import CommitOperationAdd
    api = _api()
    if api is None:
        return False
    with _lock:
        idx = load_index()
        changed = [e for e in event_ids
                   if e in idx and idx[e].get("status") != "ended"]
        if not changed:
            return True
        for e in changed:
            idx[e]["status"] = "ended"
        try:
            api.create_commit(
                repo_id=REPO, repo_type="dataset",
                operations=[CommitOperationAdd(
                    f"{PREFIX}/index.json",
                    io.BytesIO(json.dumps(idx).encode()))],
                commit_message=f"episodes ended: {', '.join(changed)}")
            return True
        except Exception:
            return False


def finalize_episodes(event_ids, log=print) -> int:
    """V30 episode final product (user directive 2026-08-14): when an
    episode ends, its running event entry is REPLACED by a consolidated
    final product — the hourly re-publishes were fragments of the episode;
    only the finished episode is worth keeping. Per episode this writes:

      events/<id>/final.json     full detail: episode stats + the
                                 episode-scoped nowcast verification
      index entry                status "ended" + summary["final"] (the
                                 card the dashboard shows instead of the
                                 live-event presentation)
      events/scorecard.json      per-gauge running list of episode scores
                                 (the model's track record where it counts)

    Idempotent: episodes already carrying "final" are skipped, so the
    hourly tick can re-call it to heal a lost index race (a worker publish
    that snapshotted the index mid-finalize). Returns episodes finalized."""
    from huggingface_hub import CommitOperationAdd, hf_hub_download
    from . import eventscore
    api = _api()
    if api is None:
        return 0
    tok = os.environ.get("HF_TOKEN")
    done = 0
    fin_by_ev = {}
    for ev in event_ids:
        idx = load_index()
        s = idx.get(ev)
        if not s or s.get("final"):
            continue
        man = _prior_manifest(ev) or {}
        gid = str(((s.get("trigger") or {}).get("gauge"))
                  or s.get("gauge") or man.get("gauge") or "")
        started = s.get("episode_started") or s.get("t0") or ""
        try:
            t_start = datetime.datetime.strptime(started, "%Y-%m-%dT%H:%MZ")
        except ValueError:
            t_start = None
        frames = man.get("frames") or []
        t_last = None
        try:
            t_last = datetime.datetime.strptime(frames[-1]["t"],
                                                "%Y-%m-%dT%H:%MZ")
        except Exception:
            pass
        ended_t = t_last or datetime.datetime.utcnow()

        # inundation stats from the episode-wide maxdepth
        depth_stats = {}
        try:
            import rasterio
            p = hf_hub_download(REPO, f"{PREFIX}/{ev}/maxdepth.tif",
                                repo_type="dataset", token=tok,
                                force_download=True)
            with rasterio.open(p) as ds:
                md = ds.read(1).astype(float) / 100.0
            gr = man.get("grid") or {}
            cell_km2 = (gr.get("dx_m", 30.0) * gr.get("dy_m", 30.0)) / 1e6
            depth_stats = {
                "peak_depth_m": round(float(md.max()), 2),
                "inundated_km2": round(float((md >= 0.10).sum()) * cell_km2, 2),
                "severe_km2": round(float((md >= 1.0).sum()) * cell_km2, 2)}
        except Exception as e:
            log(f"finalize {ev}: maxdepth stats skipped ({type(e).__name__})")

        # observed peak from the episode's merged hydrograph
        peak_obs, peak_obs_t, peak_sim = None, None, None
        for r in man.get("hydro") or []:
            if r.get("obs_q") is not None and \
                    (peak_obs is None or r["obs_q"] > peak_obs):
                peak_obs, peak_obs_t = r["obs_q"], r.get("time")
            if r.get("sim_q") is not None and \
                    (peak_sim is None or r["sim_q"] > peak_sim):
                peak_sim = r["sim_q"]

        # episode-scoped nowcast verification (user: score only where
        # there's variation — never diluted by flat days)
        score = None
        if gid and t_start:
            try:
                score = eventscore.score_episode(
                    gid, t_start - datetime.timedelta(hours=12), ended_t,
                    log=log)
            except Exception as e:
                log(f"finalize {ev}: scoring failed ({type(e).__name__}: {e})")
        # the scorer's obs series is the authority on the episode peak — the
        # manifest hydro only spans sim windows and can lag/undersample
        # (seen live: hydro peak 15.7 vs true obs peak 34.5 on 03274650)
        if score and (peak_obs is None
                      or (score.get("obs_peak_m3s") or 0) > peak_obs):
            peak_obs = score.get("obs_peak_m3s")
            peak_obs_t = score.get("obs_peak_t")

        dur_h = (round((ended_t - t_start).total_seconds() / 3600.0, 1)
                 if t_start else None)
        final = {"ended": ended_t.strftime("%Y-%m-%dT%H:%MZ"),
                 "duration_h": dur_h, "n_frames": len(frames),
                 "peak_obs_m3s": peak_obs, "peak_obs_t": peak_obs_t,
                 "peak_sim_m3s": peak_sim, **depth_stats,
                 "nse_h1": (score or {}).get("nse_h1"),
                 "nse_h6": (score or {}).get("nse_h6")}
        fin_by_ev[ev] = {"final": final, "score": score, "gauge": gid}
        done += 1

    if not fin_by_ev:
        return 0
    with _lock:
        idx = load_index()
        ops = []
        # scorecard: the model's running track record, episodes only
        card = {}
        try:
            p = hf_hub_download(REPO, f"{PREFIX}/scorecard.json",
                                repo_type="dataset", token=tok,
                                force_download=True)
            with open(p, encoding="utf-8") as fp:
                card = json.load(fp)
        except Exception:
            pass
        for ev, d in fin_by_ev.items():
            if ev in idx:
                idx[ev]["status"] = "ended"
                idx[ev]["final"] = d["final"]
            ops.append(CommitOperationAdd(
                f"{PREFIX}/{ev}/final.json",
                io.BytesIO(json.dumps(
                    {"event_id": ev, **d["final"],
                     "nowcast_verification": d["score"]}).encode())))
            if d["gauge"] and d["score"]:
                rows = [r for r in card.get(d["gauge"], [])
                        if r.get("event_id") != ev]
                rows.append({"event_id": ev, **{k: d["score"].get(k) for k in
                             ("n_issues", "obs_peak_m3s", "nse_h1",
                              "nse_h6")},
                             "peak_ratio_h6": (d["score"]["leads"].get("6")
                                               or {}).get("peak_ratio")})
                card[d["gauge"]] = rows[-20:]
        if any(d["gauge"] and d["score"] for d in fin_by_ev.values()):
            ops.append(CommitOperationAdd(
                f"{PREFIX}/scorecard.json",
                io.BytesIO(json.dumps(card).encode())))
        ops.append(CommitOperationAdd(
            f"{PREFIX}/index.json", io.BytesIO(json.dumps(idx).encode())))
        try:
            api.create_commit(repo_id=REPO, repo_type="dataset",
                              operations=ops,
                              commit_message="episode final products: "
                                             + ", ".join(fin_by_ev))
            log(f"finalized {done} episode(s): {', '.join(fin_by_ev)}")
            return done
        except Exception as e:
            log(f"finalize commit failed ({type(e).__name__})")
            return 0


def retention_sweep() -> int:
    """Hourly retention: events past EVENT_MAX_AGE_D roll OFF the list, keeping
    their simulation results (only the animation files are cleared); result
    folders past EVENT_RESULTS_MAX_AGE_D are removed entirely. One commit,
    almost always a no-op. Returns the number of events touched.

    Runs even during quiet stretches so the list ages out with no new events;
    it is also what clears the animations of events that left the list without
    a subsequent publish_event to do it (the store keeps the result folder)."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    api = _api()
    if api is None:
        return 0
    with _lock:
        idx = load_index()
        aged = _aged_out(idx)                       # leaving the list this pass
        try:
            allfiles = api.list_repo_files(REPO, repo_type="dataset")
        except Exception:
            allfiles = []
        # event ids with a result folder in the store but no longer listed
        stored = {f.split("/")[1] for f in allfiles
                  if f.startswith(f"{PREFIX}/") and f.count("/") >= 2}
        orphans = stored - set(idx.keys()) - set(aged)

        ops, stripped, purged = [], 0, 0
        for e in aged:                              # keep result, clear animation
            idx.pop(e, None)
            new = _anim_delete_ops(allfiles, e)
            ops += new
            stripped += bool(new)
        for e in orphans:                           # already off-list archives
            if (_age_days(e) or 0.0) > RESULTS_MAX_AGE_D:
                ops.append(CommitOperationDelete(f"{PREFIX}/{e}/", is_folder=True))
                purged += 1
            else:
                extra = _anim_delete_ops(allfiles, e)   # clear any stray animation
                ops += extra
                stripped += bool(extra)
        if not ops:
            return 0
        ops.append(CommitOperationAdd(f"{PREFIX}/index.json",
                                      io.BytesIO(json.dumps(idx).encode())))
        try:
            api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                              commit_message=f"retention: {stripped} animation(s) "
                                             f"cleared (results kept), {purged} "
                                             f"result(s) purged > {RESULTS_MAX_AGE_D:g} d")
            return stripped + purged
        except Exception:
            return 0


def reconcile_store(dry_run: bool = False, log=print) -> int:
    """Maintenance (one commit): make every stored manifest list only the
    frames whose depth file + overlay png are actually in the store, and
    fix the index's n_frames to match. Repairs the 404-blank frames left by
    the Delete+Add publish bug and by animation stripping. Returns the
    number of manifests rewritten."""
    from huggingface_hub import CommitOperationAdd, hf_hub_download
    api = _api()
    if api is None:
        return 0
    with _lock:
        allfiles = api.list_repo_files(REPO, repo_type="dataset")
        by_ev: dict[str, set] = {}
        for f in allfiles:
            parts = f.split("/")
            if len(parts) >= 3 and parts[0] == PREFIX and parts[1] != "queue":
                by_ev.setdefault(parts[1], set()).add(parts[-1])
        idx = load_index()
        ops, touched = [], 0
        for ev in sorted(by_ev):
            files = by_ev[ev]
            if "manifest.json" not in files:
                continue
            try:
                mp = hf_hub_download(REPO, f"{PREFIX}/{ev}/manifest.json",
                                     repo_type="dataset",
                                     token=os.environ.get("HF_TOKEN"))
                with open(mp, encoding="utf-8") as fp:
                    man = json.load(fp)
            except Exception as e:
                log(f"{ev}: manifest unreadable ({type(e).__name__})")
                continue
            frames = man.get("frames") or []
            keep = _serviceable(frames, set(), files)
            if len(keep) == len(frames):
                continue
            man["frames"] = keep
            man["n_frames_dropped"] = (man.get("n_frames_dropped") or 0) + \
                len(frames) - len(keep)
            log(f"{ev}: {len(frames)} -> {len(keep)} frames "
                f"({len(frames) - len(keep)} listed but not stored)")
            touched += 1
            if dry_run:
                continue
            ops.append(CommitOperationAdd(
                f"{PREFIX}/{ev}/manifest.json",
                io.BytesIO(json.dumps(man).encode())))
            if ev in idx:
                idx[ev]["n_frames"] = len(keep)
        if dry_run or not ops:
            return touched
        ops.append(CommitOperationAdd(f"{PREFIX}/index.json",
                                      io.BytesIO(json.dumps(idx).encode())))
        api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                          commit_message=f"reconcile: {touched} manifest(s) "
                                         f"trimmed to stored frames")
        return touched


def event_url(event_id: str, filename: str) -> str:
    return (f"https://huggingface.co/datasets/{REPO}/resolve/main/"
            f"{PREFIX}/{event_id}/{filename}")


# ---- admin archive: EVERY stored result folder, listed or not ---------------
# The public list is a rolling ~month (KEEP_EVENTS / MAX_AGE_D); the store keeps
# each result for RESULTS_MAX_AGE_D with only the animation cleared. The owner
# reaches those through /api/admin/archive (user directive 2026-09-08).
_ARCH_TTL_S = float(os.environ.get("EVENT_ARCHIVE_TTL_S", "600"))
_arch = {"built": 0.0, "building": False, "events": {}, "error": None}
_arch_lock = threading.Lock()
_ANIM_FILE = re.compile(r"^depth_\d+\.(?:png|tif)$")


def _arch_summary(ev: str, m: dict, files: list, listed: dict | None) -> dict:
    tr = m.get("trigger") or {}
    dom = m.get("domain") or {}
    pv = m.get("provenance") or {}
    fin = m.get("final") or (listed or {}).get("final")
    frames_stored = sum(1 for f in files if _ANIM_FILE.match(f) and f.endswith(".png"))
    return {
        "id": ev, "listed": listed is not None,
        "gauge": tr.get("gauge") or m.get("gauge"), "gauge_name": tr.get("name"),
        "lat": tr.get("lat"), "lon": tr.get("lon"),
        "t0": m.get("t0"), "sim_start": m.get("sim_start"), "t_end": m.get("t_end"),
        "generated": m.get("generated"), "status": m.get("status"),
        "model": m.get("model"),
        "engine": pv.get("engine") or (listed or {}).get("engine"),
        "crestimap": pv.get("crestimap"),
        "area_km2": dom.get("area_km2"),
        "n_hucs": dom.get("n_hucs") or len(m.get("huc12s") or []),
        "depth_cap_m": m.get("depth_cap_m"), "bounds": m.get("bounds"),
        "n_frames_manifest": len(m.get("frames") or []),
        "frames_stored": frames_stored,
        "has_flux": bool(m.get("maxspeed")),
        "has_maxdepth": (m.get("maxdepth_png") or "maxdepth.png") in files,
        "peak_depth_m": fin.get("peak_depth_m") if isinstance(fin, dict) else None,
        # result files present (animation excluded) -> the UI offers downloads
        "files": sorted(f for f in files
                        if not _ANIM_FILE.match(f) and f != "archive.parquet"),
    }


def _arch_build():
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    try:
        allfiles = HfApi(token=tok).list_repo_files(REPO, repo_type="dataset")
        idx = load_index()
        per: dict = {}
        for f in allfiles:
            if f.startswith(f"{PREFIX}/") and f.count("/") >= 2:
                ev, name = f.split("/", 2)[1:]
                per.setdefault(ev, []).append(name)

        def one(ev):
            try:                      # HF-cached: re-downloaded only on change
                p = hf_hub_download(REPO, f"{PREFIX}/{ev}/manifest.json",
                                    repo_type="dataset", token=tok)
                with open(p, encoding="utf-8") as fp:
                    m = json.load(fp)
            except Exception:
                m = {}
            return ev, _arch_summary(ev, m, per[ev], idx.get(ev))

        with ThreadPoolExecutor(max_workers=8) as ex:
            out = dict(ex.map(one, sorted(per)))
        with _arch_lock:
            _arch.update(events=out, built=time.time(), error=None)
    except Exception as e:
        with _arch_lock:
            _arch["error"] = f"{type(e).__name__}: {e}"
    finally:
        with _arch_lock:
            _arch["building"] = False


def archive_index(force: bool = False) -> dict:
    """Admin view of every result folder in the store. Returns the current
    snapshot immediately; a stale or missing snapshot refreshes in the
    background (the caller polls while "building")."""
    with _arch_lock:
        stale = force or (time.time() - _arch["built"]) > _ARCH_TTL_S
        if stale and not _arch["building"]:
            _arch["building"] = True
            threading.Thread(target=_arch_build, daemon=True).start()
        evs = sorted(_arch["events"].values(),
                     key=lambda s: s.get("t0") or "", reverse=True)
        return {"events": evs, "n": len(evs),
                "n_listed": sum(1 for e in evs if e["listed"]),
                "building": _arch["building"], "built": _arch["built"],
                "error": _arch["error"], "results_max_age_d": RESULTS_MAX_AGE_D,
                "base": (f"https://huggingface.co/datasets/{REPO}/resolve/main/"
                         f"{PREFIX}")}
