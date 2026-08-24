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
"""
from __future__ import annotations

import datetime
import io
import json
import os
import threading

REPO = os.environ.get("CREST_DATA_REPO", "vincewin/CREST_data")
PREFIX = "events"
# tiered retention: newest KEEP_FULL events keep their full frame stacks;
# older ones are demoted to manifest + maxdepth (tif+png) only; beyond
# KEEP_EVENTS the folder is deleted. ~8 x 35 MB + 32 x ~1.5 MB < 350 MB.
# Independently of the count caps, MAX_AGE_D bounds how long the event LIST
# grows: events older than this are removed entirely (folder + index entry)
# by the hourly retention_sweep, so the panel stays a rolling ~month.
KEEP_FULL = int(os.environ.get("EVENT_KEEP_FULL", "8"))
KEEP_EVENTS = int(os.environ.get("EVENT_KEEP", "40"))
MAX_AGE_D = float(os.environ.get("EVENT_MAX_AGE_D", "30"))

_lock = threading.Lock()


def _age_days(event_id: str) -> float | None:
    """Event age from the id's YYYYMMDDHH prefix (e.g. 2026081015_07233650)."""
    try:
        t = datetime.datetime.strptime(str(event_id)[:10], "%Y%m%d%H")
        return (datetime.datetime.utcnow() - t).total_seconds() / 86400.0
    except ValueError:
        return None


def _aged_out(idx: dict, keep: str | None = None) -> list[str]:
    return [e for e in idx
            if e != keep
            and idx[e].get("status") != "active"
            and (_age_days(e) or 0.0) > MAX_AGE_D]


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
        if prior is not None:
            # FIXED EPISODE WINDOW (user directive 2026-08-14): a re-publish
            # replaces only what this run re-simulated — frames OLDER than
            # its sim_start are carried forward, so the episode record always
            # starts at trigger-minus-backset and never slides (03190000's
            # crest was lost to the old replace-the-folder behavior).
            local = set(os.listdir(local_dir))
            new_start = str(manifest.get("sim_start") or "")
            carried = []
            try:
                allfiles = api.list_repo_files(REPO, repo_type="dataset")
            except Exception:
                allfiles = None
            if allfiles is None or not new_start:
                ops.append(CommitOperationDelete(f"{PREFIX}/{ev}/",
                                                 is_folder=True))
            else:
                pm = _prior_manifest(ev)
                carried = [f for f in (pm or {}).get("frames", [])
                           if f.get("t", "") < new_start]
                keep = {f.get("file") for f in carried} | \
                       {f.get("png") for f in carried}
                for f in allfiles:
                    if not f.startswith(f"{PREFIX}/{ev}/"):
                        continue
                    base = os.path.basename(f)
                    if base in local or base not in keep:
                        ops.append(CommitOperationDelete(f))
                if carried:
                    manifest["frames"] = carried + manifest.get("frames", [])
                    manifest["episode_start"] = carried[0].get("t")
                    summary["n_frames"] = len(manifest["frames"])
                    with open(os.path.join(local_dir, "manifest.json"),
                              "w") as fp:
                        json.dump(manifest, fp)
        for fn in sorted(os.listdir(local_dir)):
            p = os.path.join(local_dir, fn)
            if os.path.isfile(p):
                ops.append(CommitOperationAdd(f"{PREFIX}/{ev}/{fn}", p))
        # demote events beyond KEEP_FULL: drop the frame stacks, keep
        # manifest + maxdepth + dem (validation/archive tier)
        demote = [d for d in list(idx.keys())[KEEP_FULL:]
                  if not idx[d].get("demoted")]
        if demote:
            try:
                allfiles = api.list_repo_files(REPO, repo_type="dataset")
            except Exception:
                allfiles, demote = [], []
            for d in demote:
                for f in allfiles:
                    base = os.path.basename(f)
                    if (f.startswith(f"{PREFIX}/{d}/")
                            and base.startswith("depth_")):
                        ops.append(CommitOperationDelete(f))
                idx[d]["demoted"] = True
        ops.append(CommitOperationAdd(
            f"{PREFIX}/index.json",
            io.BytesIO(json.dumps(idx).encode())))
        for d in drop:
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
    """Remove events older than EVENT_MAX_AGE_D (folder + index entry) in one
    commit. Called from the hourly tick so the list ages out even during
    quiet stretches with no new events; almost always a no-op. Returns the
    number of events removed (0 on no-op or failure)."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    api = _api()
    if api is None:
        return 0
    with _lock:
        idx = load_index()
        aged = _aged_out(idx)
        if not aged:
            return 0
        for e in aged:
            idx.pop(e, None)
        ops = [CommitOperationDelete(f"{PREFIX}/{e}/", is_folder=True)
               for e in aged]
        ops.append(CommitOperationAdd(f"{PREFIX}/index.json",
                                      io.BytesIO(json.dumps(idx).encode())))
        try:
            api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                              commit_message=f"retention: -{len(aged)} events "
                                             f"older than {MAX_AGE_D:g} d")
            return len(aged)
        except Exception:
            return 0


def event_url(event_id: str, filename: str) -> str:
    return (f"https://huggingface.co/datasets/{REPO}/resolve/main/"
            f"{PREFIX}/{event_id}/{filename}")
