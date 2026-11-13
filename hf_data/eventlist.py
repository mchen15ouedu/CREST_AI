"""Public 2-D inundation case list (GET /api/events): a VIEW of the store's
events/index.json, never a change to it.

User directive 2026-11-03: "Any results older than 30 days in 2D inundation
case list, remove from the list, but keep the data in the backend", and the
list "has many duplications". Both fixes are view-only: nothing in the store
is deleted or rewritten (eventstore's age retention, EVENT_MAX_AGE_D, stays
OFF: it clears animation files). /api/admin/archive and every other reader
keep using the full index (the archive takes only its `listed` flag from
this view: mark_listed).
  * absorbed: an entry carrying merged_into (folded into another episode by
    the store-side merge script) is never listed on its own, even when its
    target is gone; it counts toward the entry its merged_into chain ends at
    (list_info.absorbed counts them all).
  * collapse: the hourly tick splits one flood into many episodes on the
    same trigger gauge (09353800 had 12 at consecutive hours on 2026-11-03).
    Episodes of one gauge whose windows overlap or sit at most
    EVENT_LIST_MERGE_GAP_H hours apart are ONE entry, keyed by a
    representative member: the newest active one, else the deepest
    (final.peak_depth_m, ties -> newest). Its own summary is kept unchanged
    (its map and stats belong together) plus merged = {n, ids newest first,
    first_start, last_end}. A representative the merge script already gave
    a merged record keeps it as is while nothing joins it; when unmerged
    fragments do, the record is rebuilt over the union (each id once, the
    span covering the old record's). groups() is this grouping, pure and
    importable, so the merge script folds exactly as the list does.
  * roll-off: a group is hidden once its flood END (the newest over the
    group of final.ended, else t_end, else the id hour) is more than
    EVENT_LIST_MAX_AGE_D days ago; a group with an active episode never is.
    Exception: a retrospective reconstruction (trigger.retrospective, a
    backdated /api/event_run) is new to users when it is published, so it
    stays until EVENT_LIST_MAX_AGE_D days after its
    trigger.retrospective.requested_at, however long ago its flood ended.
    Grouping comes first, so an old fragment of a recent flood stays folded
    instead of resurfacing on its own.
Pinned demo events (events/pinned.json) protect DATA from retention; they are
NOT exempt from this roll-off by default: the pinned set is every result
simulated before 2026-10-16, and on 2026-11-03 every listed event older than
30 days was pinned, so exempting them would have hidden nothing.
EVENT_LIST_KEEP_PINNED=1 exempts them (a group with a pinned member stays).
EVENT_LIST_MAX_AGE_D (<= 0: off) and EVENT_LIST_MERGE_GAP_H (< 0: off) fall
back to 30 / 12 when unparsable or not finite.
"""
from __future__ import annotations

import datetime
import math
import os


def _num(v, default: float) -> float:
    """float(v) when that is a finite number, else `default`."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


MAX_AGE_D = _num(os.environ.get("EVENT_LIST_MAX_AGE_D"), 30.0)      # <= 0: off
MERGE_GAP_H = _num(os.environ.get("EVENT_LIST_MERGE_GAP_H"), 12.0)  # < 0: off
KEEP_PINNED = os.environ.get("EVENT_LIST_KEEP_PINNED", "0").strip() == "1"


def _utc(t) -> datetime.datetime | None:
    """'YYYY-MM-DDTHH:MM[:SS]Z' (any ISO form, or a datetime) -> naive UTC;
    None when missing or unparsable."""
    if isinstance(t, datetime.datetime):
        d = t
    else:
        try:            # Python 3.10 fromisoformat does not take a trailing Z
            d = datetime.datetime.fromisoformat(str(t).strip().replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if d.tzinfo is not None:
        d = d.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return d


def _id_hour(event_id: str) -> datetime.datetime | None:
    """The id's YYYYMMDDHH prefix (e.g. 2026103104_09353800)."""
    try:
        return datetime.datetime.strptime(str(event_id)[:10], "%Y%m%d%H")
    except ValueError:
        return None


def _iso(t: datetime.datetime | None) -> str | None:
    return t.strftime("%Y-%m-%dT%H:%MZ") if t else None


def _span(event_id: str, s: dict):
    """(start, end) of one episode, naive UTC. start = episode_started, else
    t0, else sim_start, else the id hour; end = final.ended, else t_end, else
    the id hour (never before start). (None, None) when nothing parses."""
    fin = s.get("final") if isinstance(s.get("final"), dict) else {}
    hour = _id_hour(event_id)
    start = (_utc(s.get("episode_started")) or _utc(s.get("t0"))
             or _utc(s.get("sim_start")) or hour)
    end = _utc(fin.get("ended")) or _utc(s.get("t_end")) or hour or start
    if start is None:
        start = end
    if end is not None and end < start:
        end = start
    return start, end


def _gauge(event_id: str, s: dict) -> str | None:
    """Trigger gauge, else the summary's gauge, else the id's suffix."""
    tg = s.get("trigger") if isinstance(s.get("trigger"), dict) else {}
    g = tg.get("gauge") or s.get("gauge")
    if not g and "_" in str(event_id):
        g = str(event_id).split("_", 1)[1]
    return str(g) if g else None


def _depth(s: dict) -> float:
    fin = s.get("final") if isinstance(s.get("final"), dict) else {}
    try:
        v = float(fin.get("peak_depth_m"))
    except (TypeError, ValueError):
        return -math.inf
    return v if math.isfinite(v) else -math.inf


def _into(s) -> str | None:
    """merged_into of an entry the merge script absorbed, else None."""
    t = s.get("merged_into") if isinstance(s, dict) else None
    return str(t) if t else None


def _requested(s) -> datetime.datetime | None:
    """trigger.retrospective.requested_at of a retrospective reconstruction;
    None for an operational episode."""
    tg = s.get("trigger") if isinstance(s, dict) else None
    r = tg.get("retrospective") if isinstance(tg, dict) else None
    return _utc(r.get("requested_at")) if isinstance(r, dict) else None


def groups(index: dict, gap_h: float | None = None) -> list:
    """The list's grouping, pure (`index` is never modified): one dict per
    flood, in the index's order (a group sits where its first-listed
    episode sat, absorbed ones included, so a merge script's folding does
    not reorder the list). Entries carrying merged_into are never members;
    each joins the entry its merged_into chain ends at (dropped when the
    chain dangles or loops). A member stands for itself, the ids its own
    merged record names and the entries absorbed into it, and its span
    covers theirs (and the record's first_start/last_end). Members of one
    gauge whose spans overlap or sit at most gap_h hours apart (default
    EVENT_LIST_MERGE_GAP_H; < 0: no grouping) are one group:
      rep         newest active member, else the deepest
                  (final.peak_depth_m), ties -> newest
      members     the index ids grouped here (never an absorbed one),
                  newest first
      ids         every episode the group stands for, members and absorbed
                  ones, newest first, each once
      n           len(ids), plus episodes an earlier record counted without
                  naming them
      start, end  naive-UTC span of the whole group (None when unknown)
      active      a member or an absorbed episode is active
    merged_record(g) is the summary.merged a merge script writes on rep."""
    gap_h = MERGE_GAP_H if gap_h is None else _num(gap_h, MERGE_GAP_H)
    idx = index if isinstance(index, dict) else {}

    def summ(e):
        s = idx.get(e)
        return s if isinstance(s, dict) else {}

    span = {e: _span(e, summ(e)) for e in idx}
    held = {e: {e} for e in idx if not _into(idx[e])}
    extra = dict.fromkeys(held, 0)
    for e in idx:                       # absorbed: follow the chain to a member
        h, seen = e, set()
        while h in idx and _into(idx[h]) and h not in seen:
            seen.add(h)
            h = _into(idx[h])
        if h != e and h in held:
            held[h].add(e)
    for e in held:                      # an earlier merge's record on a member
        m = summ(e).get("merged")
        if isinstance(m, dict):
            ids = m.get("ids") if isinstance(m.get("ids"), list) else []
            held[e] |= {str(i) for i in ids if i}
            # its n only adds episodes no id here accounts for
            extra[e] = max(0, int(_num(m.get("n"), 0)) - len(held[e]))

    wide, by_gauge = {}, {}
    for e in held:
        pts = [span[i] for i in held[e] if i in span]
        m = summ(e).get("merged")
        if isinstance(m, dict):
            pts.append((_utc(m.get("first_start")), _utc(m.get("last_end"))))
        st = min((a for a, _ in pts if a is not None), default=None)
        en = max((b for _, b in pts if b is not None), default=None)
        st, en = st or en, en or st     # one side missing: a point
        if st is not None and en < st:
            en = st
        wide[e] = (st, en)
        g = _gauge(e, summ(e)) if st is not None else None
        by_gauge.setdefault(g or ("\0" + e), []).append(e)  # no gauge/time: alone

    # interval merge per gauge: chronological, a member joins the running group
    # when it starts at most gap_h after the group's latest end (1e7 h caps an
    # absurd gap below timedelta's range)
    gap = datetime.timedelta(hours=min(gap_h, 1e7)) if gap_h >= 0 else None
    runs = []
    for members in by_gauge.values():
        cur, cur_end = [], None
        for e in sorted(members, key=lambda e: (wide[e][0] or datetime.datetime.min, e)):
            st, en = wide[e]
            if (cur and gap is not None and st is not None and cur_end is not None
                    and st - cur_end <= gap):
                cur.append(e)
                cur_end = max(cur_end, en)
            else:
                if cur:
                    runs.append(cur)
                cur, cur_end = [e], en
        if cur:
            runs.append(cur)

    pos = {e: i for i, e in enumerate(idx)}

    def newest(e):
        return ((span[e][0] if e in span else _id_hour(e)) or datetime.datetime.min, e)

    out = []
    held_pos = {e: min(pos[i] for i in held[e] if i in pos) for e in held}
    for run in sorted(runs, key=lambda r: min(held_pos[e] for e in r)):
        ids = set().union(*(held[e] for e in run))
        active = [e for e in run if summ(e).get("status") == "active"]
        if active:
            rep = max(active, key=newest)
        else:
            rep = max(run, key=lambda e: (_depth(summ(e)), newest(e)))
        out.append({
            "rep": rep, "members": sorted(run, key=newest, reverse=True),
            "ids": sorted(ids, key=newest, reverse=True),
            "n": len(ids) + sum(extra[e] for e in run),
            "start": min((wide[e][0] for e in run if wide[e][0] is not None), default=None),
            "end": max((wide[e][1] for e in run if wide[e][1] is not None), default=None),
            "active": any(summ(i).get("status") == "active" for i in ids)})
    return out


def merged_record(g: dict) -> dict:
    """summary.merged for a group from groups(): {n, ids newest first,
    first_start, last_end}, as the list shows it and a merge script writes it."""
    return {"n": g["n"], "ids": list(g["ids"]),
            "first_start": _iso(g["start"]), "last_end": _iso(g["end"])}


def list_view(index: dict, now=None, pinned=frozenset(),
              max_age_d: float | None = None, merge_gap_h: float | None = None):
    """(events, info) for the public list, built on groups(). `events` is
    {representative id: summary} in the index's newest-first order; `index`
    is never modified. `pinned` = ids exempt from the roll-off (a group with
    one stays); None = that set could not be read, so nothing is hidden (as
    eventstore._aged_out does). info: max_age_d, merge_gap_h, cutoff, total
    (index size), absorbed (merged_into entries, never listed), hidden_old
    (events rolled off, fragments included), merged_fragments (events folded
    into a listed entry here):
    len(events) + merged_fragments + hidden_old + absorbed == total."""
    max_age_d = MAX_AGE_D if max_age_d is None else _num(max_age_d, MAX_AGE_D)
    gap_h = MERGE_GAP_H if merge_gap_h is None else _num(merge_gap_h, MERGE_GAP_H)
    now = (_utc(now) if now is not None else None) or datetime.datetime.utcnow()
    idx = index if isinstance(index, dict) else {}
    try:
        cutoff = (now - datetime.timedelta(days=max_age_d)) if max_age_d > 0 else None
    except OverflowError:               # an absurd age: nothing is that old
        cutoff = None
    events, hidden, folded = {}, 0, 0
    for g in groups(idx, gap_h):
        run, rep = g["members"], g["rep"]
        # age runs from the flood's end, or from a retrospective's request
        # when that is later (a rebuilt old flood is news when published)
        keep = [t for t in (g["end"], *(_requested(idx.get(i)) for i in g["ids"]))
                if t is not None]
        if (cutoff is not None and not g["active"] and keep and max(keep) < cutoff
                and pinned is not None and not any(i in pinned for i in g["ids"])):
            hidden += len(run)
            continue
        s = idx[rep]
        if len(run) == 1 and (g["n"] == 1 or (isinstance(s, dict)
                                              and isinstance(s.get("merged"), dict))):
            events[rep] = s             # alone, or the merge script's record as is
            continue
        events[rep] = {**(s if isinstance(s, dict) else {}), "merged": merged_record(g)}
        folded += len(run) - 1
    info = {"max_age_d": max_age_d, "merge_gap_h": gap_h, "cutoff": _iso(cutoff),
            "total": len(idx), "absorbed": sum(1 for s in idx.values() if _into(s)),
            "hidden_old": hidden, "merged_fragments": folded}
    return events, info


def mark_listed(archive: dict, index: dict, now=None, pinned=frozenset()) -> dict:
    """The admin archive (eventstore.archive_index) with each row's `listed`
    = shown on the public list, a key of list_view(index, now, pinned): a
    rolled-off or folded episode is off it although the index still holds
    it (the snapshot's own flag only says "in the index"), and n_listed is
    recounted. A new dict with new rows; `archive` is never modified. An
    empty index (eventstore.load_index answers {} to any failure) leaves the
    archive as it is: it can't tell what is shown."""
    if not isinstance(index, dict) or not index:
        return archive
    shown = set(list_view(index, now, pinned)[0])
    rows = [{**r, "listed": r.get("id") in shown} for r in archive.get("events") or []]
    return {**archive, "events": rows, "n_listed": sum(1 for r in rows if r["listed"])}
