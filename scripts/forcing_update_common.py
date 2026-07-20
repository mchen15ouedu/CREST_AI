"""Shared machinery for the store-forcing updaters (MRMS, PET).

Both feeds live in vincewin/CREST_data as one month tar per variable/year
(`<var>/YYYY/<var>_YYYY_MM.tar`), a flat tar of per-timestep PQF members. Their
store grids ARE the native source grids (MRMS 0.01deg CONUS 3500x7000; PET 1deg
global 360x181), so — unlike the NARR temperature updater — no regridding is
needed: each updater just fetches the source field, writes it to a PQF on the
canonical grid, and this module appends the missing members to the month tar.

The append is STREAMING (old members are copied member-by-member from the
downloaded tar into the new one, never all held in RAM at once) because a MRMS
month tar is ~0.5 GB. This mirrors scripts/update_temp_narr.py's update_month
but is generic over the source via a `produce(timestep) -> pqf_bytes | None`
callback (None = the source has no data for that step yet, i.e. lag).
"""
from __future__ import annotations

import os
import re
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone

HF_REPO = "vincewin/CREST_data"
TOKEN_PATH = r"C:\Users\chenm\Documents\EF5\CREST_token.txt"


def hf_token() -> str | None:
    """HF write token: env (Space secret) first, local token file as fallback."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    return open(TOKEN_PATH).read().strip() if os.path.exists(TOKEN_PATH) else None


def member_hours(names, pats) -> set[datetime]:
    """Timesteps already present in a tar, parsed from member names."""
    hours: set[datetime] = set()
    for n in names:
        for p in pats:
            m = re.search(p, n)
            if not m:
                continue
            g = "".join(m.groups())
            fmt = "%Y%m%d%H" if len(g) == 10 else "%Y%m%d"
            try:
                hours.add(datetime.strptime(g, fmt))
            except ValueError:
                pass
            break
    return hours


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def wanted_timesteps(year: int, month: int, freq: str, have: set[datetime],
                     lag: timedelta) -> list[datetime]:
    """Missing timesteps in [month_start, min(now-lag, month_end)).

    `lag` trims the near-real-time tail the source cannot possibly have yet, so a
    normal run doesn't churn on hours/days that are always going to come back
    empty. Anything still missing inside the window is returned for a fetch."""
    ref = datetime(year, month, 1)
    end = min(_now() - lag, datetime(year + (month == 12), month % 12 + 1, 1))
    step = timedelta(hours=1) if freq == "h" else timedelta(days=1)
    if freq == "d":
        ref = ref.replace(hour=0)
    out, t = [], ref
    while t < end:
        if t not in have:
            out.append(t)
        t += step
    return out


def scan_months(back_days: int, months_arg: str | None) -> list[tuple[int, int]]:
    """Either the explicit --months list or every month in the last back_days."""
    if months_arg:
        return [tuple(int(x) for x in m.split("-")) for m in months_arg.split(",")]
    now = _now()
    start = now - timedelta(days=back_days)
    out, y, m = [], start.year, start.month
    while (y, m) <= (now.year, now.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def update_month(*, var: str, year: int, month: int, produce, api, dry_run: bool,
                 month_fmt: str, member_fmt: str, member_pats, freq: str,
                 source: str, lag: timedelta, cap: int | None = None,
                 log=print) -> str:
    """Fetch + append every missing member of one month tar. Returns a summary.

    `produce(t)` returns PQF bytes for timestep t, or None if the source has no
    data for it yet (counted as still-lagging, retried on the next run)."""
    from huggingface_hub import hf_hub_download, CommitOperationAdd
    ref = datetime(year, month, 1)
    tar_path = ref.strftime(month_fmt)
    old_local, have = None, set()
    try:
        old_local = hf_hub_download(HF_REPO, tar_path, repo_type="dataset")
        with tarfile.open(old_local) as tf:
            have = member_hours(tf.getnames(), member_pats)
    except Exception:
        pass                                          # brand-new month tar

    want = wanted_timesteps(year, month, freq, have, lag)
    if not want:
        return f"{year}-{month:02d}: complete ({len(have)} in store), nothing to do"
    capped = cap is not None and len(want) > cap
    if capped:
        want = want[:cap]

    new: list[tuple[str, str]] = []                   # (member_name, tmp_path)
    misses = 0
    for t in want:
        try:
            data = produce(t)
        except Exception as e:
            log(f"  [{var}] {t:%Y-%m-%d %H:%M} produce error: {e}")
            data = None
        if data is None:
            misses += 1
            continue
        name = t.strftime(member_fmt)
        p = os.path.join(tempfile.gettempdir(), f"_{var}_" + name.replace("/", "_"))
        with open(p, "wb") as fh:
            fh.write(data)
        new.append((name, p))

    if not new:
        return (f"{year}-{month:02d}: {len(want)} timestep(s) missing but {source} "
                f"has no data for them yet (lag)")
    if dry_run:
        for _, p in new:
            os.remove(p)
        note = f" (capped from more)" if capped else ""
        return f"{year}-{month:02d}: would add {len(new)} {source} member(s){note}"

    out_tar = os.path.join(tempfile.gettempdir(), f"{var}_{year}_{month:02d}.tar")
    with tarfile.open(out_tar, "w") as tf:
        if old_local:                                 # stream old members across
            with tarfile.open(old_local) as old:
                for ti in old.getmembers():
                    tf.addfile(ti, old.extractfile(ti))
        for name, p in sorted(new):                   # then the freshly fetched
            tf.add(p, arcname=name)
    for _, p in new:
        os.remove(p)

    api.create_commit(
        repo_id=HF_REPO, repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=tar_path, path_or_fileobj=out_tar)],
        commit_message=f"{var} {year}-{month:02d}: +{len(new)} {source} member(s)")
    os.remove(out_tar)
    tail = f", {misses} still lagging" if misses else ""
    more = f", capped (rerun for the rest)" if capped else ""
    return (f"{year}-{month:02d}: uploaded +{len(new)} {source} member(s) "
            f"({len(have)} kept{tail}{more})")
