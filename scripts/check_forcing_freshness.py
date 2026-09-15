"""Read-only freshness double-check for the CREST_demo data feeds.

All the feeds a run depends on are refreshed weekly by their own updaters
(update_temp_narr.py, update_mrms.py, update_pet.py, update_usgs_obs.py). This
script is the routine sanity check that those updaters are actually keeping up —
it downloads nothing to the store and writes nothing anywhere. It reports, for
each feed, the latest timestep actually available and how far behind "now" that
is, and flags anything that has fallen further behind than its updater's cadence
+ source lag should allow (i.e. a stalled updater):

  * MRMS precip  — newest member in the HF store (vincewin/CREST_data mrms tars)
  * PET          — newest member in the HF store (pet tars)
  * TEMP         — newest member in the HF store (temp tars; NARR-fed)
  * USGS gauge   — live reachability of NWIS instantaneous values

The forcing feeds bound the newest date a simulation can run: the runnable window
ends at min(MRMS, PET, TEMP). USGS is fetched live per run (hf_data/obs.py), so
its check is a reachability/lag probe against a few perennial gauges rather than a
store scan.

Run (needs the HF token file + net access):
    python scripts/check_forcing_freshness.py

Exit code 0 = everything within tolerance; 1 = at least one feed flagged STALE or
a probe failed. Tolerances allow for the weekly updater cadence plus each
source's own lag (MRMS/PET a day or two, NARR temp several weeks, USGS live);
tune with --mrms-days / --pet-days / --temp-days / --usgs-hours.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tarfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import truststore
truststore.inject_into_ssl()

HF_REPO = "vincewin/CREST_data"
TOKEN_PATH = r"C:\Users\chenm\Documents\EF5\CREST_token.txt"

# member-name -> timestamp patterns (NLDAS-heritage names AND generic ones the
# updaters write); mirrors hf_data.forcing.VARS member/out formats
_MEMBER_PATS = {
    "mrms": [r"mrms_corr_(\d{10})\.pqf$", r"mrms_(\d{10})\.pqf$"],
    "pet":  [r"et(\d{8})\.bil\.pqf$", r"et(\d{8})\.pqf$"],
    "temp": [r"NLDAS_FORA0125_H\.A(\d{8})\.(\d{2})00\.", r"temp_(\d{10})\.pqf$"],
}
# NWIS gauges to probe for live reachability: perennial, widely-separated,
# rarely-offline USGS sites (San Saba TX, Cimarron OK, Potomac MD)
_USGS_PROBE = ["08144500", "07103700", "01646500"]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# TEMP's source: NOAA PSL mirrors NARR 2-m air temperature as one yearly file
# that is appended weeks after the fact and at irregular intervals (the 2026
# file was written 2026-08-24 and still ended 2026-07-31 21z on 2026-09-15; the
# 2025 file got its last write on 2026-03-20). A calendar-lag limit therefore
# alarms on a store that is PERFECTLY caught up (2026-09-15: lag 45.8 d > 45).
# Instead, ask the source how far it goes — two tiny OPeNDAP requests (.dds for
# the time length, .ascii for the last value) — and call TEMP stale only when
# NARR has data we have not ingested.
NARR_DODS = "https://psl.noaa.gov/thredds/dodsC/Datasets/NARR/monolevel/air.2m.{year}.nc"


def narr_source_end(now: datetime):
    """Last time step NOAA PSL has published for NARR air.2m (UTC), or None."""
    import re as _re
    import urllib.request
    for year in (now.year, now.year - 1):       # January: last year's file
        base = NARR_DODS.format(year=year)
        try:
            dds = urllib.request.urlopen(base + ".dds", timeout=30).read().decode()
            n = int(_re.search(r"time\s*=\s*(\d+)", dds).group(1))
            asc = urllib.request.urlopen(f"{base}.ascii?time[{n - 1}:1:{n - 1}]",
                                         timeout=30).read().decode()
            hours = float(_re.findall(r"[-\d.]+", asc.rsplit("time", 1)[-1])[-1])
            return datetime(1800, 1, 1) + timedelta(hours=hours)   # "hours since 1800-1-1"
        except Exception:
            continue
    return None


def _latest_member(names, var) -> datetime | None:
    """Newest timestep among a tar's member names, or None."""
    best = None
    for n in names:
        for p in _MEMBER_PATS[var]:
            m = re.search(p, n)
            if not m:
                continue
            g = "".join(m.groups())
            fmt = "%Y%m%d%H" if len(g) == 10 else "%Y%m%d"
            try:
                t = datetime.strptime(g, fmt)
            except ValueError:
                break
            if best is None or t > best:
                best = t
            break
    return best


def check_store_var(api, var: str) -> tuple[datetime | None, str | None]:
    """Latest available timestep for a store-backed forcing var (mrms/pet/temp).

    Lists the repo, picks the newest month tar, and (falling back to the newest
    year tar) reads the newest member inside it. Returns (latest, error)."""
    from huggingface_hub import hf_hub_download
    try:
        files = api.list_repo_files(HF_REPO, repo_type="dataset")
    except Exception as e:
        return None, f"list_repo_files failed: {e}"
    months = sorted(f for f in files
                    if re.match(rf"{var}/\d{{4}}/{var}_\d{{4}}_\d{{2}}\.tar$", f))
    years = sorted(f for f in files if re.match(rf"{var}/{var}_\d{{4}}\.tar$", f))
    candidates = ([months[-1]] if months else []) + ([years[-1]] if years else [])
    if not candidates:
        return None, "no month or year tar found in store"
    latest, err = None, None
    for path in candidates:            # newest month first; year tar as fallback
        try:
            local = hf_hub_download(HF_REPO, path, repo_type="dataset")
            with tarfile.open(local) as tf:
                latest = _latest_member(tf.getnames(), var)
            if latest is not None:
                return latest, None
        except Exception as e:
            err = f"{path}: {e}"
    return None, err or "no datable members in newest tar"


def check_mrms_recent(api) -> tuple[datetime | None, str | None]:
    """Newest loose Pass1 hour in mrms_recent/ (no download — names carry the
    timestamp). This is the nowcasting feed, refreshed every 6 h."""
    try:
        files = api.list_repo_files(HF_REPO, repo_type="dataset")
    except Exception as e:
        return None, f"list_repo_files failed: {e}"
    best = None
    for f in files:
        m = re.match(r"mrms_recent/mrms1h_pass1_(\d{10})\.pqf$", f)
        if not m:
            continue
        try:
            t = datetime.strptime(m.group(1), "%Y%m%d%H")
        except ValueError:
            continue
        if best is None or t > best:
            best = t
    if best is None:
        return None, "no mrms_recent/ members in store"
    return best, None


def check_usgs() -> tuple[datetime | None, str | None]:
    """Newest instantaneous-value timestamp across the probe gauges (live NWIS)."""
    try:
        from hf_data import obs
    except Exception as e:
        return None, f"cannot import obs: {e}"
    now = _now()
    latest, errs = None, []
    for site in _USGS_PROBE:
        try:
            s = obs.fetch_usgs_discharge(site, now - timedelta(days=2), now)
        except Exception as e:
            errs.append(f"{site}: {e}")
            continue
        if s and (latest is None or s[-1][0] > latest):
            latest = s[-1][0]
    if latest is None:
        return None, "; ".join(errs) or "no recent data from any probe gauge"
    return latest, None


def _report(label: str, latest, err, now, warn_after: timedelta,
            unit: str) -> bool:
    """Print one line; return True if OK (within tolerance and no error)."""
    if err and latest is None:
        print(f"  {label:12s} FAILED  — {err}")
        return False
    lag = now - latest
    amt = lag.total_seconds() / (3600 if unit == "h" else 86400)
    stale = lag > warn_after
    tag = "STALE " if stale else "ok    "
    print(f"  {label:12s} {tag} latest {latest:%Y-%m-%d %H:%M} UTC "
          f"(lag {amt:.1f} {unit})")
    return not stale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrms-days", type=float, default=14)   # weekly cadence + a missed run
    ap.add_argument("--pet-days", type=float, default=14)
    ap.add_argument("--temp-days", type=float, default=45)   # fallback when the source probe fails
    ap.add_argument("--temp-source-gap-days", type=float, default=5)   # stale = NARR has > this much we lack
    ap.add_argument("--temp-hard-days", type=float, default=120)       # even caught-up: a dead source is news
    ap.add_argument("--usgs-hours", type=float, default=12)
    ap.add_argument("--recent-hours", type=float, default=4)   # hourly cadence + ~2 h source lag + margin
    args = ap.parse_args()

    from huggingface_hub import HfApi
    tok = os.environ.get("HF_TOKEN") or (
        open(TOKEN_PATH).read().strip() if os.path.exists(TOKEN_PATH) else None)
    api = HfApi(token=tok)
    now = _now()
    print(f"Forcing/obs freshness check @ {now:%Y-%m-%d %H:%M} UTC")

    ok = True
    store_lat = {}
    for var, days in (("mrms", args.mrms_days), ("pet", args.pet_days)):
        latest, err = check_store_var(api, var)
        store_lat[var] = latest
        ok &= _report(var.upper(), latest, err, now, timedelta(days=days), "d")

    # TEMP: judged against its SOURCE, not the calendar (see narr_source_end)
    latest, err = check_store_var(api, "temp")
    store_lat["temp"] = latest
    src = narr_source_end(now) if latest is not None else None
    if latest is not None and src is not None:
        gap_d = (src - latest).total_seconds() / 86400.0
        lag_d = (now - latest).total_seconds() / 86400.0
        stale = gap_d > args.temp_source_gap_days or lag_d > args.temp_hard_days
        tag = "STALE " if stale else "ok    "
        why = (f"NARR source ends {src:%Y-%m-%d %H:%M} UTC — "
               + (f"{gap_d:.1f} d not yet ingested" if gap_d > args.temp_source_gap_days
                  else "store caught up to the source"
                  if lag_d <= args.temp_hard_days
                  else f"source itself silent > {args.temp_hard_days:g} d"))
        # keep "(lag N d)" intact: the updater Space parses it for self-heal
        print(f"  {'TEMP':12s} {tag} latest {latest:%Y-%m-%d %H:%M} UTC "
              f"(lag {lag_d:.1f} d) · {why}")
        ok &= not stale
    else:
        ok &= _report("TEMP", latest, err, now, timedelta(days=args.temp_days), "d")

    latest, err = check_mrms_recent(api)
    ok &= _report("MRMS recent", latest, err, now,
                  timedelta(hours=args.recent_hours), "h")

    latest, err = check_usgs()
    ok &= _report("USGS gauge", latest, err, now,
                  timedelta(hours=args.usgs_hours), "h")

    # the runnable simulation window ends where the FORCING feeds run out
    forcing = [t for t in store_lat.values() if t is not None]
    if forcing:
        end = min(forcing)
        limiter = min(store_lat, key=lambda v: store_lat[v] or now).upper() \
            if len(forcing) == 3 else "(partial)"
        print(f"  ------------\n  runnable window ends {end:%Y-%m-%d %H:%M} UTC "
              f"(bounded by {limiter})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
