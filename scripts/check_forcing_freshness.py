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
    ap.add_argument("--temp-days", type=float, default=45)   # NARR itself lags weeks
    ap.add_argument("--usgs-hours", type=float, default=12)
    args = ap.parse_args()

    from huggingface_hub import HfApi
    tok = os.environ.get("HF_TOKEN") or (
        open(TOKEN_PATH).read().strip() if os.path.exists(TOKEN_PATH) else None)
    api = HfApi(token=tok)
    now = _now()
    print(f"Forcing/obs freshness check @ {now:%Y-%m-%d %H:%M} UTC")

    ok = True
    store_lat = {}
    for var, days in (("mrms", args.mrms_days), ("pet", args.pet_days),
                      ("temp", args.temp_days)):
        latest, err = check_store_var(api, var)
        store_lat[var] = latest
        ok &= _report(var.upper(), latest, err, now, timedelta(days=days), "d")

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
