"""Ranked western backlog for CREST_autocal's idle time (PI 2026-10-27:
"The auto-calibration, we should prioritize the western part of the USA,
since those areas have more issues").

When no event (whole-event miss or over-prediction) is eligible, the autocal
loop calibrates the next gauge of this list, once each. Eligible: western
gauges (lon < pipeline.WEST_LON) with a usable basin-extent row
(basinextent.ef5_extent does not raise), a multiplier row, and ACTIVE USGS
instantaneous discharge (NWIS site service, one request per western state).
Tiers, in order:
  1  event-prone: the gauge appears in events/index.json or events/missed.json
  2  dam-free (GAGES-II STOR_NOR_2009 < 10 mm) with a suspect parameter set:
     donor-borrowed multipliers, wm < 0.01 or alpha0 == 0
  3  other dam-free gauges
  4  reservoir-affected (STOR_NOR_2009 >= 10 mm, or no GAGES-II dam record):
     held back by the loop unless AUTOCAL_LAKES_ON=1 — calibrated before the
     lake module simulates the reservoirs, the parameters would absorb them.
     Event-prone reservoir gauges land here too (same reason).
Within a tier, smaller drainage area first (cheaper EF5 runs).

Writes hf_data/data/autocal_west_backlog.json and
hf_data/data/gagesii_reservoir_stor.json (every CONUS GAGES-II gauge with
STOR_NOR_2009 >= 10 mm: the autocal loop holds the WESTERN ones back from
every trigger until AUTOCAL_LAKES_ON=1; the Space has no copy of the GAGES-II
tables), both
committed with the code the Space clones. Light: ~15 small HTTP requests, no
rasters. --reservoirs-only rewrites just the reservoir table (local file only).

    python scripts/build_autocal_backlog.py [--dams PATH] [--out PATH] [--reservoirs-only]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import truststore
truststore.inject_into_ssl()

DAMS_PATH = "D:/gageII_char/spreadsheets-in-csv-format/conterm_hydromod_dams.txt"
OUT_PATH = os.path.join(ROOT, "hf_data", "data", "autocal_west_backlog.json")
RESV_PATH = os.path.join(ROOT, "hf_data", "data", "gagesii_reservoir_stor.json")
SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
STOR_MM_MAX = 10.0         # STOR_NOR_2009 (ML/km2 == mm) below this = dam-free
WM_MIN = 0.01


def parse_rdb_sites(text: str) -> set[str]:
    """site_no column of an NWIS rdb answer (comments and the format row skipped)."""
    rows = [ln.split("\t") for ln in text.splitlines() if ln and not ln.startswith("#")]
    if not rows:
        return set()
    try:
        i = rows[0].index("site_no")
    except ValueError:
        raise RuntimeError(f"rdb answer has no site_no column: {rows[0][:6]}")
    return {r[i].strip() for r in rows[2:] if len(r) > i and r[i].strip()}


def active_iv_sites(states, tries: int = 3) -> set[str]:
    """Sites with ACTIVE instantaneous discharge (00060) in these states.
    Fails loudly: a state silently missing would drop all its gauges."""
    import requests
    out = set()
    for st in sorted(states):
        for k in range(tries):
            try:
                r = requests.get(SITE_URL, timeout=90, params={
                    "format": "rdb", "stateCd": st.lower(), "siteStatus": "active",
                    "hasDataTypeCd": "iv", "parameterCd": "00060"})
                r.raise_for_status()
                got = parse_rdb_sites(r.text)
                break
            except Exception as e:
                if k == tries - 1:
                    raise RuntimeError(f"NWIS site service failed for {st}: {e}")
                time.sleep(5 * (k + 1))
        print(f"  {st}: {len(got)} active iv discharge sites", flush=True)
        out |= got
    return out


def load_dams(path: str) -> dict[str, float]:
    """{STAID (8 digits): STOR_NOR_2009 mm} from the GAGES-II hydromod table."""
    import pandas as pd
    df = pd.read_csv(path, dtype={"STAID": str})
    df["STAID"] = df["STAID"].str.zfill(8)
    return {s: float(v) for s, v in zip(df.STAID, df.STOR_NOR_2009) if v == v and v >= 0}


def write_reservoirs(dams: dict[str, float], path: str) -> int:
    """{gauge: STOR_NOR_2009 mm} of the reservoir-affected gauges, one per
    line (small, reviewable diffs). Returns how many."""
    res = {g: round(v, 2) for g, v in sorted(dams.items()) if v >= STOR_MM_MAX}
    head = {"source": "GAGES-II conterm_hydromod_dams.txt STOR_NOR_2009 (ML/km2 == mm)",
            "stor_mm_min": STOR_MM_MAX,
            "rule": "PI 2026-10-27: western reservoir gauges (lon < WEST_LON) are not "
                    "auto-calibrated from any trigger until the lake module simulates "
                    "them (AUTOCAL_LAKES_ON=1, CREST_LAKES=west)"}
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    lines += [' "gauges": {', ",\n".join(f"  {json.dumps(g)}: {v}" for g, v in res.items()),
              " }", "}"]
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write("\n".join(lines) + "\n")
    json.load(open(path, encoding="utf-8"))           # round-trip check
    return len(res)


def event_gauges(index: dict, missed: list) -> set[str]:
    """Gauges that triggered a published event or a recorded CREST miss."""
    out = set()
    for ev, s in (index or {}).items():
        gid = s.get("gauge") or (s.get("trigger") or {}).get("gauge") or ev.split("_")[-1]
        out.add(str(gid).zfill(8))
    for r in missed or []:
        if r.get("gauge"):
            out.add(str(r["gauge"]).zfill(8))
    return out


def tier_of(gid: str, event_prone: bool, stor_mm, mult: dict) -> tuple[int, bool, list[str]]:
    """(tier, donor, reasons) for one eligible western gauge. `mult` carries
    wm, alpha0, source_station, source_dist_km from params/multipliers.parquet."""
    src = str(mult.get("source_station") or gid).zfill(8)
    donor = src != gid
    suspect = []
    if donor:
        suspect.append(f"donor {src} @ {float(mult.get('source_dist_km') or 0):.0f} km")
    if float(mult["wm"]) < WM_MIN:
        suspect.append(f"wm {float(mult['wm']):.3g} < {WM_MIN:g}")
    if float(mult["alpha0"]) == 0:
        suspect.append("alpha0 = 0")
    reasons = (["event-prone"] if event_prone else []) + suspect
    if stor_mm is None:
        return 4, donor, reasons + ["no GAGES-II dam record"]
    if stor_mm >= STOR_MM_MAX:
        return 4, donor, reasons + [f"reservoir storage {stor_mm:.0f} mm"]
    if event_prone:
        return 1, donor, reasons
    if suspect:
        return 2, donor, reasons
    return 3, donor, reasons + ["dam-free"]


def rank(entries: list[dict]) -> list[dict]:
    """Tier first, then smaller drainage area (cheaper runs), then id."""
    return sorted(entries, key=lambda e: (e["tier"], e["area_km2"], e["gauge"]))


def _hub_json(path: str):
    from huggingface_hub import hf_hub_download
    from hf_data import eventstore
    p = hf_hub_download(eventstore.REPO, path, repo_type="dataset",
                        token=os.environ.get("HF_TOKEN") or None, force_download=True)
    with open(p, encoding="utf-8") as fp:
        return json.load(fp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dams", default=DAMS_PATH, help="GAGES-II conterm_hydromod_dams.txt")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--reservoirs-out", default=RESV_PATH)
    ap.add_argument("--reservoirs-only", action="store_true",
                    help="only rewrite the reservoir table (no network)")
    a = ap.parse_args()
    dams = load_dams(a.dams)
    n = write_reservoirs(dams, a.reservoirs_out)
    print(f"wrote {a.reservoirs_out}: {n} of {len(dams)} gauges >= {STOR_MM_MAX:g} mm")
    if a.reservoirs_only:
        return 0

    from hf_data import basinextent, gauges, multipliers
    from hf_data.pipeline import WEST_LON

    df, meta = basinextent._load()
    cat = gauges.load_catalog().set_index("STAID")
    west = list(df.index[(df["kind"] == "g") & (df["lon"].astype(float) < WEST_LON)])
    print(f"{len(west)} western gauge rows in basin-extent table {meta['table_version']}")
    excluded: dict[str, int] = {}
    ok = []
    for gid in west:
        try:
            basinextent.ef5_extent(gid)
        except basinextent.BasinExtentError as e:
            excluded[f"basin extent: {e.reason}"] = excluded.get(f"basin extent: {e.reason}", 0) + 1
            continue
        ok.append(gid)
    tbl = multipliers._table()
    no_mult = [g for g in ok if g not in tbl.index]
    if no_mult:
        excluded["no multiplier row"] = len(no_mult)
    ok = [g for g in ok if g in tbl.index]
    states = {str(cat.loc[g, "STATE"]) for g in ok if g in cat.index}
    print(f"{len(ok)} with a basin outline + multipliers; NWIS active-iv query for "
          f"{len(states)} states", flush=True)
    active = active_iv_sites(states)
    inactive = [g for g in ok if g not in active]
    if inactive:
        excluded["no active USGS iv discharge"] = len(inactive)
    ok = [g for g in ok if g in active]

    index, missed = _hub_json("events/index.json"), _hub_json("events/missed.json")
    prone = event_gauges(index, missed)
    entries = []
    for gid in ok:
        m = tbl.loc[gid]
        stor = dams.get(gid)
        tier, donor, reasons = tier_of(gid, gid in prone, stor, {
            "wm": m["wm"], "alpha0": m["alpha0"], "source_station": m["source_station"],
            "source_dist_km": m["source_dist_km"]})
        entries.append({"gauge": gid, "tier": tier,
                        "area_km2": round(float(df.loc[gid, "area"]), 1),
                        "stor_mm": None if stor is None else round(stor, 2),
                        "donor": bool(donor), "reasons": reasons})
    entries = rank(entries)
    counts = {str(t): sum(1 for e in entries if e["tier"] == t) for t in (1, 2, 3, 4)}
    doc = {"built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "rule": "PI 2026-10-27 west-first + backlog: tier 1 event-prone, 2 dam-free "
                   "suspect params, 3 other dam-free, 4 reservoir (held until "
                   "AUTOCAL_LAKES_ON=1); smaller area first within a tier",
           "west_lon": WEST_LON, "stor_mm_max": STOR_MM_MAX, "wm_min": WM_MIN,
           "basin_table": meta["table_version"],
           "n_events": len(index), "n_missed_rows": len(missed),
           "counts": counts, "excluded": excluded}
    # one entry per line: small, reviewable diffs when the list is rebuilt
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in doc.items()]
    lines += [' "gauges": [', ",\n".join("  " + json.dumps(e) for e in entries), " ]", "}"]
    with open(a.out, "w", encoding="utf-8", newline="\n") as fp:
        fp.write("\n".join(lines) + "\n")
    json.load(open(a.out, encoding="utf-8"))           # round-trip check
    print(f"wrote {a.out}: {len(entries)} gauges, tiers {counts}, excluded {excluded}")
    for e in entries[:15]:
        print("  ", json.dumps(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
