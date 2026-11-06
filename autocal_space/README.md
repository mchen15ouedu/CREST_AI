---
title: CREST autocal
emoji: 🎯
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# CREST_autocal

Auto-triggered AI calibration for [CREST_demo](https://huggingface.co/spaces/vincewin/CREST_demo):
it calibrates the gauges whose **CREST 1-D simulation scores a low NSE in a
flood episode** (PI 2026-11-05: "it should be low NSCE score triggers auto
call"), **western gauges first** (PI 2026-10-27; western = longitude < -105),
then the lowest NSE.

**Trigger (kind `low_nse`).** An episode in `events/index.json` on
`vincewin/CREST_data` (status active or ended; a fragment absorbed by the
episode merger, `merged_into`, is skipped) whose manifest hydrograph at the
trigger gauge has NSE < `AUTOCAL_NSE_MAX` (0.1): `sim_q` paired with `obs_q`,
observed hours only (time <= t0), over the episode window (from its earliest
start: `sim_start`, `episode_start` or the first depth frame), with at least
`AUTOCAL_NSE_MIN_PAIRS` (12) paired hours and a non-constant observed series.
An episode that is still running qualifies — no quiet-period wait. The gauge
must not have been auto-calibrated within `AUTOCAL_COOLDOWN_D` days; the
episode must not have been handled — a calibration that *ran* (saved or not)
listed it or any episode it stands for after a merge (index `merged.ids`,
manifest `merged_from`, entries `merged_into` it); and an episode last
published before the gauge's last calibration *finished* is skipped (its
hydrograph predates the new parameters). A merge target's `generated` is the
merger's re-stamp (`provenance.merged.at`), not a new simulation: it counts
as published at the latest `generated` of the entries it absorbed and its
own t0. Each verdict is cached per (event, generated), so a poll downloads
nothing for an episode already judged.

**One candidate per gauge.** A gauge's eligible episodes are calibrated
together: the gauge ranks by its worst NSE, the window ends at the latest
`t_end` among them (an active one's current t0 when one is active, never
later than now - 2 h), and the record lists them all in `events` (each one's
NSE in `episode_nse`) — all of them are handled once it ran.

**Skips are not handled.** A calibration that did not run — observations
covering less than `AUTOCAL_OBS_MIN_FRAC` of the window or none at all,
most of a flood's paired hours missing, a calibration error (any exception
of the gauge's own calibration), no EF5 NSE, no candidate run with an NSE,
`AUTOCAL_OUTAGE_STRIKES` outages in a row on the gauge — leaves the
episodes eligible: the gauge goes on a local skip list for
`AUTOCAL_SKIP_TTL_D` days (too little obs, fetched without error, the
flood's there), about one poll (an outage streak) or
`AUTOCAL_SKIP_ERR_TTL_H` hours (anything else: an empty answer or a fetch
error may be an NWIS failure the outage checks missed), and is retried
after it, or as soon as a new eligible episode joins it.
A cover too low on the obs store's earlier answers is asked of NWIS once
more before the skip, and the store caches an empty answer as covered only
for a window that stored rows bracket on both sides, so a quiet NWIS
failure (200 with an empty series — a gap open towards now) is asked again
when the skip expires.
**Never without the flood's observations.** The hours the runner paired
`obs_q` with `sim_q` in each episode (its `nse_span`) were in NWIS then:
when most of them are missing from the window's series, they are asked of
NWIS once more, and still missing the gauge is a skip (error TTL) — a
window that is covered well enough by older observations is no calibration
on the flood that triggered it.
Skips reach the Hub only in one `low_nse_skips` summary record at most every
`AUTOCAL_SKIP_SUMMARY_H` (it re-seeds the list after a restart). After
`AUTOCAL_MAX_SKIP_RUN` skips in a row the loop waits for the next poll.

Retired triggers: outlet-Q **misses** (`events/missed.json` is no longer
written — the live runner treats a miss as advisory and the 2-D run
continues), the western **over-prediction** trigger and the **western
backlog**. `hf_data/data/autocal_west_backlog.json` and
`scripts/build_autocal_backlog.py` stay in the repo, but autocal no longer
reads the backlog (the builder still writes the reservoir table below). With
the backlog went *calibrate once*: a gauge whose next episode again scores
below the threshold after the cooldown is calibrated again, on a window
holding the new flood. `events/autocal_done.json` (gauge -> first/last/n of
its calibrations, never truncated, written with every calibration record)
still keeps the cooldown working beyond the 500 rows `events/autocal.json`
keeps.

**2-D runs come first** (PI 2026-11-05: "2D run has priority. Autocal has
lower priority than 2D run"; "Never abort a 2D run as long as the nowcasting
put the event in the queue"). Autocal runs on this Space only and never
touches the 2-D queue, a worker or an episode's runs. When it calibrates the
gauge of a running episode, a winner that beats the current parameters is
pushed to `vincewin/CREST_state` `params/`, and the episode's next hourly
re-run picks it up; whatever the NSE after calibration, the 2-D runs carry on
unchanged and the episode is not calibrated again.

**Compute ranking** (user 2026-09-25): user-initiated operations > nowcast >
2-D inundation > data update > fleet runner > autocal. Autocal yields **only
to user-initiated work**: it starts after the dashboard's `/api/health`
`user_ops` (user simulations and calibrations) has been zero for
`AUTOCAL_IDLE_FOR_S`, re-reads it between rounds and drops a calibration
(nothing recorded, retried later) when user work appears; unreadable health
counts as busy. The former 2-D gates (unpublished bundles, active episodes,
worker claims, the runner's tick) are gone: this Space shares no compute with
the 2-D solve, and those gates kept autocal idle through every flood —
exactly when low-NSE episodes appear. Nowcast, data update and the fleet run
on their own Spaces too.

**Reservoir hold (west only):** a western reservoir-affected gauge (GAGES-II
normal storage >= 10 mm, `hf_data/data/gagesii_reservoir_stor.json`) is not
calibrated until `AUTOCAL_LAKES_ON=1` — the lake module is
being switched on for the west (`CREST_LAKES=west`), and calibrating first
would bake the missing reservoirs into the parameters. The flag lifts the
hold only while the lake module is live on this Space (`hf_data.lakes`
importable, `CREST_LAKES` west or all, a lake table with a version);
otherwise the hold stays and the phase line says why. Eastern gauges are not
held. Held gauges (with their episodes) are listed under `held_reservoir` in
`/api/status`; local skips under `skipped`.

It calibrates that gauge over the six months up to the episode's end (now,
for a running episode; LLM-proposed multiplier updates within hydrologic
bounds, judged by real EF5 runs), saves the winner only if it beats the
current parameters on that window, and pushes it to `vincewin/CREST_state`
`params/`, where the dashboard, the event runner and the fleet pick it up on
their next run. Results: `events/autocal.json` (`kind` low_nse with `nse`,
`n_pairs`, `nse_max`, the episodes, their status and, when it ran,
`finished`; `low_nse_skips` summaries of the skips; older rows keep their
kinds miss / oversim / backlog / backlog_skips), shown in the dashboard's
Events panel.

Failures never look like history: an unreadable `autocal.json` /
`autocal_done.json` makes the loop wait ("history unreadable"); an obs-fetch
or network outage (connection error, timeout, an answer that is no JSON,
NWIS 5xx / 401 / 403 / 429), candidate runs that all failed that way, or —
with `AUTOCAL_REQUIRE_LLM` — no round with LLM candidates (a provider
outage or an exhausted key: the calibration stops before a heuristic-only
search, nothing is saved or pushed) records nothing and backs off a poll,
until `AUTOCAL_OUTAGE_STRIKES` in a row on one gauge (counted per gauge)
make it a skip of about one poll: the next gauge gets its turn, and a
global outage, which moves the strikes from gauge to gauge, leaves the
order as it was once it is over. A fetch error that is the site's own
(another 4xx, an answer that fails to parse) is a skip; a record whose
commit failed stays pending in the state file — counted for the daily cap,
the handled episodes and the cooldown — and is retried every poll. A winner
saved but not pushed to `vincewin/CREST_state` is unfinished: kept in the
state file (`unpushed`, counted the same way), its push retried every loop,
and recorded only once the parameters are there (`finished` at the push).
A calibration in which no EF5 run, or no candidate run, produced an NSE did
not run (a skip).

One gauge at a time, at most `AUTOCAL_MAX_PER_DAY` calibrations a day, own
Space.

Code lives in [CREST_AI](https://github.com/mchen15ouedu/CREST_AI)
(`autocal/autocal_run.py`, `hf_data/calibrate.py`) and is cloned fresh at
every boot. Secrets: `HF_TOKEN` (write), `OPENAI_API_KEY` (or the
`VLLM_BASE_URL` variable). Variables (defaults): `AUTOCAL_NSE_MAX` 0.1,
`AUTOCAL_NSE_MIN_PAIRS` 12, `AUTOCAL_WINDOW_D` 183, `AUTOCAL_COOLDOWN_D` 30,
`AUTOCAL_MAX_PER_DAY` 2, `AUTOCAL_OBS_MIN_FRAC` 0.5, `AUTOCAL_ROUNDS` 4,
`AUTOCAL_K` 3, `AUTOCAL_REQUIRE_LLM` 1, `AUTOCAL_POLL_S` 900,
`AUTOCAL_IDLE_FOR_S` 1800, `AUTOCAL_LAKES_ON` 0, `AUTOCAL_SKIP_TTL_D` 30,
`AUTOCAL_SKIP_ERR_TTL_H` 24, `AUTOCAL_SKIP_SUMMARY_H` 24,
`AUTOCAL_MAX_SKIP_RUN` 3, `AUTOCAL_OUTAGE_STRIKES` 4. No longer read:
`AUTOCAL_MIN_CHECKS`, `AUTOCAL_QUIET_H`, `AUTOCAL_STREAK_GAP_H`,
`AUTOCAL_OVERSIM_RATIO`, `AUTOCAL_OVERSIM_MIN_OBS`, `AUTOCAL_BACKLOG`.
