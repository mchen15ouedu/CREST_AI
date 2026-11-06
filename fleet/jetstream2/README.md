# Gauged fleet on the Jetstream2 VM

These files run `fleet/fleet_run.py`, the same code the HF `CREST_fleet_runner`
Spaces run, on the Jetstream2 GPU VM `crest-js2-gpu1` (NSF ACCESS EES260210).
It uses the VM's spare CPU cores; the 2-D GPU worker (`crest-worker.service`)
keeps priority. The kit also runs the hourly 2-D event tick
(`crest-events.service`): see "2-D event tick" at the end.

Why move it: the HF runners restart every few hours and lose `/tmp`. Large
basins never finish there, and the CONUS month tars are re-downloaded for
every gauge. The fleet had no upload from 2026-10-14. On the VM the cache
persists on the data volume. The month tars stay cached; the
`fleet.env.example` comments explain the other choices.

## Who runs what: a size split, not a shard split

The roster is every USGS gauge plus every HydroBASINS level-07 pour point
(the virtual points). Each is simulated on its basin outline from
`hf_data/data/basin_extent.parquet`; there is never a box, and an outlet
without an outline is skipped with the reason. The roster is split by
drainage area at `FLEET_BIG_KM2` (5,000 km², PI 2026-10-23):

| Runner | `FLEET_SIZE` | Takes |
|---|---|---|
| HF Spaces CREST_fleet_runner..6 (shards 0/6..5/6) | `small` (the default) | basins under 5,000 km² |
| this kit's `crest-fleet@K` units (shards K/`FLEET_N`) | `big` | basins of 5,000 km² or more |

Once every HF runner runs the size-split code, the two classes don't
overlap: no HF Space needs pausing and no outlet runs twice.

- **Switch-over.** The HF runners load CREST_AI main only when they boot.
  After the size split first reaches main, restart all six. Check that each
  log shows `size class small (split at 5,000 km2)` before enabling any
  `crest-fleet@K` unit.
- **Several hosts.** `FLEET_N` splits the big class across all big-class
  units: give the units on different hosts different K of the same N.
- **Uploads.** Outlets already uploaded are skipped by either side.
- **Explicit lists.** An explicit `FLEET_GAUGES` list, such as a pilot,
  needs `FLEET_SIZE=all` to run every listed outlet.

Pour-point runs upload under `vp/states/` and `vp/results/` in CREST_fleet.
The bare `states/V*` names belong to the ungauged nowcast's rolling
checkpoints.

The kit also runs on a CPU-only Jetstream2 instance. There is no 2-D worker
to share with there, so raise `crest-fleet.slice`'s MemoryMax and CPUWeight
with a drop-in.

## Install (once; idempotent)

```bash
git clone --depth 1 https://github.com/mchen15ouedu/CREST_AI.git ~/crest-ai-kit
sudo bash ~/crest-ai-kit/fleet/jetstream2/setup.sh
sudo /opt/crest-fleet/bin/setup.sh --set-token
```

`setup.sh` copies the kit into `/opt/crest-fleet/bin`, owned by root. Root
only ever runs that copy, or a fresh `~/crest-ai-kit` clone. It never runs
`/opt/crest-fleet/src`, which the `crest-fleet` user owns and which is reset to
`main` at every start.

To update, `git -C ~/crest-ai-kit pull` and re-run the first `setup.sh`. It
stops the running shards while it swaps code, venv or EF5, then starts them
again.

The token is a fine-grained HF token named `crest-js2-fleet`. It needs:
- `repo.write` on `vincewin/CREST_fleet`;
- `repo.content.read` on the private `vincewin/CREST_state`. Without it,
  calibrated gauges silently fall back to a-priori parameters.

It is stored in `/etc/crest-fleet/hf_token` (root:root 0600) and reaches the
service only as a systemd credential. Check it with
`sudo /opt/crest-fleet/bin/setup.sh --check-token`.

## Run

```bash
sudo systemctl enable --now crest-fleet@0
journalctl -u crest-fleet@0 -f
```

The first start prefetches the CONUS terrain COGs, about 7 GB, once. A gauge
takes roughly 0.5–1.5 h. Each finished gauge prints `OK {...}` and adds one
`fleet <key>` commit to CREST_fleet.

## Limits

- **Priority.** All shards share `crest-fleet.slice`: MemoryMax=28G for the
  whole fleet, CPUWeight=20 and IOWeight=20. Each shard also runs at Nice 10
  with OOMScoreAdjust=500. The 2-D worker therefore keeps the RAM and CPU it
  needs, and the fleet is the first process the OOM killer picks.
- **Threads.** EF5 is built with OpenMP. The unit sets `OMP_NUM_THREADS=2`, so
  4 workers use about 8 cores.
- **Memory.** `statebundle.pack_dir` streams a gauge's ~1,370 state grids in
  row blocks (`CREST_BUNDLE_BLOCK_MB`, default 512) into a temp file. A big
  basin's bundle never sits in RAM whole.
- **Disk.** `crest-fleet-housekeep.timer` runs hourly. It removes EF5 run
  workdirs older than 24 h and clip stores untouched for 7 days. When the
  volume runs short (`FLEET_STOP_FREE_GB`, `FLEET_MAX_INODE_PCT`), it first
  drops hub blobs unread for 7 days, then restarts the running shards.
  `run-fleet.sh` refuses to start until `FLEET_MIN_FREE_GB` is free, and
  `Restart=always` retries every 30 min. The fleet therefore pauses and
  resumes by itself.
- **Code changes.** Each gauge's partial work is stamped with the hf_data/ and
  fleet/ tree hashes (`FLEET_CODE_REV`). A restart on different code starts
  that gauge over, so one uploaded record never mixes two code versions.
- **Several shards.** They share one checkout, so `run-fleet.sh` refreshes it
  only while no other shard is running. They also share one cache, so a clip
  store that two catalog gauges map to is never pruned.
- **Maintenance.** Jetstream2 maintenance reboots restart the units by
  themselves. `run-fleet.sh` refuses to run while the volume is not mounted.

## 2-D event tick (`crest-events`)

PI 2026-11-05: the hourly 2-D event tick moves here from the HF
`CREST_runner` Space ("we have jetstream2 VM as the workhorse").
`crest-events.service` runs `scripts/event_tick_daemon.py`, which calls
`eventsim.hourly_tick()` of a pinned CREST_AI checkout. The EF5 extent and
the capacity preflight come from the basin outline, never a box, and
`EVENT_PREP_MAX_CELLS` stays 35,000,000 (the g3.large GPU holds ~37.7M). The
runner Space keeps only the always-on episode merger (`EPISODE_MERGER=1`);
its tick is switched off (`EVENT_TICK=0`) at the cutover. Ticks never run on
both hosts, and 2-D solves (the GPU worker's) are never aborted.

- **When.** A tick starts when the gauged nowcast's issue time advances, at
  most once per issue, and at least every `EVENT_TICK_MAX_GAP_MIN` (70)
  minutes. One tick at a time; the last issue survives restarts. A tick that
  falls due while the previous one still runs is not started but noted as
  an attempt (`eventsim.note_tick_attempt`, as the HF runner did for a ping
  that found its tick busy). A 2-4 h tick therefore reads as busy in
  `/api/health`, not as a dead schedule; only `busy_h` past 4 h
  (`HEALTH_WEDGE_MAX_H`) reads as wedged.
- **Bounds.** Nothing in a tick may block forever (the HF runner hung 55 h
  on one EF5 run, 11-01..11-04):
  - EF5 is killed after 45 min without output (`CREST_STALL_TIMEOUT_S`) or
    3 h of running (`CREST_RUN_TIMEOUT_S=10800` in `events.env`; 10 h on the
    Spaces). The tick records the failure and goes on with its other events.
    This is the binding cap: one event's warm-up and main run at the cap
    (6 h) stay inside the tick bound.
  - GDAL `/vsicurl` reads get 30 s to connect, 600 s per request, and are
    aborted below 1 KB/s for 60 s (`GDAL_HTTP_*` in `events.env`). Hub calls
    have finite timeouts too.
  - `EVENT_TICK_MAX_BUSY_H` (8) is the outer bound for what no inner
    watchdog sees, such as a hung read or a kill that did not take. Past it
    the daemon logs every thread's stack, publishes `"wedged"` and exits 3.
    systemd restarts it after 60 s; `KillMode=mixed` reaps a leftover EF5
    child. The saved schedule keeps the issue, and the max gap starts a new
    tick at once.
  - The main loop pings systemd every pass (`WatchdogSec=3600`). A loop hung
    for an hour is aborted with its stacks in the journal, then restarted.
- **A daemon, not a timer.** eventsim keeps state in memory between ticks:
  the native-capacity deferral of oversize basins, the CPU-timeout cooldown,
  the mosaic refresh gate and the nowcast/threshold/parameter caches.
- **Never on both hosts.** (1) The cutover order below. (2) Before each tick
  the daemon reads the runner's `/api/runner_status` (`EVENT_TICK_GUARD_URL`)
  and does not tick while that Space runs its own tick ("NOT started: the HF
  runner ..." in the log). A block is sticky: after one, only a clear read
  lets ticks resume. A Space that turns unreachable (restarting, say) keeps
  the block, and the block survives daemon restarts. An unreachable Space
  counts as clear only while the guard never blocked. If the runner Space
  is gone for good, empty `EVENT_TICK_GUARD_URL` and restart the unit.
  (3) A runner on this code refuses `/api/event_tick` while the VM's status
  file is fresh, not "stopped" and not blocked by (2). Whichever host is
  ticking keeps the tick; neither waits on the other.
- **Status.** The daemon commits `events/tick_status.json` to CREST_data at
  start, at each tick start and end, every 15 min while a tick runs, hourly
  when idle and at stop. Commits are coalesced: about 2-6 per hour against
  the 256/h shared with the merger, the worker and the fleet. With
  `EVENT_TICK` off, a Space's `/api/health` reads the tick from it
  (`tick.source: "remote"`, stale after 2 h without a refresh) and
  `/api/runner_status` returns it (`tick_source: "remote"`).
- **Not server.py.** The daemon never imports `server.py`, which would start
  persist backups and the merger supervisor that the Spaces run. Between
  ticks it runs the Spaces' cache janitor itself (datamgr LRU caps, hub
  cache, result compaction) and removes `event_*` EF5 work dirs older than
  6 h.

| What | Where |
|---|---|
| user | `crest-events` (home `/var/lib/crest-events`) |
| code | `/opt/crest-events/src` at `CREST_AI_REF`, root-owned, never fetched at start |
| EF5 | `/opt/crest-events/EF5` at `1011809` (main's pin); `/opt/crest-fleet/EF5` is not touched |
| venv | `/opt/crest-events/venv`: CPU torch, `requirements.txt`, crestimap |
| crestimap | copied from the GPU worker's current release (`/opt/crest-worker/current`, its `FORK_SHA`), else the fork at `c5184759` |
| pins | `/opt/crest-events/RELEASE` |
| settings | `/etc/crest-fleet/events.env` (from `events.env.example`) |
| token | `/etc/crest-fleet/events_hf_token` (root:root 0600), a systemd credential |
| data | `/media/volume/crest-data/events/{cache,hub,tmp,dem_cache,state}` |

**Its own user, not `crest-fleet`.** The fleet deletes a gauge's forcing clip
stores after each upload (`FLEET_PRUNE_FORCING`), and its housekeeper prunes
the fleet cache. A shared cache would lose files under a running event EF5.
The tokens differ too: this one writes CREST_data, the fleet's must not. The
first run downloads the terrain COGs (~7 GB) into its own hub cache.

**Priority** (PI ranking: user-initiated > nowcast > 2-D inundation > data
update > fleet > autocal). The unit stays in system.slice at CPUWeight and
IOWeight 100, against `crest-fleet.slice`'s 20. It runs at Nice 5 (the fleet
at 10) with MemoryHigh=12G and MemoryMax=16G; the fleet slice holds 28G and
the worker the rest. OOMScoreAdjust=200: the OOM killer takes the fleet
(500) first and the GPU worker (0) last. `OMP_NUM_THREADS=6` is for the one
EF5 run at a time.

### Token scope

What `hourly_tick` writes or reads beyond public data (audited 2026-11-05;
`events-setup.sh --check-token` enforces it):

| Repo | Access | Paths, caller |
|---|---|---|
| `vincewin/CREST_data` | **write** | `events/index.json` (`mark_ended`, `reopen_episodes`, `finalize_episodes`, `retention_sweep`); `events/<id>/final.json`, `events/scorecard.json` (`finalize_episodes`); `events/queue/<id>.json` + `.forcing.tar.gz`, deleting that id's old `.claim`/`.failed.json` and idle entries past `EVENT_QUEUE_MAX_AGE_H` (`eventqueue.enqueue`); deleting `events/<id>/` animation files or folders (`retention_sweep`); `events/tick_status.json` (the daemon) |
| `vincewin/CREST_fleet` | **write**, unless `CREST_MOSAIC_REFRESH=0` | `mosaic/index.json` (`conusstate.maybe_refresh`: at most daily, only when it is a day old) |
| `vincewin/CREST_state` (private) | **read** | `params/` (`paramstore.get`: calibrated parameters; without read access calibrated gauges silently run a-priori) |

All other reads (nowcast, thresholds, forcing, terrain, WBD, the fleet's
states, the queue, USGS, 3DEP) are public. A CREST_data-only token is
therefore not enough. Use a fine-grained token, e.g. `crest-js2-events`,
with:
- `repo.write` on `vincewin/CREST_data`;
- `repo.content.read` on `vincewin/CREST_state`;
- `repo.write` on `vincewin/CREST_fleet`.

Without the CREST_fleet write, set `CREST_MOSAIC_REFRESH=0`. The tick then
skips the refresh, and it is skipped on the HF runner too once its tick is
off, so the CONUS warm-start mosaic index stops following new fleet bundles
until another host refreshes it. The two existing VM tokens do not fit: the
worker's is CREST_data-only and the fleet's must stay CREST_fleet-only.

### Install (once; idempotent; never starts the unit)

```bash
git -C ~/crest-ai-kit pull     # the fleet's kit clone (a copy of main with this kit)
sudo CREST_AI_REF=<full commit SHA> bash ~/crest-ai-kit/fleet/jetstream2/events-setup.sh
sudo /opt/crest-events/bin/events-setup.sh --set-token
sudo /opt/crest-events/bin/events-setup.sh --check-token
```

The first run creates `/etc/crest-fleet/events.env` from the example; review
it. A `CREST_AI_REF=` given on the command line is written into it, so the
pin holds on later runs. `events-setup.sh` refuses a CREST_AI ref without
`scripts/event_tick_daemon.py`.
It ends with `run-events.sh --check` as `crest-events`. The check imports
everything, finds EF5, reads the nowcast issue, the guard and the status
file, and needs no token. Before the cutover it reports the HF runner's own
tick; that is expected.

### Cutover (in this order)

1. **HF first.** `CREST_runner` Space > Settings > Variables: `EVENT_TICK=0`.
   Leave `EPISODE_MERGER=1`. The Space restarts. An HF tick in flight loses
   only its EF5 run; the GPU worker's 2-D solves are untouched. Check:
   `curl -s https://vincewin-crest-runner.hf.space/api/runner_status`.
   You should see `tick.running` false and `tick.last_attempt` null, or
   `tick_source: "remote"` on this code. Also
   `curl -s -X POST https://vincewin-crest-runner.hf.space/api/event_tick`
   should answer "event tick disabled on this Space".
2. **Then the VM.** Run `sudo systemctl enable --now crest-events`, then
   `journalctl -u crest-events -f`. Look for the hub timeouts line, then
   "tick #1 started (start; ...)" and "tick #1 ok after ... s".
3. **Check.** Run
   `curl -s https://huggingface.co/datasets/vincewin/CREST_data/resolve/main/events/tick_status.json`.
   Within the hour, `/api/health` on the runner should show `tick.source`
   "remote" and ok. That needs this branch's `hf_data/health.py`,
   `hf_data/tickstatus.py` and `server.py` on the runner; older code reports
   its own idle tick.

**Rollback**, also in order:
1. `sudo systemctl disable --now crest-events`. It waits for a running tick
   (up to `EVENT_TICK_STOP_GRACE_S`, 600 s from the signal) and publishes
   "stopped".
2. `sudo /opt/crest-events/bin/events-setup.sh --rollback-check` must print
   "rollback check: OK". That means the unit is inactive, and
   `events/tick_status.json` shows `daemon.state` "stopped" or is more than
   2 h old (a daemon that died without its stop commit). By hand:
   `curl -s https://huggingface.co/datasets/vincewin/CREST_data/resolve/main/events/tick_status.json`
   and read `published_at` and `daemon.state`. Until then CREST_runner
   refuses `/api/event_tick` ("event tick runs on js2-events").
3. Set `EVENT_TICK=1` on CREST_runner.

**Update / move the pin:** set `CREST_AI_REF` in `/etc/crest-fleet/events.env`
(or prefix `CREST_AI_REF=<sha>`), then run
`sudo /opt/crest-events/bin/events-setup.sh`. A running unit is stopped for
the update, after its tick. That includes a unit between two restarts
("activating (auto-restart)"), which systemd would otherwise start in the
middle of the update. Afterwards the unit starts again if it was running
or is enabled.

### Operation

- **Logs.** `journalctl -u crest-events`: one line per tick start and end
  with the result summary, plus status-commit failures, guard blocks and
  janitor passes. eventsim's own lines and its watchdog's stack dumps appear
  there too.
- **Stop.** SIGTERM between ticks exits at once, and no tick starts after
  the signal, even from a pass already under way. During a tick the daemon
  waits up to `EVENT_TICK_STOP_GRACE_S` (600 s, counted from the signal) for
  the tick and its finalize threads. A second SIGTERM skips the rest of the
  wait. Then comes the "stopped" commit: one attempt and one retry, 60 s at
  most. 600 + 60 + 60 s for the exit = `TimeoutStopSec=720`; the daemon
  cuts a larger grace to fit (`EVENT_TICK_TIMEOUT_STOP_S`, set in the unit
  equal to `TimeoutStopSec`). `KillMode=mixed` then reaps a leftover EF5.
  An abandoned tick loses only its EF5 work: Hub commits are atomic, and
  the next tick redoes the visit.
- **Restarts.** `Restart=always`, also after a wedged tick (exit 3) or a
  watchdog abort. `run-events.sh` backs off 2, 4, 8, 16, then 30 min when
  starts come within 15 min of each other; systemd 249 has no
  `RestartSteps=`. A second daemon on the same state dir refuses to start.
- **Disk.** The janitor bounds the forcing/frames/states caches and the hub
  cache (`CREST_*_CACHE_GB`). `dem_cache` (3DEP 1" tiles) is not pruned. It
  is bounded by CONUS coverage; watch it with `du -sh`.
- **Manual runs.** With `EVENT_TICK=0` and no `EVENT_RUNNER_URL`, the
  runner's `/api/event_run` still runs EF5 on the Space. Those runs are
  user-initiated and not ticks; they show in `/api/runner_status` as
  `local_runner`. While the VM's fresh status shows a tick or a run in
  progress, the runner refuses the gauge-less auto-detect (the tick's own
  job) and the gauge the VM is running; other gauges run.
