# Gauged fleet on the Jetstream2 VM

These files run `fleet/fleet_run.py`, the same code the HF `CREST_fleet_runner`
Spaces run, on the Jetstream2 GPU VM `crest-js2-gpu1` (NSF ACCESS EES260210).
It uses the VM's spare CPU cores; the 2-D GPU worker (`crest-worker.service`)
keeps priority.

Why move it: the HF runners restart every few hours and lose `/tmp`. Large
basins never finish there, and the CONUS month tars are re-downloaded for
every gauge. The fleet had no upload from 2026-10-14. On the VM the cache
persists on the data volume. The month tars stay cached; the
`fleet.env.example` comments explain the other choices.

## One writer per shard

A shard is `catalog index % FLEET_N == K`. Before starting shard K on the VM,
PAUSE the HF Space that runs K/6:

| HF Space | FLEET_SHARD | VM unit |
|---|---|---|
| CREST_fleet_runner | 0/6 | crest-fleet@0 |
| CREST_fleet_runner2 | 1/6 | crest-fleet@1 |
| CREST_fleet_runner3 | 2/6 | crest-fleet@2 |
| CREST_fleet_runner4 | 3/6 | crest-fleet@3 |
| CREST_fleet_runner5 | 4/6 | crest-fleet@4 |
| CREST_fleet_runner6 | 5/6 | crest-fleet@5 |

Read the Space's `FLEET_SHARD` variable before pausing it rather than trusting
this table. To hand a shard back, stop and disable the unit, then restart the
Space. Gauges already uploaded are skipped by either side.

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
