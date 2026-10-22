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
sudo /opt/crest-fleet/src/fleet/jetstream2/setup.sh --set-token
```

The token is a fine-grained HF token named `crest-js2-fleet`. It needs:
- write on `vincewin/CREST_fleet`;
- read on the private `vincewin/CREST_state`. Without it, calibrated gauges
  silently fall back to a-priori parameters.

It is stored in `/etc/crest-fleet/hf_token` (root:root 0600) and reaches the
service only as a systemd credential.

## Run

```bash
sudo systemctl enable --now crest-fleet@0
journalctl -u crest-fleet@0 -f
```

The first start prefetches the CONUS terrain COGs, about 7 GB, once. A gauge
takes roughly 0.5–1.5 h. Each finished gauge prints `OK {...}` and adds one
`fleet <key>` commit to CREST_fleet.

## Limits

- **Priority.** The unit runs at Nice 10, with CPUWeight=20, IOWeight=20 and
  MemoryMax=28G, so the 2-D worker always wins the CPU and RAM it needs.
- **Disk.** `crest-fleet-housekeep.timer` runs hourly. It removes old EF5 run
  workdirs and stops every shard when the data volume has less than
  `FLEET_STOP_FREE_GB` free. `run-fleet.sh` then refuses to start until
  `FLEET_MIN_FREE_GB` is free.
- **Maintenance.** Jetstream2 maintenance reboots restart the units by
  themselves. `run-fleet.sh` refuses to run while the volume is not mounted.
