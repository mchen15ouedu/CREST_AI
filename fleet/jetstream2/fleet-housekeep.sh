#!/usr/bin/env bash
# fleet-housekeep.sh - hourly, as root (crest-fleet-housekeep.timer).
# 1. Removes finished EF5 run workdirs (crest_* under TMPDIR, older than
#    FLEET_WORKDIR_TTL_H). The datamgr janitor that does this on the dashboard
#    does not run in the fleet, and HF Spaces lose /tmp at restart instead.
# 2. Disk guard: stops every crest-fleet@ shard when free space on the data
#    volume drops below FLEET_STOP_FREE_GB, so the fleet can never fill the
#    disk the 2-D GPU worker uses. run-fleet.sh refuses to start again until
#    FLEET_MIN_FREE_GB is free.
set -euo pipefail
# shellcheck source=/dev/null
. /etc/crest-fleet/fleet.env

ttl_min=$((${FLEET_WORKDIR_TTL_H:-6} * 60))
if [ -d "${TMPDIR:?}" ]; then
  n=$(find "$TMPDIR" -mindepth 1 -maxdepth 1 -name 'crest_*' -mmin +"$ttl_min" -print | wc -l)
  find "$TMPDIR" -mindepth 1 -maxdepth 1 -name 'crest_*' -mmin +"$ttl_min" -exec rm -rf {} +
  [ "$n" -eq 0 ] || echo "housekeep: removed $n run workdir(s) older than ${FLEET_WORKDIR_TTL_H:-6} h"
fi

free_gb=$(df -P -BG "${CREST_CACHE_DIR:?}" | awk 'NR==2 {gsub("G", "", $4); print $4}')
if [ "${free_gb:-0}" -lt "${FLEET_STOP_FREE_GB:-30}" ]; then
  echo "housekeep: DISK GUARD: only ${free_gb} GB free under $CREST_CACHE_DIR - stopping all fleet shards" >&2
  systemctl stop 'crest-fleet@*.service' || true
fi
