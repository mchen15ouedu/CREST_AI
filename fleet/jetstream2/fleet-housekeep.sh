#!/usr/bin/env bash
# fleet-housekeep.sh - hourly, as root (crest-fleet-housekeep.timer).
# 1. Removes finished EF5 run workdirs (crest_* under TMPDIR, older than
#    FLEET_WORKDIR_TTL_H and not used by a running process). The datamgr
#    janitor that does this on the dashboard does not run in the fleet.
# 2. Removes clip stores (forcing/<var>/<bbox>, basic/<bbox>) untouched for
#    FLEET_STORE_TTL_D days: leftovers of crashes; a retry re-clips them.
# 3. Disk guard: below FLEET_STOP_FREE_GB free (or above FLEET_MAX_INODE_PCT
#    inodes used) it first drops hub-cache blobs unread for FLEET_HUB_KEEP_DAYS,
#    then restarts the running shards: run-fleet.sh refuses to start while
#    space is short, and Restart=always retries every 30 min, so the fleet
#    resumes by itself once space is back. The 2-D worker shares the volume.
set -euo pipefail

ENV=/etc/crest-fleet/fleet.env
# KEY=VALUE lookup; the file is systemd EnvironmentFile syntax, never sourced
envget() {
  local v
  v=$(sed -n "s/^[[:space:]]*$1=//p" "$ENV" | tail -n 1)
  v=${v%\"}
  v=${v#\"}
  echo "${v:-$2}"
}
TMPD=$(envget TMPDIR "")
CACHE=$(envget CREST_CACHE_DIR "")
HUB=$(envget HF_HOME "")
[ -n "$TMPD" ] && [ -n "$CACHE" ] && [ -n "$HUB" ] || {
  echo "housekeep: TMPDIR/CREST_CACHE_DIR/HF_HOME missing in $ENV" >&2
  exit 1
}

ttl_min=$(($(envget FLEET_WORKDIR_TTL_H 24) * 60))
n=0
if [ -d "$TMPD" ]; then
  while IFS= read -r -d '' d; do
    pgrep -f -- "$d" > /dev/null && continue  # a live ef5/python still uses it
    rm -rf -- "$d"
    n=$((n + 1))
  done < <(find "$TMPD" -mindepth 1 -maxdepth 1 -name 'crest_*' -mmin +"$ttl_min" -print0)
fi
[ "$n" -eq 0 ] || echo "housekeep: removed $n run workdir(s) older than $((ttl_min / 60)) h"

store_days=$(envget FLEET_STORE_TTL_D 7)
n=0
for root in "$CACHE/forcing/mrms" "$CACHE/forcing/pet" "$CACHE/forcing/temp" "$CACHE/basic"; do
  [ -d "$root" ] || continue
  while IFS= read -r -d '' d; do
    # newest file inside, not the dir mtime (a merge adds files to old dirs)
    if [ -z "$(find "$d" -type f -mtime -"$store_days" -print -quit)" ]; then
      rm -rf -- "$d"
      n=$((n + 1))
    fi
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -print0)
done
[ "$n" -eq 0 ] || echo "housekeep: removed $n clip store(s) untouched for ${store_days} d"

short() {
  local free iuse
  free=$(df -P -BG "$CACHE" | awk 'NR==2 {gsub("G", "", $4); print $4}')
  iuse=$(df -P -i "$CACHE" | awk 'NR==2 {gsub("%", "", $5); print $5}')
  [ "${free:-0}" -lt "$(envget FLEET_STOP_FREE_GB 30)" ] || [ "${iuse:-0}" -gt "$(envget FLEET_MAX_INODE_PCT 85)" ]
}
if short; then
  echo "housekeep: DISK GUARD: $(df -P -h "$CACHE" | awk 'NR==2 {print $4}') free, inodes $(df -P -i "$CACHE" | awk 'NR==2 {print $5}') used - pruning the fleet hub cache" >&2
  if [ -d "$HUB/hub" ]; then
    find "$HUB/hub" -type f -path '*/blobs/*' -atime +"$(envget FLEET_HUB_KEEP_DAYS 7)" -delete || true
    find "$HUB/hub" -xtype l -delete || true          # snapshot links to deleted blobs
  fi
  if short; then
    echo "housekeep: DISK GUARD: still short - restarting running shards (they refuse to start until space is back, then retry every 30 min)" >&2
    systemctl try-restart 'crest-fleet@*.service' || true
  fi
fi
