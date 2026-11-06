#!/usr/bin/env bash
# run-events.sh - ExecStart of crest-events.service (runs as crest-events).
# Runs scripts/event_tick_daemon.py of the PINNED CREST_AI checkout in
# /opt/crest-events/src: this never fetches code (events-setup.sh moves the
# pin). The token comes only from the systemd credential: ExecStart sees
# $CREDENTIALS_DIRECTORY on systemd 249, ExecStartPre does not.
#   run-events.sh            the daemon (systemd)
#   run-events.sh --check    imports, EF5, nowcast / guard / status reads;
#                            no tick, no write, no token (events-setup.sh)
set -euo pipefail

EH=${EVENTS_HOME:-/opt/crest-events}
fail() {
  echo "run-events: REFUSING TO START: $*" >&2
  exit 1
}
CHECK=0
[ "${1:-}" = --check ] && CHECK=1

# data on the attached volume, never the 60 GB root disk
for d in "${CREST_CACHE_DIR:?}" "${HF_HOME:?}" "${TMPDIR:?}" "${EVENT_DEM_CACHE:?}" "${EVENT_TICK_STATE_DIR:?}"; do
  case $d in /media/volume/*)
    a=$d
    while [ ! -e "$a" ] && [ "$a" != / ]; do a=$(dirname "$a"); done
    [ "$(findmnt -n -o TARGET -T "$a" 2> /dev/null || echo /)" != / ] ||
      fail "$d is under /media/volume/ but no volume is mounted there"
    ;;
  esac
  [ -d "$d" ] && [ -w "$d" ] || fail "$d is missing or not writable (re-run events-setup.sh; after a reboot: getfacl on the volume root needs user:$(id -un):--x)"
done
free_gb=$(df -P -BG "$CREST_CACHE_DIR" | awk 'NR==2 {gsub("G", "", $4); print $4}')
# 2-D inundation outranks the fleet (PI ranking): warn, never refuse - the
# fleet's housekeeper frees space by pausing the fleet
[ "${free_gb:-0}" -ge "${EVENTS_MIN_FREE_GB:-20}" ] ||
  echo "run-events: WARNING: only ${free_gb} GB free under $CREST_CACHE_DIR" >&2

cd "$EH/src"
[ -x EF5/bin/ef5 ] || fail "EF5/bin/ef5 missing (re-run events-setup.sh)"
nf=$(ldd EF5/bin/ef5 2>&1 | grep -c 'not found' || true)
[ "${nf:-0}" -eq 0 ] || fail "EF5/bin/ef5 has $nf unresolved libraries (an apt upgrade of Arrow? re-run events-setup.sh)"
# code identity for provenance.stamp and the tick status (crestimap is a copy
# without .git; events-setup.sh records its commit)
CRESTIMAP_SHA=$(sed -n 's/^CRESTIMAP_SHA=//p' "$EH/RELEASE" 2> /dev/null | head -n 1 || true)
export CRESTIMAP_SHA
if [ "$CHECK" = 1 ]; then
  exec "$EH/venv/bin/python" -u scripts/event_tick_daemon.py --check
fi

# Restart=always with backoff (systemd 249 has no RestartSteps=): a start
# within 15 min of the previous one waits 2, 4, 8, 16, then 30 min
bo=$EVENT_TICK_STATE_DIR/restart_backoff
now=$(date +%s)
read -r last n 2> /dev/null < "$bo" || { last=0; n=0; }
[[ $last =~ ^[0-9]+$ ]] && [[ $n =~ ^[0-9]+$ ]] || { last=0; n=0; }
if [ $((now - last)) -lt 900 ]; then n=$((n + 1)); else n=0; fi
echo "$now $n" > "$bo"
if [ "$n" -gt 0 ]; then
  wait_s=$((60 << (n < 5 ? n : 5)))
  [ "$wait_s" -le 1800 ] || wait_s=1800
  echo "run-events: restart $n within 15 min - waiting ${wait_s} s"
  sleep "$wait_s"
fi

tokf=${CREDENTIALS_DIRECTORY:-/nonexistent}/hf_token
[ -s "$tokf" ] || fail "credential hf_token is empty or missing"
echo "run-events: CREST_AI $(sed -n 's/^CREST_AI_SHA=//p' "$EH/RELEASE" | cut -c1-12), EF5 $(sed -n 's/^EF5_SHA=//p' "$EH/RELEASE" | cut -c1-7), crestimap ${CRESTIMAP_SHA:0:12}, ${free_gb} GB free"
HF_TOKEN=$(< "$tokf")
export HF_TOKEN
exec "$EH/venv/bin/python" -u scripts/event_tick_daemon.py
