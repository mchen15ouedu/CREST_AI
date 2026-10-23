#!/usr/bin/env bash
# run-fleet.sh K - ExecStart of crest-fleet@K.service (runs as crest-fleet).
# Same code path as the HF CREST_fleet_runner Spaces: refresh CREST_AI main,
# then run fleet/fleet_run.py for shard K/FLEET_N. The token comes only from
# the systemd credential (ExecStart sees it, also on systemd 249).
set -euo pipefail

K=${1:?usage: run-fleet.sh <shard index>}
N=${FLEET_N:-6}
FH=${FLEET_HOME:-/opt/crest-fleet}
fail() {
  echo "run-fleet: REFUSING TO START: $*" >&2
  exit 1
}

[[ $K =~ ^[0-9]+$ ]] && [ "$K" -lt "$N" ] || fail "bad shard '$K' for FLEET_N=$N"
tokf=${CREDENTIALS_DIRECTORY:-/nonexistent}/hf_token
[ -s "$tokf" ] || fail "credential hf_token is empty or missing"

# data must be on the attached volume, never the 60 GB root disk
for d in "${CREST_CACHE_DIR:?}" "${HF_HOME:?}" "${TMPDIR:?}"; do
  case $d in /media/volume/*)
    a=$d
    while [ ! -e "$a" ] && [ "$a" != / ]; do a=$(dirname "$a"); done
    [ "$(findmnt -n -o TARGET -T "$a" 2> /dev/null || echo /)" != / ] ||
      fail "$d is under /media/volume/ but no volume is mounted there"
    ;;
  esac
  [ -d "$d" ] && [ -w "$d" ] || fail "$d is missing or not writable (re-run setup.sh)"
done
free_gb=$(df -P -BG "$CREST_CACHE_DIR" | awk 'NR==2 {gsub("G", "", $4); print $4}')
[ "${free_gb:-0}" -ge "${FLEET_MIN_FREE_GB:-60}" ] ||
  fail "only ${free_gb} GB free under $CREST_CACHE_DIR (FLEET_MIN_FREE_GB=${FLEET_MIN_FREE_GB:-60})"
# the clip stores are ~89k small files per gauge: inodes can run out first
iuse=$(df -P -i "$CREST_CACHE_DIR" | awk 'NR==2 {gsub("%", "", $5); print $5}')
[ "${iuse:-0}" -le "${FLEET_MAX_INODE_PCT:-85}" ] ||
  fail "inodes ${iuse}% used under $CREST_CACHE_DIR (FLEET_MAX_INODE_PCT=${FLEET_MAX_INODE_PCT:-85})"

cd "$FH/src"
# All shards share this checkout: refresh it only when no other shard's
# fleet_run.py is running from it (a reset under live workers would swap
# their modules mid-gauge).
if pgrep -u "$(id -u)" -f 'fleet/fleet_run\.py' > /dev/null; then
  echo "run-fleet: another shard is running: keeping CREST_AI $(git rev-parse --short=12 HEAD)"
elif git fetch -q origin main && git reset -q --hard origin/main; then :; else
  echo "run-fleet: WARNING: could not refresh CREST_AI main; running the checked-out copy" >&2
fi
[ -x EF5/bin/ef5 ] || fail "EF5/bin/ef5 missing (re-run setup.sh)"
nf=$(ldd EF5/bin/ef5 2>&1 | grep -c 'not found' || true)
[ "${nf:-0}" -eq 0 ] || fail "EF5/bin/ef5 has $nf unresolved libraries (an apt upgrade of Arrow? re-run setup.sh)"
# per-gauge code stamp: partial work from other hf_data/ or fleet/ code is
# dropped instead of being mixed into one record (fleet_run._check_code_rev)
FLEET_CODE_REV="$(git rev-parse HEAD:hf_data):$(git rev-parse HEAD:fleet)"
export FLEET_CODE_REV
echo "run-fleet: shard $K/$N, CREST_AI $(git rev-parse --short=12 HEAD), workers ${FLEET_WORKERS:-4}, ${free_gb} GB free, inodes ${iuse}%"

HF_TOKEN=$(< "$tokf")
export HF_TOKEN
exec "$FH/venv/bin/python" -u fleet/fleet_run.py \
  --workers "${FLEET_WORKERS:-4}" \
  --gauges "${FLEET_GAUGES:-all}" \
  --start "${FLEET_START:-2021-07-01}" \
  --end "${FLEET_END:-2026-06-30}" \
  --shard "$K/$N"
