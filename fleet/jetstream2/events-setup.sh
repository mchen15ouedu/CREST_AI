#!/usr/bin/env bash
# events-setup.sh - install the hourly 2-D event tick (crest-events.service,
# scripts/event_tick_daemon.py) on the Jetstream2 VM, beside the gauged fleet
# and the GPU worker. Idempotent. It installs the unit but NEVER enables or
# starts it (cutover: README "2-D event tick"); a unit that was running (or
# between two restarts) is stopped for the update (a running tick gets its
# stop grace), and started again if it was running or enabled. From a
# root-trusted copy, never from /opt/crest-events/src:
#   sudo bash ~/crest-ai-kit/fleet/jetstream2/events-setup.sh       # install / update
#   sudo CREST_AI_REF=<sha> /opt/crest-events/bin/events-setup.sh   # move the code pin
#   sudo /opt/crest-events/bin/events-setup.sh --set-token          # fine-grained HF token, hidden input
#   sudo /opt/crest-events/bin/events-setup.sh --check-token        # its scope (read-only)
#   sudo /opt/crest-events/bin/events-setup.sh --steps              # the cutover / rollback steps
#   sudo /opt/crest-events/bin/events-setup.sh --rollback-check     # OK before CREST_runner EVENT_TICK=1
# Optional overrides: CRESTIMAP_REF=<sha> (else the GPU worker's current
# release on this host, else the pinned default), TORCH_SPEC, TORCH_INDEX_URL
# (default: the CPU wheel index). Never touches /opt/crest-fleet (its EF5 stays
# at its own commit) or /opt/crest-worker.
# Runbook: fleet/jetstream2/README.md "2-D event tick"
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EH=/opt/crest-events
ETC=/etc/crest-fleet
ENVF=$ETC/events.env
USR=crest-events
UHOME=/var/lib/crest-events
UNIT=crest-events.service
# CREST_AI main's EF5 pin (Dockerfile); built into $EH/EF5 for the events only
EF5_REPO=https://github.com/mchen15ouedu/EF5.git
EF5_REF=1011809
SRC_REPO=https://github.com/mchen15ouedu/CREST_AI.git
IMAP_REPO=https://github.com/mchen15ouedu/CREST-iMAP.git
# the Spaces' Dockerfile CRESTIMAP_REF and the worker kit's default FORK_REF;
# used only when this host has no worker release
IMAP_DEFAULT_REF=c518475956c933a9c0fc46962b3825685a54b181
WORKER_CURRENT=/opt/crest-worker/current
WORKER_USER=crest
TOKEN_FILE=$ETC/events_hf_token
ME=$EH/bin/events-setup.sh
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}
TORCH_SPEC=${TORCH_SPEC:-torch}
GPU_MAX_CELLS=37700000
STATUS_URL=https://huggingface.co/datasets/vincewin/CREST_data/resolve/main/events/tick_status.json
# a status file older than this: its daemon is gone (hf_data/tickstatus.py FRESH_H)
STATUS_FRESH_H=2

log() { echo "[$(date -u +%FT%TZ)] $*"; }
die() {
  echo "[$(date -u +%FT%TZ)] ERROR: $*" >&2
  exit 1
}
[ "$(id -u)" -eq 0 ] || die "run this with sudo"

# KEY=VALUE lookup in events.env (systemd EnvironmentFile syntax, never sourced)
envget() {
  local v
  v=$(sed -n "s/^[[:space:]]*$1=//p" "$ENVF" 2> /dev/null | tail -n 1)
  v=${v%\"}
  v=${v#\"}
  echo "${v:-${2:-}}"
}

as_usr() { (cd / && runuser -u "$USR" -- env HOME="$UHOME" "$@"); }

steps() {
  cat << EOF

Cutover (README "2-D event tick"; ticks never run on both hosts):
  1. token:  sudo $ME --set-token   (then --check-token says OK)
  2. HF:     CREST_runner Space > Settings > Variables: EVENT_TICK=0 (the Space
             restarts; an in-flight HF tick loses only its EF5 run, the GPU
             worker's 2-D solves are untouched). Keep EPISODE_MERGER=1.
             Check: curl -s https://vincewin-crest-runner.hf.space/api/runner_status
             -> tick.running false and tick.last_attempt null (or tick_source "remote")
  3. VM:     sudo systemctl enable --now crest-events
             journalctl -u crest-events -f   # "tick #1 started", "tick #1 ok after ..."
             (until step 2 lands the daemon logs "NOT started: the HF runner ...")
  4. check:  curl -s $STATUS_URL
             curl -s https://vincewin-crest-runner.hf.space/api/health   # tick.source "remote", ok
Rollback (in this order too):
  1. VM:     sudo systemctl disable --now crest-events
             (waits for a running tick, up to EVENT_TICK_STOP_GRACE_S from the
             signal, then commits "stopped"; the stop never outlasts TimeoutStopSec=720)
  2. check:  sudo $ME --rollback-check   -> "rollback check: OK"
             = $UNIT inactive, and events/tick_status.json shows daemon.state
             "stopped" or is more than ${STATUS_FRESH_H} h old (a daemon that died without
             its stop commit). Until then CREST_runner refuses /api/event_tick
             ("event tick runs on js2-events"); do not force it.
  3. HF:     CREST_runner Space > Settings > Variables: EVENT_TICK=1
EOF
}

# before CREST_runner EVENT_TICK=1: nothing ticks here and the status file says so
rollback_check() {
  local state rc=0
  state=$(systemctl is-active "$UNIT" 2> /dev/null || true)
  case $state in
    active | activating | reloading | deactivating)
      echo "  NOT YET: $UNIT is $state - sudo systemctl disable --now $UNIT first"
      rc=1
      ;;
    *) echo "  $UNIT: ${state:-inactive}" ;;
  esac
  python3 - "$STATUS_URL" "$STATUS_FRESH_H" << 'PY' || rc=1
import datetime, json, sys, urllib.error, urllib.request
url, fresh_h = sys.argv[1], float(sys.argv[2])
try:
    with urllib.request.urlopen(url, timeout=30) as r:
        doc = json.load(r)
except urllib.error.HTTPError as exc:
    if exc.code == 404:
        print("  tick_status.json: never published - no daemon ever ticked from it")
        sys.exit(0)
    print(f"  NOT YET: tick_status.json unreadable (HTTP {exc.code}) - retry, do not guess")
    sys.exit(1)
except Exception as exc:
    print(f"  NOT YET: tick_status.json unreadable ({type(exc).__name__}) - retry, do not guess")
    sys.exit(1)
if not isinstance(doc, dict):
    print("  NOT YET: tick_status.json is not a JSON object - retry, do not guess")
    sys.exit(1)
daemon = doc.get("daemon") or {}
state, host, pub = daemon.get("state"), doc.get("host"), doc.get("published_at")
try:
    t = datetime.datetime.strptime(pub or "", "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)
    age_h = (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds() / 3600
except ValueError:
    age_h = None
where = f"tick_status.json: host {host}, daemon {state}, published {pub}"
if state == "stopped":
    print(f"  {where}: stopped")
elif age_h is None or age_h > fresh_h:
    print(f"  {where}: older than {fresh_h:g} h - its daemon is gone")
else:
    print(f"  NOT YET: {where} ({age_h:.1f} h ago) - a live daemon; wait for its "
          f"\"stopped\" commit, or until the file is {fresh_h:g} h old")
    sys.exit(1)
PY
  if [ "$rc" -eq 0 ]; then
    echo "rollback check: OK - set EVENT_TICK=1 on CREST_runner now"
  else
    echo "rollback check: NOT OK - do not set EVENT_TICK=1 yet"
  fi
  return "$rc"
}

check_token() {
  [ -s "$TOKEN_FILE" ] || die "no token at $TOKEN_FILE (sudo $ME --set-token)"
  [ "$(stat -c '%U:%G %a' "$TOKEN_FILE")" = "root:root 600" ] || die "$TOKEN_FILE must be root:root 0600"
  [ -x "$EH/venv/bin/python" ] || die "venv not installed yet - run the install first"
  # root reads the root-only file inside Python and sends it only to the
  # read-only whoami endpoint; the token is never printed
  env -i PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8 HF_HUB_DISABLE_TELEMETRY=1 \
    "$EH/venv/bin/python" - "$TOKEN_FILE" "$(envget CREST_MOSAIC_REFRESH 1)" << 'PY'
import sys
from huggingface_hub import HfApi
tok = open(sys.argv[1]).read().strip()
mosaic = sys.argv[2].strip() != "0"
info = HfApi().whoami(token=tok)
del tok
at = (info.get("auth") or {}).get("accessToken") or {}
print(f"account: {info.get('name')}  token: '{at.get('displayName')}' role={at.get('role')}")
ok = at.get("role") == "fineGrained"
if not ok:
    print(f"  FAIL: token role is {at.get('role')!r}; create a FINE-GRAINED token")
# what the tick touches (README "Token scope"): writes CREST_data events/;
# reads calibrated parameters from the private CREST_state; rebuilds the
# CREST_fleet mosaic index unless CREST_MOSAIC_REFRESH=0
have = {}
for s in (at.get("fineGrained") or {}).get("scoped") or []:
    ent, perms = s.get("entity") or {}, set(s.get("permissions") or [])
    name = ent.get("name") or ent.get("_id")
    print(f"  scoped {ent.get('type')} {name}: {', '.join(sorted(perms))}")
    if ent.get("type") in ("user", "org"):
        if "repo.write" in perms:
            ok = False
            print(f"  FAIL: account-wide repo.write on {name}; scope the token to the datasets")
        continue
    have[name] = "write" if "repo.write" in perms else ("read" if "repo.content.read" in perms else "")
if have.get("vincewin/CREST_data") != "write":
    ok = False
    print("  FAIL: needs repo.write on vincewin/CREST_data (events/ index, queue, finals, tick_status.json)")
if have.get("vincewin/CREST_state") != "read":
    ok = False
    print("  FAIL: needs repo.content.read (read only) on the private vincewin/CREST_state "
          "(else calibrated gauges silently run with a-priori parameters)")
if mosaic and have.get("vincewin/CREST_fleet") != "write":
    ok = False
    print("  FAIL: CREST_MOSAIC_REFRESH=1 needs repo.write on vincewin/CREST_fleet (mosaic/index.json); "
          "or set CREST_MOSAIC_REFRESH=0 in /etc/crest-fleet/events.env")
allowed = {"vincewin/CREST_data", "vincewin/CREST_state"} | ({"vincewin/CREST_fleet"} if mosaic else set())
for repo, lvl in have.items():
    if repo not in allowed and lvl == "write":
        ok = False
        print(f"  FAIL: write on {repo} is not needed - remove it")
print("token scope:", "OK" if ok else "NOT OK")
sys.exit(0 if ok else 1)
PY
}

set_token() {
  install -d -o root -g root -m 0755 "$ETC"
  [ -r /dev/tty ] || die "--set-token needs an interactive terminal"
  local tok tmp
  printf 'Paste the fine-grained HF token for the event tick (input hidden), then press Enter: ' > /dev/tty
  IFS= read -rs tok < /dev/tty
  printf '\n' > /dev/tty
  tok=${tok//[[:space:]]/}
  [ -n "$tok" ] || die "empty input - nothing stored"
  # a hidden prompt over `ssh -tt` has dropped the FIRST pasted character
  # (seen 2026-10-22: 'f_...'); HF tokens always start with hf_
  if [[ $tok =~ ^f_[A-Za-z0-9]{30,}$ ]]; then
    tok="h$tok"
    log "the first character was lost in the paste; restored the 'h' of 'hf_'"
  fi
  [[ $tok =~ ^hf_[A-Za-z0-9]{20,}$ ]] || log "WARNING: input does not look like an hf_... token; storing it anyway"
  tmp=$(mktemp "$ETC/.events_hf_token.XXXXXX")
  printf '%s\n' "$tok" > "$tmp"
  tok=
  unset tok
  chown root:root "$tmp"
  chmod 0600 "$tmp"
  mv -f "$tmp" "$TOKEN_FILE"
  log "token stored in $TOKEN_FILE (root:root 0600); checking its scope ..."
  check_token || log "WARNING: token scope check did not pass - fix the token before the cutover"
}

on_unmounted_volume() { # path under /media/volume/ whose volume is not mounted
  local a=$1
  case $a in /media/volume/*) ;; *) return 1 ;; esac
  while [ ! -e "$a" ] && [ "$a" != / ]; do a=$(dirname "$a"); done
  [ "$(findmnt -n -o TARGET -T "$a" 2> /dev/null || echo /)" = / ]
}

install_all() {
  # shellcheck source=/dev/null
  . /etc/os-release
  [ "${ID:-}:${VERSION_ID:-}" = ubuntu:22.04 ] ||
    die "needs Ubuntu 22.04 (GDAL 3.4.1 for the gdal==3.4.1 pin; same base as the Dockerfile)"

  # the daemon imports hf_data lazily from $EH/src: never swap code under it.
  # Running = active, reloading, deactivating, or activating - which is also
  # Restart=always's pause between two starts ("activating (auto-restart)"):
  # left alone, systemd would start the daemon in the middle of the update.
  # Afterwards it starts again if it was running or is enabled.
  local unit_state unit_enabled
  unit_state=$(systemctl is-active "$UNIT" 2> /dev/null || true)
  unit_enabled=$(systemctl is-enabled "$UNIT" 2> /dev/null || true)
  was_running=0
  case $unit_state in active | activating | reloading | deactivating) was_running=1 ;; esac
  restart=0
  if [ "$was_running" = 1 ] || [ "$unit_enabled" = enabled ]; then restart=1; fi
  if [ "$was_running" = 1 ]; then
    log "stopping $UNIT ($unit_state) for the update (a running tick gets its stop grace, up to 10 min)"
    systemctl stop "$UNIT"
  fi
  if [ "$restart" = 1 ]; then
    trap 'rc=$?; [ "$rc" -eq 0 ] || echo "[$(date -u +%FT%TZ)] ERROR: the update failed and $UNIT is STOPPED - no 2-D event ticks run anywhere. Fix and re-run, or roll back (README: CREST_runner EVENT_TICK=1 after --rollback-check)." >&2' EXIT
  fi

  local p missing=()
  for p in git gcc g++ build-essential make autoconf automake libtool dh-autoreconf \
    autotools-dev pkg-config libgeotiff-dev libtiff-dev zlib1g-dev python3 python3-dev \
    python3-pip python3-venv wget ca-certificates lsb-release gnupg acl libgeos-dev \
    libproj-dev libgdal-dev; do
    dpkg -s "$p" > /dev/null 2>&1 || missing+=("$p")
  done
  export DEBIAN_FRONTEND=noninteractive
  if [ ${#missing[@]} -gt 0 ]; then
    log "apt: ${missing[*]}"
    apt-get update -q
    apt-get install -y -q --no-install-recommends "${missing[@]}"
  fi
  if ! dpkg -s libarrow-dev libparquet-dev > /dev/null 2>&1; then
    log "apt: Apache Arrow C++ + Parquet"
    local deb=/tmp/apache-arrow-apt-source-latest-jammy.deb
    wget -q -O "$deb" https://apache.jfrog.io/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-jammy.deb
    apt-get install -y -q -o Acquire::Retries=8 "$deb"
    apt-get update -q -o Acquire::Retries=8
    apt-get install -y -q -o Acquire::Retries=8 libarrow-dev libparquet-dev
    rm -f "$deb"
  fi

  if ! id "$USR" > /dev/null 2>&1; then
    useradd --system --user-group --home-dir "$UHOME" --create-home --shell /usr/sbin/nologin "$USR"
    log "created system user $USR"
  fi
  install -d -o "$USR" -g "$USR" -m 0750 "$UHOME"

  install -d -o root -g root -m 0755 "$ETC" "$EH" "$EH/bin"
  if [ ! -f "$ENVF" ]; then
    install -o root -g root -m 0644 "$HERE/events.env.example" "$ENVF"
    log "created $ENVF from the example (no secrets go there) - review it"
  fi
  grep -qiE '^[[:space:]]*(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=' "$ENVF" &&
    die "$ENVF contains a token line - remove it; the token belongs only in $TOKEN_FILE"
  local cells
  cells=$(envget EVENT_PREP_MAX_CELLS 35000000)
  [[ $cells =~ ^[0-9]+$ ]] && [ "$cells" -le "$GPU_MAX_CELLS" ] ||
    die "EVENT_PREP_MAX_CELLS=$cells: the g3.large GPU holds ~$GPU_MAX_CELLS cells; jobs above it would never be claimed (PI 2026-11-05: 35000000)"
  [ "$(envget EVENT_QUEUE_MODE)" = on ] || log "WARNING: EVENT_QUEUE_MODE is not 'on' (the HF runner runs 'on')"
  [ -n "$(envget EVENT_ENGINE_IDENT)" ] || log "WARNING: EVENT_ENGINE_IDENT is empty (the status host is then the hostname)"
  for v in CREST_RUN_TIMEOUT_S GDAL_HTTP_TIMEOUT GDAL_HTTP_CONNECTTIMEOUT GDAL_HTTP_LOW_SPEED_TIME GDAL_HTTP_LOW_SPEED_LIMIT; do
    [ -n "$(envget $v)" ] || log "WARNING: $v is empty in $ENVF (see events.env.example: unbounded EF5 runs / /vsicurl reads)"
  done

  # ---- data dirs on the volume; the volume root needs u:crest-events:--x
  local d a root vroots=() cache hub tmpd dem state
  cache=$(envget CREST_CACHE_DIR)
  hub=$(envget HF_HOME)
  tmpd=$(envget TMPDIR)
  dem=$(envget EVENT_DEM_CACHE)
  state=$(envget EVENT_TICK_STATE_DIR)
  for d in "$cache" "$hub" "$tmpd" "$dem" "$state"; do
    [ -n "$d" ] || die "CREST_CACHE_DIR, HF_HOME, TMPDIR, EVENT_DEM_CACHE and EVENT_TICK_STATE_DIR must be set in $ENVF"
    case $d in /home/* | /root/*) die "$d: /home and /root are hidden from the service (ProtectHome=yes)" ;; esac
    on_unmounted_volume "$d" && die "$d is under /media/volume/ but no volume is mounted there"
  done
  for d in "$cache" "$hub" "$tmpd" "$dem" "$state"; do
    a=$d
    while [ ! -e "$a" ] && [ "$a" != / ]; do a=$(dirname "$a"); done
    root=$(findmnt -n -o TARGET -T "$a")
    if [ "$root" != / ] && [[ " ${vroots[*]} " != *" $root "* ]]; then
      # ALWAYS, even if it is traversable today: the volume root comes back
      # exouser 0750 after a reboot, and a named ACL entry is what survives
      # (2026-11-03: the GPU worker's user had none and stopped)
      setfacl -m "u:$USR:--x" "$root"
      getfacl -p "$root" 2> /dev/null | grep -q "^user:$USR:..x" ||
        die "the ACL entry u:$USR:--x did not stick on $root"
      log "ACL: $USR may traverse the volume root $root"
      vroots+=("$root")
    fi
    # a missing parent only: install -d would re-mode an existing one (the volume root)
    [ -d "$(dirname "$d")" ] || install -d -m 0755 "$(dirname "$d")"
    a=$(dirname "$d")
    while [ "$a" != / ]; do
      if ! as_usr test -x "$a"; then
        setfacl -m "u:$USR:--x" "$a"
        log "ACL: $USR may traverse $a"
      fi
      a=$(dirname "$a")
    done
    install -d -o "$USR" -g "$USR" -m 0750 "$d"
    as_usr test -w "$d" || die "$USR cannot write $d (check the permissions of its parents)"
  done

  # ---- CREST_AI at the pinned ref (root-owned: the service cannot change its code)
  local ref sha
  ref=${CREST_AI_REF:-$(envget CREST_AI_REF main)}
  [[ $ref =~ ^[A-Za-z0-9._/-]+$ ]] || die "CREST_AI_REF '$ref' is not a plain ref"
  if [ "$(envget CREST_AI_REF)" != "$ref" ]; then  # the pin holds on later runs
    if grep -q '^[[:space:]]*CREST_AI_REF=' "$ENVF"; then
      sed -i "s|^[[:space:]]*CREST_AI_REF=.*|CREST_AI_REF=$ref|" "$ENVF"
    else
      echo "CREST_AI_REF=$ref" >> "$ENVF"
    fi
    log "$ENVF: CREST_AI_REF=$ref"
  fi
  if [ ! -d "$EH/src/.git" ]; then
    log "CREST_AI -> $EH/src"
    git clone -q "$SRC_REPO" "$EH/src"
  fi
  git -C "$EH/src" fetch -q origin
  sha=$(git -C "$EH/src" rev-parse --verify --quiet "origin/$ref^{commit}") ||
    sha=$(git -C "$EH/src" rev-parse --verify --quiet "$ref^{commit}") ||
    die "CREST_AI ref '$ref' not found"
  git -C "$EH/src" -c advice.detachedHead=false checkout -q --detach -f "$sha"
  [ -f "$EH/src/scripts/event_tick_daemon.py" ] && [ -f "$EH/src/hf_data/tickstatus.py" ] ||
    die "CREST_AI $sha ($ref) predates scripts/event_tick_daemon.py - pin a newer ref"
  ln -sfn "$EH/EF5" "$EH/src/EF5"
  chown -R root:root "$EH/src"
  chmod -R u+rwX,go+rX,go-w "$EH/src"
  log "CREST_AI $ref -> ${sha:0:12}"

  # ---- EF5 at CREST_AI main's pin (rebuilt on another ref or broken libraries)
  if [ ! -x "$EH/EF5/bin/ef5" ] ||
    [ "$(git -C "$EH/EF5" rev-parse --short=7 HEAD 2> /dev/null)" != "$EF5_REF" ] ||
    [ "$(ldd "$EH/EF5/bin/ef5" 2>&1 | grep -c 'not found' || true)" -ne 0 ]; then
    log "EF5 fork @ $EF5_REF: build into $EH/EF5"
    rm -rf "$EH/EF5"
    git clone -q "$EF5_REPO" "$EH/EF5"
    (
      cd "$EH/EF5"
      git checkout -q "$EF5_REF"
      autoreconf --force --install > /dev/null 2>&1
      ./configure --with-arrow CXXFLAGS="-std=c++20 -Wall -O2" CFLAGS="-Wall -O2" > /dev/null
      sed -i 's/-Werror//g' Makefile
      make -j"$(nproc)" > /dev/null
    )
    [ -x "$EH/EF5/bin/ef5" ] || die "EF5 build produced no bin/ef5"
  fi
  chmod -R go+rX "$EH/EF5"
  log "EF5: $EH/EF5/bin/ef5 @ $(git -C "$EH/EF5" rev-parse --short=7 HEAD)"
  # provenance runs `git rev-parse` as the service user on these root-owned clones
  local safe
  safe=$(as_usr git config --global --get-all safe.directory 2> /dev/null || true)
  for d in "$EH/src" "$EH/EF5"; do
    grep -qxF "$d" <<< "$safe" || as_usr git config --global --add safe.directory "$d"
  done

  # ---- venv: CPU torch first (no CUDA wheels via a dependency), then main's requirements
  local req_hash
  req_hash=$({
    sha256sum "$EH/src/requirements.txt"
    echo "$TORCH_SPEC $TORCH_INDEX_URL"
  } | sha256sum | cut -c1-16)
  [ -x "$EH/venv/bin/python" ] || python3 -m venv "$EH/venv"
  if [ "$(cat "$EH/venv/.req_hash" 2> /dev/null)" != "$req_hash" ]; then
    log "venv: $EH/venv ($TORCH_SPEC from $TORCH_INDEX_URL + requirements.txt of CREST_AI ${sha:0:12})"
    "$EH/venv/bin/pip" install -q --no-cache-dir --upgrade pip
    "$EH/venv/bin/pip" install -q --no-cache-dir "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
    "$EH/venv/bin/pip" install -q --no-cache-dir -r "$EH/src/requirements.txt"
    echo "$req_hash" > "$EH/venv/.req_hash"
  fi

  # ---- crestimap: the GPU worker's current release on this host, else the pin
  local site imap_sha imap_src imap_from tmp="" got
  site=$("$EH/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  if [ -z "${CRESTIMAP_REF:-}" ] && [ -f "$WORKER_CURRENT/RELEASE.txt" ] &&
    [ -f "$WORKER_CURRENT/CREST-iMAP/crestimap/__init__.py" ]; then
    imap_sha=$(sed -n 's/^FORK_SHA=//p' "$WORKER_CURRENT/RELEASE.txt" | head -n 1)
    imap_src="$(readlink -f "$WORKER_CURRENT")/CREST-iMAP"
    imap_from="worker release $(readlink -f "$WORKER_CURRENT")"
    if id "$WORKER_USER" > /dev/null 2>&1; then
      # as its owner (git refuses a root caller on another user's clone)
      got=$(cd / && runuser -u "$WORKER_USER" -- env HOME=/var/lib/crest-worker \
        git -C "$imap_src" rev-parse HEAD 2> /dev/null || true)
      [ -z "$got" ] || [ "$got" = "$imap_sha" ] ||
        die "$imap_src is at $got, not its RELEASE.txt FORK_SHA $imap_sha"
    fi
  else
    tmp=$(mktemp -d)
    git clone -q --filter=blob:none --no-checkout "$IMAP_REPO" "$tmp/CREST-iMAP"
    git -C "$tmp/CREST-iMAP" sparse-checkout init --cone
    git -C "$tmp/CREST-iMAP" sparse-checkout set crestimap
    imap_sha=$(git -C "$tmp/CREST-iMAP" rev-parse --verify --quiet "${CRESTIMAP_REF:-$IMAP_DEFAULT_REF}^{commit}") ||
      die "CREST-iMAP ref '${CRESTIMAP_REF:-$IMAP_DEFAULT_REF}' not found"
    git -C "$tmp/CREST-iMAP" -c advice.detachedHead=false checkout -q --detach "$imap_sha"
    imap_src=$tmp/CREST-iMAP
    imap_from=$IMAP_REPO
  fi
  [[ $imap_sha =~ ^[0-9a-f]{40}$ ]] || die "no usable crestimap commit (got '$imap_sha')"
  if [ "$(sed -n 's/^CRESTIMAP_SHA=//p' "$EH/RELEASE" 2> /dev/null)" != "$imap_sha" ] ||
    [ ! -f "$site/crestimap/__init__.py" ]; then
    log "crestimap ${imap_sha:0:12} from $imap_from -> $site"
    rm -rf "$site/crestimap"
    cp -r "$imap_src/crestimap" "$site/"
    find "$site/crestimap" -name __pycache__ -prune -exec rm -rf {} +
    chmod -R go+rX "$site/crestimap"
  fi
  [ -z "$tmp" ] || rm -rf "$tmp"

  {
    echo "INSTALLED=$(date -u +%FT%TZ)"
    echo "CREST_AI_REF=$ref"
    echo "CREST_AI_SHA=$sha"
    echo "EF5_REF=$EF5_REF"
    echo "EF5_SHA=$(git -C "$EH/EF5" rev-parse HEAD)"
    echo "CRESTIMAP_SHA=$imap_sha"
    echo "CRESTIMAP_FROM=$imap_from"
  } > "$EH/RELEASE"
  chmod 0644 "$EH/RELEASE"

  # root-owned copy of the kit: what root runs later (--set-token, updates)
  if [ "$HERE" != "$EH/bin" ]; then
    install -o root -g root -m 0755 "$HERE/events-setup.sh" "$HERE/run-events.sh" "$EH/bin/"
    install -o root -g root -m 0644 "$HERE/crest-events.service" "$HERE/events.env.example" "$EH/bin/"
  fi
  install -o root -g root -m 0644 "$EH/bin/crest-events.service" /etc/systemd/system/
  systemctl daemon-reload

  log "smoke test as $USR: run-events.sh --check (reads only, no token)"
  local kv=() line key val
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    [[ $line =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key=${BASH_REMATCH[1]}
    val=${BASH_REMATCH[2]}
    val=${val%\"}
    val=${val#\"}
    case $key in HF_TOKEN | HUGGING_FACE_HUB_TOKEN) continue ;; esac
    kv+=("$key=$val")
  done < "$ENVF"
  (cd / && runuser -u "$USR" -- env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$UHOME" \
    LANG=C.UTF-8 EVENTS_HOME="$EH" EF5_DIR="$EH/EF5" PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=6 "${kv[@]}" "$EH/bin/run-events.sh" --check) ||
    die "smoke test failed (above) - the unit is installed but must not be started"

  if [ "$restart" = 1 ]; then
    log "starting $UNIT again (before the update: ${unit_state:-inactive}, ${unit_enabled:-disabled})"
    systemctl start "$UNIT"
  fi
  echo
  log "setup complete: CREST_AI ${sha:0:12}, EF5 $EF5_REF, crestimap ${imap_sha:0:12}; $UNIT is $(systemctl is-active "$UNIT" || true), $(systemctl is-enabled "$UNIT" 2> /dev/null || echo disabled)"
  [ -s "$TOKEN_FILE" ] || log "next: sudo $ME --set-token"
  [ "$restart" = 1 ] || steps
}

case ${1:-} in
  '') install_all ;;
  --set-token) set_token ;;
  --check-token) check_token ;;
  --steps) steps ;;
  --rollback-check) rollback_check ;;
  -h | --help) sed -n '2,19p' "$0" ;;
  *) die "unknown option $1 (see --help)" ;;
esac
