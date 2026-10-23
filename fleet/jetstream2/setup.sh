#!/usr/bin/env bash
# setup.sh - install the CREST gauged-fleet runner on the Jetstream2 VM.
# Same stack as fleet_space/Dockerfile (Ubuntu 22.04: GDAL 3.4.1 matches the
# gdal==3.4.1 pin; Arrow C++ for the EF5 fork's Parquet forcing reader).
# Idempotent; never enables a shard (it stops running shards while it updates
# their code and restarts them afterwards). From a root-trusted copy, never
# from /opt/crest-fleet/src (owned by crest-fleet, reset to main at each start):
#   sudo bash ~/crest-ai-kit/fleet/jetstream2/setup.sh   # first install / update
#   sudo /opt/crest-fleet/bin/setup.sh                    # update (root-owned copy)
#   sudo /opt/crest-fleet/bin/setup.sh --set-token        # fine-grained HF token, hidden input
#   sudo /opt/crest-fleet/bin/setup.sh --check-token      # show its scope (read-only)
# Then, after PAUSING the HF Space with FLEET_SHARD=K/6:
#   sudo systemctl enable --now crest-fleet@K
# Runbook: fleet/jetstream2/README.md
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FH=/opt/crest-fleet
ETC=/etc/crest-fleet
USR=crest-fleet
UHOME=/var/lib/crest-fleet
EF5_REPO=https://github.com/mchen15ouedu/EF5.git
EF5_REF=5a26a86
SRC_REPO=https://github.com/mchen15ouedu/CREST_AI.git
TOKEN_FILE=$ETC/hf_token
ME=$FH/bin/setup.sh

log() { echo "[$(date -u +%FT%TZ)] $*"; }
die() {
  echo "[$(date -u +%FT%TZ)] ERROR: $*" >&2
  exit 1
}
[ "$(id -u)" -eq 0 ] || die "run this with sudo"

# KEY=VALUE lookup in fleet.env (systemd EnvironmentFile syntax, never sourced)
envget() {
  local v
  v=$(sed -n "s/^[[:space:]]*$1=//p" "$ETC/fleet.env" | tail -n 1)
  v=${v%\"}
  v=${v#\"}
  echo "${v:-${2:-}}"
}

check_token() {
  [ -s "$TOKEN_FILE" ] || die "no token at $TOKEN_FILE (sudo $ME --set-token)"
  [ "$(stat -c '%U:%G %a' "$TOKEN_FILE")" = "root:root 600" ] || die "$TOKEN_FILE must be root:root 0600"
  [ -x "$FH/venv/bin/python" ] || die "venv not installed yet - run the install first"
  # root reads the root-only file inside Python and sends it only to the
  # read-only whoami endpoint; the token is never printed
  env -i PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8 HF_HUB_DISABLE_TELEMETRY=1 \
    "$FH/venv/bin/python" - "$TOKEN_FILE" << 'PY'
import sys
from huggingface_hub import HfApi
tok = open(sys.argv[1]).read().strip()
info = HfApi().whoami(token=tok)
del tok
at = (info.get("auth") or {}).get("accessToken") or {}
print(f"account: {info.get('name')}  token: '{at.get('displayName')}' role={at.get('role')}")
ok = at.get("role") == "fineGrained"
if not ok:
    print(f"  FAIL: token role is {at.get('role')!r}; create a FINE-GRAINED token")
# exact HF permission names: repo.write = content write; repo.content.read = read
# (discussion.write and the like grant no content access and are ignored)
have = {}
for s in (at.get("fineGrained") or {}).get("scoped") or []:
    ent, perms = s.get("entity") or {}, set(s.get("permissions") or [])
    name = ent.get("name") or ent.get("_id")
    print(f"  scoped {ent.get('type')} {name}: {', '.join(sorted(perms))}")
    if ent.get("type") in ("user", "org"):
        if "repo.write" in perms:
            ok = False
            print(f"  FAIL: account-wide repo.write on {name}; scope the token to the two datasets")
        continue
    have[name] = "write" if "repo.write" in perms else ("read" if "repo.content.read" in perms else "")
if have.get("vincewin/CREST_fleet") != "write":
    ok = False
    print("  FAIL: needs repo.write on vincewin/CREST_fleet")
if have.get("vincewin/CREST_state") not in ("read", "write"):
    ok = False
    print("  FAIL: needs repo.content.read on vincewin/CREST_state (else calibrated gauges silently use a-priori parameters)")
if have.get("vincewin/CREST_state") == "write":
    ok = False
    print("  FAIL: write on vincewin/CREST_state is not needed - make it read-only")
for repo, lvl in have.items():
    if repo not in ("vincewin/CREST_fleet", "vincewin/CREST_state") and lvl == "write":
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
  printf 'Paste the fine-grained HF token (input hidden), then press Enter: ' > /dev/tty
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
  tmp=$(mktemp "$ETC/.hf_token.XXXXXX")
  printf '%s\n' "$tok" > "$tmp"
  tok=
  unset tok
  chown root:root "$tmp"
  chmod 0600 "$tmp"
  mv -f "$tmp" "$TOKEN_FILE"
  log "token stored in $TOKEN_FILE (root:root 0600); checking its scope ..."
  check_token || log "WARNING: token scope check did not pass - fix the token before starting a shard"
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
    die "needs Ubuntu 22.04 (GDAL 3.4.1 for the gdal==3.4.1 pin; same base as fleet_space/Dockerfile)"

  # shards run from $FH/src, $FH/venv and $FH/EF5: never swap them underneath
  local active
  active=$(systemctl list-units --plain --no-legend --state=active,activating 'crest-fleet@*.service' | awk '{print $1}' | tr '\n' ' ')
  if [ -n "${active// /}" ]; then
    log "stopping running shards for the update: $active"
    # shellcheck disable=SC2086
    systemctl stop $active
  fi

  log "apt: build + geo stack (as fleet_space/Dockerfile)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q --no-install-recommends \
    git gcc g++ build-essential make \
    autoconf automake libtool dh-autoreconf autotools-dev pkg-config \
    libgeotiff-dev libtiff-dev zlib1g-dev \
    python3 python3-dev python3-pip python3-venv \
    wget ca-certificates lsb-release gnupg acl \
    libgeos-dev libproj-dev libgdal-dev
  if ! dpkg -s libarrow-dev libparquet-dev > /dev/null 2>&1; then
    log "apt: Apache Arrow C++ + Parquet"
    local deb=/tmp/apache-arrow-apt-source-latest-jammy.deb
    wget -q -O "$deb" https://apache.jfrog.io/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-jammy.deb
    apt-get install -y -q -o Acquire::Retries=8 "$deb"
    apt-get update -q -o Acquire::Retries=8
    apt-get install -y -q -o Acquire::Retries=8 libarrow-dev libparquet-dev
    rm -f "$deb"
  fi

  install -d -m 0755 "$FH" "$FH/bin"
  # rebuild on a missing binary, another ref, or libraries an apt upgrade moved
  if [ ! -x "$FH/EF5/bin/ef5" ] ||
    [ "$(git -C "$FH/EF5" rev-parse --short=7 HEAD 2> /dev/null)" != "$EF5_REF" ] ||
    [ "$(ldd "$FH/EF5/bin/ef5" 2>&1 | grep -c 'not found' || true)" -ne 0 ]; then
    log "EF5 fork @ $EF5_REF: build"
    rm -rf "$FH/EF5"
    git clone -q "$EF5_REPO" "$FH/EF5"
    (
      cd "$FH/EF5"
      git checkout -q "$EF5_REF"
      autoreconf --force --install > /dev/null 2>&1
      ./configure --with-arrow CXXFLAGS="-std=c++20 -Wall -O2" CFLAGS="-Wall -O2" > /dev/null
      sed -i 's/-Werror//g' Makefile
      make -j"$(nproc)" > /dev/null
    )
    [ -x "$FH/EF5/bin/ef5" ] || die "EF5 build produced no bin/ef5"
  fi
  log "EF5: $FH/EF5/bin/ef5 @ $(git -C "$FH/EF5" rev-parse --short=7 HEAD)"

  if ! id "$USR" > /dev/null 2>&1; then
    useradd --system --home-dir "$UHOME" --create-home --shell /usr/sbin/nologin "$USR"
    log "created system user $USR"
  fi

  install -d -o root -g root -m 0755 "$ETC"
  if [ ! -f "$ETC/fleet.env" ]; then
    install -o root -g root -m 0644 "$HERE/fleet.env.example" "$ETC/fleet.env"
    log "created $ETC/fleet.env from the example (no secrets go there)"
  fi
  local d a cache hub tmpd
  cache=$(envget CREST_CACHE_DIR)
  hub=$(envget HF_HOME)
  tmpd=$(envget TMPDIR)
  [ -n "$cache" ] && [ -n "$hub" ] && [ -n "$tmpd" ] || die "CREST_CACHE_DIR, HF_HOME and TMPDIR must be set in $ETC/fleet.env"
  for d in "$cache" "$hub" "$tmpd"; do
    on_unmounted_volume "$d" && die "$d is under /media/volume/ but no volume is mounted there"
    # the 2-D worker kit makes the volume root crest:crest 0750: give
    # crest-fleet search (x) on every ancestor it cannot pass. A named ACL
    # entry survives the worker bootstrap's chmod (that only resets the mask).
    a=$(dirname "$d")
    while [ "$a" != / ]; do
      if [ -d "$a" ] && ! sudo -u "$USR" test -x "$a"; then
        setfacl -m "u:$USR:--x" "$a"
        log "ACL: $USR may traverse $a"
      fi
      a=$(dirname "$a")
    done
    install -d -o "$USR" -g "$USR" -m 0750 "$d"
    sudo -u "$USR" test -w "$d" || die "$USR cannot write $d (check the permissions of its parents)"
  done

  if [ ! -d "$FH/src/.git" ]; then
    log "CREST_AI main -> $FH/src"
    git clone -q "$SRC_REPO" "$FH/src"
  fi
  ln -sfn "$FH/EF5" "$FH/src/EF5"
  chown -R "$USR:$USR" "$FH/src"
  sudo -u "$USR" git -C "$FH/src" fetch -q origin main
  sudo -u "$USR" git -C "$FH/src" reset -q --hard origin/main
  log "CREST_AI $(sudo -u "$USR" git -C "$FH/src" rev-parse --short=12 HEAD)"

  local req_hash
  req_hash=$(sha256sum "$FH/src/requirements.txt" | cut -c1-16)
  if [ ! -x "$FH/venv/bin/python" ] || [ "$(cat "$FH/venv/.req_hash" 2> /dev/null)" != "$req_hash" ]; then
    log "venv: $FH/venv (requirements.txt of CREST_AI main, as the Spaces)"
    [ -x "$FH/venv/bin/python" ] || python3 -m venv "$FH/venv"
    "$FH/venv/bin/pip" install -q --no-cache-dir --upgrade pip
    "$FH/venv/bin/pip" install -q --no-cache-dir -r "$FH/src/requirements.txt"
    echo "$req_hash" > "$FH/venv/.req_hash"
  fi

  # root-owned copy of the kit: what root runs later (--set-token, updates)
  if [ "$HERE" != "$FH/bin" ]; then
    install -o root -g root -m 0755 "$HERE/setup.sh" "$HERE/run-fleet.sh" "$HERE/fleet-housekeep.sh" "$FH/bin/"
    install -o root -g root -m 0644 "$HERE/fleet.env.example" "$HERE/crest-fleet@.service" \
      "$HERE/crest-fleet.slice" "$HERE/crest-fleet-housekeep.service" "$HERE/crest-fleet-housekeep.timer" "$FH/bin/"
  fi
  install -o root -g root -m 0644 "$FH/bin/crest-fleet@.service" "$FH/bin/crest-fleet.slice" \
    "$FH/bin/crest-fleet-housekeep.service" "$FH/bin/crest-fleet-housekeep.timer" /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable --now crest-fleet-housekeep.timer > /dev/null

  log "smoke test as $USR"
  sudo -u "$USR" env CREST_DEMO_MOCK=0 CREST_CACHE_DIR="$cache" HF_HOME="$hub" \
    TMPDIR="$tmpd" HOME="$UHOME" "$FH/venv/bin/python" -c "
import os, sys
sys.path.insert(0, '$FH/src')
from osgeo import gdal
import rasterio, numpy, pyarrow
from hf_data import pipeline, forcing, basic, statecache, statebundle
print('imports OK: gdal', gdal.__version__, 'rasterio', rasterio.__version__, 'numpy', numpy.__version__,
      'cache', statecache.CACHE_DIR, 'bundle block', statebundle.PACK_BLOCK_BYTES)
"
  "$FH/src/EF5/bin/ef5" 2>&1 | head -n 3 || true

  if [ -n "${active// /}" ]; then
    log "restarting the shards stopped for the update: $active"
    # shellcheck disable=SC2086
    systemctl start $active
  fi
  echo
  log "setup complete. Enabled shards: $(systemctl list-unit-files --plain --no-legend 'crest-fleet@*.service' --state=enabled | awk '{print $1}' | tr '\n' ' ')"
  [ -s "$TOKEN_FILE" ] || log "next: sudo $ME --set-token"
  log "to add shard K: pause the HF Space with FLEET_SHARD=K/6, then: sudo systemctl enable --now crest-fleet@K"
}

case ${1:-} in
  '') install_all ;;
  --set-token) set_token ;;
  --check-token) check_token ;;
  -h | --help) sed -n '2,15p' "$0" ;;
  *) die "unknown option $1 (see --help)" ;;
esac
