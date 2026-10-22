#!/usr/bin/env bash
# setup.sh - install the CREST gauged-fleet runner on the Jetstream2 VM.
# Same stack as fleet_space/Dockerfile (Ubuntu 22.04: GDAL 3.4.1 matches the
# gdal==3.4.1 pin; Arrow C++ for the EF5 fork's Parquet forcing reader).
# Idempotent; never starts or enables a shard. As root:
#   sudo bash fleet/jetstream2/setup.sh                # install / update
#   sudo bash fleet/jetstream2/setup.sh --set-token    # fine-grained HF token, hidden input
#   sudo bash fleet/jetstream2/setup.sh --check-token  # show its scope (read-only)
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

log() { echo "[$(date -u +%FT%TZ)] $*"; }
die() {
  echo "[$(date -u +%FT%TZ)] ERROR: $*" >&2
  exit 1
}
[ "$(id -u)" -eq 0 ] || die "run this with sudo"

check_token() {
  [ -s "$TOKEN_FILE" ] || die "no token at $TOKEN_FILE (sudo $0 --set-token)"
  [ "$(stat -c '%U:%G %a' "$TOKEN_FILE")" = "root:root 600" ] || die "$TOKEN_FILE must be root:root 0600"
  [ -x "$FH/venv/bin/python" ] || die "venv not installed yet - run sudo $0 first"
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
need = {"vincewin/CREST_fleet": "write", "vincewin/CREST_state": "read"}
have = {}
for s in (at.get("fineGrained") or {}).get("scoped") or []:
    ent, perms = s.get("entity") or {}, s.get("permissions") or []
    name = ent.get("name") or ent.get("_id")
    print(f"  scoped {ent.get('type')} {name}: {', '.join(perms)}")
    w = any("write" in p for p in perms)
    if ent.get("type") in ("user", "org") and w:
        ok = False
        print(f"  FAIL: account-wide write on {name}; scope the token to the two datasets")
    have[name] = "write" if w else ("read" if any("read" in p for p in perms) else "")
for repo, lvl in need.items():
    got = have.get(repo, "")
    if not got or (lvl == "write" and got != "write"):
        ok = False
        print(f"  FAIL: needs {lvl} on {repo} (has: {got or 'nothing'})")
for repo, lvl in have.items():
    if repo not in need and lvl == "write":
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

  log "apt: build + geo stack (as fleet_space/Dockerfile)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q --no-install-recommends \
    git gcc g++ build-essential make \
    autoconf automake libtool dh-autoreconf autotools-dev pkg-config \
    libgeotiff-dev libtiff-dev zlib1g-dev \
    python3 python3-dev python3-pip python3-venv \
    wget ca-certificates lsb-release gnupg \
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
  if [ ! -x "$FH/EF5/bin/ef5" ] || [ "$(git -C "$FH/EF5" rev-parse --short=7 HEAD 2> /dev/null)" != "$EF5_REF" ]; then
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
  # shellcheck source=/dev/null
  . "$ETC/fleet.env"
  for d in "${CREST_CACHE_DIR:?}" "${HF_HOME:?}" "${TMPDIR:?}"; do
    on_unmounted_volume "$d" && die "$d is under /media/volume/ but no volume is mounted there"
    install -d -o "$USR" -g "$USR" -m 0750 "$d"
  done

  if [ ! -d "$FH/src/.git" ]; then
    log "CREST_AI main -> $FH/src"
    git clone -q "$SRC_REPO" "$FH/src"
  fi
  ln -sfn "$FH/EF5" "$FH/src/EF5"
  chown -R "$USR:$USR" "$FH/src"
  sudo -u "$USR" git -C "$FH/src" fetch -q origin main
  sudo -u "$USR" git -C "$FH/src" reset -q --hard origin/main
  log "CREST_AI $(git -C "$FH/src" rev-parse --short=12 HEAD)"

  local req_hash
  req_hash=$(sha256sum "$FH/src/requirements.txt" | cut -c1-16)
  if [ ! -x "$FH/venv/bin/python" ] || [ "$(cat "$FH/venv/.req_hash" 2> /dev/null)" != "$req_hash" ]; then
    log "venv: $FH/venv (requirements.txt of CREST_AI main, as the Spaces)"
    [ -x "$FH/venv/bin/python" ] || python3 -m venv "$FH/venv"
    "$FH/venv/bin/pip" install -q --no-cache-dir --upgrade pip
    "$FH/venv/bin/pip" install -q --no-cache-dir -r "$FH/src/requirements.txt"
    echo "$req_hash" > "$FH/venv/.req_hash"
  fi

  install -o root -g root -m 0755 "$HERE/run-fleet.sh" "$HERE/fleet-housekeep.sh" "$FH/bin/"
  install -o root -g root -m 0644 "$HERE/crest-fleet@.service" \
    "$HERE/crest-fleet-housekeep.service" "$HERE/crest-fleet-housekeep.timer" /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable --now crest-fleet-housekeep.timer > /dev/null

  log "smoke test as $USR"
  sudo -u "$USR" env CREST_DEMO_MOCK=0 CREST_CACHE_DIR="$CREST_CACHE_DIR" HF_HOME="$HF_HOME" \
    TMPDIR="$TMPDIR" HOME="$UHOME" "$FH/venv/bin/python" -c "
import os, sys
sys.path.insert(0, '$FH/src')
from osgeo import gdal
import rasterio, numpy, pyarrow
from hf_data import pipeline, forcing, basic, statecache
print('imports OK: gdal', gdal.__version__, 'rasterio', rasterio.__version__, 'numpy', numpy.__version__,
      'cache', statecache.CACHE_DIR)
"
  "$FH/src/EF5/bin/ef5" 2>&1 | head -n 3 || true

  echo
  log "setup complete. No shard is enabled or started."
  [ -s "$TOKEN_FILE" ] || log "next: sudo $0 --set-token"
  log "then pause the HF Space with FLEET_SHARD=K/6 and: sudo systemctl enable --now crest-fleet@K"
}

case ${1:-} in
  '') install_all ;;
  --set-token) set_token ;;
  --check-token) check_token ;;
  -h | --help) sed -n '2,12p' "$0" ;;
  *) die "unknown option $1 (see --help)" ;;
esac
