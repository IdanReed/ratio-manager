#!/usr/bin/env bash
# Local end-to-end rig: three real qBittorrent-nox instances (home, seedit4me,
# pulsed), the fakes (TL, tracker, Prowlarr, autobrr, ntfy) and the manager.
#
#   dev/run.sh up      # start everything under $RIG (default /tmp/ratio-rig)
#   dev/run.sh down    # stop it
#
# The WebUI login of every box is admin / adminadmin.
set -euo pipefail
RIG=${RIG:-/tmp/ratio-rig}
HERE=$(cd "$(dirname "$0")" && pwd)

qbit_conf() { # dir port listen
  mkdir -p "$1/qBittorrent/config" "$1/dl"
  python3 - "$1" "$2" "$3" <<'PY'
import base64, hashlib, os, sys
d, port, listen = sys.argv[1:]
salt = os.urandom(16)
key = hashlib.pbkdf2_hmac("sha512", b"adminadmin", salt, 100000, 64)
pw = f"@ByteArray({base64.b64encode(salt).decode()}:{base64.b64encode(key).decode()})"
open(f"{d}/qBittorrent/config/qBittorrent.conf", "w").write(f"""[LegalNotice]
Accepted=true

[BitTorrent]
Session\\DefaultSavePath={d}/dl
Session\\Port={listen}
Session\\DHTEnabled=true
Session\\PeXEnabled=true
Session\\LSDEnabled=true
Session\\SSRFMitigation=false

[Preferences]
WebUI\\Port={port}
WebUI\\Username=admin
WebUI\\Password_PBKDF2="{pw}"
WebUI\\LocalHostAuth=true
WebUI\\CSRFProtection=false
WebUI\\HostHeaderValidation=false
""")
PY
}

case "${1:-up}" in
up)
  mkdir -p "$RIG"
  python3 "$HERE/fakes.py" > "$RIG/fakes.log" 2>&1 & echo $! > "$RIG/fakes.pid"
  n=0
  for box in home:18083:18103 seedit4me:18081:18101 pulsed:18082:18102; do
    IFS=: read -r name port listen <<<"$box"
    qbit_conf "$RIG/$name" "$port" "$listen"
    nix shell nixpkgs#qbittorrent-nox -c qbittorrent-nox --profile="$RIG/$name" --webui-port="$port" --confirm-legal-notice \
      > "$RIG/$name.log" 2>&1 & echo $! > "$RIG/$name.pid"
    n=$((n + 1))
  done
  sed "s#@RIG@#$RIG#g" "$HERE/registry.yaml" > "$RIG/registry.yaml"
  echo "rig up under $RIG"
  ;;
down)
  for f in "$RIG"/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null || true; rm -f "$f"; done
  pkill -f "qbittorrent-nox --profile=$RIG" 2>/dev/null || true
  echo "rig down"
  ;;
esac
