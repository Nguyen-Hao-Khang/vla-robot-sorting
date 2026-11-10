#!/usr/bin/env bash
set -euo pipefail

BRIDGE_MOUNT="${DAMH_DATA_MOUNT:-/mnt/data}"
BRIDGE="${BRIDGE_MOUNT}/DAMH-Bridge"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if ! findmnt -rn "$BRIDGE_MOUNT" >/dev/null 2>&1; then
  device="$(blkid -L DATA || true)"
  if [[ -z "$device" ]]; then
    echo "DATA volume label not found."
    exit 2
  fi
  sudo mkdir -p "$BRIDGE_MOUNT"
  sudo mount -t ntfs3 "$device" "$BRIDGE_MOUNT"
fi

SESSION_NAME="unknown-session"
if [[ -f "$BRIDGE/outgoing/LATEST.txt" ]]; then
  SESSION_NAME="$(tr -d '\r\n' < "$BRIDGE/outgoing/LATEST.txt")"
fi

OUT="$BRIDGE/incoming/${SESSION_NAME}-host-${STAMP}"
mkdir -p "$OUT"

uname -a > "$OUT/uname.txt"
cat /etc/os-release > "$OUT/os-release.txt"
lsblk -f > "$OUT/lsblk.txt"
df -h > "$OUT/df.txt"
free -h > "$OUT/free.txt"
nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1 || true
vulkaninfo --summary > "$OUT/vulkaninfo-summary.txt" 2>&1 || true
ubuntu-drivers list > "$OUT/ubuntu-drivers-list.txt" 2>&1 || true

if [[ -d "$HOME/damh-host-bootstrap" ]]; then
  cp -a "$HOME/damh-host-bootstrap/." "$OUT/"
fi

cat > "$OUT/RESULT.md" <<EOF
# Host Bootstrap Result

- Session: $SESSION_NAME
- Timestamp UTC: $STAMP
- Result classification: FILL_ME

## What happened

## First failing command or gate

## Recommended next action
EOF

(
  cd "$OUT"
  find . -maxdepth 1 -type f ! -name SHA256SUMS.txt -printf '%f\n' |
    sort |
    xargs -r sha256sum > SHA256SUMS.txt
)

echo "Host evidence exported to: $OUT"
echo "Edit $OUT/RESULT.md before rebooting Windows."
