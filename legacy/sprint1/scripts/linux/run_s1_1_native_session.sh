#!/usr/bin/env bash
set -u -o pipefail

BRIDGE_MOUNT="${DAMH_DATA_MOUNT:-/mnt/data}"
BRIDGE="${BRIDGE_MOUNT}/DAMH-Bridge"
PROJECT_ROOT="${DAMH_LINUX_PROJECT:-$HOME/projects/DAMH}"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run RUN_S1_1_NATIVE.sh with sudo/root."
  echo "Run it as the normal desktop user."
  exit 7
fi

if ! findmnt -rn "$BRIDGE_MOUNT" >/dev/null 2>&1; then
  echo "DATA is not mounted at $BRIDGE_MOUNT."
  echo "Mount it for the normal user before continuing:"
  echo "  sudo mount -t ntfs3 -o uid=$(id -u),gid=$(id -g),umask=022 /dev/nvme0n1p4 $BRIDGE_MOUNT"
  exit 2
fi

LATEST_FILE="${BRIDGE}/outgoing/LATEST.txt"
if [[ ! -f "$LATEST_FILE" ]]; then
  echo "Missing $LATEST_FILE"
  exit 3
fi

if [[ ! -w "$BRIDGE/incoming" ]]; then
  echo "$BRIDGE/incoming is not writable by $(id -un)."
  echo "Remount DATA with the normal user's uid/gid before continuing."
  echo "Example:"
  echo "  cd ~"
  echo "  sudo umount $BRIDGE_MOUNT"
  echo "  sudo mount -t ntfs3 -o uid=$(id -u),gid=$(id -g),umask=022 /dev/nvme0n1p4 $BRIDGE_MOUNT"
  exit 8
fi

SESSION_NAME="$(tr -d '\r\n' < "$LATEST_FILE")"
SESSION_DIR="${BRIDGE}/outgoing/${SESSION_NAME}"

echo "=== S1.1 native session ==="
echo "session=$SESSION_NAME"
echo "project=$PROJECT_ROOT"
(
  cd "$SESSION_DIR" || exit 4
  sha256sum -c SHA256SUMS.txt
) || exit 4

echo "=== Updating ext4 project from reviewed Windows bundle ==="
bash "${SESSION_DIR}/bootstrap_from_bridge.sh" || exit 5

echo "=== Network preflight ==="
if ! getent hosts github.com >/dev/null 2>&1; then
  echo "DNS preflight failed: github.com cannot be resolved."
  echo "Connect Wi-Fi/network and rerun this launcher; no simulator gate has started."
  exit 9
fi
if ! curl -fsSI --connect-timeout 10 --max-time 20 https://github.com >/dev/null; then
  echo "HTTPS preflight failed: github.com is not reachable."
  echo "Fix network connectivity and rerun; no simulator gate has started."
  exit 10
fi
echo "NETWORK_PREFLIGHT_OK"

cd "$PROJECT_ROOT" || exit 6
echo "=== Running canonical S1.1 harness ==="
set +e
bash scripts/linux/run_s1_1_smoke.sh
SMOKE_RC=$?

S12_RC=99
S12_STATUS="SKIPPED"
if [[ "$SMOKE_RC" -eq 0 ]]; then
  echo "=== Running non-actuating S1.2 project motion probe ==="
  S12_DIR="experiments/results/s1_2_adapter_probe_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$S12_DIR"

  CONDA_EXE="$(command -v conda || true)"
  if [[ -z "$CONDA_EXE" && -x "$HOME/miniforge3/bin/conda" ]]; then
    CONDA_EXE="$HOME/miniforge3/bin/conda"
  fi

  if [[ -z "$CONDA_EXE" ]]; then
    echo "conda executable not found for S1.2 probe"
    S12_RC=90
    S12_STATUS="FAIL_NO_CONDA"
  else
    timeout --signal=TERM --kill-after=20s 600s \
      "$CONDA_EXE" run -n RoboTwin python -u scripts/linux/probe_panthera_project_motion.py \
        --robotwin-root "$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin" \
        --project-root "$PROJECT_ROOT" \
        --output "$S12_DIR/project-motion.json"
    S12_RC=$?
    S12_STATUS="$([[ "$S12_RC" -eq 0 ]] && echo PASS || echo FAIL)"
  fi
fi

echo "=== Exporting evidence to DATA bridge ==="
bash scripts/linux/collect_to_bridge.sh
COLLECT_RC=$?
sync
echo
echo "S1.1 smoke exit code: $SMOKE_RC"
echo "S1.2 planning probe status: $S12_STATUS (exit=$S12_RC)"
echo "Evidence export exit code: $COLLECT_RC"
echo "Newest bridge result:"
ls -1dt "${BRIDGE}/incoming/"* 2>/dev/null | head -n 1 || true

if [[ "$COLLECT_RC" -ne 0 ]]; then
  echo "Evidence export failed; do not reboot until the bridge result is preserved."
  exit "$COLLECT_RC"
fi

if [[ "$SMOKE_RC" -ne 0 ]]; then
  echo "S1.1 harness failed, but evidence was exported."
  echo "Reboot to Windows for diagnosis; do not patch dependencies manually."
  exit "$SMOKE_RC"
fi

if [[ "$S12_RC" -ne 0 ]]; then
  echo "S1.1 PASS, but the supplemental S1.2 planning probe failed."
  echo "Evidence was exported; return to Windows for audit without manual dependency changes."
  exit "$S12_RC"
fi

echo "S1.1 PASS and supplemental S1.2 planning probe PASS."
echo "You may sync/reboot to Windows for audit."
exit 0
