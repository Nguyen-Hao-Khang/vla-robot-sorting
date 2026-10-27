#!/usr/bin/env bash
set -u -o pipefail

BRIDGE_MOUNT="${DAMH_DATA_MOUNT:-/mnt/data}"
BRIDGE="${BRIDGE_MOUNT}/DAMH-Bridge"
PROJECT_ROOT="${DAMH_LINUX_PROJECT:-$HOME/projects/DAMH}"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run RUN_S1_2_NATIVE.sh with sudo/root."
  exit 7
fi

if ! findmnt -rn "$BRIDGE_MOUNT" >/dev/null 2>&1; then
  echo "DATA is not mounted at $BRIDGE_MOUNT."
  exit 2
fi
if [[ ! -w "$BRIDGE/incoming" ]]; then
  echo "$BRIDGE/incoming is not writable by $(id -un)."
  exit 8
fi

LATEST_FILE="$BRIDGE/outgoing/LATEST.txt"
SESSION_NAME="$(tr -d '\r\n' < "$LATEST_FILE")"
SESSION_DIR="$BRIDGE/outgoing/$SESSION_NAME"

echo "=== S1.2 native backend session ==="
echo "session=$SESSION_NAME"
(
  cd "$SESSION_DIR" || exit 4
  sha256sum -c SHA256SUMS.txt
) || exit 4

echo "=== Updating ext4 project from reviewed Windows bundle ==="
bash "$SESSION_DIR/bootstrap_from_bridge.sh" || exit 5
cd "$PROJECT_ROOT" || exit 6

CONDA_EXE="$(command -v conda || true)"
if [[ -z "$CONDA_EXE" && -x "$HOME/miniforge3/bin/conda" ]]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
fi
if [[ -z "$CONDA_EXE" ]]; then
  echo "conda executable not found"
  exit 90
fi

nvidia-smi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
S12_DIR="experiments/results/s1_2_native_$STAMP"
mkdir -p "$S12_DIR"

echo "=== Project motion contract probe ==="
set +e
timeout --signal=TERM --kill-after=20s 600s \
  "$CONDA_EXE" run -n RoboTwin python -u scripts/linux/probe_panthera_project_motion.py \
    --robotwin-root "$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin" \
    --project-root "$PROJECT_ROOT" \
    --output "$S12_DIR/project-motion.json" \
  > >(tee "$S12_DIR/project-motion.log") 2>&1
MOTION_RC=$?

if [[ "$MOTION_RC" -eq 0 ]]; then
  PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin \
    python scripts/audit_s1_2_probe.py "$S12_DIR/project-motion.json" \
    > >(tee "$S12_DIR/project-motion-audit.log") 2>&1
  MOTION_AUDIT_RC=$?
else
  MOTION_AUDIT_RC=99
fi

echo "=== Repeated B1/B2/B3 backend matrix ==="
timeout --signal=TERM --kill-after=30s 1200s \
  "$CONDA_EXE" run -n RoboTwin python -u scripts/linux/probe_panthera_backend_matrix.py \
    --robotwin-root "$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin" \
    --project-root "$PROJECT_ROOT" \
    --output "$S12_DIR/backend-matrix.json" \
    --curobo-runs 10 \
    --coexistence-runs 30 \
    --mplib-runs 10 \
  > >(tee "$S12_DIR/backend-matrix.log") 2>&1
MATRIX_RC=$?

echo "=== B4 timing + splice probe ==="
timeout --signal=TERM --kill-after=30s 1200s \
  "$CONDA_EXE" run -n RoboTwin python -u scripts/linux/probe_panthera_timing_splice.py \
    --robotwin-root "$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin" \
    --project-root "$PROJECT_ROOT" \
    --output "$S12_DIR/timing-splice.json" \
    --runs 20 \
  > >(tee "$S12_DIR/timing-splice.log") 2>&1
B4_RC=$?
set -e

SESSION_RC=0
FIRST_FAIL=""
if [[ "$MOTION_RC" -ne 0 ]]; then
  SESSION_RC="$MOTION_RC"
  FIRST_FAIL="project-motion exit=$MOTION_RC"
elif [[ "$MOTION_AUDIT_RC" -ne 0 ]]; then
  SESSION_RC="$MOTION_AUDIT_RC"
  FIRST_FAIL="project-motion-audit exit=$MOTION_AUDIT_RC"
elif [[ "$MATRIX_RC" -ne 0 ]]; then
  SESSION_RC="$MATRIX_RC"
  FIRST_FAIL="backend-matrix exit=$MATRIX_RC"
elif [[ "$B4_RC" -ne 0 ]]; then
  SESSION_RC="$B4_RC"
  FIRST_FAIL="timing-splice B4 exit=$B4_RC"
fi

if [[ "$SESSION_RC" -eq 0 ]]; then
  RESULT_CLASSIFICATION="PASS"
else
  RESULT_CLASSIFICATION="FAIL"
fi

printf 'MOTION_RC=%s\nMOTION_AUDIT_RC=%s\nMATRIX_RC=%s\nB4_RC=%s\nSESSION_RC=%s\n' \
  "$MOTION_RC" "$MOTION_AUDIT_RC" "$MATRIX_RC" "$B4_RC" "$SESSION_RC" \
  > "$S12_DIR/SESSION_RESULT.txt"

S12_EVIDENCE="$HOME/damh-evidence/s1_2_$STAMP"
mkdir -p "$S12_EVIDENCE"
cp -a "$S12_DIR/." "$S12_EVIDENCE/"
if [[ "$SESSION_RC" -eq 0 ]]; then
  echo "PASS" > "$S12_EVIDENCE/PASS"
else
  printf 'FAILED_STAGE=%s\nEXIT_CODE=%s\n' "$FIRST_FAIL" "$SESSION_RC" \
    > "$S12_EVIDENCE/FAILED"
fi
(
  cd "$S12_EVIDENCE"
  find . -type f ! -path './SHA256SUMS.txt' -print0 |
    sort -z |
    xargs -0 sha256sum > SHA256SUMS.txt
)

echo "=== Exporting S1.2 evidence to DATA bridge ==="
DAMH_RESULT_CLASSIFICATION_OVERRIDE="$RESULT_CLASSIFICATION" \
DAMH_RESULT_SCOPE_OVERRIDE="S1.2 native motion + B1/B2/B3 + B4 timing/splice" \
DAMH_FIRST_FAIL_OVERRIDE="$FIRST_FAIL" \
DAMH_EVIDENCE_DIR_OVERRIDE="$S12_EVIDENCE" \
  bash scripts/linux/collect_to_bridge.sh
COLLECT_RC=$?
sync

echo
echo "Project motion exit: $MOTION_RC"
echo "Project motion audit exit: $MOTION_AUDIT_RC"
echo "Backend matrix exit: $MATRIX_RC"
echo "B4 timing/splice exit: $B4_RC"
echo "Session result: $RESULT_CLASSIFICATION (exit=$SESSION_RC)"
echo "Evidence export exit code: $COLLECT_RC"
ls -1dt "$BRIDGE/incoming/"* 2>/dev/null | head -n 1 || true

if [[ "$COLLECT_RC" -ne 0 ]]; then
  echo "Evidence export failed; do not reboot yet."
  exit "$COLLECT_RC"
fi
if [[ "$SESSION_RC" -ne 0 ]]; then
  echo "S1.2 has a measured failure; return to Windows for audit without changing dependencies."
  exit "$SESSION_RC"
fi

echo "S1.2 project motion, B1/B2/B3 backend matrix, and B4 timing/splice PASS."
echo "You may sync/reboot to Windows for audit."
