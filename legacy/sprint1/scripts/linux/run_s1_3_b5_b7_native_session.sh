#!/usr/bin/env bash
set -u -o pipefail

BRIDGE_MOUNT="${DAMH_DATA_MOUNT:-/mnt/data}"
BRIDGE="${BRIDGE_MOUNT}/DAMH-Bridge"
PROJECT_ROOT="${DAMH_LINUX_PROJECT:-$HOME/projects/DAMH}"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run RUN_S1_3_B5_B7_NATIVE.sh with sudo/root."
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

echo "=== B5/B7 native grasp/recovery session ==="
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
RESULT_DIR="experiments/results/sprint1_native_consolidated_$STAMP"
mkdir -p "$RESULT_DIR"
echo "=== B5 geometry-first 6-DoF grasp + B7 stale recovery (non-actuating) ==="
set +e
timeout --signal=TERM --kill-after=30s 2700s \
  "$CONDA_EXE" run -n RoboTwin python -u scripts/linux/probe_panthera_grasp_recovery.py \
    --robotwin-root "$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin" \
    --project-root "$PROJECT_ROOT" \
    --output "$RESULT_DIR/grasp-recovery.json" \
    --runs 20 \
  > >(tee "$RESULT_DIR/grasp-recovery.log") 2>&1
B57_RC=$?

if [[ "$B57_RC" -eq 0 ]]; then
  PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin \
    python scripts/audit_s1_3_b5_b7.py "$RESULT_DIR/grasp-recovery.json" \
    > >(tee "$RESULT_DIR/grasp-recovery-audit.log") 2>&1
  AUDIT_RC=$?
else
  AUDIT_RC=99
fi

echo "=== S1-A-007 controlled planner-failure fail-closed probe ==="
timeout --signal=TERM --kill-after=30s 600s \
  "$CONDA_EXE" run -n RoboTwin python -u scripts/linux/probe_panthera_safe_failure.py \
    --robotwin-root "$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin" \
    --project-root "$PROJECT_ROOT" \
    --output "$RESULT_DIR/safe-failure.json" \
  > >(tee "$RESULT_DIR/safe-failure.log") 2>&1
A007_RC=$?

if [[ "$A007_RC" -eq 0 ]]; then
  PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin \
    python scripts/audit_s1_a007_safe_failure.py "$RESULT_DIR/safe-failure.json" \
    > >(tee "$RESULT_DIR/safe-failure-audit.log") 2>&1
  A007_AUDIT_RC=$?
else
  A007_AUDIT_RC=99
fi

echo "=== Integrated live grounding + bounded pregrasp execution + destination recovery ==="
nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/integrated-gpu.csv" 2>&1 &
INTEGRATED_GPU_PID=$!
/usr/bin/time -v -o "$RESULT_DIR/integrated-time.txt" \
  timeout --signal=TERM --kill-after=30s 1200s \
  "$CONDA_EXE" run -n RoboTwin python -u scripts/linux/probe_panthera_integrated_pregrasp.py \
    --robotwin-root "$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin" \
    --project-root "$PROJECT_ROOT" \
    --output "$RESULT_DIR/integrated-pregrasp.json" \
  > >(tee "$RESULT_DIR/integrated-pregrasp.log") 2>&1
INTEGRATED_RC=$?
kill "$INTEGRATED_GPU_PID" >/dev/null 2>&1 || true
wait "$INTEGRATED_GPU_PID" 2>/dev/null || true

if [[ "$INTEGRATED_RC" -eq 0 ]]; then
  PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin \
    python scripts/audit_integrated_pregrasp.py "$RESULT_DIR/integrated-pregrasp.json" \
    > >(tee "$RESULT_DIR/integrated-pregrasp-audit.log") 2>&1
  INTEGRATED_AUDIT_RC=$?
else
  INTEGRATED_AUDIT_RC=99
fi

PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin \
  python scripts/summarize_resource_usage.py \
    --gpu-csv "$RESULT_DIR/integrated-gpu.csv" \
    --time-file "$RESULT_DIR/integrated-time.txt" \
    --output "$RESULT_DIR/integrated-resource-summary.json" \
  > >(tee "$RESULT_DIR/integrated-resource-summary.log") 2>&1
INTEGRATED_RESOURCE_RC=$?

echo "=== Assemble fixed 12-case Sprint-1 evaluation matrix ==="
if [[ -f "$RESULT_DIR/safe-failure.json" ]]; then
  PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin \
    python scripts/build_sprint1_evaluation.py \
      --output-dir "$RESULT_DIR/sprint1-evaluation" \
      --a007-probe "$RESULT_DIR/safe-failure.json" \
      --require-complete \
    > >(tee "$RESULT_DIR/sprint1-evaluation.log") 2>&1
  EVAL_RC=$?
else
  PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin \
    python scripts/build_sprint1_evaluation.py \
      --output-dir "$RESULT_DIR/sprint1-evaluation" \
      --require-complete \
    > >(tee "$RESULT_DIR/sprint1-evaluation.log") 2>&1
  EVAL_RC=$?
fi
set -e

SESSION_RC=0
FIRST_FAIL=""
if [[ "$B57_RC" -ne 0 ]]; then
  SESSION_RC="$B57_RC"
  FIRST_FAIL="B5/B7 grasp-recovery exit=$B57_RC"
elif [[ "$AUDIT_RC" -ne 0 ]]; then
  SESSION_RC="$AUDIT_RC"
  FIRST_FAIL="B5/B7 independent audit exit=$AUDIT_RC"
elif [[ "$A007_RC" -ne 0 ]]; then
  SESSION_RC="$A007_RC"
  FIRST_FAIL="S1-A-007 safe-failure probe exit=$A007_RC"
elif [[ "$A007_AUDIT_RC" -ne 0 ]]; then
  SESSION_RC="$A007_AUDIT_RC"
  FIRST_FAIL="S1-A-007 independent audit exit=$A007_AUDIT_RC"
elif [[ "$INTEGRATED_RC" -ne 0 ]]; then
  SESSION_RC="$INTEGRATED_RC"
  FIRST_FAIL="integrated grounding/pregrasp/recovery exit=$INTEGRATED_RC"
elif [[ "$INTEGRATED_AUDIT_RC" -ne 0 ]]; then
  SESSION_RC="$INTEGRATED_AUDIT_RC"
  FIRST_FAIL="integrated independent audit exit=$INTEGRATED_AUDIT_RC"
elif [[ "$INTEGRATED_RESOURCE_RC" -ne 0 ]]; then
  SESSION_RC="$INTEGRATED_RESOURCE_RC"
  FIRST_FAIL="integrated resource summary exit=$INTEGRATED_RESOURCE_RC"
elif [[ "$EVAL_RC" -ne 0 ]]; then
  SESSION_RC="$EVAL_RC"
  FIRST_FAIL="Sprint-1 evaluation assembly exit=$EVAL_RC"
fi

if [[ "$SESSION_RC" -eq 0 ]]; then
  RESULT_CLASSIFICATION="PASS"
else
  RESULT_CLASSIFICATION="FAIL"
fi

printf 'B57_RC=%s\nAUDIT_RC=%s\nA007_RC=%s\nA007_AUDIT_RC=%s\nINTEGRATED_RC=%s\nINTEGRATED_AUDIT_RC=%s\nINTEGRATED_RESOURCE_RC=%s\nEVAL_RC=%s\nSESSION_RC=%s\n' \
  "$B57_RC" "$AUDIT_RC" "$A007_RC" "$A007_AUDIT_RC" \
  "$INTEGRATED_RC" "$INTEGRATED_AUDIT_RC" "$INTEGRATED_RESOURCE_RC" \
  "$EVAL_RC" "$SESSION_RC" \
  > "$RESULT_DIR/SESSION_RESULT.txt"

EVIDENCE="$HOME/damh-evidence/sprint1_native_consolidated_$STAMP"
mkdir -p "$EVIDENCE"
cp -a "$RESULT_DIR/." "$EVIDENCE/"
if [[ "$SESSION_RC" -eq 0 ]]; then
  echo "PASS" > "$EVIDENCE/PASS"
else
  printf 'FAILED_STAGE=%s\nEXIT_CODE=%s\n' "$FIRST_FAIL" "$SESSION_RC" \
    > "$EVIDENCE/FAILED"
fi
(
  cd "$EVIDENCE"
  find . -type f ! -path './SHA256SUMS.txt' -print0 |
    sort -z |
    xargs -0 sha256sum > SHA256SUMS.txt
)

echo "=== Exporting consolidated Sprint-1 native evidence to DATA bridge ==="
set +e
DAMH_RESULT_CLASSIFICATION_OVERRIDE="$RESULT_CLASSIFICATION" \
DAMH_RESULT_SCOPE_OVERRIDE="Native B5/B7 + A-007 + integrated grounding/pregrasp/destination recovery" \
DAMH_FIRST_FAIL_OVERRIDE="$FIRST_FAIL" \
DAMH_EVIDENCE_DIR_OVERRIDE="$EVIDENCE" \
  bash scripts/linux/collect_to_bridge.sh
COLLECT_RC=$?
set -e
sync

echo
echo "B5/B7 probe exit: $B57_RC"
echo "B5/B7 independent audit exit: $AUDIT_RC"
echo "S1-A-007 probe exit: $A007_RC"
echo "S1-A-007 independent audit exit: $A007_AUDIT_RC"
echo "Integrated pregrasp probe exit: $INTEGRATED_RC"
echo "Integrated pregrasp independent audit exit: $INTEGRATED_AUDIT_RC"
echo "Integrated resource summary exit: $INTEGRATED_RESOURCE_RC"
echo "Sprint-1 evaluation assembly exit: $EVAL_RC"
echo "Session result: $RESULT_CLASSIFICATION (exit=$SESSION_RC)"
echo "Evidence export exit code: $COLLECT_RC"
ls -1dt "$BRIDGE/incoming/"* 2>/dev/null | head -n 1 || true

if [[ "$COLLECT_RC" -ne 0 ]]; then
  echo "Evidence export failed; do not reboot yet."
  exit "$COLLECT_RC"
fi
if [[ "$SESSION_RC" -ne 0 ]]; then
  echo "Native batch has a measured failure; return to Windows for audit without changing dependencies."
  exit "$SESSION_RC"
fi

echo "Consolidated Sprint-1 native batch PASS. Evidence is preserved."
echo "You may sync/reboot to Windows for audit."
