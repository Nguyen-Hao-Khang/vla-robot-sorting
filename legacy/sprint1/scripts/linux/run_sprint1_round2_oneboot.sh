#!/usr/bin/env bash
# Single-boot Round2: source lock, plan-only gate, conditional physical diagnostics.
# No retry.  No fixture promotion.  Exit/failure evidence is preserved.
set -Eeuo pipefail
BRIDGE=/mnt/data/DAMH-Bridge
PROJECT="$HOME/projects/DAMH"
CONDA="$HOME/miniforge3/bin/conda"
if (( EUID == 0 )); then echo "Refuse root"; exit 3; fi
if ! findmnt -rn /mnt/data >/dev/null; then
  DEV="$(blkid -L DATA || true)"
  [[ -n "$DEV" ]] || { echo "DATA partition not detected"; exit 4; }
  sudo mkdir -p /mnt/data
  sudo mount -t ntfs3 -o "uid=$(id -u),gid=$(id -g),umask=022" "$DEV" /mnt/data
fi
[[ -w "$BRIDGE/incoming" ]] || { echo "Bridge not writable"; exit 5; }
[[ -x "$CONDA" ]] || { echo "Conda binary missing"; exit 6; }
SESSION="$(tr -d '\r\n' < "$BRIDGE/outgoing/LATEST.txt")"
[[ "$SESSION" =~ ^linux-session-[A-Za-z0-9-]+$ ]] || { echo "Invalid session"; exit 7; }
SESSION_DIR="$BRIDGE/outgoing/$SESSION"
SOURCE_LOCK="$SESSION_DIR/manipulation-source-lock.json"
[[ -f "$SOURCE_LOCK" ]] || { echo "Source lock missing"; exit 8; }
(cd "$SESSION_DIR" && sha256sum -c SHA256SUMS.txt) || exit 9
ROUND="$BRIDGE/incoming/sprint1_round2_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -- "$ROUND"
trap 'sync' EXIT
printf '%s\n' "$SESSION" > "$ROUND/session.txt"
printf '%s\n' "ROUND2_DIAGNOSTICS_NOT_ACCEPTANCE" > "$ROUND/classification.txt"
inventory() {
  (cd "$ROUND" && find . -type f ! -path './SHA256SUMS.txt' ! -path './checksum-verification.txt' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt)
  (cd "$ROUND" && sha256sum -c SHA256SUMS.txt > checksum-verification.txt)
  sync
}
stop_here() {
  printf '%s\n' "$1" > "$ROUND/result.txt"
  inventory
  echo "ROUND2_EVIDENCE=$ROUND"
  echo "STOP: $1; do not retry; STAY IN UBUNTU for read-only evidence review"
  exit "$2"
}
echo "=== 1: bootstrap ==="
set +e
bash "$SESSION_DIR/bootstrap_from_bridge.sh" > "$ROUND/bootstrap.stdout.log" 2> "$ROUND/bootstrap.stderr.log"
BOOT_RC=$?
set -e
printf '%s\n' "$BOOT_RC" > "$ROUND/bootstrap.exit.txt"
if (( BOOT_RC != 0 )); then stop_here "BOOTSTRAP_BLOCKED" 21; fi
EXPECTED="$(python3 - "$SESSION_DIR/session-manifest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8-sig"))["git_head"])
PY
)"
ACTUAL="$(git -C "$PROJECT" rev-parse HEAD)"
printf '%s\n' "$ACTUAL" > "$ROUND/git-head.txt"
git -C "$PROJECT" status --short --branch > "$ROUND/git-status.txt"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then stop_here "SOURCE_MISMATCH" 22; fi
echo "SOURCE_HEAD=$ACTUAL"
echo "=== 2: source lock preboot ==="
set +e
"$CONDA" run --no-capture-output -n RoboTwin python \
  "$PROJECT/scripts/check_final_manipulation_preboot.py" \
  --project-root "$PROJECT" --source-lock "$SOURCE_LOCK" --dry-run \
  > "$ROUND/preboot.stdout.log" 2> "$ROUND/preboot.stderr.log"
PRE_RC=$?
set -e
printf '%s\n' "$PRE_RC" > "$ROUND/preboot.exit.txt"
if (( PRE_RC != 0 )); then stop_here "PREBOOT_BLOCKED" 23; fi

echo "=== 2b: independent native environment diagnostic; inspect ALL gates ==="
ROBOTWIN="$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin"
git -C "$ROBOTWIN" rev-parse HEAD \
  > "$ROUND/robotwin-head.stdout.txt" 2> "$ROUND/robotwin-head.stderr.txt" || true
git -C "$ROBOTWIN" status --porcelain=v1 --untracked-files=all \
  > "$ROUND/robotwin-git-status.stdout.txt" \
  2> "$ROUND/robotwin-git-status.stderr.txt" || true
git -C "$ROBOTWIN" diff --name-only HEAD -- \
  > "$ROUND/robotwin-tracked-diff.stdout.txt" \
  2> "$ROUND/robotwin-tracked-diff.stderr.txt" || true
set +e
"$CONDA" run --no-capture-output -n RoboTwin python \
  "$PROJECT/scripts/check_native_manipulation_dependencies.py" \
  --robotwin-root "$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin" --diagnose \
  > "$ROUND/dependency-diagnostic.stdout.log" \
  2> "$ROUND/dependency-diagnostic.stderr.log"
DEPENDENCY_RC=$?
set -e
printf '%s\n' "$DEPENDENCY_RC" > "$ROUND/dependency-diagnostic.exit.txt"
tail -n 80 "$ROUND/dependency-diagnostic.stdout.log" || true
tail -n 30 "$ROUND/dependency-diagnostic.stderr.log" || true
if (( DEPENDENCY_RC != 0 )); then
  stop_here "DEPENDENCY_DIAGNOSTIC_BLOCKED_NO_NATIVE_MOTION" 26
fi

echo "=== 3: native plan-only for seed0,1,2 ==="
PLAN_DIR="$ROUND/native-plan"
set +e
bash "$PROJECT/scripts/linux/run_final_manipulation_recovery.sh" \
  --source-lock "$SOURCE_LOCK" --mode plan-only --output-dir "$PLAN_DIR" \
  > "$ROUND/plan.stdout.log" 2> "$ROUND/plan.stderr.log"
PLAN_RC=$?
set -e
printf '%s\n' "$PLAN_RC" > "$ROUND/plan.exit.txt"
tail -n 15 "$ROUND/plan.stdout.log" || true
tail -n 20 "$ROUND/plan.stderr.log" || true
if (( PLAN_RC != 0 )); then stop_here "PLAN_ONLY_BLOCKED" 24; fi
echo "=== 4: independent plan-only provenance + trace audit ==="
set +e
"$CONDA" run --no-capture-output -n RoboTwin python -c \
  'import json,sys;sys.path.insert(0,sys.argv[3]);from vla_robot_sorting.planning.manipulation_recovery import plan_only_readiness;r=plan_only_readiness(sys.argv[1],sys.argv[2]);print(json.dumps(r,sort_keys=True));sys.exit(0 if r["ready"] else 2)' \
  "$PLAN_DIR" "$ACTUAL" "$PROJECT/src" \
  > "$ROUND/plan-gate.stdout.log" 2> "$ROUND/plan-gate.stderr.log"
GATE_RC=$?
set -e
printf '%s\n' "$GATE_RC" > "$ROUND/plan-gate.exit.txt"
cat "$ROUND/plan-gate.stdout.log"
if (( GATE_RC != 0 )); then stop_here "PLAN_PROVENANCE_BLOCKED_NO_PHYSICAL" 25; fi
echo "=== 5: bounded physical diagnostics plus opt-in full native telemetry ==="
PHYSICAL_DIR="$ROUND/native-physical"
set +e
bash "$PROJECT/scripts/linux/run_final_manipulation_recovery.sh" \
  --source-lock "$SOURCE_LOCK" --mode physical-diagnostic \
  --plan-readiness-dir "$PLAN_DIR" --output-dir "$PHYSICAL_DIR" \
  --additional-recovery-diagnostics \
  > "$ROUND/physical.stdout.log" 2> "$ROUND/physical.stderr.log"
PHYS_RC=$?
set -e
printf '%s\n' "$PHYS_RC" > "$ROUND/physical.exit.txt"
tail -n 30 "$ROUND/physical.stdout.log" || true
tail -n 30 "$ROUND/physical.stderr.log" || true

echo "=== 6: audit three physical seeds BEFORE separately opening strict ==="
PHYSICAL_GATE_RC=99
if (( PHYS_RC == 0 )); then
  set +e
  "$CONDA" run --no-capture-output -n RoboTwin python -c \
    'import json,sys;sys.path.insert(0,sys.argv[3]);from vla_robot_sorting.planning.manipulation_recovery import acceptance_readiness;r=acceptance_readiness(sys.argv[1],sys.argv[2]);print(json.dumps(r,sort_keys=True));sys.exit(0 if r["ready"] else 2)' \
    "$PHYSICAL_DIR" "$ACTUAL" "$PROJECT/src" \
    > "$ROUND/physical-gate.stdout.log" 2> "$ROUND/physical-gate.stderr.log"
  PHYSICAL_GATE_RC=$?
  set -e
  cat "$ROUND/physical-gate.stdout.log"
fi
printf '%s\n' "$PHYSICAL_GATE_RC" > "$ROUND/physical-gate.exit.txt"

STRICT_RC=99
STRICT_DIR="$ROUND/native-strict"
if (( PHYS_RC == 0 && PHYSICAL_GATE_RC == 0 )); then
  echo "=== 7: strict campaign allowed by independent 3/3 physical-readiness gate ==="
  set +e
  bash "$PROJECT/scripts/linux/run_final_manipulation_recovery.sh" \
    --source-lock "$SOURCE_LOCK" --mode strict-acceptance \
    --readiness-dir "$PHYSICAL_DIR" --output-dir "$STRICT_DIR" \
    > "$ROUND/strict.stdout.log" 2> "$ROUND/strict.stderr.log"
  STRICT_RC=$?
  set -e
  tail -n 30 "$ROUND/strict.stdout.log" || true
  tail -n 30 "$ROUND/strict.stderr.log" || true
  if (( STRICT_RC == 0 )); then
    echo "=== 8: independent strict audit from source-bound campaign evidence ==="
    set +e
    "$CONDA" run --no-capture-output -n RoboTwin python \
      "$PROJECT/scripts/check_final_manipulation_preboot.py" \
      --project-root "$PROJECT" --source-lock "$SOURCE_LOCK" \
      --campaign-dir "$STRICT_DIR" \
      > "$ROUND/strict-audit.stdout.log" 2> "$ROUND/strict-audit.stderr.log"
    STRICT_RC=$?
    set -e
  fi
else
  echo "Strict execution BLOCKED: no independently verified 3/3 physical diagnostics."
fi
printf '%s\n' "$STRICT_RC" > "$ROUND/strict-audit.exit.txt"
if (( PHYS_RC == 0 && STRICT_RC == 0 )); then
  printf '%s\n' "STRICT_AUDIT_CANDIDATE_PASS_REVIEW_ON_WINDOWS" > "$ROUND/result.txt"
else
  printf '%s\n' "NATIVE_NOT_ACCEPTED_REVIEW_ON_WINDOWS" > "$ROUND/result.txt"
fi
inventory
echo "ROUND2_EVIDENCE=$ROUND"
echo "PLAN_RC=$PLAN_RC PLAN_GATE=$GATE_RC PHYSICAL_RC=$PHYS_RC STRICT_AUDIT_RC=$STRICT_RC"
cat "$ROUND/result.txt"
echo "No automatic rerun. Preserve evidence; STAY IN UBUNTU for read-only review."
if (( PHYS_RC != 0 || STRICT_RC != 0 )); then
  exit 40  # evidence exported, native Sprint 1 strict acceptance NOT achieved
fi
exit 0  # independently audited strict campaign candidate; Windows review still required
