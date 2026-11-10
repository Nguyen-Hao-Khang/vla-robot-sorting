#!/usr/bin/env bash
set -u -o pipefail

BRIDGE_MOUNT="${DAMH_DATA_MOUNT:-/mnt/data}"
BRIDGE="${BRIDGE_MOUNT}/DAMH-Bridge"
PROJECT_ROOT="${DAMH_LINUX_PROJECT:-$HOME/projects/DAMH}"
ROBOTWIN_ROOT="$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run RUN_SPRINT1_PROGRESSIVE_NATIVE.sh with sudo/root."
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
if [[ ! -f "$LATEST_FILE" ]]; then
  echo "Missing $LATEST_FILE"
  exit 3
fi
SESSION_NAME="$(tr -d '\r\n' < "$LATEST_FILE")"
SESSION_DIR="$BRIDGE/outgoing/$SESSION_NAME"

echo "=== Sprint-1 progressive native validation ==="
echo "session=$SESSION_NAME"
(
  cd "$SESSION_DIR" || exit 4
  sha256sum -c SHA256SUMS.txt
) || exit 4

echo "=== Fast-forward ext4 project from reviewed Windows bundle ==="
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

EXPECTED_HEAD="$(python3 - "$SESSION_DIR/session-manifest.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8-sig") as handle:
    print(json.load(handle)["git_head"])
PY
)"
ACTUAL_HEAD="$(git rev-parse HEAD)"
if [[ "$ACTUAL_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "HEAD mismatch after bootstrap: expected=$EXPECTED_HEAD actual=$ACTUAL_HEAD"
  exit 91
fi

echo "=== Runtime provenance ==="
git status --short --branch
git rev-parse HEAD
nvidia-smi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="experiments/results/sprint1_progressive_native_$STAMP"
EVIDENCE="$HOME/damh-evidence/sprint1_progressive_native_$STAMP"
mkdir -p "$RESULT_DIR" "$EVIDENCE"

nvidia-smi   --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total   --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/gpu.csv" 2>&1 &
GPU_PID=$!

SMOKE_RC=98
SMOKE_B5=0
SMOKE_B7=0
INTEGRATED_RC=98
INTEGRATED_AUDIT_RC=98
FULL_B57_RC=98
FULL_B57_AUDIT_RC=98
MANIP_RC=98
MANIP_AUDIT_RC=98
MANIP_PASS_COUNT=0
MANIP_RUN_COUNT=0
EVAL_RC=98

echo "=== Stage 1: one-seed B5/B7 world-sync smoke ==="
set +e
/usr/bin/time -v -o "$RESULT_DIR/b57-smoke-time.txt"   timeout --signal=TERM --kill-after=30s 1200s   "$CONDA_EXE" run -n RoboTwin python -u scripts/linux/probe_panthera_grasp_recovery.py     --robotwin-root "$ROBOTWIN_ROOT"     --project-root "$PROJECT_ROOT"     --output "$RESULT_DIR/b57-smoke.json"     --seed-start 0     --runs 1   > >(tee "$RESULT_DIR/b57-smoke.log") 2>&1
SMOKE_RC=$?
set -e

if [[ -f "$RESULT_DIR/b57-smoke.json" ]]; then
  read -r SMOKE_B5 SMOKE_B7 < <(
    python3 - "$RESULT_DIR/b57-smoke.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
print(int(data.get("B5_pass") is True), int(data.get("B7_pass") is True))
PY
  )
fi
echo "smoke_rc=$SMOKE_RC B5=$SMOKE_B5 B7=$SMOKE_B7"

if [[ -f "$RESULT_DIR/b57-smoke.json" ]]; then
  python3 - "$RESULT_DIR/b57-smoke.json" > "$RESULT_DIR/b57-smoke-summary.txt" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)

print(f"schema={data.get('schema')}")
print(f"B5_pass={data.get('B5_pass')} B7_pass={data.get('B7_pass')}")
records = data.get("records")
if not isinstance(records, list) or not records or not isinstance(records[0], dict):
    print("records=missing")
    raise SystemExit(0)

record = records[0]
print(f"seed={record.get('seed')}")
print(f"error={record.get('error')}")
print(f"mplib_world_initial={record.get('mplib_world_initial')}")
print(f"geometry_proxy={record.get('geometry_proxy')}")
print(f"gripper={record.get('gripper')}")
b5 = record.get("B5")
if not isinstance(b5, dict):
    print("B5=missing")
    raise SystemExit(0)

print(
    "B5 "
    f"candidate_count={b5.get('candidate_count')} "
    f"non_top_down_count={b5.get('non_top_down_count')} "
    f"evaluated_candidate_count={b5.get('evaluated_candidate_count')} "
    f"selected={b5.get('selected')}"
)
print(f"candidate_evaluation_policy={b5.get('candidate_evaluation_policy')}")
print(f"rejection_histogram={b5.get('rejection_histogram')}")
for index, evaluation in enumerate(b5.get("evaluations") or []):
    if not isinstance(evaluation, dict):
        continue
    candidate = evaluation.get("candidate")
    if not isinstance(candidate, dict):
        candidate = {}
    metadata = candidate.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    print(
        f"eval[{index}] "
        f"id={candidate.get('candidate_id')} "
        f"shape={metadata.get('shape')} "
        f"height={metadata.get('grasp_height_fraction')} "
        f"clearance={metadata.get('table_clearance_m')} "
        f"planner={evaluation.get('pregrasp_selected_planner_id')} "
        f"terminal={evaluation.get('terminal_reachability', {}).get('success') if isinstance(evaluation.get('terminal_reachability'), dict) else None} "
        f"plan={evaluation.get('pregrasp_plan_success')} "
        f"reachable={evaluation.get('reachable')} "
        f"collision_free={evaluation.get('collision_free')}"
    )

b7 = record.get("B7")
if isinstance(b7, dict):
    print(
        "B7 "
        f"passed={b7.get('passed')} "
        f"reason={b7.get('reason')} "
        f"change={b7.get('scene_change_kind')} "
        f"decision={b7.get('decision_action')} "
        f"stale_status={b7.get('stale_feedback_status')} "
        f"stale_steps={b7.get('stale_physics_steps')} "
        f"replacement_passed={b7.get('replacement_passed')} "
        f"replacement_ready={b7.get('replacement_segment_ready')} "
        f"replacement_collision_free={b7.get('replacement_collision_free')}"
    )
    print(
        "B7_mplib_world_after_perturbation="
        f"{b7.get('mplib_world_after_perturbation')}"
    )
PY
  cat "$RESULT_DIR/b57-smoke-summary.txt"
fi

echo "=== Stage 2: bounded integrated pregrasp if B5 smoke passed ==="
if [[ "$SMOKE_B5" -eq 1 ]]; then
  set +e
  /usr/bin/time -v -o "$RESULT_DIR/integrated-time.txt"     timeout --signal=TERM --kill-after=30s 1200s     "$CONDA_EXE" run -n RoboTwin python -u scripts/linux/probe_panthera_integrated_pregrasp.py       --robotwin-root "$ROBOTWIN_ROOT"       --project-root "$PROJECT_ROOT"       --output "$RESULT_DIR/integrated-pregrasp.json"       --seed 0     > >(tee "$RESULT_DIR/integrated-pregrasp.log") 2>&1
  INTEGRATED_RC=$?
  if [[ "$INTEGRATED_RC" -eq 0 ]]; then
    PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin       python scripts/audit_integrated_pregrasp.py "$RESULT_DIR/integrated-pregrasp.json"       > >(tee "$RESULT_DIR/integrated-pregrasp-audit.log") 2>&1
    INTEGRATED_AUDIT_RC=$?
  else
    INTEGRATED_AUDIT_RC=99
  fi
  set -e
else
  echo "Integrated pregrasp skipped because B5 smoke did not pass."
fi

echo "=== Stage 3: full B5/B7 20-run gate if one-seed B5+B7 passed ==="
if [[ "$SMOKE_B5" -eq 1 && "$SMOKE_B7" -eq 1 ]]; then
  set +e
  /usr/bin/time -v -o "$RESULT_DIR/b57-full-time.txt"     timeout --signal=TERM --kill-after=30s 3600s     "$CONDA_EXE" run -n RoboTwin python -u scripts/linux/probe_panthera_grasp_recovery.py       --robotwin-root "$ROBOTWIN_ROOT"       --project-root "$PROJECT_ROOT"       --output "$RESULT_DIR/grasp-recovery-20.json"       --seed-start 0       --runs 20     > >(tee "$RESULT_DIR/grasp-recovery-20.log") 2>&1
  FULL_B57_RC=$?
  if [[ "$FULL_B57_RC" -eq 0 ]]; then
    PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin       python scripts/audit_s1_3_b5_b7.py "$RESULT_DIR/grasp-recovery-20.json"       > >(tee "$RESULT_DIR/grasp-recovery-20-audit.log") 2>&1
    FULL_B57_AUDIT_RC=$?
  else
    FULL_B57_AUDIT_RC=99
  fi
  set -e
else
  echo "20-run B5/B7 skipped because one-seed B5+B7 did not both pass."
fi

echo "=== Stage 4: project-owned manipulation only after all prerequisite gates ==="
if [[ "$INTEGRATED_RC" -eq 0 && "$INTEGRATED_AUDIT_RC" -eq 0    && "$FULL_B57_RC" -eq 0 && "$FULL_B57_AUDIT_RC" -eq 0 ]]; then
  MANIP_RUN_COUNT=3
  MANIP_RC=0
  MANIP_AUDIT_RC=0
  for seed in 0 1 2; do
    echo "--- project manipulation seed=$seed ---"
    set +e
    /usr/bin/time -v -o "$RESULT_DIR/manipulation-seed-${seed}-time.txt"       timeout --signal=TERM --kill-after=30s 1800s       "$CONDA_EXE" run -n RoboTwin python -u scripts/linux/probe_panthera_project_manipulation.py         --robotwin-root "$ROBOTWIN_ROOT"         --project-root "$PROJECT_ROOT"         --output "$RESULT_DIR/manipulation-seed-${seed}.json"         --seed "$seed"       > >(tee "$RESULT_DIR/manipulation-seed-${seed}.log") 2>&1
    seed_rc=$?
    if [[ "$seed_rc" -eq 0 ]]; then
      PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin         python scripts/audit_project_manipulation.py           "$RESULT_DIR/manipulation-seed-${seed}.json"         > >(tee "$RESULT_DIR/manipulation-seed-${seed}-audit.log") 2>&1
      seed_audit_rc=$?
    else
      seed_audit_rc=99
    fi
    set -e

    printf 'seed=%s probe_rc=%s audit_rc=%s\n'       "$seed" "$seed_rc" "$seed_audit_rc"       >> "$RESULT_DIR/manipulation-summary.txt"

    if [[ "$seed_rc" -eq 0 && "$seed_audit_rc" -eq 0 ]]; then
      MANIP_PASS_COUNT=$((MANIP_PASS_COUNT + 1))
    else
      if [[ "$MANIP_RC" -eq 0 ]]; then
        MANIP_RC="$seed_rc"
        [[ "$MANIP_RC" -eq 0 ]] && MANIP_RC=99
      fi
      if [[ "$MANIP_AUDIT_RC" -eq 0 ]]; then
        MANIP_AUDIT_RC="$seed_audit_rc"
        [[ "$MANIP_AUDIT_RC" -eq 0 ]] && MANIP_AUDIT_RC=99
      fi
    fi
  done
else
  echo "Project manipulation skipped because prerequisite native gates are not all PASS."
fi

echo "=== Stage 5: fixed Sprint-1 evaluation assembly ==="
set +e
PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin   python scripts/build_sprint1_evaluation.py     --output-dir "$RESULT_DIR/sprint1-evaluation"     --require-complete   > >(tee "$RESULT_DIR/sprint1-evaluation.log") 2>&1
EVAL_RC=$?
set -e

kill "$GPU_PID" >/dev/null 2>&1 || true
wait "$GPU_PID" 2>/dev/null || true

SESSION_RC=0
FIRST_FAIL=""
if [[ "$SMOKE_B5" -ne 1 ]]; then
  SESSION_RC=20
  FIRST_FAIL="B5 one-seed smoke failed"
elif [[ "$SMOKE_B7" -ne 1 ]]; then
  SESSION_RC=21
  FIRST_FAIL="B7 one-seed smoke failed"
elif [[ "$INTEGRATED_RC" -ne 0 ]]; then
  SESSION_RC="$INTEGRATED_RC"
  FIRST_FAIL="integrated pregrasp probe exit=$INTEGRATED_RC"
elif [[ "$INTEGRATED_AUDIT_RC" -ne 0 ]]; then
  SESSION_RC="$INTEGRATED_AUDIT_RC"
  FIRST_FAIL="integrated pregrasp audit exit=$INTEGRATED_AUDIT_RC"
elif [[ "$FULL_B57_RC" -ne 0 ]]; then
  SESSION_RC="$FULL_B57_RC"
  FIRST_FAIL="B5/B7 20-run probe exit=$FULL_B57_RC"
elif [[ "$FULL_B57_AUDIT_RC" -ne 0 ]]; then
  SESSION_RC="$FULL_B57_AUDIT_RC"
  FIRST_FAIL="B5/B7 20-run audit exit=$FULL_B57_AUDIT_RC"
elif [[ "$MANIP_PASS_COUNT" -ne "$MANIP_RUN_COUNT" || "$MANIP_RUN_COUNT" -eq 0 ]]; then
  SESSION_RC=30
  FIRST_FAIL="project manipulation pass count=$MANIP_PASS_COUNT/$MANIP_RUN_COUNT"
elif [[ "$EVAL_RC" -ne 0 ]]; then
  SESSION_RC="$EVAL_RC"
  FIRST_FAIL="Sprint-1 evaluation assembly exit=$EVAL_RC"
fi

if [[ "$SESSION_RC" -eq 0 ]]; then
  RESULT_CLASSIFICATION="PASS"
else
  RESULT_CLASSIFICATION="FAIL"
fi

cat > "$RESULT_DIR/SESSION_RESULT.txt" <<EOF
SMOKE_RC=$SMOKE_RC
SMOKE_B5=$SMOKE_B5
SMOKE_B7=$SMOKE_B7
INTEGRATED_RC=$INTEGRATED_RC
INTEGRATED_AUDIT_RC=$INTEGRATED_AUDIT_RC
FULL_B57_RC=$FULL_B57_RC
FULL_B57_AUDIT_RC=$FULL_B57_AUDIT_RC
MANIP_RC=$MANIP_RC
MANIP_AUDIT_RC=$MANIP_AUDIT_RC
MANIP_PASS_COUNT=$MANIP_PASS_COUNT
MANIP_RUN_COUNT=$MANIP_RUN_COUNT
EVAL_RC=$EVAL_RC
SESSION_RC=$SESSION_RC
RESULT_CLASSIFICATION=$RESULT_CLASSIFICATION
FIRST_FAIL=$FIRST_FAIL
EOF

cp -a "$RESULT_DIR/." "$EVIDENCE/"
if [[ "$SESSION_RC" -eq 0 ]]; then
  echo "PASS" > "$EVIDENCE/PASS"
else
  printf 'FAILED_STAGE=%s\nEXIT_CODE=%s\n' "$FIRST_FAIL" "$SESSION_RC"     > "$EVIDENCE/FAILED"
fi
(
  cd "$EVIDENCE"
  find . -type f ! -path './SHA256SUMS.txt' -print0 |
    sort -z |
    xargs -0 sha256sum > SHA256SUMS.txt
)

echo "=== Exporting progressive native evidence ==="
set +e
DAMH_RESULT_CLASSIFICATION_OVERRIDE="$RESULT_CLASSIFICATION" DAMH_RESULT_SCOPE_OVERRIDE="Progressive B5/B7 + integrated pregrasp + project-owned manipulation" DAMH_FIRST_FAIL_OVERRIDE="$FIRST_FAIL" DAMH_EVIDENCE_DIR_OVERRIDE="$EVIDENCE"   bash scripts/linux/collect_to_bridge.sh
COLLECT_RC=$?
set -e
sync

echo
cat "$RESULT_DIR/SESSION_RESULT.txt"
echo "Evidence export exit: $COLLECT_RC"
ls -1dt "$BRIDGE/incoming/"* 2>/dev/null | head -n 1 || true

if [[ "$COLLECT_RC" -ne 0 ]]; then
  echo "Evidence export failed; do not reboot yet."
  exit "$COLLECT_RC"
fi

echo "Evidence preserved. Return to Windows for audit; do not patch Ubuntu source/dependencies."
exit "$SESSION_RC"
