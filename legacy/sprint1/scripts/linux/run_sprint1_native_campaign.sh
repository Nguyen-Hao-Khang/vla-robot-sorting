#!/usr/bin/env bash
set -u -o pipefail

BRIDGE_MOUNT="${DAMH_DATA_MOUNT:-/mnt/data}"
BRIDGE="${BRIDGE_MOUNT}/DAMH-Bridge"
PROJECT_ROOT="${DAMH_LINUX_PROJECT:-$HOME/projects/DAMH}"
ROBOTWIN_ROOT="$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run the native campaign with sudo/root."
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

SESSION_NAME="$(tr -d '\r\n' < "$BRIDGE/outgoing/LATEST.txt")"
SESSION_DIR="$BRIDGE/outgoing/$SESSION_NAME"
(
  cd "$SESSION_DIR" || exit 4
  sha256sum -c SHA256SUMS.txt
) || exit 4

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

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="experiments/results/sprint1_native_campaign_$STAMP"
EVIDENCE="$HOME/damh-evidence/sprint1_native_campaign_$STAMP"
mkdir -p "$RESULT_DIR" "$EVIDENCE"

git status --short --branch > "$RESULT_DIR/git-status-before.txt"
git rev-parse HEAD > "$RESULT_DIR/git-head.txt"
nvidia-smi > "$RESULT_DIR/nvidia-smi-start.txt"
nvidia-smi   --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total   --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/gpu.csv" 2>&1 &
GPU_PID=$!

run_native() {
  local label="$1"
  local timeout_s="$2"
  shift 2
  echo
  echo "=== $label ==="
  set +e
  /usr/bin/time -v -o "$RESULT_DIR/${label}-time.txt"     timeout --signal=TERM --kill-after=30s "${timeout_s}s"     "$@" > >(tee "$RESULT_DIR/${label}.log") 2>&1
  local rc=$?
  printf '%s=%s\n' "$label" "$rc" >> "$RESULT_DIR/stage-return-codes.txt"
  echo "$label rc=$rc"
  return "$rc"
}

audit_manip() {
  local seed="$1"
  local json="$RESULT_DIR/manipulation-seed-${seed}.json"
  local log="$RESULT_DIR/manipulation-seed-${seed}-audit.log"
  set +e
  PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin     python scripts/audit_project_manipulation.py "$json"     > >(tee "$log") 2>&1
  local rc=$?
  printf 'manipulation-seed-%s-audit=%s\n' "$seed" "$rc"     >> "$RESULT_DIR/stage-return-codes.txt"
  return "$rc"
}

SYNC_PASS_COUNT=0
SYNC0_RC=98
SYNC1_RC=98
SMOKE3_RC=98
SMOKE3_B5_COUNT=0
SMOKE3_B7_COUNT=0
INTEGRATED0_RC=98
INTEGRATED0_AUDIT_RC=98
INTEGRATED1_RC=98
INTEGRATED1_AUDIT_RC=98
MANIP0_RC=98
MANIP0_AUDIT_RC=98
FULL_B57_RC=98
FULL_B57_AUDIT_RC=98
MANIP_ACCEPT_PASS_COUNT=0
EVAL_RC=98

echo "=== Native campaign provenance ==="
echo "session=$SESSION_NAME"
echo "head=$ACTUAL_HEAD"
nvidia-smi

echo
echo "### Stage A - MPlib scaled-target preflight (seeds 0,1)"
for seed in 0 1; do
  label="target-sync-seed-${seed}"
  run_native "$label" 600     "$CONDA_EXE" run -n RoboTwin python -u     scripts/linux/probe_panthera_mplib_target_sync.py     --robotwin-root "$ROBOTWIN_ROOT"     --project-root "$PROJECT_ROOT"     --output "$RESULT_DIR/${label}.json"     --seed "$seed"
  rc=$?
  if [[ "$seed" -eq 0 ]]; then SYNC0_RC=$rc; else SYNC1_RC=$rc; fi
  if [[ "$rc" -eq 0 ]]; then
    SYNC_PASS_COUNT=$((SYNC_PASS_COUNT + 1))
  fi
done

echo
echo "### Stage B - 3-seed B5/B7 diagnostic matrix"
if [[ "$SYNC_PASS_COUNT" -ge 1 ]]; then
  run_native "b57-smoke-3seed" 1800     "$CONDA_EXE" run -n RoboTwin python -u     scripts/linux/probe_panthera_grasp_recovery.py     --robotwin-root "$ROBOTWIN_ROOT"     --project-root "$PROJECT_ROOT"     --output "$RESULT_DIR/b57-smoke-3seed.json"     --seed-start 0     --runs 3
  SMOKE3_RC=$?
  if [[ -f "$RESULT_DIR/b57-smoke-3seed.json" ]]; then
    read -r SMOKE3_B5_COUNT SMOKE3_B7_COUNT < <(
      python3 - "$RESULT_DIR/b57-smoke-3seed.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
records = data.get("records") or []
b5 = sum(isinstance(r, dict) and (r.get("B5") or {}).get("passed") is True for r in records)
b7 = sum(isinstance(r, dict) and (r.get("B7") or {}).get("passed") is True for r in records)
print(b5, b7)
PY
    )
  fi
else
  echo "B5/B7 diagnostic skipped: both target-sync preflights failed."
fi

echo
echo "### Stage C - integrated pregrasp diagnostics (seeds 0,1)"
if [[ "$SYNC_PASS_COUNT" -ge 1 ]]; then
  for seed in 0 1; do
    label="integrated-seed-${seed}"
    run_native "$label" 1500       "$CONDA_EXE" run -n RoboTwin python -u       scripts/linux/probe_panthera_integrated_pregrasp.py       --robotwin-root "$ROBOTWIN_ROOT"       --project-root "$PROJECT_ROOT"       --output "$RESULT_DIR/${label}.json"       --seed "$seed"
    rc=$?
    audit_rc=99
    if [[ "$rc" -eq 0 ]]; then
      set +e
      PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin         python scripts/audit_integrated_pregrasp.py "$RESULT_DIR/${label}.json"         > >(tee "$RESULT_DIR/${label}-audit.log") 2>&1
      audit_rc=$?
    fi
    if [[ "$seed" -eq 0 ]]; then
      INTEGRATED0_RC=$rc
      INTEGRATED0_AUDIT_RC=$audit_rc
    else
      INTEGRATED1_RC=$rc
      INTEGRATED1_AUDIT_RC=$audit_rc
    fi
    printf 'integrated-seed-%s-audit=%s\n' "$seed" "$audit_rc"       >> "$RESULT_DIR/stage-return-codes.txt"
  done
else
  echo "Integrated diagnostics skipped: target-sync preflight unavailable."
fi

echo
echo "### Stage D - project-owned manipulation diagnostic seed 0"
if [[ "$INTEGRATED0_RC" -eq 0 && "$INTEGRATED0_AUDIT_RC" -eq 0 ]]; then
  run_native "manipulation-seed-0" 2400     "$CONDA_EXE" run -n RoboTwin python -u     scripts/linux/probe_panthera_project_manipulation.py     --robotwin-root "$ROBOTWIN_ROOT"     --project-root "$PROJECT_ROOT"     --output "$RESULT_DIR/manipulation-seed-0.json"     --seed 0
  MANIP0_RC=$?
  if [[ "$MANIP0_RC" -eq 0 ]]; then
    audit_manip 0
    MANIP0_AUDIT_RC=$?
  else
    MANIP0_AUDIT_RC=99
  fi
else
  echo "Manipulation diagnostic skipped: integrated seed 0 is not audited PASS."
fi

echo
echo "### Stage E - full B5/B7 20-run acceptance"
if [[ "$SMOKE3_B5_COUNT" -eq 3 && "$SMOKE3_B7_COUNT" -eq 3 ]]; then
  run_native "b57-full-20" 5400     "$CONDA_EXE" run -n RoboTwin python -u     scripts/linux/probe_panthera_grasp_recovery.py     --robotwin-root "$ROBOTWIN_ROOT"     --project-root "$PROJECT_ROOT"     --output "$RESULT_DIR/grasp-recovery-20.json"     --seed-start 0     --runs 20
  FULL_B57_RC=$?
  if [[ "$FULL_B57_RC" -eq 0 ]]; then
    set +e
    PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin       python scripts/audit_s1_3_b5_b7.py "$RESULT_DIR/grasp-recovery-20.json"       > >(tee "$RESULT_DIR/grasp-recovery-20-audit.log") 2>&1
    FULL_B57_AUDIT_RC=$?
  else
    FULL_B57_AUDIT_RC=99
  fi
else
  echo "20-run acceptance skipped: 3-seed B5/B7 diagnostic is not 3/3 + 3/3."
fi

echo
echo "### Stage F - manipulation acceptance seeds 0,1,2"
if [[ "$FULL_B57_RC" -eq 0 && "$FULL_B57_AUDIT_RC" -eq 0    && "$INTEGRATED0_RC" -eq 0 && "$INTEGRATED0_AUDIT_RC" -eq 0 ]]; then
  if [[ "$MANIP0_RC" -eq 0 && "$MANIP0_AUDIT_RC" -eq 0 ]]; then
    MANIP_ACCEPT_PASS_COUNT=1
  fi
  for seed in 1 2; do
    label="manipulation-seed-${seed}"
    run_native "$label" 2400       "$CONDA_EXE" run -n RoboTwin python -u       scripts/linux/probe_panthera_project_manipulation.py       --robotwin-root "$ROBOTWIN_ROOT"       --project-root "$PROJECT_ROOT"       --output "$RESULT_DIR/${label}.json"       --seed "$seed"
    rc=$?
    audit_rc=99
    if [[ "$rc" -eq 0 ]]; then
      audit_manip "$seed"
      audit_rc=$?
    fi
    if [[ "$rc" -eq 0 && "$audit_rc" -eq 0 ]]; then
      MANIP_ACCEPT_PASS_COUNT=$((MANIP_ACCEPT_PASS_COUNT + 1))
    fi
  done
else
  echo "Manipulation acceptance skipped; seed-0 diagnostic evidence is still retained."
fi

echo
echo "### Stage G - fixed Sprint-1 matrix assembly"
set +e
PYTHONPATH="$PROJECT_ROOT/src" "$CONDA_EXE" run -n RoboTwin   python scripts/build_sprint1_evaluation.py   --output-dir "$RESULT_DIR/sprint1-evaluation"   --require-complete   > >(tee "$RESULT_DIR/sprint1-evaluation.log") 2>&1
EVAL_RC=$?

kill "$GPU_PID" >/dev/null 2>&1 || true
wait "$GPU_PID" 2>/dev/null || true

python3 - "$RESULT_DIR" <<'PY'
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
gpu = root / "gpu.csv"
rows = []
if gpu.is_file():
    for row in csv.reader(gpu.read_text(encoding="utf-8", errors="replace").splitlines()):
        if len(row) >= 6:
            try:
                rows.append({
                    "gpu_util_percent": float(row[2].strip()),
                    "memory_util_percent": float(row[3].strip()),
                    "memory_used_mb": float(row[4].strip()),
                    "memory_total_mb": float(row[5].strip()),
                })
            except ValueError:
                pass
summary = {
    "samples": len(rows),
    "peak_gpu_util_percent": max((r["gpu_util_percent"] for r in rows), default=None),
    "peak_memory_util_percent": max((r["memory_util_percent"] for r in rows), default=None),
    "peak_memory_used_mb": max((r["memory_used_mb"] for r in rows), default=None),
    "memory_total_mb": max((r["memory_total_mb"] for r in rows), default=None),
}
(root / "resource-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
)
PY

FIRST_BLOCKER=""
if [[ "$SYNC_PASS_COUNT" -lt 2 ]]; then
  FIRST_BLOCKER="MPlib scaled-target sync preflight"
elif [[ "$SMOKE3_B5_COUNT" -lt 3 ]]; then
  FIRST_BLOCKER="B5 3-seed diagnostic"
elif [[ "$SMOKE3_B7_COUNT" -lt 3 ]]; then
  FIRST_BLOCKER="B7 3-seed diagnostic"
elif [[ "$INTEGRATED0_RC" -ne 0 || "$INTEGRATED0_AUDIT_RC" -ne 0 ]]; then
  FIRST_BLOCKER="integrated pregrasp seed 0"
elif [[ "$FULL_B57_RC" -ne 0 || "$FULL_B57_AUDIT_RC" -ne 0 ]]; then
  FIRST_BLOCKER="B5/B7 20-run acceptance"
elif [[ "$MANIP_ACCEPT_PASS_COUNT" -lt 3 ]]; then
  FIRST_BLOCKER="project-owned manipulation acceptance"
elif [[ "$EVAL_RC" -ne 0 ]]; then
  FIRST_BLOCKER="fixed Sprint-1 matrix assembly"
fi

CAMPAIGN_ACCEPTED=0
if [[ -z "$FIRST_BLOCKER" ]]; then
  CAMPAIGN_ACCEPTED=1
fi

cat > "$RESULT_DIR/CAMPAIGN_RESULT.txt" <<EOF
SYNC0_RC=$SYNC0_RC
SYNC1_RC=$SYNC1_RC
SYNC_PASS_COUNT=$SYNC_PASS_COUNT
SMOKE3_RC=$SMOKE3_RC
SMOKE3_B5_COUNT=$SMOKE3_B5_COUNT
SMOKE3_B7_COUNT=$SMOKE3_B7_COUNT
INTEGRATED0_RC=$INTEGRATED0_RC
INTEGRATED0_AUDIT_RC=$INTEGRATED0_AUDIT_RC
INTEGRATED1_RC=$INTEGRATED1_RC
INTEGRATED1_AUDIT_RC=$INTEGRATED1_AUDIT_RC
MANIP0_RC=$MANIP0_RC
MANIP0_AUDIT_RC=$MANIP0_AUDIT_RC
FULL_B57_RC=$FULL_B57_RC
FULL_B57_AUDIT_RC=$FULL_B57_AUDIT_RC
MANIP_ACCEPT_PASS_COUNT=$MANIP_ACCEPT_PASS_COUNT
EVAL_RC=$EVAL_RC
CAMPAIGN_ACCEPTED=$CAMPAIGN_ACCEPTED
FIRST_BLOCKER=$FIRST_BLOCKER
EOF

git status --short --branch > "$RESULT_DIR/git-status-after.txt"
git diff > "$RESULT_DIR/native-working-tree.patch"
git diff --cached > "$RESULT_DIR/native-staged.patch"

cp -a "$RESULT_DIR/." "$EVIDENCE/"
if [[ "$CAMPAIGN_ACCEPTED" -eq 1 ]]; then
  echo "PASS" > "$EVIDENCE/PASS"
  RESULT_CLASSIFICATION="PASS"
  CAMPAIGN_RC=0
else
  printf 'FIRST_BLOCKER=%s\n' "$FIRST_BLOCKER" > "$EVIDENCE/PARTIAL"
  RESULT_CLASSIFICATION="PARTIAL"
  CAMPAIGN_RC=40
fi

(
  cd "$EVIDENCE"
  find . -type f ! -path './SHA256SUMS.txt' -print0     | sort -z     | xargs -0 sha256sum > SHA256SUMS.txt
)

echo
echo "=== Exporting complete native campaign evidence ==="
set +e
DAMH_RESULT_CLASSIFICATION_OVERRIDE="$RESULT_CLASSIFICATION" DAMH_RESULT_SCOPE_OVERRIDE="Sprint1 native campaign: target sync, B5/B7, integrated, manipulation, matrix" DAMH_FIRST_FAIL_OVERRIDE="$FIRST_BLOCKER" DAMH_EVIDENCE_DIR_OVERRIDE="$EVIDENCE"   bash scripts/linux/collect_to_bridge.sh
COLLECT_RC=$?
sync

echo
cat "$RESULT_DIR/CAMPAIGN_RESULT.txt"
echo "Evidence export exit: $COLLECT_RC"
ls -1dt "$BRIDGE/incoming/"* 2>/dev/null | head -n 1 || true

if [[ "$COLLECT_RC" -ne 0 ]]; then
  echo "Evidence export failed. Preserve/export evidence before reboot."
  exit "$COLLECT_RC"
fi

if [[ "$CAMPAIGN_ACCEPTED" -eq 1 ]]; then
  echo "Campaign acceptance PASS."
else
  echo "Campaign completed with partial evidence; downstream diagnostics were preserved."
fi
exit "$CAMPAIGN_RC"
