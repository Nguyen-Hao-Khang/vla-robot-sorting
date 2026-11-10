#!/usr/bin/env bash
set -u -o pipefail

BRIDGE_MOUNT="${DAMH_DATA_MOUNT:-/mnt/data}"
BRIDGE="${BRIDGE_MOUNT}/DAMH-Bridge"
PROJECT_ROOT="${DAMH_LINUX_PROJECT:-$HOME/projects/DAMH}"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run RUN_B5_COLLISION_IDENTITY_SMOKE.sh with sudo/root."
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

echo "=== One-seed B5 collision-identity diagnostic ==="
echo "session=$SESSION_NAME"
(
  cd "$SESSION_DIR" || exit 4
  sha256sum -c SHA256SUMS.txt
) || exit 4

echo "=== Fast-forwarding ext4 project from reviewed Windows bundle ==="
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

echo "=== Native/runtime provenance ==="
git rev-parse HEAD
git status --short --branch
nvidia-smi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="experiments/results/b5_collision_identity_smoke_$STAMP"
EVIDENCE="$HOME/damh-evidence/b5_collision_identity_smoke_$STAMP"
mkdir -p "$RESULT_DIR" "$EVIDENCE"

nvidia-smi   --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total   --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/gpu.csv" 2>&1 &
GPU_PID=$!

set +e
/usr/bin/time -v -o "$RESULT_DIR/time.txt"   timeout --signal=TERM --kill-after=30s 1200s   "$CONDA_EXE" run -n RoboTwin python -u scripts/linux/probe_panthera_grasp_recovery.py     --robotwin-root "$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin"     --project-root "$PROJECT_ROOT"     --output "$RESULT_DIR/grasp-recovery.json"     --seed-start 0     --runs 1   > >(tee "$RESULT_DIR/grasp-recovery.log") 2>&1
PROBE_RC=$?
set -e

kill "$GPU_PID" >/dev/null 2>&1 || true
wait "$GPU_PID" 2>/dev/null || true

python3 - "$RESULT_DIR/grasp-recovery.json" > "$RESULT_DIR/collision-identity-summary.txt" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("grasp-recovery.json missing")
    raise SystemExit(0)

data = json.loads(path.read_text(encoding="utf-8"))
print(f"schema={data.get('schema')}")
print(f"runs={data.get('runs')}")
print(f"B5_pass={data.get('B5_pass')}")
print(f"B7_pass={data.get('B7_pass')}")
records = data.get("records")
if not isinstance(records, list) or not records:
    print("records missing")
    raise SystemExit(0)

record = records[0] if isinstance(records[0], dict) else {}
print(f"seed={record.get('seed')}")
b5 = record.get("B5") if isinstance(record.get("B5"), dict) else {}
print(f"candidate_count={b5.get('candidate_count')}")
print(f"non_top_down_count={b5.get('non_top_down_count')}")
print(f"selected={b5.get('selected')}")

evaluations = b5.get("evaluations")
if not isinstance(evaluations, list):
    print("evaluations missing")
    raise SystemExit(0)

for index, evaluation in enumerate(evaluations):
    if not isinstance(evaluation, dict):
        continue
    candidate = evaluation.get("candidate")
    candidate_id = candidate.get("candidate_id") if isinstance(candidate, dict) else None
    collision = evaluation.get("collision_recheck")
    print()
    print(f"evaluation[{index}].candidate_id={candidate_id}")
    print(f"evaluation[{index}].planner={evaluation.get('pregrasp_selected_planner_id')}")
    print(f"evaluation[{index}].collision_free={evaluation.get('collision_free')}")
    if not isinstance(collision, dict):
        print(f"evaluation[{index}].collision_recheck=missing")
        continue
    collisions = collision.get("collisions")
    print(f"evaluation[{index}].provenance_schema={collision.get('provenance_schema')}")
    print(f"evaluation[{index}].scene_entities={collision.get('scene_entities')}")
    print(
        f"evaluation[{index}].collision_waypoints="
        f"{len(collisions) if isinstance(collisions, list) else 'missing'}"
    )
    if not isinstance(collisions, list) or not collisions:
        continue
    first = collisions[0] if isinstance(collisions[0], dict) else {}
    last = collisions[-1] if isinstance(collisions[-1], dict) else {}
    print(f"evaluation[{index}].first_collision={first}")
    if last is not first:
        print(f"evaluation[{index}].last_collision={last}")
PY

cp -a "$RESULT_DIR/." "$EVIDENCE/"
if [[ "$PROBE_RC" -eq 0 ]]; then
  echo "PASS" > "$EVIDENCE/PASS"
  RESULT_CLASSIFICATION="PASS"
  FIRST_FAIL=""
else
  printf 'PROBE_EXIT=%s\n' "$PROBE_RC" > "$EVIDENCE/FAILED"
  RESULT_CLASSIFICATION="FAIL"
  FIRST_FAIL="one-seed B5/B7 diagnostic exit=$PROBE_RC"
fi
(
  cd "$EVIDENCE"
  find . -type f ! -path './SHA256SUMS.txt' -print0 |
    sort -z |
    xargs -0 sha256sum > SHA256SUMS.txt
)

echo "=== Exporting immutable diagnostic evidence ==="
set +e
DAMH_RESULT_CLASSIFICATION_OVERRIDE="$RESULT_CLASSIFICATION" DAMH_RESULT_SCOPE_OVERRIDE="One-seed B5 collision-identity diagnostic" DAMH_FIRST_FAIL_OVERRIDE="$FIRST_FAIL" DAMH_EVIDENCE_DIR_OVERRIDE="$EVIDENCE"   bash scripts/linux/collect_to_bridge.sh
COLLECT_RC=$?
set -e
sync

echo
echo "Probe exit: $PROBE_RC"
echo "Evidence export exit: $COLLECT_RC"
echo "Summary:"
cat "$RESULT_DIR/collision-identity-summary.txt" || true
ls -1dt "$BRIDGE/incoming/"* 2>/dev/null | head -n 1 || true

if [[ "$COLLECT_RC" -ne 0 ]]; then
  echo "Evidence export failed; do not reboot yet."
  exit "$COLLECT_RC"
fi

echo "Diagnostic evidence is preserved. Return to Windows for collision-identity audit."
exit "$PROBE_RC"
