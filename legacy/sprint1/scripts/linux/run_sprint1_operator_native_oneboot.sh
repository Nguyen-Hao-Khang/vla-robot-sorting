#!/usr/bin/env bash
# One source-locked native campaign. Camera gate first, no-motion. Then one EN and one VI physical attempt.
# Simulator actor registry provides task identity; RGB-D detector/Gemini do not in this baseline.
set -Eeuo pipefail
PROJECT="$HOME/projects/DAMH"
ROBOTWIN="$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin"
BRIDGE="/mnt/data/DAMH-Bridge"
CONDA="$HOME/miniforge3/bin/conda"
if (( EUID == 0 )); then echo "ROOT_FORBIDDEN"; exit 3; fi
findmnt -M /mnt/data >/dev/null || { echo "DATA_MOUNT_REQUIRED"; exit 4; }
test -w "$BRIDGE/incoming" || { echo "BRIDGE_NOT_WRITABLE"; exit 5; }
test -x "$CONDA" || { echo "CONDA_MISSING"; exit 6; }
test -d "$ROBOTWIN" || { echo "ROBOTWIN_NOT_FOUND"; exit 7; }
SESSION="$(tr -d '\r\n' < "$BRIDGE/outgoing/LATEST.txt")"
[[ "$SESSION" =~ ^linux-session-[A-Za-z0-9-]+$ ]] || exit 8
RELEASE="$BRIDGE/outgoing/$SESSION"
(cd "$RELEASE" && sha256sum -c SHA256SUMS.txt) || exit 9
EXPECTED="$(python3 - "$RELEASE/session-manifest.json" <<'PY'
import json,sys
with open(sys.argv[1],encoding="utf-8-sig") as f: print(json.load(f)["git_head"])
PY
)"
test -n "$EXPECTED" || exit 10
echo "=== SOURCE-LOCKED BOOTSTRAP ==="
bash "$RELEASE/bootstrap_from_bridge.sh"
cd "$PROJECT"
ACTUAL="$(git rev-parse HEAD)"
test "$ACTUAL" = "$EXPECTED" || { echo "SOURCE_HEAD_MISMATCH"; exit 11; }
test -z "$(git status --porcelain)" || { echo "GIT_NOT_CLEAN"; exit 12; }
echo "SOURCE_HEAD=$ACTUAL"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BRIDGE/incoming/sprint1_operator_native_$STAMP"
mkdir -- "$OUT"
printf '%s\n' "$ACTUAL" > "$OUT/git-head.txt"
printf '%s\n' "$SESSION" > "$OUT/source-session.txt"
printf '%s\n' "NATIVE_OPERATOR_PENDING_INDEPENDENT_AUDIT" > "$OUT/classification.txt"
inventory() {
  local code=$?
  trap - EXIT
  set +e
  (cd "$OUT" && find . -type f ! -name SHA256SUMS.txt ! -name checksum-verification.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt && sha256sum -c SHA256SUMS.txt > checksum-verification.txt)
  sync
  echo "OPERATOR_NATIVE_EXPORT=$OUT"
  echo "EXIT_STATUS=$code"
  echo "NO_AUTOMATIC_RETRY=true"
  echo "STAY_IN_UBUNTU"
}
trap inventory EXIT
export PYTHONPATH="$PROJECT/src:$PROJECT"

# Stage A: isolated two-camera capture, with zero robot/physics motion.
CAPTURE_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CAPTURE_OUT="$BRIDGE/incoming/sprint1_operator_rgbd_only_$CAPTURE_STAMP"
mkdir -- "$CAPTURE_OUT"
set +e
"$CONDA" run --no-capture-output -n RoboTwin python -u \
  "$PROJECT/scripts/linux/probe_panthera_project_manipulation.py" \
  --project-root "$PROJECT" --robotwin-root "$ROBOTWIN" \
  --task move_pillbottle_pad --task-config move_pillbottle_pad_panthera_single \
  --seed 0 --output "$CAPTURE_OUT/observation-only.json" \
  --operator-capture-only --operator-rgbd-dir "$CAPTURE_OUT/rgbd" \
  > "$CAPTURE_OUT/capture.stdout.log" 2> "$CAPTURE_OUT/capture.stderr.log"
CAPTURE_RC=$?
set -e
printf '%s\n' "$CAPTURE_RC" > "$CAPTURE_OUT/capture.exit.txt"
if (( CAPTURE_RC != 0 )); then
  (cd "$CAPTURE_OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt)
  sync
  echo "CAPTURE_ONLY_FAILED=$CAPTURE_OUT; NO_PHYSICAL_ATTEMPTS_MADE"
  exit "$CAPTURE_RC"
fi
"$CONDA" run --no-capture-output -n RoboTwin python -B - "$CAPTURE_OUT/observation-only.json" <<'PY'
import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
o=p.get("operator_rgbd_observation",{})
assert o.get("camera_count")==2
assert o.get("controller_eligible_from_rgbd") is False
assert o.get("visual_object_identity_verified") is False
assert p.get("native_execution_attempted") is False
print("RGBD_ONLY_GATE=PASS_NON_CONTROL")
PY
(cd "$CAPTURE_OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.txt && sha256sum -c SHA256SUMS.txt > checksum-verification.txt)
sync
echo "RGBD_ONLY_EVIDENCE=$CAPTURE_OUT"
echo "CAMERA_ONLY_PASS; starting one EN and one VI physical attempt"

run_one() {
  local language="$1" seed="$2" command="$3"
  local label="$language-seed$seed"
  echo "=== NATIVE $label (one physical attempt) ==="
  local output="$OUT/manipulation-$label.json"
  local rgbd="$OUT/rgbd-$label"
  local rc=0
  # B5 candidate policy parity with the accepted S1-M strict campaign
  # (scripts/run_final_manipulation_session.py): seed1 evaluates all height
  # groups and prefers the same historical up75 grasp. Preference only; B5
  # still requires the candidate to be eligible, and every safety gate is unchanged.
  local -a b5_policy=()
  if [[ "$seed" == "1" ]]; then
    b5_policy=(--b5-evaluate-all-height-groups --preferred-candidate-id "pillbottle:cyl:side:0:s-1:up75")
  fi
  set +e
  "$CONDA" run --no-capture-output -n RoboTwin python -u \
    "$PROJECT/scripts/linux/probe_panthera_project_manipulation.py" \
    --project-root "$PROJECT" --robotwin-root "$ROBOTWIN" \
    --task move_pillbottle_pad --task-config move_pillbottle_pad_panthera_single \
    --seed "$seed" --output "$output" \
    --operator-language "$language" --operator-command "$command" \
    --operator-rgbd-dir "$rgbd" "${b5_policy[@]}" \
    > "$OUT/probe-$label.stdout.log" 2> "$OUT/probe-$label.stderr.log"
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$OUT/probe-$label.exit.txt"
  tail -n 18 "$OUT/probe-$label.stdout.log" || true
  if (( rc != 0 )); then echo "NATIVE_OPERATOR_FAILED label=$label STOP_NO_RETRY"; return "$rc"; fi
  local audit_rc=0
  set +e
  "$CONDA" run --no-capture-output -n RoboTwin python -B \
    "$PROJECT/scripts/audit_project_manipulation.py" "$output" \
    > "$OUT/audit-$label.stdout.log" 2> "$OUT/audit-$label.stderr.log"
  audit_rc=$?
  set -e
  printf '%s\n' "$audit_rc" > "$OUT/audit-$label.exit.txt"
  if (( audit_rc != 0 )); then echo "NATIVE_OPERATOR_AUDIT_FAILED label=$label"; return "$audit_rc"; fi
  "$CONDA" run --no-capture-output -n RoboTwin python -B - \
    "$output" "$rgbd" "$language" "$seed" <<'PY'
import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rgbd=Path(sys.argv[2]); lang=sys.argv[3]; seed=int(sys.argv[4])
op=p.get("operator_pipeline_result",{}); pre=p.get("operator_command_preflight",{}); obs=p.get("operator_rgbd_observation",{})
assert p.get("passed") is True and p.get("task_success_final") is True and p.get("seed")==seed
assert op.get("status")=="native_passed" and op.get("general_vla_end_to_end_accepted") is False
assert pre.get("status")=="native_pending" and pre.get("candidate_geometry_binding_passed") is True
assert pre.get("selected_native_candidate_id") in pre.get("candidate_ids",[])
assert obs.get("camera_count")==2 and obs.get("visual_object_identity_verified") is False
assert obs.get("sensor_timestamp_sync_verified") is False
assert len(p.get("stages",[]))==8 and all(x.get("passed") is True for x in p["stages"])
assert rgbd.is_dir()
print("OPERATOR_"+lang+"_NATIVE_TASK_AUDIT=PASS; VISUAL_IDENTITY=NOT_VERIFIED")
PY
  echo "OPERATOR_INDEPENDENT_AUDIT_PASS label=$label"
}
# Each language uses a fixed seed; failures stop, no retries.
run_one en 0 "Pick up the pill bottle and place it on the pad"
run_one vi 1 "Gắp chai thuốc và đặt lên tấm đệm"
printf '%s\n' "TWO_COMMAND_NATIVE_GROUND_TRUTH_BASELINE_PASS" > "$OUT/classification.txt"
echo "OPERATOR_NATIVE_FIXED_SCENE_BASELINE=PASS"
echo "NOT_GEMINI_NOT_VISION_DETECTION_NOT_FULL_VLA"
