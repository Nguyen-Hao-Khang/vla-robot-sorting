#!/usr/bin/env bash
# Two bounded native IK studies in one boot; diagnostic-only, never accepted.
set -euo pipefail

BRIDGE="/mnt/data/DAMH-Bridge"
PROJECT="$HOME/projects/DAMH"
ROBOTWIN="$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin"
if (( EUID == 0 )); then echo "Run as normal Ubuntu user" >&2; exit 3; fi
findmnt -rn /mnt/data >/dev/null || { echo "DATA not mounted" >&2; exit 4; }
[[ -w "$BRIDGE/incoming" ]] || { echo "Bridge incoming not writable" >&2; exit 5; }
[[ -d "$PROJECT/.git" && -d "$ROBOTWIN" ]] || {
  echo "Native source / simulator missing" >&2; exit 6;
}
ROBOTWIN_PIN="2aee31ad75cb7b0cd779540441c6025853a5b60d"
[[ "$(git -C "$ROBOTWIN" rev-parse HEAD)" == "$ROBOTWIN_PIN" ]] || {
  echo "RoboTwin upstream commit does not match pinned native source" >&2; exit 11;
}
SESSION="$(tr -d '\r\n' < "$BRIDGE/outgoing/LATEST.txt")"
SESSION_DIR="$BRIDGE/outgoing/$SESSION"
( cd "$SESSION_DIR" && sha256sum -c SHA256SUMS.txt > /dev/null )
EXPECTED_HEAD="$(python3 - "$SESSION_DIR/session-manifest.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8-sig") as handle:
    print(json.load(handle)["git_head"])
PY
)"
cd "$PROJECT"
ACTUAL_HEAD="$(git rev-parse HEAD)"
[[ "$ACTUAL_HEAD" == "$EXPECTED_HEAD" ]] || {
  echo "Unexpected Git revision, run verified bridge bootstrap first" >&2; exit 7;
}
[[ -z "$(git status --porcelain --untracked-files=all | grep -vE '^\?\? \.runtime/' || true)" ]] || {
  echo "Linux Git tree is dirty" >&2; exit 8;
}
CONDA_EXE="$(command -v conda || true)"
if [[ -z "$CONDA_EXE" && -x "$HOME/miniforge3/bin/conda" ]]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
fi
[[ -n "$CONDA_EXE" ]] || { echo "Conda not found" >&2; exit 9; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NAME="s1m_seed12_ik_blockers_$STAMP"
RESULT="$PROJECT/experiments/results/$NAME"
EXPORT="$BRIDGE/incoming/$NAME"
[[ ! -e "$RESULT" && ! -e "$EXPORT" ]] || {
  echo "Refusing overwrite" >&2; exit 10;
}
mkdir -p "$RESULT" "$EXPORT"
printf '%s\n' "$ACTUAL_HEAD" > "$RESULT/git-head.txt"
printf '%s\n' "$SESSION" > "$RESULT/bridge-session.txt"
printf '%s\n' "DIAGNOSTIC_ONLY_NOT_ACCEPTED" > "$RESULT/classification.txt"
git status --short --branch > "$RESULT/git-status.txt"
printf '%s\n' "$ROBOTWIN_PIN" > "$RESULT/robotwin-upstream-pin.txt"
sha256sum scripts/linux/probe_panthera_project_manipulation.py > "$RESULT/probe-sha256.txt"
sha256sum scripts/linux/probe_panthera_observation.py > "$RESULT/observation-probe-sha256.txt"
sha256sum src/vla_robot_sorting/perception/native_depth_capture.py > "$RESULT/native-depth-adapter-sha256.txt"
sha256sum src/vla_robot_sorting/perception/native_depth_audit.py > "$RESULT/native-depth-auditor-sha256.txt"
nvidia-smi > "$RESULT/nvidia-smi.txt" 2>&1 || true

echo "=== S1-M native IK blocker diagnostics, HEAD=$ACTUAL_HEAD ==="
echo "Runs: read-only head/wrist RGB-D probe, then seed1 up75 and seed2 canonical."
echo "=== S1-V4 RGB-D native source-typed capture (one synchronous observation) ==="
DEPTH_PROBE_RC=99
if timeout --signal=TERM --kill-after=30s 1200s \
   "$CONDA_EXE" run -n RoboTwin python -u \
   scripts/linux/probe_panthera_observation.py \
   --robotwin-root "$ROBOTWIN" --project-root "$PROJECT" \
   --output "$RESULT/native-observation.json" --seed 0 \
   --capture-native-depth-dir "$RESULT/native-depth" \
   > "$RESULT/native-depth-probe.log" 2>&1; then
  DEPTH_PROBE_RC=0
else
  DEPTH_PROBE_RC=$?
fi
printf '%s\n' "$DEPTH_PROBE_RC" > "$RESULT/native-depth-probe-exit.txt"
echo "=== RGB-D probe exit=$DEPTH_PROBE_RC; IK diagnostics continue independently ==="
for case_id in seed1-up75 seed2-canonical; do
  args=(--robotwin-root "$ROBOTWIN" --project-root "$PROJECT"
        --output "$RESULT/${case_id}.json"
        --diagnose-native-ik-blockers)
  if [[ "$case_id" == "seed1-up75" ]]; then
    args+=(--seed 1 --b5-evaluate-all-height-groups
           --candidate-id 'pillbottle:cyl:side:0:s-1:up75')
  else
    args+=(--seed 2)
  fi
  echo "=== Running $case_id ==="
  PROBE_RC=99
  if timeout --signal=TERM --kill-after=30s 3600s \
     "$CONDA_EXE" run -n RoboTwin python -u \
     scripts/linux/probe_panthera_project_manipulation.py "${args[@]}" \
     > "$RESULT/${case_id}.log" 2>&1; then
    PROBE_RC=0
  else
    PROBE_RC=$?
  fi
  printf '%s\n' "$PROBE_RC" > "$RESULT/${case_id}-probe-exit.txt"
  AUDIT_RC=99
  if [[ -s "$RESULT/${case_id}.json" ]]; then
    if PYTHONPATH="$PROJECT/src" "$CONDA_EXE" run -n RoboTwin \
       python scripts/audit_project_manipulation.py "$RESULT/${case_id}.json" \
       > "$RESULT/${case_id}-independent-audit.log" 2>&1; then
      AUDIT_RC=0
    else
      AUDIT_RC=$?
    fi
    python3 - "$RESULT/${case_id}.json" > "$RESULT/${case_id}-summary.json" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        payload = json.load(stream)
except (OSError, ValueError) as exc:
    # Diagnostic payload can be malformed after partial/failed native runs.
    # Preserve this failure marker and continue exporting every other log.
    print(json.dumps({
        "classification": "DIAGNOSTIC_ONLY_NOT_ACCEPTED",
        "status": "SUMMARY_PARSE_FAILED",
        "error_type": type(exc).__name__,
    }))
    sys.exit(0)
if not isinstance(payload, dict):
    print(json.dumps({
        "classification": "DIAGNOSTIC_ONLY_NOT_ACCEPTED",
        "status": "SUMMARY_PARSE_FAILED",
        "error_type": "NotObject",
    }))
    sys.exit(0)
diagnostic = payload.get("retreat_direction_diagnostic")
placement = payload.get("place_ik_blocker_diagnostic")
b5 = payload.get("B5")
selected = b5.get("selected") if isinstance(b5, dict) else None
candidate_id = selected.get("candidate_id") if isinstance(selected, dict) else None
print(json.dumps({
    "classification": "DIAGNOSTIC_ONLY_NOT_ACCEPTED",
    "seed": payload.get("seed"),
    "task_passed": payload.get("passed"),
    "failure_phase": payload.get("failure_phase"),
    "selected_candidate_id": candidate_id,
    "retreat_direction_diagnostic": diagnostic,
    "place_ik_blocker_diagnostic": placement,
}, indent=2, sort_keys=True))
PY
  else
    echo "No native JSON produced" > "$RESULT/${case_id}-independent-audit.log"
  fi
  printf '%s\n' "$AUDIT_RC" > "$RESULT/${case_id}-audit-exit.txt"
  echo "=== $case_id: probe=$PROBE_RC audit=$AUDIT_RC ==="
done

cat > "$RESULT/RESULT.md" <<EOF
# Native seed1/seed2 IK diagnostics (NOT ACCEPTED)
- Git HEAD: $ACTUAL_HEAD
- Session: $SESSION
- Tests: read-only head/wrist Color/Position capture; seed1 selected up75 and seed2 canonical, IK flag on
- RGB-D native capture requires per-camera successful manifest and verified single-frame association
- A failed RGB-D probe does not prevent independent seed1/seed2 IK diagnostics
- Planning-only diagnostic hypotheses never certify physical feasibility
- See per-case native JSON, independent audit and checksum manifests
EOF
cp -a "$RESULT/." "$EXPORT/"
(
  cd "$EXPORT"
  find . -type f ! -name SHA256SUMS.txt ! -name checksum-verification.txt \
     -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
  sha256sum -c SHA256SUMS.txt > checksum-verification.txt
)
sync
echo "EXPORT_VERIFIED=$EXPORT"
echo "S1M_SEED12_DIAGNOSTIC_EXPORTED"
echo "DIAGNOSTIC_ONLY_NOT_ACCEPTED"
