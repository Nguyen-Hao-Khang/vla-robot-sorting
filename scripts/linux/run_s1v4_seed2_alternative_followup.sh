#!/usr/bin/env bash
# Evidence-driven FOLLOW-UP only: RGB-D Python 3.10 fix + distinct B5 seed2 grasp.
# Historical seed1 and seed2 failures are NOT rerun in this script.
set -euo pipefail

BRIDGE=/mnt/data/DAMH-Bridge
PROJECT="$HOME/projects/DAMH"
ROBOTWIN="$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin"
PIN=2aee31ad75cb7b0cd779540441c6025853a5b60d
BASELINE="$BRIDGE/incoming/s1m_seed12_ik_blockers_20261012T054448Z"
ALT='pillbottle:cyl:side:3:s-1'

[[ "$EUID" -ne 0 ]] || { echo "Run as the normal Ubuntu user" >&2; exit 3; }
findmnt -rn /mnt/data >/dev/null || { echo "DATA mount missing" >&2; exit 4; }
[[ -w "$BRIDGE/incoming" ]] || { echo "Bridge incoming not writable" >&2; exit 5; }
[[ -d "$PROJECT/.git" && -d "$ROBOTWIN" ]] || { echo "Source missing" >&2; exit 6; }
[[ "$(git -C "$ROBOTWIN" rev-parse HEAD)" == "$PIN" ]] || {
  echo "Pinned RoboTwin HEAD mismatch" >&2; exit 7;
}
SESSION="$(tr -d '\r\n' < "$BRIDGE/outgoing/LATEST.txt")"
SESSION_DIR="$BRIDGE/outgoing/$SESSION"
( cd "$SESSION_DIR" && sha256sum -c SHA256SUMS.txt >/dev/null )
EXPECTED_HEAD="$(python3 - "$SESSION_DIR/session-manifest.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8-sig") as stream:
    print(json.load(stream)["git_head"])
PY
)"
cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || {
  echo "Project HEAD does not match bundled source" >&2; exit 8;
}
[[ -z "$(git status --porcelain --untracked-files=all | grep -vE '^\?\? \.runtime/' || true)" ]] || {
  echo "Project Git tree is dirty" >&2; exit 9;
}
( cd "$BASELINE" && sha256sum -c SHA256SUMS.txt >/dev/null )
[[ "$(tr -d '\r\n' < "$BASELINE/git-head.txt")" == \
  5f855a2e8e7a456e9c1194f1a67e0274d3798fa5 ]] || {
  echo "Baseline source SHA mismatch" >&2; exit 10;
}

# Gate next candidate against independently exported baseline; do NOT replay
# original seed2. Also verify the precise stage and vendor failure.
python3 - "$BASELINE/seed2-canonical.json" "$ALT" <<'PY'
import json, sys
from math import isfinite
with open(sys.argv[1], encoding="utf-8") as stream:
    data = json.load(stream)
alternative = sys.argv[2]
b5 = data["B5"]
selected = b5["selected"]["candidate_id"]
stage = next(
    (row for row in data["stages"] if row.get("stage") == "place_above_support"),
    None,
)
diag = data["place_ik_blocker_diagnostic"]
elevated = data["placement_recovery_diagnostic"]["staged_descent"]["elevated_motion"]
joint4 = elevated["qpos_after_rad"][3]
eligible = [
    item["candidate_id"]
    for item in b5["ranking"]
    if item.get("eligible") is True
]
ok = (
    data["seed"] == 2 and data["passed"] is False
    and selected == "pillbottle:cyl:side:2:s-1"
    and alternative != selected and alternative in eligible
    and stage is not None and stage.get("passed") is False
    and diag["start_state"]["status"] == "Start state is out of joint limits"
    and isfinite(joint4) and 1.6 <= joint4 <= 1.601
)
if not ok:
    raise SystemExit("STOP: baseline cannot authorize distinct seed2 diagnostic")
print(f"BASELINE_VERIFIED selected={selected} alternative={alternative} joint4={joint4}")
PY

CONDA_EXE="$(command -v conda || true)"
if [[ -z "$CONDA_EXE" && -x "$HOME/miniforge3/bin/conda" ]]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
fi
[[ -n "$CONDA_EXE" ]] || { echo "Conda not found" >&2; exit 11; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NAME="s1v4_seed2_alt_$STAMP"
RESULT="$PROJECT/experiments/results/$NAME"
EXPORT="$BRIDGE/incoming/$NAME"
[[ ! -e "$RESULT" && ! -e "$EXPORT" ]] || { echo "Refusing overwrite" >&2; exit 12; }
mkdir -p "$RESULT" "$EXPORT"
printf '%s\n' "$EXPECTED_HEAD" > "$RESULT/git-head.txt"
printf '%s\n' "$SESSION" > "$RESULT/bridge-session.txt"
printf '%s\n' "$PIN" > "$RESULT/robotwin-upstream-pin.txt"
printf '%s\n' "$ALT" > "$RESULT/seed2-expected-alternative.txt"
printf '%s\n' "DIAGNOSTIC_ONLY_NOT_ACCEPTED" > "$RESULT/classification.txt"
git status --short --branch > "$RESULT/git-status.txt"
sha256sum "$BASELINE/seed2-canonical.json" > "$RESULT/baseline-seed2-sha256.txt"
sha256sum scripts/linux/probe_panthera_observation.py \
  src/vla_robot_sorting/perception/native_depth_capture.py \
  src/vla_robot_sorting/perception/native_depth_audit.py \
  scripts/linux/probe_panthera_project_manipulation.py \
  > "$RESULT/source-sha256.txt"
nvidia-smi > "$RESULT/nvidia-smi.txt" 2>&1 || true

echo "=== FOLLOW-UP depth capture: Python 3.10 compatible import ==="
DEPTH_RC=99
if timeout --signal=TERM --kill-after=30s 1200s \
  "$CONDA_EXE" run -n RoboTwin python -u \
  scripts/linux/probe_panthera_observation.py \
  --robotwin-root "$ROBOTWIN" --project-root "$PROJECT" \
  --output "$RESULT/native-observation.json" --seed 0 \
  --capture-native-depth-dir "$RESULT/native-depth" \
  > "$RESULT/native-depth-probe.log" 2>&1; then DEPTH_RC=0
else DEPTH_RC=$?
fi
printf '%s\n' "$DEPTH_RC" > "$RESULT/native-depth-probe-exit.txt"
DEPTH_AUDIT_RC=98
if [[ -d "$RESULT/native-depth" ]]; then
  if PYTHONPATH="$PROJECT/src" "$CONDA_EXE" run -n RoboTwin python \
      scripts/audit_native_depth_capture.py "$RESULT/native-depth" \
      > "$RESULT/native-depth-independent-audit.log" 2>&1; then DEPTH_AUDIT_RC=0
  else DEPTH_AUDIT_RC=$?
  fi
else
  echo "No native depth directory produced" > "$RESULT/native-depth-independent-audit.log"
fi
printf '%s\n' "$DEPTH_AUDIT_RC" > "$RESULT/native-depth-audit-exit.txt"
echo "DEPTH_PROBE_EXIT=$DEPTH_RC DEPTH_AUDIT_EXIT=$DEPTH_AUDIT_RC"

echo "=== FOLLOW-UP fixed seed2, distinct eligible B5 grasp ==="
ALT_RC=99
if timeout --signal=TERM --kill-after=30s 3600s \
  "$CONDA_EXE" run -n RoboTwin python -u \
  scripts/linux/probe_panthera_project_manipulation.py \
  --robotwin-root "$ROBOTWIN" --project-root "$PROJECT" \
  --output "$RESULT/seed2-alternative.json" \
  --seed 2 --b5-evaluate-all-height-groups \
  --candidate-id "$ALT" --diagnose-native-ik-blockers \
  > "$RESULT/seed2-alternative.log" 2>&1; then ALT_RC=0
else ALT_RC=$?
fi
printf '%s\n' "$ALT_RC" > "$RESULT/seed2-alternative-probe-exit.txt"
ALT_AUDIT_RC=98
if [[ -s "$RESULT/seed2-alternative.json" ]]; then
  if PYTHONPATH="$PROJECT/src" "$CONDA_EXE" run -n RoboTwin \
    python scripts/audit_project_manipulation.py "$RESULT/seed2-alternative.json" \
    > "$RESULT/seed2-alternative-independent-audit.log" 2>&1; then ALT_AUDIT_RC=0
  else ALT_AUDIT_RC=$?
  fi
else
  echo "No native JSON produced" > "$RESULT/seed2-alternative-independent-audit.log"
fi
printf '%s\n' "$ALT_AUDIT_RC" > "$RESULT/seed2-alternative-audit-exit.txt"

# The recorded selected candidate must be genuinely B5-eligible on this
# fresh native scene; passing this gate alone NEVER grants task acceptance.
PROVENANCE_RC=98
if [[ -s "$RESULT/seed2-alternative.json" ]]; then
  if python3 - "$RESULT/seed2-alternative.json" "$ALT" \
      > "$RESULT/seed2-provenance-check.log" 2>&1 <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        doc = json.load(stream)
    b5 = doc["B5"]
    alt = sys.argv[2]
    ok = (doc["seed"] == 2 and b5.get("passed") is True
          and b5["selected"]["candidate_id"] == alt
          and any(row.get("candidate_id") == alt and row.get("eligible") is True
                  for row in b5["ranking"]))
except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
    print(f"PROVENANCE_ERROR={type(exc).__name__}: {exc}")
    raise SystemExit(2)
print(f"B5_ALTERNATIVE_PROVENANCE_PASS={ok}")
raise SystemExit(0 if ok else 1)
PY
  then PROVENANCE_RC=0
  else PROVENANCE_RC=$?
  fi
else
  echo "No native JSON available" > "$RESULT/seed2-provenance-check.log"
fi
printf '%s\n' "$PROVENANCE_RC" > "$RESULT/seed2-provenance-exit.txt"
echo "SEED2_PROBE_EXIT=$ALT_RC SEED2_AUDIT_EXIT=$ALT_AUDIT_RC SEED2_PROVENANCE_EXIT=$PROVENANCE_RC"

cat > "$RESULT/RESULT.md" <<EOF
# S1-V4 RGB-D + seed2 distinct B5 alternative (diagnostic only)
- New project Git HEAD: $EXPECTED_HEAD
- Pinned RoboTwin: $PIN
- Historical source: $BASELINE/seed2-canonical.json, source HEAD 5f855a2
- RGB-D: read-only head and wrist; depth auditor independent
- Seed2: \`$ALT\`, not historical failed \`pillbottle:cyl:side:2:s-1\`
- Original canonical target, joint limits and collision policy unchanged
- Seed1 original/up75 NOT rerun. No full three-seed campaign performed.
- Never infer native acceptance from process completion or source eligibility.
- Classification: DIAGNOSTIC_ONLY_NOT_ACCEPTED
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
echo "S1V4_SEED2_ALTERNATIVE_DIAGNOSTIC_EXPORTED"
echo "DIAGNOSTIC_ONLY_NOT_ACCEPTED"
