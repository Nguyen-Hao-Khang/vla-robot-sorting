#!/usr/bin/env bash
# Narrow S1-V5 follow-up: read-only SAPIEN RGB-D capture only; no manipulation.
set -euo pipefail

BRIDGE=/mnt/data/DAMH-Bridge
PROJECT="$HOME/projects/DAMH"
ROBOTWIN="$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin"
PIN=2aee31ad75cb7b0cd779540441c6025853a5b60d
PREVIOUS="$BRIDGE/incoming/s1v4_seed2_alt_20261012T133414Z"

[[ "$EUID" -ne 0 ]] || { echo "STOP: use normal Ubuntu user" >&2; exit 2; }
findmnt -rn /mnt/data >/dev/null || { echo "STOP: DATA mount absent" >&2; exit 3; }
[[ -w "$BRIDGE/incoming" ]] || { echo "STOP: incoming not writable" >&2; exit 4; }
[[ -d "$PROJECT/.git" && -d "$ROBOTWIN" ]] || { echo "STOP: project missing" >&2; exit 5; }
[[ "$(git -C "$ROBOTWIN" rev-parse HEAD)" == "$PIN" ]] || {
    echo "STOP: pinned RoboTwin SHA mismatch" >&2; exit 6;
}
SESSION="$(tr -d '\r\n' < "$BRIDGE/outgoing/LATEST.txt")"
DIR="$BRIDGE/outgoing/$SESSION"
( cd "$DIR" && sha256sum -c SHA256SUMS.txt >/dev/null )
HEAD="$(python3 - "$DIR/session-manifest.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8-sig") as handle:
    print(json.load(handle)["git_head"])
PY
)"
cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$HEAD" ]] || {
    echo "STOP: Git HEAD is not source-locked to bundle" >&2; exit 7;
}
[[ -z "$(git status --porcelain --untracked-files=all | grep -vE '^\?\? \.runtime/' || true)" ]] || {
    echo "STOP: modified checkout" >&2; exit 8;
}
( cd "$PREVIOUS" && sha256sum -c SHA256SUMS.txt >/dev/null )
python3 - "$PREVIOUS/native-depth" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for cam in ("head_camera", "wrist_camera"):
    data = json.loads((root / f"{cam}_frame_000000_depth-manifest.json").read_text())
    assert data["depth_convention"] == (
        "depth_mm = max(0, -Position[...,2] * 1000 * Color_alpha/255)"
    ), "old observation is not expected legacy source; stop"
print("PREVIOUS_FAILED_METRIC_SOURCE_VERIFIED")
PY

CONDA="$(command -v conda || true)"
if [[ -z "$CONDA" && -x "$HOME/miniforge3/bin/conda" ]]; then
    CONDA="$HOME/miniforge3/bin/conda"
fi
[[ -n "$CONDA" ]] || { echo "STOP: conda unavailable" >&2; exit 9; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NAME="s1v5_depth_repair_$STAMP"
RESULT="$PROJECT/experiments/results/$NAME"
EXPORT="$BRIDGE/incoming/$NAME"
[[ ! -e "$RESULT" && ! -e "$EXPORT" ]] || { echo "STOP: overwrite forbidden" >&2; exit 10; }
mkdir -p "$RESULT" "$EXPORT"
printf '%s\n' "$HEAD" > "$RESULT/git-head.txt"
printf '%s\n' "$SESSION" > "$RESULT/bridge-session.txt"
printf '%s\n' "$PIN" > "$RESULT/robotwin-upstream-pin.txt"
printf '%s\n' "DIAGNOSTIC_ONLY_NOT_TASK_ACCEPTANCE" > "$RESULT/classification.txt"
sha256sum "$PREVIOUS/native-depth/"*_depth-manifest.json > "$RESULT/previous-manifest-sha256.txt"
sha256sum src/vla_robot_sorting/perception/native_depth_capture.py \
  src/vla_robot_sorting/perception/native_depth_audit.py \
  src/vla_robot_sorting/perception/native_metric_quality.py \
  scripts/linux/probe_panthera_observation.py > "$RESULT/source-sha256.txt"

PROBE_RC=99
if timeout --signal=TERM --kill-after=30s 1200s \
    "$CONDA" run -n RoboTwin python -u \
    scripts/linux/probe_panthera_observation.py \
    --robotwin-root "$ROBOTWIN" --project-root "$PROJECT" \
    --output "$RESULT/native-observation.json" --seed 0 \
    --capture-native-depth-dir "$RESULT/native-depth" \
    > "$RESULT/native-depth-probe.log" 2>&1; then
    PROBE_RC=0
else
    PROBE_RC=$?
fi
printf '%s\n' "$PROBE_RC" > "$RESULT/probe-exit.txt"

ARTIFACT_RC=98
METRIC_RC=98
if [[ -d "$RESULT/native-depth" ]]; then
    if PYTHONPATH="$PROJECT/src" "$CONDA" run -n RoboTwin \
        python scripts/audit_native_depth_capture.py "$RESULT/native-depth" \
        > "$RESULT/artifact-audit.log" 2>&1; then
        ARTIFACT_RC=0
    else
        ARTIFACT_RC=$?
    fi
    if PYTHONPATH="$PROJECT/src" "$CONDA" run -n RoboTwin \
        python -m vla_robot_sorting.perception.native_metric_quality \
        "$RESULT/native-depth" > "$RESULT/metric-quality-audit.log" 2>&1; then
        METRIC_RC=0
    else
        METRIC_RC=$?
    fi
else
    echo "No native source depth directory" > "$RESULT/artifact-audit.log"
    echo "No native source depth directory" > "$RESULT/metric-quality-audit.log"
fi
printf '%s\n' "$ARTIFACT_RC" > "$RESULT/artifact-audit-exit.txt"
printf '%s\n' "$METRIC_RC" > "$RESULT/metric-quality-exit.txt"

cat > "$RESULT/RESULT.md" <<EOF
# S1-V5 repaired capture (source diagnostic only)
- project git: $HEAD
- RoboTwin git: $PIN
- prior evidence verified: $PREVIOUS
- no manip, planner, IK, physical grasp, Gemini API, or canonical task run
- probe exit: $PROBE_RC
- artifact audit exit: $ARTIFACT_RC
- metric quality audit exit: $METRIC_RC
- metric quality pass is NOT object identity, true sensor accuracy or task acceptance
- classification: DIAGNOSTIC_ONLY_NOT_TASK_ACCEPTANCE
EOF

cp -a "$RESULT/." "$EXPORT/"
(
    cd "$EXPORT"
    find . -type f ! -name SHA256SUMS.txt ! -name checksum-verification.txt \
      -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
    sha256sum -c SHA256SUMS.txt > checksum-verification.txt
)
sync
echo "PROBE_EXIT=$PROBE_RC"
echo "ARTIFACT_AUDIT_EXIT=$ARTIFACT_RC"
echo "METRIC_QUALITY_EXIT=$METRIC_RC"
echo "EXPORT_VERIFIED=$EXPORT"
echo "DIAGNOSTIC_ONLY_NOT_TASK_ACCEPTANCE"
