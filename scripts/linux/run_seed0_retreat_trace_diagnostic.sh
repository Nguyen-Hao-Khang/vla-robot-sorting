#!/usr/bin/env bash
# Seed0 native diagnostics only. Never promote to an accepted manipulation manifest.
set -euo pipefail
BRIDGE="/mnt/data/DAMH-Bridge"
PROJECT="$HOME/projects/DAMH"
ROBOTWIN="$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin"
if (( EUID == 0 )); then echo "Run as normal Ubuntu user" >&2; exit 3; fi
if ! findmnt -rn /mnt/data >/dev/null; then echo "Mount DATA at /mnt/data" >&2; exit 4; fi
if [[ ! -w "$BRIDGE/incoming" ]]; then echo "Bridge incoming not writable" >&2; exit 5; fi
if [[ ! -d "$PROJECT/.git" || ! -d "$ROBOTWIN" ]]; then
  echo "Native project / RoboTwin worktree missing" >&2; exit 6
fi
SESSION="$(tr -d '\r\n' < "$BRIDGE/outgoing/LATEST.txt")"
SESSION_DIR="$BRIDGE/outgoing/$SESSION"
( cd "$SESSION_DIR"; sha256sum -c SHA256SUMS.txt >/dev/null )
EXPECTED_HEAD="$(python3 - "$SESSION_DIR/session-manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8-sig") as handle:
    print(json.load(handle)["git_head"])
PY
)"
cd "$PROJECT"
ACTUAL_HEAD="$(git rev-parse HEAD)"
if [[ "$ACTUAL_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "Wrong Linux Git HEAD; bootstrap the current bridge bundle first" >&2
  exit 7
fi
DIRTY="$(git status --porcelain --untracked-files=all | grep -vE '^\?\? \.runtime/' || true)"
if [[ -n "$DIRTY" ]]; then
  echo "Dirty Linux worktree; refusing uncertain provenance" >&2
  printf '%s\n' "$DIRTY"; exit 8
fi
CONDA_EXE="$(command -v conda || true)"
if [[ -z "$CONDA_EXE" && -x "$HOME/miniforge3/bin/conda" ]]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
fi
if [[ -z "$CONDA_EXE" ]]; then echo "Conda missing" >&2; exit 9; fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NAME="s1m_seed0_retreat_trace_$STAMP"
RESULT="$PROJECT/experiments/results/$NAME"
EXPORT="$BRIDGE/incoming/$NAME"
if [[ -e "$RESULT" || -e "$EXPORT" ]]; then
  echo "Output already exists; refusing overwrite" >&2; exit 10
fi
mkdir -p "$RESULT" "$EXPORT"
printf '%s\n' "$ACTUAL_HEAD" > "$RESULT/git-head.txt"
printf '%s\n' "$SESSION" > "$RESULT/bridge-session.txt"
printf '%s\n' "DIAGNOSTIC_ONLY_NOT_ACCEPTED" > "$RESULT/classification.txt"
git status --short --branch > "$RESULT/git-status.txt"
sha256sum scripts/linux/probe_panthera_project_manipulation.py > "$RESULT/probe-sha256.txt"
nvidia-smi > "$RESULT/nvidia-smi.txt" 2>&1 || true

echo "Running PROJECT-OWNED seed0 ONLY, source HEAD: $ACTUAL_HEAD"
PROBE_RC=99
if timeout --signal=TERM --kill-after=30s 3600s \
  "$CONDA_EXE" run -n RoboTwin python -u \
  scripts/linux/probe_panthera_project_manipulation.py \
  --robotwin-root "$ROBOTWIN" \
  --project-root "$PROJECT" \
  --output "$RESULT/seed0.json" --seed 0 \
  > "$RESULT/seed0.log" 2>&1; then
  PROBE_RC=0
else
  PROBE_RC=$?
fi
printf '%s\n' "$PROBE_RC" > "$RESULT/probe-exit-code.txt"
AUDIT_RC=99
if [[ -s "$RESULT/seed0.json" ]]; then
  if PYTHONPATH="$PROJECT/src" "$CONDA_EXE" run -n RoboTwin \
    python scripts/audit_project_manipulation.py "$RESULT/seed0.json" \
    > "$RESULT/independent-audit.log" 2>&1; then
    AUDIT_RC=0
  else
    AUDIT_RC=$?
  fi
  python3 - "$RESULT/seed0.json" > "$RESULT/trace-summary.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
retreat = next(
    (stage for stage in data.get("stages", []) if stage.get("stage") == "retreat"),
    {},
)
trace = retreat.get("runtime_contact_trace") or {}
motion = retreat.get("runtime_motion_frame_trace") or {}
print(json.dumps({
    "schema": "s1m-seed0-retreat-diagnostic-v1",
    "accepted": False,
    "seed": data.get("seed"),
    "task_passed": data.get("passed"),
    "retreat_passed": retreat.get("passed"),
    "contact_counts": retreat.get("contacts"),
    "trace_frame_count": len(trace.get("frames") or []),
    "trace_truncated": trace.get("truncated"),
    "trace_schema": trace.get("schema"),
    "motion_frame_count": len(motion.get("frames") or []),
    "motion_frame_truncated": motion.get("truncated"),
    "motion_frame_schema": motion.get("schema"),
}, indent=2, sort_keys=True))
PY
else
  echo "No native JSON produced" > "$RESULT/independent-audit.log"
fi
printf '%s\n' "$AUDIT_RC" > "$RESULT/audit-exit-code.txt"
cat > "$RESULT/RESULT.md" <<EOF
# Seed0 retreat diagnostic (NOT ACCEPTED)
- Git HEAD: $ACTUAL_HEAD
- Bridge input session: $SESSION
- Native probe exit: $PROBE_RC
- Independent audit exit: $AUDIT_RC
- Claim boundary: diagnostic data only. Check seed0.json retreat.runtime_contact_trace.
EOF

cp -a "$RESULT/." "$EXPORT/"
(
  cd "$EXPORT"
  find . -type f ! -name SHA256SUMS.txt ! -name checksum-verification.txt \
    -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
  sha256sum -c SHA256SUMS.txt > checksum-verification.txt
)
sync
echo "Export verified: $EXPORT"
echo "Probe RC=$PROBE_RC, audit RC=$AUDIT_RC"
echo "DIAGNOSTIC_ONLY_NOT_ACCEPTED. Return to Windows for independent review."
