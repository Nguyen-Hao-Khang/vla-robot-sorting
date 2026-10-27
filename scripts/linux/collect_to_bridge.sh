#!/usr/bin/env bash
set -euo pipefail

BRIDGE_MOUNT="${DAMH_DATA_MOUNT:-/mnt/data}"
BRIDGE="${BRIDGE_MOUNT}/DAMH-Bridge"
PROJECT_ROOT="${DAMH_LINUX_PROJECT:-$HOME/projects/DAMH}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
  echo "Linux project not found: $PROJECT_ROOT"
  exit 2
fi

cd "$PROJECT_ROOT"

if [[ -f .runtime/current_session ]]; then
  SESSION_NAME="$(tr -d '\r\n' < .runtime/current_session)"
else
  SESSION_NAME="unknown-session"
fi

if [[ -n "${DAMH_EVIDENCE_DIR_OVERRIDE:-}" ]]; then
  LATEST_EVIDENCE="$DAMH_EVIDENCE_DIR_OVERRIDE"
else
  LATEST_EVIDENCE="$(find "$HOME/damh-evidence" -mindepth 1 -maxdepth 1 -type d 2>/dev/null |
    sort | tail -n 1 || true)"
fi
RESULT_CLASSIFICATION="${DAMH_RESULT_CLASSIFICATION_OVERRIDE:-FILL_ME}"
RESULT_SCOPE="${DAMH_RESULT_SCOPE_OVERRIDE:-S1.1 harness}"
FIRST_FAIL="${DAMH_FIRST_FAIL_OVERRIDE:-}"
if [[ -z "${DAMH_RESULT_CLASSIFICATION_OVERRIDE:-}" && -n "$LATEST_EVIDENCE" && -f "$LATEST_EVIDENCE/PASS" ]]; then
  RESULT_CLASSIFICATION="PASS"
elif [[ -z "${DAMH_RESULT_CLASSIFICATION_OVERRIDE:-}" && -n "$LATEST_EVIDENCE" && -f "$LATEST_EVIDENCE/FAILED" ]]; then
  RESULT_CLASSIFICATION="FAIL"
  FIRST_FAIL="$(tr '\n' ' ' < "$LATEST_EVIDENCE/FAILED" | sed 's/[[:space:]]*$//')"
fi

OUT="${BRIDGE}/incoming/${SESSION_NAME}-${STAMP}"
mkdir -p "$OUT"


{
  echo "# Linux Session Result"
  echo
  echo "- Session: $SESSION_NAME"
  echo "- Timestamp UTC: $STAMP"
  echo "- Host: $(hostname)"
  echo "- Project: $PROJECT_ROOT"
  echo "- Git HEAD: $(git rev-parse HEAD)"
  echo "- Branch: $(git branch --show-current)"
  echo "- Result classification: $RESULT_CLASSIFICATION"
  echo "- Result scope: $RESULT_SCOPE"
  echo "- Evidence directory: ${LATEST_EVIDENCE:-none}"
  echo
  echo "## Commands / gates completed"
  echo
  echo "See evidence.tar.gz and the manifest/checksum files inside it."
  echo
  echo "## First failing gate"
  echo
  echo "${FIRST_FAIL:-None captured by the S1.1 harness.}"
  echo
  echo "## Recommended next action"
  echo
  echo "Return to Windows for evidence audit."
} > "$OUT/RESULT.md"

uname -a > "$OUT/uname.txt"
cat /etc/os-release > "$OUT/os-release.txt"
lsblk -f > "$OUT/lsblk.txt"
df -h > "$OUT/df.txt"
free -h > "$OUT/free.txt"


nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1 || true
vulkaninfo --summary > "$OUT/vulkaninfo-summary.txt" 2>&1 || true
git status --short --branch > "$OUT/git-status.txt"
git log -10 --oneline --decorate > "$OUT/git-log.txt"
git diff > "$OUT/uncommitted.patch"
git diff --cached > "$OUT/staged.patch"

if command -v conda >/dev/null 2>&1; then
  conda info > "$OUT/conda-info.txt" 2>&1 || true
  conda list --explicit > "$OUT/conda-explicit.txt" 2>&1 || true
fi

python3 --version > "$OUT/python.txt" 2>&1 || true
python3 -m pip freeze > "$OUT/pip-freeze.txt" 2>&1 || true

git bundle create "$OUT/linux-project.bundle" --all
git bundle verify "$OUT/linux-project.bundle" > "$OUT/git-bundle-verify.txt" 2>&1

if [[ -n "$LATEST_EVIDENCE" ]]; then
  tar -C "$(dirname "$LATEST_EVIDENCE")" -czf "$OUT/evidence.tar.gz"     "$(basename "$LATEST_EVIDENCE")"
fi

if [[ -d experiments/results ]]; then
  tar -czf "$OUT/experiment-results.tar.gz" experiments/results
fi

(
  cd "$OUT"
  sha256sum     $(find . -maxdepth 1 -type f ! -name SHA256SUMS.txt -printf '%f\n' | sort)     > SHA256SUMS.txt
)

echo "Results exported to: $OUT"
if [[ "$RESULT_CLASSIFICATION" == "FILL_ME" ]]; then
  echo "No S1.1 PASS/FAILED marker was found; review $OUT/RESULT.md before rebooting."
else
  echo "RESULT.md classified as $RESULT_CLASSIFICATION for scope: $RESULT_SCOPE."
fi
