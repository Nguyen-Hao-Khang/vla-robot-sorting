#!/usr/bin/env bash
set -u -o pipefail

BRIDGE_MOUNT="${DAMH_DATA_MOUNT:-/mnt/data}"
BRIDGE="${BRIDGE_MOUNT}/DAMH-Bridge"
PROJECT_ROOT="${DAMH_LINUX_PROJECT:-$HOME/projects/DAMH}"
LABEL="${1:-manual-native-checkpoint}"

if ! findmnt -rn "$BRIDGE_MOUNT" >/dev/null 2>&1; then
  echo "DATA is not mounted at $BRIDGE_MOUNT."
  exit 2
fi
if [[ ! -w "$BRIDGE/incoming" ]]; then
  echo "$BRIDGE/incoming is not writable."
  exit 3
fi

cd "$PROJECT_ROOT" || exit 4
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE="$HOME/damh-evidence/native_checkpoint_$STAMP"
mkdir -p "$EVIDENCE"

printf '%s\n' "$LABEL" > "$EVIDENCE/CHECKPOINT_LABEL.txt"
git rev-parse HEAD > "$EVIDENCE/git-head.txt"
git status --short --branch > "$EVIDENCE/git-status.txt"
git log -15 --oneline --decorate > "$EVIDENCE/git-log.txt"
git diff > "$EVIDENCE/uncommitted.patch"
git diff --cached > "$EVIDENCE/staged.patch"
git diff --check > "$EVIDENCE/git-diff-check.txt" 2>&1
nvidia-smi > "$EVIDENCE/nvidia-smi.txt" 2>&1 || true

if [[ -d experiments/results ]]; then
  find experiments/results -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n'     | sort -nr     | head -n 5     | cut -d' ' -f2-     > "$EVIDENCE/recent-result-directories.txt"
fi

(
  cd "$EVIDENCE"
  find . -type f ! -path './SHA256SUMS.txt' -print0     | sort -z     | xargs -0 sha256sum > SHA256SUMS.txt
)

DAMH_RESULT_CLASSIFICATION_OVERRIDE="CHECKPOINT" DAMH_RESULT_SCOPE_OVERRIDE="Native in-boot checkpoint: $LABEL" DAMH_FIRST_FAIL_OVERRIDE="" DAMH_EVIDENCE_DIR_OVERRIDE="$EVIDENCE"   bash scripts/linux/collect_to_bridge.sh
RC=$?
sync

echo "Native checkpoint export exit: $RC"
echo "Evidence: $EVIDENCE"
exit "$RC"
