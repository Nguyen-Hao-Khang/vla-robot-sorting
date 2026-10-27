#!/usr/bin/env bash
# One Ubuntu boot: preflight -> optional offline tests -> native -> verify export.
# Does not alter motion/acceptance policy, promote manifests or reboot.
set -Eeuo pipefail

MODE=run
UNIT_TESTS=0
RERUN=0
for arg in "$@"; do
  case "$arg" in
    --preflight-only) MODE=preflight ;;
    --verify-only) MODE=verify ;;
    --unit-tests) UNIT_TESTS=1 ;;
    --rerun) RERUN=1 ;;
    -h|--help)
      echo "Usage: bash RUN_SPRINT1_ONE_BOOT.sh [--preflight-only|--verify-only] [--unit-tests] [--rerun]"
      echo "Default: one campaign plus export verification; --rerun explicitly permits a repeat."
      exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 64 ;;
  esac
done
if (( EUID == 0 )); then
  echo "Refusing to run as root." >&2
  exit 65
fi

BRIDGE_MOUNT="/mnt/data"
BRIDGE="$BRIDGE_MOUNT/DAMH-Bridge"
SESSION_DIR="$(cd "$(dirname "$0")" && pwd -P)"
SESSION="$(basename "$SESSION_DIR")"
PROJECT_ROOT="$HOME/projects/DAMH"
ROBOTWIN_PARENT="$HOME/worktrees/Panthera-HT_RoboTwin"
SOURCE="$ROBOTWIN_PARENT/RoboTwin/envs/move_pillbottle_pad.py"
EXPECTED_SOURCE_SHA="20e7b9d954fd42c3b78e4a9431aee68b3ff10f70f20104e7dc7e41c83fc8cffe"

if [[ ! "$SESSION" =~ ^linux-session-[0-9]{8}-[0-9]{6}$ ]]; then
  echo "Run the packaged script from its outgoing session directory." >&2
  exit 66
fi
if ! findmnt -rn "$BRIDGE_MOUNT" >/dev/null 2>&1; then
  echo "DATA is not mounted. Mount by label DATA before invoking this script." >&2
  exit 67
fi
if [[ ! -w "$BRIDGE/incoming" ]]; then
  echo "Bridge incoming is not writable; do not run native." >&2
  exit 68
fi
if [[ "$(tr -d '\r\n' < "$BRIDGE/outgoing/LATEST.txt")" != "$SESSION" ]]; then
  echo "LATEST.txt does not point to this immutable session ($SESSION)." >&2
  exit 69
fi

echo "=== Verify immutable package: $SESSION ==="
(cd "$SESSION_DIR" && sha256sum -c SHA256SUMS.txt)
EXPECTED_HEAD="$(python3 - "$SESSION_DIR/session-manifest.json" <<'PY'
import json,sys
with open(sys.argv[1], encoding="utf-8-sig") as handle:
    print(json.load(handle)["git_head"])
PY
)"
MANIFEST_SESSION="$(python3 - "$SESSION_DIR/session-manifest.json" <<'PY'
import json,sys
with open(sys.argv[1], encoding="utf-8-sig") as handle:
    print(json.load(handle)["session_name"])
PY
)"
if [[ "$MANIFEST_SESSION" != "$SESSION" ]]; then
  echo "Session manifest mismatch." >&2
  exit 70
fi

latest_export() {
  find "$BRIDGE/incoming" -mindepth 1 -maxdepth 1 -type d \
    -name "$SESSION-*" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}
verify_export() {
  local folder
  folder="$(latest_export)"
  if [[ -z "$folder" || ! -f "$folder/SHA256SUMS.txt" ]]; then
    echo "No complete checksum manifest for this session."
    return 1
  fi
  (cd "$folder" && sha256sum -c SHA256SUMS.txt)
  echo "VERIFIED_EXPORT=$folder"
}
if [[ "$MODE" == verify ]]; then
  verify_export
  echo "VERIFY_ONLY_DONE: native simulator was not launched."
  exit 0
fi

echo "=== Bootstrap exact project HEAD ==="
bash "$SESSION_DIR/bootstrap_from_bridge.sh"
cd "$PROJECT_ROOT"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
  echo "Linux Git HEAD does not match session manifest." >&2
  exit 71
fi
DIRTY="$(git status --porcelain --untracked-files=all | grep -vE '^\?\? \.runtime/' || true)"
if [[ -n "$DIRTY" ]]; then
  echo "Project dirty outside .runtime: preserve Linux changes and stop."
  printf '%s\n' "$DIRTY"
  exit 72
fi

echo "=== Verify pinned upstream source bytes ==="
if [[ ! -f "$SOURCE" ]]; then
  echo "Missing pinned upstream file: $SOURCE"
  exit 73
fi
ACTUAL_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SOURCE_SHA" ]]; then
  echo "Pinned source mismatch expected=$EXPECTED_SOURCE_SHA actual=$ACTUAL_SHA"
  exit 74
fi
if [[ -d "$ROBOTWIN_PARENT/.git" ]]; then
  if [[ "$(git -C "$ROBOTWIN_PARENT" rev-parse HEAD)" != "2aee31ad75cb7b0cd779540441c6025853a5b60d" ]]; then
    echo "Pinned upstream repository HEAD mismatch."
    exit 75
  fi
fi
echo "PREFLIGHT_PASS session=$SESSION head=$EXPECTED_HEAD"
if [[ "$MODE" == preflight ]]; then
  echo "PREFLIGHT_ONLY_DONE: no simulator was launched."
  exit 0
fi

if (( UNIT_TESTS )); then
  echo "=== Optional offline tests (not native acceptance) ==="
  CONDA_EXE="$(command -v conda || true)"
  if [[ -z "$CONDA_EXE" && -x "$HOME/miniforge3/bin/conda" ]]; then
    CONDA_EXE="$HOME/miniforge3/bin/conda"
  fi
  if [[ -n "$CONDA_EXE" ]]; then
    set +e
    "$CONDA_EXE" run -n RoboTwin python -m pytest -q tests/unit
    OFFLINE_RC=$?
    set -e
    echo "LINUX_OFFLINE_TESTS_RC=$OFFLINE_RC (not native evidence)"
  else
    echo "LINUX_OFFLINE_TESTS_SKIPPED: conda missing"
  fi
fi

BASE_LOG="$BRIDGE/incoming/native-console-$SESSION.log"
if [[ -e "$BASE_LOG" && "$RERUN" -eq 0 ]]; then
  echo "Native console log already exists. Refusing accidental duplicate run."
  echo "Use --verify-only, or --rerun explicitly if genuinely needed."
  exit 76
fi
if (( RERUN )); then
  LOG="$BRIDGE/incoming/native-console-$SESSION-rerun-$(date -u +%Y%m%dT%H%M%SZ).log"
else
  LOG="$BASE_LOG"
fi
MARKER="$(mktemp)"
touch "$MARKER"
echo "=== Run exactly one native manipulation campaign ==="
echo "LOG=$LOG"
set +e
bash "$SESSION_DIR/RUN_SPRINT1_MANIPULATION_CAMPAIGN.sh" 2>&1 | tee "$LOG"
CAMPAIGN_RC=${PIPESTATUS[0]}
set -e
sync
echo "CAMPAIGN_RC=$CAMPAIGN_RC (40 means PARTIAL)"
if ! grep -Fq "Evidence export exit: 0" "$LOG"; then
  echo "EXPORT_NOT_CONFIRMED: preserve Ubuntu session and inspect evidence."
  bash "$SESSION_DIR/EXPORT_NATIVE_CHECKPOINT.sh" "one-boot-export-missing" || true
  rm -f "$MARKER"
  exit 77
fi
FRESH="$(find "$BRIDGE/incoming" -mindepth 1 -maxdepth 1 -type d \
  -name "$SESSION-*" -newer "$MARKER" -printf '%T@ %p\n' \
  | sort -nr | head -n 1 | cut -d' ' -f2-)"
rm -f "$MARKER"
if [[ -z "$FRESH" ]]; then
  echo "No fresh export folder despite export confirmation."
  exit 78
fi
(cd "$FRESH" && sha256sum -c SHA256SUMS.txt)
echo "EVIDENCE_EXPORT_CONFIRMED=$FRESH"

# The digest was included in dedicated evidence by the campaign.
LATEST_RUN="$(find "$PROJECT_ROOT/experiments/results" -mindepth 1 -maxdepth 1 \
  -type d -name 'sprint1_manipulation_campaign_*' -printf '%T@ %p\n' \
  | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -n "$LATEST_RUN" && -f "$LATEST_RUN/DIAGNOSTIC_DIGEST.txt" ]]; then
  echo "=== Diagnostic digest (not independent acceptance) ==="
  cat "$LATEST_RUN/DIAGNOSTIC_DIGEST.txt"
fi
echo "Do not promote native acceptance until independent Windows audit."
echo "No reboot was performed. Return to Windows after preserving this evidence."
exit "$CAMPAIGN_RC"
