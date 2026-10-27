#!/usr/bin/env bash
set -euo pipefail

BRIDGE_MOUNT="${DAMH_DATA_MOUNT:-/mnt/data}"
BRIDGE="${BRIDGE_MOUNT}/DAMH-Bridge"
PROJECT_ROOT="${DAMH_LINUX_PROJECT:-$HOME/projects/DAMH}"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run bootstrap_from_bridge.sh with sudo/root."
  echo "Run it as the normal desktop user so HOME and project ownership stay correct."
  exit 7
fi

mount_data_if_needed() {
  local device uid gid
  device="$(blkid -L DATA || true)"
  uid="$(id -u)"
  gid="$(id -g)"
  if [[ -z "$device" ]]; then
    echo "DATA volume label not found."
    return 1
  fi

  if ! findmnt -rn "$BRIDGE_MOUNT" >/dev/null 2>&1; then
    sudo mkdir -p "$BRIDGE_MOUNT"
    sudo mount -t ntfs3 -o "uid=${uid},gid=${gid},umask=022" "$device" "$BRIDGE_MOUNT"
  fi

  findmnt "$BRIDGE_MOUNT"

  if [[ ! -w "$BRIDGE/incoming" ]]; then
    echo "DATA bridge is mounted but $BRIDGE/incoming is not writable by $(id -un)."
    echo "Remount DATA from a shell whose cwd is outside /mnt/data:"
    echo "  cd ~"
    echo "  sudo umount $BRIDGE_MOUNT"
    echo "  sudo mount -t ntfs3 -o uid=$(id -u),gid=$(id -g),umask=022 $device $BRIDGE_MOUNT"
    return 1
  fi
}


mount_data_if_needed

LATEST_FILE="${BRIDGE}/outgoing/LATEST.txt"
if [[ ! -f "$LATEST_FILE" ]]; then
  echo "Missing $LATEST_FILE"
  exit 2
fi

SESSION_NAME="$(tr -d '\r\n' < "$LATEST_FILE")"
SESSION_DIR="${BRIDGE}/outgoing/${SESSION_NAME}"
if [[ ! -d "$SESSION_DIR" ]]; then
  echo "Missing session directory: $SESSION_DIR"
  exit 3
fi

echo "Using bridge session: $SESSION_NAME"
(
  cd "$SESSION_DIR"
  sha256sum -c SHA256SUMS.txt
)


EXPECTED_HEAD="$(python3 - "${SESSION_DIR}/session-manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8-sig") as handle:
    print(json.load(handle)["git_head"])
PY
)"
BRANCH="$(python3 - "${SESSION_DIR}/session-manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8-sig") as handle:
    print(json.load(handle)["git_branch"])
PY
)"

BUNDLE="${SESSION_DIR}/project-main.bundle"
mkdir -p "$(dirname "$PROJECT_ROOT")"


if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
  git clone -b "$BRANCH" "$BUNDLE" "$PROJECT_ROOT"
else
  cd "$PROJECT_ROOT"
  DIRTY_STATUS="$(git status --porcelain --untracked-files=all |
    grep -vE '^\?\? \.runtime/' || true)"
  if [[ -n "$DIRTY_STATUS" ]]; then
    echo "Existing Linux project is dirty outside local .runtime metadata."
    printf '%s\n' "$DIRTY_STATUS"
    echo "Export/commit Linux work before updating from Windows."
    exit 4
  fi

  git fetch "$BUNDLE" "${BRANCH}:refs/remotes/bridge/${BRANCH}"
  INCOMING="$(git rev-parse "refs/remotes/bridge/${BRANCH}")"
  CURRENT="$(git rev-parse HEAD)"
  if [[ "$CURRENT" != "$INCOMING" ]]; then
    if ! git merge-base --is-ancestor "$CURRENT" "$INCOMING"; then
      echo "Linux project contains history not present in the Windows bundle."
      echo "Export Linux work before continuing."
      exit 5
    fi
    git checkout "$BRANCH"
    git merge --ff-only "refs/remotes/bridge/${BRANCH}"
  fi
fi


cd "$PROJECT_ROOT"
ACTUAL_HEAD="$(git rev-parse HEAD)"
if [[ "$ACTUAL_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "Project HEAD mismatch: expected $EXPECTED_HEAD, got $ACTUAL_HEAD"
  exit 6
fi

mkdir -p .runtime
printf '%s\n' "$SESSION_NAME" > .runtime/current_session
printf '%s\n' "$SESSION_DIR" > .runtime/session_dir

echo "Linux project ready:"
echo "  project=$PROJECT_ROOT"
echo "  branch=$BRANCH"
echo "  head=$ACTUAL_HEAD"
echo "  session=$SESSION_NAME"
git status --short --branch
