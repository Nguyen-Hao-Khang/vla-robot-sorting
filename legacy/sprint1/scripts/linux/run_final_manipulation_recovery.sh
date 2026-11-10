#!/usr/bin/env bash
# Called only after chat main locks the shared HEAD and makes its single bundle.
set -euo pipefail
if (( EUID == 0 )); then echo "Run as normal Ubuntu user" >&2; exit 3; fi
BRIDGE="/mnt/data/DAMH-Bridge"
PROJECT="$HOME/projects/DAMH"
ROBOTWIN="$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin"
findmnt -rn /mnt/data >/dev/null || { echo "DATA not mounted" >&2; exit 4; }
[[ -w "$BRIDGE/incoming" ]] || { echo "Incoming unavailable" >&2; exit 5; }
SESSION="$(tr -d '\r\n' < "$BRIDGE/outgoing/LATEST.txt")"
[[ "$SESSION" =~ ^linux-session-[A-Za-z0-9-]+$ ]] || { echo "Invalid session name" >&2; exit 6; }
SESSION_DIR="$BRIDGE/outgoing/$SESSION"
(cd "$SESSION_DIR" && sha256sum -c SHA256SUMS.txt)
EXPECTED="$(python3 - "$SESSION_DIR/session-manifest.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8-sig") as f:
    print(json.load(f)["git_head"])
PY
)"
[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$EXPECTED" ]] || { echo "Project source mismatch" >&2; exit 7; }
CONDA_EXE="$(command -v conda || true)"
if [[ -z "$CONDA_EXE" && -x "$HOME/miniforge3/bin/conda" ]]; then CONDA_EXE="$HOME/miniforge3/bin/conda"; fi
[[ -n "$CONDA_EXE" ]] || { echo "Conda unavailable" >&2; exit 8; }
cd "$PROJECT"
exec "$CONDA_EXE" run --no-capture-output -n RoboTwin python -u \
  scripts/run_final_manipulation_session.py --project-root "$PROJECT" \
  --robotwin-root "$ROBOTWIN" "$@"
