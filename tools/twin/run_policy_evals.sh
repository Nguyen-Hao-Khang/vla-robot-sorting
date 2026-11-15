#!/usr/bin/env bash
# Serve one checkpoint with policy_bridge.py, evaluate it in the twin in English and Vietnamese,
# stop the server and print the summary table. Run in the RoboTwin environment; the server runs
# in the LeRobot environment given by LEROBOT_PYTHON.
#
#   bash tools/twin/run_policy_evals.sh <label> <checkpoint dir> [policy_bridge serve options]
#   N=10 OUT=~/vrs-data/eval_n10 bash tools/twin/run_policy_evals.sh smolvla_seed1_n10 \
#       ~/vrs-models/smolvla_seed1 --camera-map wrist=camera1,workspace=camera2 --n-action-steps 10
#
# Environment: N (episodes per language, 30), SEED (scene seed, 9), OUT (result folder),
# LANGS ("en vi"), EVAL_ARGS (extra twin_eval.py run options, e.g. --no-safety),
# LEROBOT_PYTHON (~/venvs/lerobot07/bin/python), PORT (5555).
set -u
if [ $# -lt 2 ]; then
  sed -n '2,12p' "$0"
  exit 2
fi
label=$1
ckpt=$2
shift 2
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LR=${LEROBOT_PYTHON:-$HOME/venvs/lerobot07/bin/python}
OUT=${OUT:-$HOME/vrs-data/eval}
N=${N:-30}
SEED=${SEED:-9}
PORT=${PORT:-5555}
LANGS=${LANGS:-"en vi"}
mkdir -p "$OUT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1  # an expired stored login breaks public downloads
cd "$REPO" || exit 1
log="$OUT/server_$label.log"
"$LR" tools/twin/policy_bridge.py serve --checkpoint "$ckpt" --port "$PORT" "$@" > "$log" 2>&1 &
server=$!
for _ in $(seq 1 180); do
  if grep -q POLICY_SERVER_READY "$log" || ! kill -0 "$server" 2>/dev/null; then break; fi
  sleep 5
done
if ! grep -q POLICY_SERVER_READY "$log"; then
  echo "SERVER_FAILED $label"
  tail -n 20 "$log"
  kill "$server" 2>/dev/null
  exit 1
fi
grep POLICY_SERVER_READY "$log"
for lang in $LANGS; do
  # shellcheck disable=SC2086
  python tools/twin/twin_eval.py run --executor policy --checkpoint "$label" \
    --policy-server "127.0.0.1:$PORT" --episodes "$N" --seed "$SEED" --language "$lang" \
    --out "$OUT/${label}_${lang}.jsonl" ${EVAL_ARGS:-} 2>&1 \
    | sed -u 's/.*\r//' | grep --line-buffered -E "^\[|EVAL_DONE|Error|Traceback"
done
kill "$server"
wait "$server" 2>/dev/null
python tools/twin/twin_eval.py summarise "$OUT"/"$label"_*.jsonl --by checkpoint,language
echo "MODEL_DONE $label"
