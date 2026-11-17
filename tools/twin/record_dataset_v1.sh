#!/usr/bin/env bash
# Overnight recording for dataset v1 (plan item 2.1), in the RoboTwin environment.
#
# Part A: one scripted-expert demonstration per fresh scene (twin_eval --executor expert
#         --record-dir), English and Vietnamese, generated and built-in command wording.
# Part B: corrective demonstrations (twin_eval --executor dagger): the served policy drives for
#         1-10 s, then the expert completes the command; only the expert part is saved.
#
#   bash tools/twin/record_dataset_v1.sh <out root> <policy checkpoint> [A end HH:MM] [B end HH:MM]
#
# Each part stops at its end time (next occurrence of HH:MM); runs resume where they stopped,
# so the script can simply be started again. Scene seeds 31-34 (A) and 41-42 (B) do not overlap
# the evaluation seed 9. Environment: LEROBOT_PYTHON (~/venvs/lerobot07/bin/python), PORT (5555),
# CAMERA_MAP (wrist=camera1,workspace=camera2), N_A and N_B (episodes per run, 150).
set -u
if [ $# -lt 2 ]; then
  sed -n '2,16p' "$0"
  exit 2
fi
out=$1
ckpt=$2
a_end=${3:-02:30}
b_end=${4:-06:10}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LR=${LEROBOT_PYTHON:-$HOME/venvs/lerobot07/bin/python}
PORT=${PORT:-5555}
CAMERA_MAP=${CAMERA_MAP:-wrist=camera1,workspace=camera2}
N_A=${N_A:-150}
N_B=${N_B:-150}
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
cd "$REPO" || exit 1
mkdir -p "$out/layouts" "$out/dagger" "$out/logs"

until_ts() {  # seconds until the next HH:MM
  local t
  t=$(date -d "$1" +%s)
  [ "$t" -le "$(date +%s)" ] && t=$(date -d "tomorrow $1" +%s)
  echo $((t - $(date +%s)))
}

run_part() {  # deadline label args...
  local left
  left=$(until_ts "$1")
  shift
  local label=$1
  shift
  if [ "$left" -le 60 ]; then
    echo "SKIP $label (deadline reached)"
    return
  fi
  echo "RUN $label ($((left / 60)) min left in this part)"
  timeout --signal=INT "${left}s" python tools/twin/twin_eval.py run "$@" 2>&1 \
    | sed -u 's/.*\r//' | grep --line-buffered -E "^\[|EVAL_DONE|Error|Traceback" \
    | tee -a "$out/logs/$label.log"
}

echo "PART_A until $a_end"
for spec in "31 en generated" "32 vi generated" "33 en builtin" "34 vi builtin"; do
  set -- $spec
  run_part "$a_end" "A_s$1_$2_$3" --executor expert --episodes "$N_A" --seed "$1" \
    --language "$2" --command-source "$3" --record-dir "$out/layouts" \
    --out "$out/layouts/run_s$1_$2_$3.jsonl"
done

echo "PART_B until $b_end"
log="$out/logs/server.log"
"$LR" tools/twin/policy_bridge.py serve --checkpoint "$ckpt" --port "$PORT" \
  --camera-map "$CAMERA_MAP" > "$log" 2>&1 &
server=$!
for _ in $(seq 1 180); do
  if grep -q POLICY_SERVER_READY "$log" || ! kill -0 "$server" 2>/dev/null; then break; fi
  sleep 5
done
if ! grep -q POLICY_SERVER_READY "$log"; then
  echo "SERVER_FAILED"
  tail -n 20 "$log"
else
  for spec in "41 en" "42 vi"; do
    set -- $spec
    run_part "$b_end" "B_s$1_$2" --executor dagger --policy-server "127.0.0.1:$PORT" \
      --checkpoint "$(basename "$ckpt")" --episodes "$N_B" --seed "$1" --language "$2" \
      --command-source generated --record-dir "$out/dagger" --out "$out/dagger/run_s$1_$2.jsonl"
  done
fi
kill "$server" 2>/dev/null
wait "$server" 2>/dev/null
for part in layouts dagger; do
  ok=$(cat "$out/$part"/run_*.jsonl 2>/dev/null | grep -c '"success": true')
  all=$(cat "$out/$part"/run_*.jsonl 2>/dev/null | wc -l)
  echo "SUMMARY $part: $ok successful of $all episodes; folders: $(ls -d "$out/$part"/episode_* 2>/dev/null | wc -l)"
done
git rev-parse HEAD > "$out/SOURCE_COMMIT"
echo "ALL_DONE"
