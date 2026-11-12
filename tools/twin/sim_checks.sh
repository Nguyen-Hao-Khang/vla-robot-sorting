#!/usr/bin/env bash
# First simulator checks of the sorting task (plan items 1.3 and 1.8).
#
# Run on the Ubuntu simulation workstation inside the RoboTwin conda environment:
#   conda activate RoboTwin
#   bash tools/twin/sim_checks.sh
#
# Steps:
#   1. record 50 demonstrations twice with the same seed and compare them (item 1.3);
#   2. evaluate the scripted expert on the evaluation scenes (seed 9) with built-in English
#      and Vietnamese commands (30 episodes each) and on the 160 held-out test commands;
#   3. print the result tables.
# The expert results are the upper reference for the learned policies and check the
# evaluation script itself. Outputs go to $WORK/sim_checks_<time>/.
#
# Settings: EPISODES=50 EVAL_EPISODES=30 SKIP_EVAL=0 ROBOTWIN_ROOT=... WORK=~/vrs-data
# SKIP_EVAL=1 runs only the reproducibility check (step 1).
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EPISODES=${EPISODES:-50}
EVAL_EPISODES=${EVAL_EPISODES:-30}
SKIP_EVAL=${SKIP_EVAL:-0}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin}
WORK=${WORK:-$HOME/vrs-data}
OUT="$WORK/sim_checks_$(date +%Y%m%d-%H%M)"
TWIN="$REPO_DIR/tools/twin"

log() { printf '\n=== %s: %s\n' "$(date +%H:%M:%S)" "$*"; }

python -c "import sapien" || { echo "Activate the RoboTwin conda environment first."; exit 2; }
[ -d "$ROBOTWIN_ROOT" ] || { echo "RoboTwin not found at $ROBOTWIN_ROOT (set ROBOTWIN_ROOT)."; exit 2; }
mkdir -p "$OUT"
echo "repo=$REPO_DIR commit=$(git -C "$REPO_DIR" rev-parse --short HEAD) out=$OUT"

for run in a b; do
    log "1/3 recording run $run ($EPISODES episodes, seed 0)"
    python "$TWIN/twin_record.py" --robotwin-root "$ROBOTWIN_ROOT" --out "$OUT/repro_$run" \
        --episodes "$EPISODES" --seed 0 2>&1 | tee "$OUT/record_$run.log"
done
set +e
python "$TWIN/compare_recordings.py" "$OUT/repro_a" "$OUT/repro_b" --json "$OUT/compare.json" \
    | tee "$OUT/compare.txt"
set -e
[ "$SKIP_EVAL" = 1 ] && { echo "Done (reproducibility only). Results in $OUT"; exit 0; }

log "2/3 expert evaluation"
for lang in en vi; do
    python "$TWIN/twin_eval.py" run --robotwin-root "$ROBOTWIN_ROOT" --executor expert \
        --episodes "$EVAL_EPISODES" --seed 9 --language "$lang" --command-source builtin \
        --out "$OUT/eval/expert_builtin_$lang.jsonl" 2>&1 | tee "$OUT/eval_builtin_$lang.log"
done
python "$TWIN/twin_eval.py" run --robotwin-root "$ROBOTWIN_ROOT" --executor expert \
    --seed 9 --command-source human --out "$OUT/eval/expert_human.jsonl" 2>&1 | tee "$OUT/eval_human.log"

log "3/3 tables"
python "$TWIN/twin_eval.py" summarise "$OUT"/eval/*.jsonl --by command_source,language,command_kind \
    --json "$OUT/summary.json" | tee "$OUT/summary.md"
echo "Done. Results in $OUT"
