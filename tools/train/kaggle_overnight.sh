#!/usr/bin/env bash
# Record simulated demonstrations, upload them to Kaggle and start the training notebook.
#
# Run on the Ubuntu simulation workstation inside the RoboTwin conda environment:
#   conda activate RoboTwin
#   bash tools/train/kaggle_overnight.sh
#
# Settings (environment variables, all optional):
#   EPISODES=80        successful episodes to record
#   LEVELS=2,5         scene levels (2: colours, 5: colours x sizes)
#   SEED=0             random seed for layouts, tasks and command language
#   LANGUAGES=en,vi    command languages
#   MAX_MINUTES=150    stop recording after this long and upload what exists
#   ROBOTWIN_ROOT      Panthera-HT RoboTwin checkout
#   WORK=~/vrs-data    where raw episodes and archives are written
#
# Kaggle access: create an API token at kaggle.com > Settings > API and save it as
# ~/.kaggle/kaggle.json (chmod 600). Without it the script stops after packing the data.
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EPISODES=${EPISODES:-80}
LEVELS=${LEVELS:-2,5}
SEED=${SEED:-0}
LANGUAGES=${LANGUAGES:-en,vi}
MAX_MINUTES=${MAX_MINUTES:-150}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-$HOME/worktrees/Panthera-HT_RoboTwin/RoboTwin}
WORK=${WORK:-$HOME/vrs-data}
DATASET_SLUG=${DATASET_SLUG:-panthera-twin-raw}
KERNEL_SLUG=${KERNEL_SLUG:-panthera-twin-train}
STAMP=$(date +%Y%m%d-%H%M)
RAW="$WORK/twin_raw_$STAMP"
STAGE="$WORK/kaggle_dataset_$STAMP"

log() { printf '\n=== %s: %s\n' "$(date +%H:%M:%S)" "$*"; }

log "1/4 checks"
python -c "import sapien" || { echo "Activate the RoboTwin conda environment first."; exit 2; }
[ -d "$ROBOTWIN_ROOT" ] || { echo "RoboTwin not found at $ROBOTWIN_ROOT (set ROBOTWIN_ROOT)."; exit 2; }
mkdir -p "$WORK"
echo "repo=$REPO_DIR commit=$(git -C "$REPO_DIR" rev-parse --short HEAD)"
echo "episodes=$EPISODES levels=$LEVELS seed=$SEED languages=$LANGUAGES max_minutes=$MAX_MINUTES"

log "2/4 recording into $RAW"
set +e
timeout --signal=INT "${MAX_MINUTES}m" python "$REPO_DIR/tools/twin/twin_record.py" \
    --robotwin-root "$ROBOTWIN_ROOT" --out "$RAW" --episodes "$EPISODES" \
    --levels "$LEVELS" --seed "$SEED" --languages "$LANGUAGES" 2>&1 | tee "$WORK/record_$STAMP.log"
set -e
# An interrupted run can leave one episode without meta.json; drop it.
find "$RAW" -maxdepth 1 -name 'episode_*' -type d ! -exec test -e '{}/meta.json' ';' -print -exec rm -rf '{}' +
COUNT=$(find "$RAW" -maxdepth 1 -name 'episode_*' -type d | wc -l)
echo "recorded episodes: $COUNT"
[ "$COUNT" -gt 0 ] || { echo "No episode recorded; see $WORK/record_$STAMP.log"; exit 3; }
git -C "$REPO_DIR" rev-parse HEAD > "$RAW/SOURCE_COMMIT"

log "3/4 packing"
mkdir -p "$STAGE"
(cd "$WORK" && python -m zipfile -c "$STAGE/raw.zip" "$(basename "$RAW")")
ls -lh "$STAGE/raw.zip"

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "No ~/.kaggle/kaggle.json: upload $STAGE/raw.zip as a Kaggle dataset by hand,"
    echo "then run notebooks/kaggle_train_panthera_twin.ipynb with a GPU."
    exit 0
fi
command -v kaggle >/dev/null || python -m pip install -q kaggle
KAGGLE_USER=$(python -c "import json, pathlib; print(json.loads((pathlib.Path.home() / '.kaggle/kaggle.json').read_text())['username'])")

log "4/4 uploading dataset $KAGGLE_USER/$DATASET_SLUG and starting the notebook"
cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "$DATASET_SLUG", "id": "$KAGGLE_USER/$DATASET_SLUG", "licenses": [{"name": "CC0-1.0"}]}
JSON
if kaggle datasets status "$KAGGLE_USER/$DATASET_SLUG" >/dev/null 2>&1; then
    kaggle datasets version -p "$STAGE" -m "raw episodes $STAMP"
else
    kaggle datasets create -p "$STAGE"
fi
for _ in $(seq 1 60); do
    STATUS=$(kaggle datasets status "$KAGGLE_USER/$DATASET_SLUG" 2>/dev/null || true)
    echo "dataset status: $STATUS"
    [ "$STATUS" = "ready" ] && break
    sleep 20
done

KERNEL_DIR="$WORK/kaggle_kernel_$STAMP"
mkdir -p "$KERNEL_DIR"
cp "$REPO_DIR/notebooks/kaggle_train_panthera_twin.ipynb" "$KERNEL_DIR/"
cat > "$KERNEL_DIR/kernel-metadata.json" <<JSON
{
  "id": "$KAGGLE_USER/$KERNEL_SLUG",
  "title": "$KERNEL_SLUG",
  "code_file": "kaggle_train_panthera_twin.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["$KAGGLE_USER/$DATASET_SLUG"],
  "competition_sources": [],
  "kernel_sources": []
}
JSON
kaggle kernels push -p "$KERNEL_DIR"

echo
echo "Training started: https://www.kaggle.com/code/$KAGGLE_USER/$KERNEL_SLUG"
echo "Check:    kaggle kernels status $KAGGLE_USER/$KERNEL_SLUG"
echo "Download: kaggle kernels output $KAGGLE_USER/$KERNEL_SLUG -p $WORK/checkpoints_$STAMP"
