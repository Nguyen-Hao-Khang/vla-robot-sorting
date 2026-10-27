#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_STAGE="init"
GPU_SAMPLER_PID=""
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

S1_TAG="s1_1_$(date -u +%Y%m%dT%H%M%SZ)"
S1_EVIDENCE="$HOME/damh-evidence/${S1_TAG}"
mkdir -p "${S1_EVIDENCE}"/{logs,telemetry,manifest,checksums}

cleanup() {
  if [[ -n "${GPU_SAMPLER_PID}" ]]; then
    kill "${GPU_SAMPLER_PID}" 2>/dev/null || true
    wait "${GPU_SAMPLER_PID}" 2>/dev/null || true
  fi
}

write_failed_marker() {
  local rc="$1"
  if [[ ! -f "${S1_EVIDENCE}/FAILED" && ! -f "${S1_EVIDENCE}/PASS" ]]; then
    echo "FAILED_STAGE=${CURRENT_STAGE}" | tee "${S1_EVIDENCE}/FAILED"
    echo "EXIT_CODE=${rc}" | tee -a "${S1_EVIDENCE}/FAILED"
  fi
}

on_error() {
  local rc=$?
  write_failed_marker "${rc}"
  cleanup
  exit "${rc}"
}

on_exit() {
  local rc=$?
  cleanup
  if [[ "${rc}" -ne 0 ]]; then
    write_failed_marker "${rc}"
  fi
}

trap on_error ERR
trap on_exit EXIT
exec > >(tee "${S1_EVIDENCE}/logs/session.log") 2>&1

conda_hook_compatible() {
  # NVIDIA cuda-nvcc activate.d scripts reference variables that may be unset.
  # Run conda shell-function operations with nounset temporarily disabled, then
  # restore the harness's strict mode immediately afterward.
  set +u
  conda "$@"
  local rc=$?
  set -u
  return "${rc}"
}


CURRENT_STAGE="host-preflight"
. /etc/os-release
test "${ID}" = "ubuntu"
test "${VERSION_ID}" = "22.04"
test "$(stat -f -c %T "$HOME")" != "fuseblk"

nvidia-smi | tee "${S1_EVIDENCE}/manifest/nvidia-smi.txt"
nvidia-smi   --query-gpu=name,driver_version,memory.total,compute_cap   --format=csv,noheader |
  tee "${S1_EVIDENCE}/manifest/gpu.csv"

if lsmod | grep -q '^nouveau'; then
  echo "nouveau is loaded; proprietary NVIDIA driver is required."
  false
fi

lspci -nnk | grep -A3 -i nvidia |
  tee "${S1_EVIDENCE}/manifest/lspci-nvidia.txt"

vulkaninfo --summary |
  tee "${S1_EVIDENCE}/manifest/vulkaninfo-summary.txt"


CURRENT_STAGE="miniforge"
if ! command -v curl >/dev/null 2>&1; then
  sudo apt update
  sudo apt install -y curl
fi

if ! command -v conda >/dev/null 2>&1 && [[ ! -x "$HOME/miniforge3/bin/conda" ]]; then
  MINIFORGE_VERSION="26.7.2-0"
  MINIFORGE_INSTALLER="Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"
  MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_INSTALLER}"
  MINIFORGE_SHA256="281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05"

  curl -fL "${MINIFORGE_URL}" -o "/tmp/${MINIFORGE_INSTALLER}"
  echo "${MINIFORGE_SHA256}  /tmp/${MINIFORGE_INSTALLER}" | sha256sum -c -
  bash "/tmp/${MINIFORGE_INSTALLER}" -b -p "$HOME/miniforge3"
  rm -f "/tmp/${MINIFORGE_INSTALLER}"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  CONDA_BASE="$HOME/miniforge3"
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"


CURRENT_STAGE="conda-environment"
conda --version | tee "${S1_EVIDENCE}/manifest/conda-version.txt"

if ! conda env list | awk '{print $1}' | grep -qx RoboTwin; then
  conda_hook_compatible create -n RoboTwin python=3.10.20 -y
fi

conda_hook_compatible activate RoboTwin
python --version | tee "${S1_EVIDENCE}/manifest/python-version.txt"
which python | tee "${S1_EVIDENCE}/manifest/python-path.txt"
conda info | tee "${S1_EVIDENCE}/manifest/conda-info.txt"

CURRENT_STAGE="disk-preflight"
{
  echo "=== filesystem before cache cleanup ==="
  df -h /
  echo "=== selected cache/worktree sizes ==="
  du -sh "$HOME/.cache/pip" "$HOME/miniforge3/pkgs" "$HOME/worktrees" 2>/dev/null || true
} | tee "${S1_EVIDENCE}/manifest/disk-preflight-before.txt"

ROOT_AVAIL_KIB="$(df -Pk / | awk 'NR==2 {print $4}')"
MIN_FREE_KIB=$((15 * 1024 * 1024))
if [[ "$ROOT_AVAIL_KIB" -lt "$MIN_FREE_KIB" ]]; then
  echo "Root filesystem has less than 15 GiB free; clearing package download caches."
  python -m pip cache purge || true
  conda_hook_compatible clean --all -y || true
fi

{
  echo "=== filesystem after cache cleanup ==="
  df -h /
  echo "=== selected cache/worktree sizes ==="
  du -sh "$HOME/.cache/pip" "$HOME/miniforge3/pkgs" "$HOME/worktrees" 2>/dev/null || true
} | tee "${S1_EVIDENCE}/manifest/disk-preflight-after.txt"

CURRENT_STAGE="clone-panthera"
S1_WORKROOT="$HOME/worktrees"
mkdir -p "$S1_WORKROOT"
cd "$S1_WORKROOT"

if [[ ! -d Panthera-HT_RoboTwin/.git ]]; then
  git clone https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin.git
fi

cd Panthera-HT_RoboTwin
git fetch origin 2aee31ad75cb7b0cd779540441c6025853a5b60d --depth 1
git checkout --detach 2aee31ad75cb7b0cd779540441c6025853a5b60d


git rev-parse HEAD | tee "${S1_EVIDENCE}/manifest/panthera-robotwin.sha"
git rev-parse HEAD:RoboTwin | tee "${S1_EVIDENCE}/manifest/vendored-robotwin-tree.sha"
git status --porcelain=v1 | tee "${S1_EVIDENCE}/manifest/checkout-status.txt"

test "$(git rev-parse HEAD)" = "2aee31ad75cb7b0cd779540441c6025853a5b60d"
test "$(git rev-parse HEAD:RoboTwin)" = "3b0e2538a956c4c10c61c9311646a8d38e576598"

cd RoboTwin
pwd | tee "${S1_EVIDENCE}/manifest/robotwin-workdir.txt"

CURRENT_STAGE="torch-cu128"
python -m pip install --upgrade pip
python -m pip install   torch==2.7.1 torchvision==0.22.1   --index-url https://download.pytorch.org/whl/cu128

python - <<'PY' | tee "${S1_EVIDENCE}/manifest/torch-cuda.txt"
import torch
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device=", torch.cuda.get_device_name(0))
    print("capability=", torch.cuda.get_device_capability(0))
PY


python - <<'PY'
import torch
assert torch.__version__.startswith("2.7.1")
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (7, 5)
PY

CURRENT_STAGE="cuda-compiler"
if ! command -v nvcc >/dev/null 2>&1 || ! nvcc --version | grep -q 'release 12\.8'; then
  conda_hook_compatible install -y -c nvidia/label/cuda-12.8.0 cuda-compiler=12.8
  hash -r
fi
nvcc --version | tee "${S1_EVIDENCE}/manifest/nvcc.txt"
nvcc --version | grep -q 'release 12\.8'

CURRENT_STAGE="core-runtime"
if ! python - <<'PY'
from importlib.metadata import version

expected = {
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "sapien": "3.0.0b1",
    "mplib": "0.2.1",
    "open3d": "0.18.0",
    "toppra": "0.6.9",
    "h5py": "3.16.0",
    "opencv-python": "4.11.0.86",
}
for package, wanted in expected.items():
    actual = version(package)
    assert actual == wanted, f"{package}: expected {wanted}, got {actual}"
PY
then
  PIP_NO_CACHE_DIR=1 python -m pip install \
    numpy==1.26.4 scipy==1.15.3 sapien==3.0.0b1 mplib==0.2.1 \
    transforms3d==0.4.2 trimesh==4.4.3 open3d==0.18.0 \
    gymnasium==0.29.1 toppra==0.6.9 h5py==3.16.0 \
    opencv-python==4.11.0.86 imageio==2.34.2 Pillow==11.3.0 \
    PyYAML==6.0.3 huggingface-hub==0.25.0 warp-lang==1.12.0 \
    setuptools==69.5.1 setuptools-scm==10.2.1 wheel==0.47.0 ninja==1.13.0
else
  echo "Core runtime versions already match; skipping reinstall."
fi


CURRENT_STAGE="sapien-offscreen"
mkdir -p "${S1_EVIDENCE}/sapien-offscreen"
pushd "${S1_EVIDENCE}/sapien-offscreen" >/dev/null
python -m sapien.example.offscreen
test -s sapien_offscreen.png
sha256sum sapien_offscreen.png |
  tee "${S1_EVIDENCE}/checksums/sapien-offscreen.sha256"
popd >/dev/null

cd "$S1_WORKROOT/Panthera-HT_RoboTwin/RoboTwin"

CURRENT_STAGE="assets"
OBJECT_SAMPLE="$(find assets/objects -mindepth 1 -type f -print -quit 2>/dev/null || true)"
BACKGROUND_SAMPLE="$(find assets/background_texture -mindepth 1 -type f -print -quit 2>/dev/null || true)"
if [[ -z "$OBJECT_SAMPLE" || -z "$BACKGROUND_SAMPLE" ]]; then
  bash script/_download_assets.sh
else
  echo "Bulk RoboTwin assets already present; skipping dataset re-download."
  python ./script/update_embodiment_config_path.py
fi

CURRENT_STAGE="panthera-lfs-assets"
python "${PROJECT_ROOT}/scripts/linux/materialize_git_lfs_pointers.py" \
  --repo-root "${S1_WORKROOT}/Panthera-HT_RoboTwin" \
  --subtree "RoboTwin/assets/embodiments/panthera-6dof" \
  --github-repo "HighTorque-Robotics/Panthera-HT_RoboTwin" \
  --commit "2aee31ad75cb7b0cd779540441c6025853a5b60d" \
  --output "${S1_EVIDENCE}/manifest/panthera-lfs-materialization.json"

test -f assets/embodiments/panthera-6dof/panthera_6dof.urdf
test -f assets/embodiments/panthera-6dof/panthera_6dof.srdf
test -f assets/embodiments/panthera-6dof/config.yml
test -f assets/embodiments/panthera-6dof/curobo.yml
test -f assets/embodiments/panthera-6dof/collision_panthera.yml
test -d assets/embodiments/panthera-6dof/meshes
test -d assets/objects
test -d assets/background_texture

python - <<'PY' | tee "${S1_EVIDENCE}/manifest/panthera-urdf-validation.txt"
from pathlib import Path
import xml.etree.ElementTree as ET

urdf = Path("assets/embodiments/panthera-6dof/panthera_6dof.urdf")
root = ET.parse(urdf).getroot()
assert root.tag == "robot"
print("urdf_xml=PASS")
print("urdf_size_bytes=", urdf.stat().st_size)
PY

find assets/embodiments/panthera-6dof -maxdepth 1 -type f -printf '%f\n' |
  sort |
  tee "${S1_EVIDENCE}/manifest/panthera-assets.txt"

CURRENT_STAGE="curobo"
if [[ ! -d envs/curobo/.git ]]; then
  git clone --branch v0.7.8 --depth 1     https://github.com/NVlabs/curobo.git envs/curobo
fi

git -C envs/curobo rev-parse HEAD |
  tee "${S1_EVIDENCE}/manifest/curobo.sha"
test "$(git -C envs/curobo rev-parse HEAD)" = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"

if ! python - "$(pwd)/envs/curobo" <<'PY'
from pathlib import Path
import sys
import curobo

expected_root = Path(sys.argv[1]).resolve()
module_path = Path(curobo.__file__).resolve()
assert getattr(curobo, "__version__", None) == "0.7.8"
assert module_path.is_relative_to(expected_root), (module_path, expected_root)
PY
then
  PIP_NO_CACHE_DIR=1 python -m pip install -e envs/curobo --no-build-isolation
else
  echo "CuRobo 0.7.8 editable install already points at the pinned checkout; skipping rebuild."
fi

SAPIEN_LOCATION="$(python -m pip show sapien | awk '/^Location:/{print $2}')/sapien"
sed -i -E   's/("r")(\))( as)/\1, encoding="utf-8") as/g'   "$SAPIEN_LOCATION/wrapper/urdf_loader.py"

MPLIB_LOCATION="$(python -m pip show mplib | awk '/^Location:/{print $2}')/mplib"
sed -i -E   's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g'   "$MPLIB_LOCATION/planner.py"


CURRENT_STAGE="runtime-imports"
python - <<'PY' | tee "${S1_EVIDENCE}/manifest/runtime-imports.txt"
import cv2
import curobo
import h5py
import mplib
import open3d
import sapien
import torch
import toppra

print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("curobo", getattr(curobo, "__version__", "unknown"))
print("sapien", getattr(sapien, "__version__", "unknown"))
print("mplib", getattr(mplib, "__version__", "unknown"))
print("open3d", open3d.__version__)
print("cv2", cv2.__version__)
print("h5py", h5py.__version__)
print("toppra", getattr(toppra, "__version__", "unknown"))
PY

python -m pip check | tee "${S1_EVIDENCE}/manifest/pip-check.txt"
python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)" |
  tee "${S1_EVIDENCE}/manifest/cuda-home.txt"
conda list --explicit > "${S1_EVIDENCE}/manifest/conda-explicit.txt"
python -m pip freeze --all > "${S1_EVIDENCE}/manifest/pip-freeze.txt"

if command -v sapien >/dev/null 2>&1; then
  set +e
  sapien info --all > "${S1_EVIDENCE}/manifest/sapien-info.txt" 2>&1
  SAPIEN_INFO_RC=$?
  set -e
  echo "sapien-info exit=$SAPIEN_INFO_RC" >> "${S1_EVIDENCE}/manifest/sapien-info.txt"
fi

CURRENT_STAGE="project-observation-probe"
nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits -l 1 \
  > "${S1_EVIDENCE}/telemetry/observation-nvidia-smi.csv" &
GPU_SAMPLER_PID=$!

set +e
timeout --signal=TERM --kill-after=15s 300s \
  python -u "${PROJECT_ROOT}/scripts/linux/probe_panthera_observation.py" \
    --robotwin-root "${S1_WORKROOT}/Panthera-HT_RoboTwin/RoboTwin" \
    --project-root "${PROJECT_ROOT}" \
    --output "${S1_EVIDENCE}/manifest/panthera-observation.json"
OBSERVATION_RC=$?
set -e
cleanup
GPU_SAMPLER_PID=""
test "$OBSERVATION_RC" -eq 0

python - "${S1_EVIDENCE}/manifest/panthera-observation.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)

assert data["joint_names"] == [f"joint{i}" for i in range(1, 7)]
assert len(data["actual_joint_positions_rad"]) == 6
assert data["proprioception_source"] == "Robot.get_left_arm_real_jointState"
assert {item["camera_id"] for item in data["cameras"]} == {"head_camera", "wrist_camera"}
assert all(item["width_px"] > 0 and item["height_px"] > 0 for item in data["cameras"])
assert data["single_arm_setup"]["robot_articulation_count"] == 1
assert data["single_arm_setup"]["wrist_camera_count"] == 1
assert data["planner_initialization"] == "bypassed-observation-only"
assert data["render_profile"] == {
    "render_spp": 1,
    "render_path_depth": 2,
    "render_denoiser": "none",
}
PY

CURRENT_STAGE="planner-init-probe"
timeout --signal=TERM --kill-after=20s 600s \
  python -u "${PROJECT_ROOT}/scripts/linux/probe_panthera_planner_init.py" \
    --robotwin-root "${S1_WORKROOT}/Panthera-HT_RoboTwin/RoboTwin" \
    --project-root "${PROJECT_ROOT}" \
    --output "${S1_EVIDENCE}/manifest/panthera-planner-init.json"

CURRENT_STAGE="canonical-smoke"
nvidia-smi   --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw   --format=csv,noheader,nounits -l 1   > "${S1_EVIDENCE}/telemetry/nvidia-smi.csv" &
GPU_SAMPLER_PID=$!

S1_SMOKE_ROOT="${S1_WORKROOT}/smoke-output"
mkdir -p "$S1_SMOKE_ROOT"

set +e
/usr/bin/time -v -o "${S1_EVIDENCE}/telemetry/time.txt" \
  timeout --signal=TERM --kill-after=30s 1200s \
  python -u "${PROJECT_ROOT}/scripts/linux/run_panthera_collection.py" \
    --robotwin-root "${S1_WORKROOT}/Panthera-HT_RoboTwin/RoboTwin" \
    --project-root "${PROJECT_ROOT}" \
    --task move_pillbottle_pad \
    --task-config move_pillbottle_pad_panthera_single \
    --episode-num 1 \
    --save-path "$S1_SMOKE_ROOT" \
  2>&1 | tee "${S1_EVIDENCE}/logs/panthera-single-smoke.log"
SMOKE_RC=${PIPESTATUS[0]}
set -e

cleanup
GPU_SAMPLER_PID=""
test "$SMOKE_RC" -eq 0


CURRENT_STAGE="artifact-validation"
S1_EP_ROOT="${S1_SMOKE_ROOT}/move_pillbottle_pad/move_pillbottle_pad_panthera_single"
export S1_EP_ROOT

find "$S1_EP_ROOT" -maxdepth 3 -type f -printf '%p %s bytes\n' |
  sort |
  tee "${S1_EVIDENCE}/manifest/episode-files.txt"

test -s "$S1_EP_ROOT/seed.txt"
test -s "$S1_EP_ROOT/scene_info.json"
test -s "$S1_EP_ROOT/collection_manifest.json"
test -s "$S1_EP_ROOT/_traj_data/episode0.pkl"
test -s "$S1_EP_ROOT/instructions/episode0.json"

python - <<'PY' | tee "${S1_EVIDENCE}/manifest/hdf5-schema.txt"
from pathlib import Path
import h5py
import os

root = Path(os.environ["S1_EP_ROOT"]) / "data"
episodes = sorted(root.glob("episode*.hdf5"))
assert episodes, f"No HDF5 under {root}"

with h5py.File(episodes[0], "r") as handle:
    print("HDF5", episodes[0], episodes[0].stat().st_size, "bytes")
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape, obj.dtype)
    handle.visititems(show)
PY


find "$S1_EP_ROOT/video" -type f -size +0c -printf '%p\n' |
  sort |
  tee "${S1_EVIDENCE}/manifest/video-files.txt"

VIDEO_COUNT="$(wc -l < "${S1_EVIDENCE}/manifest/video-files.txt")"
test "$VIDEO_COUNT" -ge 2

while IFS= read -r video; do
  echo "--- $video"
  ffprobe -v error -select_streams v:0     -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames     -of default=noprint_wrappers=1 "$video"
done < "${S1_EVIDENCE}/manifest/video-files.txt" |
  tee "${S1_EVIDENCE}/manifest/video-probe.txt"

python -u "${PROJECT_ROOT}/scripts/summarize_resource_usage.py" \
  --gpu-csv "${S1_EVIDENCE}/telemetry/nvidia-smi.csv" \
  --time-file "${S1_EVIDENCE}/telemetry/time.txt" \
  --output "${S1_EVIDENCE}/manifest/resource-summary.json"

CURRENT_STAGE="finalize"
git -C "$S1_WORKROOT/Panthera-HT_RoboTwin" status --porcelain=v1 |
  tee "${S1_EVIDENCE}/manifest/final-checkout-status.txt"

echo "PASS" | tee "${S1_EVIDENCE}/PASS"
(
  cd "${S1_EVIDENCE}"
  find . -type f ! -path './checksums/SHA256SUMS.txt' -print0 |
    sort -z |
    xargs -0 sha256sum > checksums/SHA256SUMS.txt
)

tar -C "$(dirname "$S1_EVIDENCE")"   -czf "$HOME/${S1_TAG}.tar.gz"   "$(basename "$S1_EVIDENCE")"
sha256sum "$HOME/${S1_TAG}.tar.gz" |
  tee "${S1_EVIDENCE}/checksums/result-bundle.sha256"

echo
echo "S1.1 canonical smoke PASS"
echo "Evidence: $S1_EVIDENCE"
echo "Archive: $HOME/${S1_TAG}.tar.gz"
echo "Next: run scripts/linux/collect_to_bridge.sh before leaving Ubuntu."
