# Getting started

## 1. Which machine does what

| Task | Windows | Ubuntu 22.04 + NVIDIA GPU | Kaggle / Colab |
|---|---|---|---|
| Code, unit tests, data processing, documentation | Yes | Yes | Not needed |
| RoboTwin simulation, demonstration recording, policy rollouts | No (WSL lacks the required Vulkan support) | Yes | No |
| Fine-tuning SmolVLA or Evo-1 | Yes, GPU with 16 GB or more | Yes, GPU with 16 GB or more | Yes |

## 2. Development environment

Install [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/). uv downloads a matching Python (3.10 to 3.12).

```sh
git clone https://github.com/Nguyen-Hao-Khang/vla-robot-sorting.git
cd vla-robot-sorting
uv sync
uv run ruff check .
uv run pytest
uv run pre-commit install
```

Expected: ruff reports `All checks passed!` and pytest reports about 190 passed tests.

On Windows, `tools/setup/bootstrap_windows.ps1` runs the same steps and `tools/setup/check_windows_env.ps1` prints the Git, Python, uv and GPU versions.

Keep the clone outside OneDrive, Dropbox or Google Drive. Their partial syncs corrupt `.git`.

## 3. Simulation environment (Ubuntu 22.04)

### 3.1 Requirements

- Ubuntu 22.04 on bare metal. Virtual machines and WSL are not supported.
- NVIDIA GPU with Vulkan. Tested with driver 595 on a GTX 1660 Ti (6 GB).
- About 40 GB of free disk for RoboTwin, assets and data.

### 3.2 Pinned versions

| Component | Source | Version |
|---|---|---|
| Panthera-HT RoboTwin | `HighTorque-Robotics/Panthera-HT_RoboTwin` | commit `2aee31ad` |
| CuRobo | `NVlabs/curobo` | `v0.7.8` (`d64c4b00`) |
| SAPIEN | PyPI | `3.0.0b1` |
| PyTorch | PyPI | CUDA 12.8 build |

Full list with commit hashes: [`configs/repos.lock.yaml`](../configs/repos.lock.yaml).

### 3.3 Steps

1. Install the driver and system tools, then reboot:

   ```sh
   bash tools/setup/bootstrap_ubuntu_host.sh
   ```

   The script installs `build-essential`, `vulkan-tools`, `ffmpeg` and the recommended NVIDIA driver, and logs to `~/vrs-host-bootstrap/`.

2. Install Miniforge, create the conda environment `RoboTwin`, and clone Panthera-HT RoboTwin at the pinned commit. Follow that repository's installation guide, including the asset download.
3. Install CuRobo `v0.7.8` into the `RoboTwin` environment following the NVIDIA instructions.
4. Clone this repository onto the Linux file system, for example `~/projects/vla-robot-sorting`. NTFS partitions break file permissions.
5. Open a scene:

   ```sh
   conda activate RoboTwin
   cd ~/projects/vla-robot-sorting
   python tools/twin/twin_console.py \
       --robotwin-root ~/worktrees/Panthera-HT_RoboTwin/RoboTwin --level 1
   ```

   A SAPIEN window shows the arm and one cube on the table. Session logs go to `~/twin_logs/`; change this with `--log-dir`.

### 3.4 Language model API key

The interactive console calls Gemini to plan typed commands. They read the key from `GEMINI_API_KEY` only.

```sh
mkdir -p ~/.config/vla-robot-sorting
printf 'GEMINI_API_KEY=<your key>\n' > ~/.config/vla-robot-sorting/gemini.env
chmod 600 ~/.config/vla-robot-sorting/gemini.env
set -a; . ~/.config/vla-robot-sorting/gemini.env; set +a
```

Never put the key in code, notebooks, issues or chat messages.

## 4. Training environment

### 4.1 Local NVIDIA GPU (Windows or Linux)

```sh
uv venv .venv-train --python 3.10
uv pip install --python .venv-train "lerobot[smolvla]==0.4.4"
.venv-train/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On Windows the interpreter is `.venv-train\Scripts\python`. RTX 50-series GPUs need a PyTorch build for CUDA 12.8 or newer.

### 4.2 Kaggle

1. Verify your phone number on Kaggle to enable GPUs (30 hours per week).
2. Upload `notebooks/kaggle_train_panthera_twin.ipynb`, attach the raw dataset, select GPU T4 ×2.
3. Save checkpoints to the Kaggle output or the Hugging Face Hub. Do not commit checkpoints.
