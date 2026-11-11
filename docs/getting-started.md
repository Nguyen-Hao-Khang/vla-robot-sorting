# Getting started

## 1. Which machine does what

| Task | Windows | Ubuntu 22.04 + NVIDIA GPU | Kaggle / Colab |
|---|---|---|---|
| Code, unit tests, data processing, documentation | Yes | Yes | Not needed |
| RoboTwin simulation, demonstration recording, policy rollouts | No (WSL lacks the required Vulkan support) | Yes | No |
| Fine-tuning SmolVLA or Evo-1 | Smoke tests only (section 4.3) | Yes | Yes, main runs |

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

Expected: ruff reports `All checks passed!` and pytest reports about 210 passed tests.

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

SmolVLA runs on LeRobot 0.4.4. Evo-1 is not in that release; it needs LeRobot `main` at commit `e581c1d` or later, so it gets its own environment.

```sh
# SmolVLA
uv venv .venv-train --python 3.12
uv pip install --python .venv-train "lerobot[smolvla]==0.4.4"
# Evo-1
uv venv .venv-evo1 --python 3.12
uv pip install --python .venv-evo1 \
    "lerobot[training,evo1] @ git+https://github.com/huggingface/lerobot@e581c1d5701f846788662079d4ac08835c96def9"
```

Installing LeRobot can replace a CUDA build of PyTorch with the CPU build. Check `torch.cuda.is_available()` and, if it is `False`, reinstall the CUDA build, for example `torch==2.10.0 torchvision==0.25.0` from `https://download.pytorch.org/whl/cu126`. RTX 50-series GPUs need CUDA 12.8 or newer. On Windows the interpreter is `.venv-train\Scripts\python`.

### 4.2 Kaggle

1. Verify your phone number on Kaggle to enable GPUs (30 hours per week).
2. Upload `notebooks/kaggle_train_panthera_twin.ipynb`, attach the raw dataset, select GPU T4 Ă—2.
3. Save checkpoints to the Kaggle output or the Hugging Face Hub. Do not commit checkpoints.

### 4.3 Smoke tests (plan item 1.6)

A smoke test trains 100 steps on the public dataset `lerobot/svla_so100_pickplace` (50 episodes, two cameras) to check that the environment, the model download and one optimisation loop work. It says nothing about task performance.

```sh
lerobot-train --policy.path=<local copy of lerobot/smolvla_base> \
    --dataset.repo_id=lerobot/svla_so100_pickplace \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --policy.empty_cameras=1 --batch_size=2 --steps=100 --num_workers=0 \
    --save_checkpoint=false --policy.push_to_hub=false --policy.device=cuda
```

Results on 12 November 2026, laptop GTX 1660 Ti Max-Q (6 GB, Windows 11, driver 555.97, PyTorch 2.10.0+cu126):

| Model | Environment | Setting | Result |
|---|---|---|---|
| SmolVLA (`lerobot/smolvla_base`, 450 M parameters, 100 M trainable) | LeRobot 0.4.4 | batch 2, fp32 | 100 steps in 1 min 53 s (1.1 s per step); loss 0.12 at step 100; about 2.3 GB GPU memory |
| Evo-1 (InternVL3-1B, 775 M parameters, 122 M trainable in stage 1) | LeRobot `e581c1d` | batch 1, fp16, no flash attention | Fails before the first step: out of GPU memory with eager attention; with SDPA attention, `CUBLAS_STATUS_EXECUTION_FAILED` in the action head |

Notes for Windows:

- Pass a local directory to `--policy.path`. LeRobot 0.4.4 turns a Hub ID such as `lerobot/smolvla_base` into a Windows path and rejects it. Download it first with `huggingface_hub.snapshot_download`.
- `--num_workers=0` keeps host memory low. With less than about 5 GB of free commit memory, loading the 2 GB SmolVLM2 weights crashes with an access violation.
- If an expired Hugging Face token is stored, public downloads fail with HTTP 401. Set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.
- Evo-1 defaults to bfloat16 and flash attention. Turing GPUs (GTX 16 series, T4) support neither, so set `--policy.dtype=float16 --policy.use_amp=false --policy.use_flash_attn=false`.

The Evo-1 smoke test is repeated on Kaggle (T4, 16 GB) and on Ubuntu with the same GPU before ADR 0005 is decided.
