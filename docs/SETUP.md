# Setup

## What runs where

| Task | Windows | Ubuntu 22.04 + NVIDIA GPU | Kaggle / Colab |
|---|---|---|---|
| Unit tests, offline scenarios, data processing, docs | Yes | Yes | — |
| RoboTwin / SAPIEN simulation, digital twin, demo recording, policy evaluation in sim | **No** (WSL is not supported for this stack) | Yes | No |
| Training SmolVLA / Evo-1 | With an NVIDIA GPU ≥ 16 GB (RTX 50-series needs PyTorch built for CUDA 12.8 or later) | With ≥ 16 GB VRAM | **Yes** |
| Training ACT | With an NVIDIA GPU ≥ 4 GB | Yes | Yes |

## Windows (development and tests)

Prerequisites: Git, and [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 by itself.

```powershell
git clone <repo-url> DAMH
cd DAMH
uv sync --dev --extra native-camera-replay
uv pip install opencv-python-headless
uv run pytest
uv run ruff check .
```

Expected result:
- About 1,090 tests pass. The tests in `test_native_depth_foreground_unittest.py` are skipped unless `S1I_NATIVE_DIR` points to a native RGB-D capture.

Do not place the clone inside a OneDrive/Dropbox folder. Cloud sync can corrupt `.git`.

## Ubuntu (simulation)

The pinned versions are in `configs/s1.1/repos.lock.yaml`:
- Panthera-HT RoboTwin `2aee31ad…`
- CuRobo `d64c4b0`
- SAPIEN 3.0.0b1
- PyTorch cu128

To set up and run:
1. Run `scripts/linux/bootstrap_ubuntu_host.sh`. It prepares the conda environment `RoboTwin`.
2. Clone the repository to `~/projects/DAMH` on the Linux disk. Do not use an NTFS partition.
3. Run the twin:

```bash
cd ~/projects/DAMH
set -a; . ~/.config/damh/gemini.env; set +a   # GEMINI_API_KEY lives only in this file (mode 600)
python scripts/linux/twin_console.py --level 3
```

## Secrets

- API keys are read from environment variables only.
- Never commit `.env` files or keys, and never paste keys into issues or chat.
