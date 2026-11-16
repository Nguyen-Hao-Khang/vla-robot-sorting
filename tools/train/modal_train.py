"""Train LeRobot policies on Modal (paid GPUs) from datasets kept on a Modal Volume.

One-time setup (local machine):
  pip install modal && python -m modal setup
  modal volume create vrs-data
  modal volume put vrs-data <local LeRobot dataset folder> /datasets/<name>

Runs (``--detach`` keeps the run alive if the local terminal closes):
  modal run --detach tools/train/modal_train.py::train --dataset blocks_v1 --run smolvla_blocks_v1 \
      --steps 20000 --batch 64 --save-freq 2000
  modal run tools/train/modal_train.py::status --run smolvla_blocks_v1
  modal volume get vrs-data /runs/smolvla_blocks_v1/checkpoints/020000/pretrained_model <local dir>

Rerunning ``train`` with the same ``--run`` resumes from the newest checkpoint. Checkpoints and
the log are committed to the Volume every 10 minutes, so a stopped run loses at most that much.
Costs follow Modal's per-second prices (GPU + CPU + memory); every function has a timeout.
"""

from __future__ import annotations

import modal

LEROBOT = (
    "lerobot[training,smolvla] @ "
    "git+https://github.com/huggingface/lerobot@e581c1d5701f846788662079d4ac08835c96def9"
)
app = modal.App("vrs-train")
volume = modal.Volume.from_name("vrs-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(LEROBOT, "num2words")
    # an expired stored Hugging Face login breaks public downloads; this image has none anyway
    .env({"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "TOKENIZERS_PARALLELISM": "false"})
)
DATA = "/data"
POLICY_ARGS = {
    # smolvla_base expects camera1..camera3: wrist -> camera1, workspace -> camera2, camera3 empty
    "smolvla": [
        "--policy.path=lerobot/smolvla_base",
        "--policy.empty_cameras=1",
        '--rename_map={"observation.images.wrist": "observation.images.camera1", '
        '"observation.images.workspace": "observation.images.camera2"}',
    ],
    "act": ["--policy.type=act"],
}


def _commit_every(seconds: int, stop) -> None:
    while not stop.wait(seconds):
        volume.commit()


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8.0,
    memory=32768,
    volumes={DATA: volume},
    timeout=8 * 3600,
)
def train(
    dataset: str,
    run: str,
    steps: int = 20000,
    batch: int = 64,
    save_freq: int = 2000,
    policy: str = "smolvla",
    extra: str = "",
) -> dict:
    """Train ``policy`` on /data/datasets/<dataset>; output in /data/runs/<run>."""
    import json
    import subprocess
    import threading
    import time
    from pathlib import Path

    root = Path(DATA) / "datasets" / dataset
    out = Path(DATA) / "runs" / run
    if not (root / "meta" / "info.json").is_file():
        raise SystemExit(f"no LeRobot dataset at {root}")
    last = out / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    if last.is_file():
        cmd = ["lerobot-train", f"--config_path={last}", "--resume=true", f"--steps={steps}"]
    else:
        cmd = [
            "lerobot-train",
            *POLICY_ARGS[policy],
            f"--dataset.repo_id=local/{dataset}",
            f"--dataset.root={root}",
            "--policy.device=cuda",
            "--policy.push_to_hub=false",
            f"--batch_size={batch}",
            f"--steps={steps}",
            f"--save_freq={save_freq}",
            "--log_freq=100",
            "--num_workers=8",
            f"--output_dir={out}",
            f"--job_name={run}",
        ]
    cmd += extra.split()
    out.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd), flush=True)
    stop = threading.Event()
    committer = threading.Thread(target=_commit_every, args=(600, stop), daemon=True)
    committer.start()
    started = time.time()
    with open(Path(DATA) / "runs" / f"{run}.log", "a") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            if "step:" in line or "Error" in line or "Traceback" in line:
                print(line.rstrip(), flush=True)
        code = proc.wait()
    stop.set()
    summary = {
        "run": run,
        "exit": code,
        "minutes": round((time.time() - started) / 60, 1),
        "checkpoints": sorted(p.name for p in (out / "checkpoints").glob("[0-9]*")),
    }
    (Path(DATA) / "runs" / f"{run}.summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    print(summary, flush=True)
    return summary


@app.function(image=image, volumes={DATA: volume}, timeout=600)
def status(run: str) -> None:
    """Last training lines and checkpoints of a run (CPU only)."""
    from pathlib import Path

    volume.reload()
    log = Path(DATA) / "runs" / f"{run}.log"
    lines = log.read_text(errors="replace").splitlines() if log.is_file() else []
    for line in [x for x in lines if "step:" in x][-5:] + lines[-3:]:
        print(line)
    ckpts = sorted(p.name for p in (Path(DATA) / "runs" / run / "checkpoints").glob("[0-9]*"))
    print("checkpoints:", ckpts)


@app.function(image=image, cpu=16.0, memory=32768, volumes={DATA: volume}, timeout=12 * 3600)
def convert(raw: str, dataset: str, vcodec: str = "h264") -> None:
    """Convert raw episodes in /data/raw/<raw> into /data/datasets/<dataset> (CPU only)."""
    import subprocess
    from pathlib import Path

    src = Path(DATA) / "raw" / raw
    cmd = [
        "python",
        str(src / "lerobot_convert.py"),
        "--raw",
        str(src),
        "--root",
        str(Path(DATA) / "datasets" / dataset),
        "--repo-id",
        f"local/{dataset}",
    ]
    if vcodec:
        cmd += ["--vcodec", vcodec]
    subprocess.run(cmd, check=True)
    volume.commit()
