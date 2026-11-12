#!/usr/bin/env python3
# ruff: noqa: E501
"""Convert raw twin episodes (twin_record.py output) into a LeRobot dataset for ACT / SmolVLA.

Runs in a LeRobot environment (e.g. a Kaggle notebook), not in the RoboTwin env. Tested with
lerobot 0.4.4 and 0.7.1 (commit e581c1d). Features written:
  observation.state                 float32 [D]  arm joints + gripper
  action                            float32 [D]  next joint-position target
  observation.images.<camera>       video (or image with --images) per recorded camera
plus the per-frame language task, so the same dataset trains ACT (ignores language) and
SmolVLA (uses it). Example:
  python lerobot_convert.py --raw twin_raw --root /kaggle/working/panthera_twin --repo-id local/panthera_twin
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np


def load_episode(folder: Path) -> tuple[dict, np.ndarray, np.ndarray, dict[str, list[Path]]]:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    states = np.load(folder / "states.npy").astype(np.float32)
    actions = np.load(folder / "actions.npy").astype(np.float32)
    frames = {cam: sorted((folder / cam).glob("*.jpg")) for cam in meta.get("cameras", [])}
    for cam, files in frames.items():
        if len(files) != len(states):
            raise ValueError(
                f"{folder.name}: camera {cam} has {len(files)} frames for {len(states)} states"
            )
    return meta, states, actions, frames


def episode_folders(raw: Path, include_failures: bool) -> list[Path]:
    folders = []
    for folder in sorted(raw.glob("episode_*")):
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if meta.get("success") or include_failures:
            folders.append(folder)
    return folders


def build_features(
    state_dim: int, cameras: list[str], image_hw: tuple[int, int], use_videos: bool
) -> dict:
    names = [f"joint_{i}" for i in range(state_dim - 1)] + ["gripper"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": names},
        "action": {"dtype": "float32", "shape": (state_dim,), "names": names},
    }
    h, w = image_hw
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw", type=Path, required=True, help="folder with episode_* from twin_record.py"
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="output dataset folder (must not exist)"
    )
    parser.add_argument("--repo-id", default="local/panthera_twin")
    parser.add_argument("--robot-type", default="panthera_ht_sim")
    parser.add_argument("--images", action="store_true", help="store PNG frames instead of videos")
    parser.add_argument("--vcodec", default="", help="video codec, e.g. h264 (default: lerobot's)")
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args(argv)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    folders = episode_folders(args.raw, args.include_failures)
    if args.max_episodes:
        folders = folders[: args.max_episodes]
    if not folders:
        print("NO_EPISODES: nothing to convert.")
        return 4
    meta0, states0, _, frames0 = load_episode(folders[0])
    cameras = sorted(frames0)
    image_hw = read_rgb(frames0[cameras[0]][0]).shape[:2] if cameras else (0, 0)
    fps = round(float(meta0["fps"]))
    features = build_features(states0.shape[1], cameras, image_hw, not args.images)
    kwargs = {
        "repo_id": args.repo_id,
        "fps": fps,
        "features": features,
        "root": args.root,
        "robot_type": args.robot_type,
        "use_videos": not args.images,
    }
    if args.vcodec and "vcodec" in inspect.signature(LeRobotDataset.create).parameters:
        kwargs["vcodec"] = args.vcodec
    dataset = LeRobotDataset.create(**kwargs)
    total = 0
    for folder in folders:
        meta, states, actions, frames = load_episode(folder)
        if (
            sorted(frames) != cameras
            or states.shape[1] != states0.shape[1]
            or round(float(meta["fps"])) != fps
        ):
            print(f"SKIP {folder.name}: cameras/state size/fps differ from the first episode")
            continue
        for t in range(len(states)):
            frame = {"observation.state": states[t], "action": actions[t], "task": meta["task"]}
            for cam in cameras:
                frame[f"observation.images.{cam}"] = read_rgb(frames[cam][t])
            dataset.add_frame(frame)
        dataset.save_episode()
        total += len(states)
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    print(
        f"CONVERT_DONE episodes={dataset.meta.total_episodes} frames={total} root={args.root} cameras={','.join(cameras)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
