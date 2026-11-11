#!/usr/bin/env python3
# ruff: noqa: E501
"""Record pick-and-place demonstrations from the digital twin for imitation learning (ACT/SmolVLA).

The scripted expert (RoboTwin grasp/place + closed-loop alignment, the same executor as the
twin console) performs one randomly sampled single-step task per episode on a randomly
jittered layout. While it moves, a physics-step hook samples, at a fixed simulated rate
(default 15 Hz): the arm joint state (joints + gripper) and RGB images from the wrist camera
and the configurable workspace camera (configs/twin_cameras.json). Only successful episodes
are kept unless --keep-failures is given.

Output is a dependency-free raw format (convert it to a LeRobot dataset with
tools/twin/lerobot_convert.py, e.g. on Kaggle):
  <out>/episode_000000/meta.json        command, language, level, object, destination, success, ...
  <out>/episode_000000/states.npy       float32 [T, D]  (arm joints + gripper)
  <out>/episode_000000/actions.npy      float32 [T, D]  (next state = joint-position target)
  <out>/episode_000000/<camera>/000000.jpg ...

Run headless with the RoboTwin conda python (no viewer window):
  python tools/twin/twin_record.py --out data/twin_raw --episodes 50 --levels 1,2,3
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sorting_task as st
import twin_scene as ts

RAW_FORMAT_VERSION = 1
TEMPLATES = (
    "pick up the {obj} and put it in the {dest}",
    "place the {obj} into the {dest}",
    "move the {obj} to the {dest}",
    "put the {obj} in the {dest}",
)
PAD_TEMPLATES = (
    "put the {obj} on the {dest}",
    "move the {obj} onto the {dest}",
)


VI_TEMPLATES = (
    "gắp {obj} bỏ vào {dest}",
    "đặt {obj} vào {dest}",
    "chuyển {obj} sang {dest}",
    "cho {obj} vào {dest}",
)
VI_PAD_TEMPLATES = (
    "đặt {obj} lên {dest}",
    "chuyển {obj} ra {dest}",
)
LANGUAGES = ("en", "vi")

# Vietnamese names are built word by word: noun, then colour, then size ("khối trụ đỏ nhỏ").
VI_NOUNS = {
    "cube": "khối vuông",
    "cylinder": "khối trụ",
    "triangular prism": "khối lăng trụ tam giác",
    "tetrahedron": "khối tứ diện",
    "tray": "khay",
    "wide tray": "khay rộng",
    "deep bin": "thùng sâu",
    "staging pad": "ô trung chuyển",
    "table": "bàn",
}
VI_COLOURS = {
    "red": "đỏ",
    "green": "xanh lá",
    "blue": "xanh dương",
    "yellow": "vàng",
    "white": "trắng",
    "purple": "tím",
    "orange": "cam",
}
VI_SIZES = {"small": "nhỏ", "medium": "vừa", "large": "lớn"}


def vietnamese_name(name: str) -> str | None:
    """Vietnamese for an English object or container name, or None if a word is unknown."""
    words = name.split()
    size = VI_SIZES.get(words[0]) if words else None
    if size:
        words = words[1:]
    colour = VI_COLOURS.get(words[0]) if words else None
    if colour:
        words = words[1:]
    noun = VI_NOUNS.get(" ".join(words))
    if noun is None:
        return None
    return " ".join(part for part in (noun, colour, size) if part)


def instruction_for(
    obj: str, dest: str, kind: str, rng: random.Random, language: str = "en"
) -> str:
    """One command for moving obj to dest; falls back to English when a name has no translation."""
    if language == "vi":
        obj_vi, dest_vi = vietnamese_name(obj), vietnamese_name(dest)
        if obj_vi and dest_vi:
            template = rng.choice(VI_PAD_TEMPLATES if kind == "pad" else VI_TEMPLATES)
            return template.format(obj=obj_vi, dest=dest_vi)
    template = rng.choice(PAD_TEMPLATES if kind == "pad" else TEMPLATES)
    return template.format(obj=obj, dest=dest)


def command_language(text: str) -> str:
    return "vi" if any(ord(ch) > 127 for ch in text) else "en"


def sample_task(
    level: ts.Level, objects_xyz: dict[str, list[float]], table_top: float, rng: random.Random
) -> tuple[str, str] | None:
    """Random (object, destination) that the expert can attempt, or None if nothing is possible."""
    options = []
    for obj in level.objects:
        where = ts.location_of(level, objects_xyz[obj.name], table_top)
        source = None if where == "table" else level.container(where)
        if source is not None and not ts.graspable_in(obj, source):
            continue
        for dest in level.containers:
            if dest.name == where:
                continue
            if len(ts.occupants(level, dest, objects_xyz, table_top)) >= dest.capacity:
                continue
            options.append((obj.name, dest.name))
    return rng.choice(options) if options else None


def derive_actions(states: np.ndarray) -> np.ndarray:
    """Joint-position targets: the state at the next sample (last frame repeats itself)."""
    if len(states) == 0:
        return states.copy()
    return np.concatenate([states[1:], states[-1:]], axis=0)


@dataclass
class FrameRecorder:
    """Samples state/images from inside the physics loop at a fixed simulated rate."""

    grab: Callable[[], tuple[np.ndarray, dict[str, np.ndarray]]]
    step_dt: float
    fps: float
    active: bool = False
    sim_time: float = 0.0
    next_sample: float = 0.0
    states: list[np.ndarray] = field(default_factory=list)
    images: dict[str, list[np.ndarray]] = field(default_factory=dict)
    times: list[float] = field(default_factory=list)

    def attach(self, task: object) -> None:
        original = task._step_scene

        def stepped() -> None:
            original()
            self.on_step()

        task._step_scene = stepped

    def start(self) -> None:
        self.active, self.sim_time, self.next_sample = True, 0.0, 0.0
        self.states, self.images, self.times = [], {}, []
        self.sample()

    def stop(self) -> None:
        self.active = False

    def on_step(self) -> None:
        if not self.active:
            return
        self.sim_time += self.step_dt
        if self.sim_time + 1e-9 >= self.next_sample:
            self.sample()

    def sample(self) -> None:
        state, images = self.grab()
        self.states.append(np.asarray(state, np.float32))
        for name, image in images.items():
            self.images.setdefault(name, []).append(image)
        self.times.append(self.sim_time)
        self.next_sample += 1.0 / self.fps

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        states = np.stack(self.states) if self.states else np.zeros((0, 0), np.float32)
        return states, derive_actions(states), np.asarray(self.times, np.float32)


def write_episode(
    root: Path,
    index: int,
    states: np.ndarray,
    actions: np.ndarray,
    times: np.ndarray,
    images: dict[str, list[np.ndarray]],
    meta: dict,
    save_image: Callable[[Path, np.ndarray], None],
) -> Path:
    folder = root / f"episode_{index:06d}"
    folder.mkdir(parents=True, exist_ok=False)
    np.save(folder / "states.npy", states.astype(np.float32))
    np.save(folder / "actions.npy", actions.astype(np.float32))
    np.save(folder / "timestamps.npy", times.astype(np.float32))
    for camera, frames in images.items():
        (folder / camera).mkdir()
        for i, frame in enumerate(frames):
            save_image(folder / camera / f"{i:06d}.jpg", frame)
    meta = {
        **meta,
        "format": RAW_FORMAT_VERSION,
        "frames": len(states),
        "state_dim": int(states.shape[1]) if states.ndim == 2 else 0,
        "cameras": sorted(images),
    }
    (folder / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return folder


def next_episode_index(root: Path) -> int:
    existing = sorted(p for p in root.glob("episode_*") if p.is_dir())
    return int(existing[-1].name.split("_")[1]) + 1 if existing else 0


def cv2_saver(quality: int = 92) -> Callable[[Path, np.ndarray], None]:
    import cv2

    def save(path: Path, rgb: np.ndarray) -> None:
        cv2.imwrite(
            str(path), np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

    return save


def scene_state(twin, scene: st.Scene) -> tuple[set[str], set[str]]:
    """Objects still on the table and containers with free room, from the simulator."""
    xyz = twin.objects_xyz()
    level = twin.level
    available = {
        o.name
        for o in scene.objects
        if ts.location_of(level, xyz[o.name], twin.table_top) == "table"
    }
    open_containers = {
        c.name
        for c in level.containers
        if len(ts.occupants(level, c, xyz, twin.table_top)) < c.capacity
    }
    return available, open_containers


def record_sorting(twin, args, rng, languages, size, fps, save, index) -> tuple[int, int]:
    """Procedural sorting scenes: rebuild per layout, several commands per layout."""
    kept = attempted = 0
    while kept < args.episodes and attempted < args.episodes * 3:
        scene = st.sample_scene(rng)
        twin.custom_level = st.to_level(scene)
        print(
            f"Building sorting scene ({len(scene.objects)} objects, {len(scene.containers)} containers)...",
            flush=True,
        )
        twin.setup()
        recorder = FrameRecorder(lambda: twin.record_frame(size), twin.step_dt(), fps)
        recorder.attach(twin.task)
        spawns = twin.reset_layout(rng)
        for _ in range(args.episodes_per_layout):
            if kept >= args.episodes:
                break
            available, open_containers = scene_state(twin, scene)
            command = st.make_command(
                scene, rng, rng.choice(languages), args.class_share, available, open_containers
            )
            if command is None:
                break
            obj = rng.choice(command.targets)
            attempted += 1
            recorder.start()
            outcome = twin.pick_and_place(obj, command.destination, echo=lambda _m: None)
            recorder.stop()
            states, actions, times = recorder.arrays()
            print(
                f"[{attempted}] {command.kind}: {command.text} -> {'ok' if outcome.ok else 'FAILED'} ({len(states)} frames)",
                flush=True,
            )
            meta = {
                "task": command.text,
                "language": command.language,
                "command_kind": command.kind,
                "object_reference": dict(command.object_reference),
                "container_reference": list(command.container_reference),
                "targets": list(command.targets),
                "object": obj,
                "destination": command.destination,
                "scene": {
                    "objects": [
                        {"name": o.name, "shape": o.shape, "colour": o.colour, "size": o.size}
                        for o in scene.objects
                    ],
                    "containers": [
                        {
                            "name": c.name,
                            "colour": c.colour,
                            "outline": c.outline,
                            "depth": c.depth,
                            "position": c.position,
                        }
                        for c in scene.containers
                    ],
                },
                "success": outcome.ok,
                "summary": outcome.summary,
                "fps": fps,
                "seed": args.seed,
                "spawns": {k: list(v) for k, v in spawns.items()},
                "image_size": list(size),
                "source": "robotwin_twin_scripted_expert",
            }
            if outcome.ok or args.keep_failures:
                write_episode(args.out, index, states, actions, times, recorder.images, meta, save)
                index += 1
                kept += outcome.ok
            if not outcome.ok:
                break
    return kept, attempted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=Path.home() / "worktrees/Panthera-HT_RoboTwin/RoboTwin",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--levels", default="1,2,3,4,5,6")
    parser.add_argument(
        "--episodes-per-layout",
        type=int,
        default=3,
        help="tasks before the layout is re-randomised",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fps", type=float, default=0.0, help="0 = configs/twin_cameras.json record.fps"
    )
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument(
        "--task",
        choices=("sorting", "levels"),
        default="sorting",
        help="sorting: procedural scenes from docs/task-spec.md; levels: the fixed scene levels",
    )
    parser.add_argument(
        "--class-share",
        type=float,
        default=0.3,
        help="sorting task: fraction of class commands (any member of the class is correct)",
    )
    parser.add_argument(
        "--languages",
        default="en,vi",
        help="command languages, sampled uniformly per episode (en, vi)",
    )
    args = parser.parse_args(argv)
    languages = [v.strip() for v in args.languages.split(",") if v.strip()]
    if not languages or any(v not in LANGUAGES for v in languages):
        parser.error(f"languages must be a subset of {','.join(LANGUAGES)}")
    levels = [int(v) for v in args.levels.split(",") if v.strip()]
    if any(v not in ts.LEVELS for v in levels) or args.episodes < 1:
        parser.error("levels must be in 1..7 and episodes >= 1")

    from twin_console import Twin
    from twin_geometry import CAMERA_CONFIG

    record_cfg = {"width": 320, "height": 240, "fps": 15}
    if CAMERA_CONFIG.is_file():
        record_cfg.update(json.loads(CAMERA_CONFIG.read_text()).get("record", {}))
    fps = args.fps or float(record_cfg["fps"])
    size = (int(record_cfg["width"]), int(record_cfg["height"]))
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for helper in ("lerobot_convert.py", "policy_bridge.py"):  # travel with the dataset (Kaggle)
        shutil.copy2(Path(__file__).resolve().parent / helper, args.out / helper)
    save = cv2_saver()
    index = next_episode_index(args.out)
    kept = attempted = 0
    started = time.perf_counter()
    twin = Twin(args.robotwin_root.resolve(), args.seed, 0, "free", levels[0], headless=True)
    recorder: FrameRecorder | None = None
    current_level = None
    if args.task == "sorting":
        kept, attempted = record_sorting(twin, args, rng, languages, size, fps, save, index)
        elapsed = time.perf_counter() - started
        print(
            f"RECORD_DONE kept={kept} attempted={attempted} out={args.out} elapsed_s={elapsed:.0f}"
        )
        return 0 if kept else 4
    while kept < args.episodes and attempted < args.episodes * 3:
        level = rng.choice(levels)
        if level != current_level:
            twin.level_number = level
            print(f"Building level {level} (headless)...", flush=True)
            twin.setup()
            current_level = level
            recorder = FrameRecorder(lambda: twin.record_frame(size), twin.step_dt(), fps)
            recorder.attach(twin.task)
        spawns = twin.reset_layout(rng)
        for _ in range(args.episodes_per_layout):
            if kept >= args.episodes:
                break
            task = sample_task(twin.level, twin.objects_xyz(), twin.table_top, rng)
            if task is None:
                break
            obj, dest = task
            kind = twin.level.container(dest).kind
            text = instruction_for(obj, dest, kind, rng, rng.choice(languages))
            attempted += 1
            recorder.start()
            outcome = twin.pick_and_place(obj, dest, echo=lambda _m: None)
            recorder.stop()
            states, actions, times = recorder.arrays()
            status = "ok" if outcome.ok else "FAILED"
            print(
                f"[{attempted}] level {level}: {text} -> {status} ({len(states)} frames) {'' if outcome.ok else outcome.summary}",
                flush=True,
            )
            if not outcome.ok:
                twin.reset_layout(rng)
                if not args.keep_failures:
                    break
            meta = {
                "task": text,
                "language": command_language(text),
                "level": level,
                "object": obj,
                "destination": dest,
                "success": outcome.ok,
                "summary": outcome.summary,
                "fps": fps,
                "seed": args.seed,
                "spawns": {k: list(v) for k, v in spawns.items()},
                "image_size": list(size),
                "source": "robotwin_twin_scripted_expert",
            }
            if outcome.ok or args.keep_failures:
                write_episode(args.out, index, states, actions, times, recorder.images, meta, save)
                index += 1
                kept += outcome.ok
    elapsed = time.perf_counter() - started
    print(f"RECORD_DONE kept={kept} attempted={attempted} out={args.out} elapsed_s={elapsed:.0f}")
    return 0 if kept else 4


if __name__ == "__main__":
    raise SystemExit(main())
