#!/usr/bin/env python3
"""Open a RoboTwin/Panthera-HT scene in the SAPIEN viewer and keep it open (no task motion).

Step 1 of the interactive digital twin: proves that the GUI window works on this
machine and lists every entity in the scene (name + position), which is what the
command layer will ground against. The robot only settles to its home pose during
upstream setup; no planner or task motion runs. Close the window or press Ctrl+C to exit.
Run with the RoboTwin conda python:
  ~/miniforge3/bin/conda run --no-capture-output -n RoboTwin python scripts/linux/twin_view.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=Path.home() / "worktrees/Panthera-HT_RoboTwin/RoboTwin",
    )
    parser.add_argument("--task", default="move_pillbottle_pad")
    parser.add_argument("--task-config", default="move_pillbottle_pad_panthera_single")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-freq", type=int, default=1)
    parser.add_argument("--fps", type=float, default=20.0, help="viewer redraw cap (frames/s)")
    parser.add_argument(
        "--clutter", action="store_true", help="keep RoboTwin random clutter objects"
    )
    parser.add_argument(
        "--ray-tracing",
        action="store_true",
        help="keep upstream ray-traced viewer shader (slow on a GTX 1660 Ti)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="auto-close after N seconds (0 = until closed)",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.render_freq <= 60:
        parser.error("--render-freq must be 1..60")
    return args


def scene_entities(task: object) -> list[dict]:
    rows = []
    for entity in task.scene.get_entities():
        pose = entity.get_pose()
        rows.append({"name": entity.get_name(), "xyz_m": [round(float(v), 3) for v in pose.p]})
    return rows


def force_raster_shaders() -> list:
    """Make upstream setup create the viewer/cameras with SAPIEN's raster shader.

    Display-only: the ray-traced viewer shader is far too slow on a 6 GB GTX 1660 Ti
    and leaves the window "not responding". Returns callables that restore the API.
    """
    import sapien

    render = sapien.render
    restore = []
    for name in ("set_viewer_shader_dir", "set_camera_shader_dir"):
        original = getattr(render, name, None)
        if original is None:
            continue
        original("default")

        def forced(_shader: str, _original=original) -> None:
            _original("default")

        setattr(render, name, forced)
        restore.append(lambda _name=name, _original=original: setattr(render, _name, _original))
    return restore


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.environ.get("DISPLAY"):
        print("NO_DISPLAY: run inside a graphical session (DISPLAY is empty).")
        return 3
    robotwin = args.robotwin_root.resolve()
    os.chdir(robotwin)
    sys.path.insert(0, str(robotwin))
    sys.path.insert(0, str(PROJECT / "src"))
    from script.collect_data import prepare_task_and_args

    task, task_args = prepare_task_and_args(args.task, args.task_config)
    task_args.update(
        render_freq=args.render_freq, save_data=False, collect_data=False, eval_video_log=False
    )
    randomization = task_args.get("domain_randomization") or {}
    if not args.clutter:
        for key in ("cluttered_table", "random_background", "random_light"):
            if key in randomization:
                randomization[key] = False
    restore = [] if args.ray_tracing else force_raster_shaders()
    started = time.perf_counter()
    try:
        task.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
    finally:
        for undo in restore:
            undo()
    viewer = getattr(task, "viewer", None)
    if viewer is None:
        print("VIEWER_NOT_CREATED")
        return 4
    print(
        f"VIEWER_OPEN task={args.task} seed={args.seed} setup_s={time.perf_counter() - started:.1f}"
        f" shader={'ray_tracing' if args.ray_tracing else 'raster'} clutter={args.clutter}"
    )
    print("SCENE_ENTITIES=" + json.dumps(scene_entities(task), ensure_ascii=False))
    period = 1.0 / max(args.fps, 1.0)
    frames, window_start, slowest = 0, time.perf_counter(), 0.0
    try:
        while not getattr(viewer, "closed", False):
            tick = time.perf_counter()
            viewer.render()
            spent = time.perf_counter() - tick
            slowest = max(slowest, spent)
            frames += 1
            if tick - window_start >= 5.0:
                fps = frames / (tick - window_start)
                print(f"VIEWER_FPS={fps:.1f} slowest_frame_s={slowest:.3f}", flush=True)
                frames, window_start, slowest = 0, tick, 0.0
            if args.max_seconds and tick - started > args.max_seconds:
                break
            time.sleep(max(0.0, period - spent))
    except KeyboardInterrupt:
        pass
    print("VIEWER_CLOSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
