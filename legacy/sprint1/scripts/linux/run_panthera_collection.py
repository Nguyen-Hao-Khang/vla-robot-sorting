#!/usr/bin/env python3
"""Run one RoboTwin collection episode with project-owned runtime overrides."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--task", default="move_pillbottle_pad")
    parser.add_argument("--task-config", default="move_pillbottle_pad_panthera_single")
    parser.add_argument("--episode-num", type=int, default=1)
    parser.add_argument("--save-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robotwin_root = args.robotwin_root.resolve()
    project_root = args.project_root.resolve()

    if not robotwin_root.is_dir():
        raise RuntimeError(f"RoboTwin root does not exist: {robotwin_root}")
    if not (project_root / "src").is_dir():
        raise RuntimeError(f"Project src does not exist: {project_root / 'src'}")
    if args.episode_num < 1:
        raise ValueError("episode-num must be at least 1")

    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(project_root / "src"))

    from script.collect_data import prepare_task_and_args, run

    from vla_robot_sorting.simulation.render_profile import (
        FAST_PREVIEW_RENDER_PROFILE,
        apply_fast_preview_render_profile,
    )

    task, task_args = prepare_task_and_args(args.task, args.task_config)
    task_args["episode_num"] = args.episode_num
    task_args["data_root"] = os.path.abspath(args.save_path)
    task_args["save_path"] = os.path.join(
        task_args["data_root"],
        str(task_args["task_name"]),
        task_args["task_config"],
    )
    apply_fast_preview_render_profile(task_args)

    print(
        "[collection-wrapper] render_profile=fast_preview "
        f"settings={FAST_PREVIEW_RENDER_PROFILE}",
        flush=True,
    )
    run(task, task_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
