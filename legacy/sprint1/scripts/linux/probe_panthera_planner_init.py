#!/usr/bin/env python3
"""Diagnose Panthera vendor planner initialization without executing motion.

The probe stops immediately after RoboTwin's load_robot() returns. It records
fine-grained CuRobo/MPLib constructor and MotionGen warmup progress so a native
timeout identifies the exact blocking phase instead of hanging the whole S1.1 run.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any


class _PlannerInitComplete(RuntimeError):
    """Internal sentinel used to stop task setup after planner initialization."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="move_pillbottle_pad")
    parser.add_argument("--task-config", default="move_pillbottle_pad_panthera_single")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traceback-after-s", type=float, default=90.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    events: list[dict[str, Any]] = []

    def event(label: str, **details: Any) -> None:
        payload = {
            "label": label,
            "elapsed_s": round(time.monotonic() - started_at, 6),
            **details,
        }
        events.append(payload)
        print(f"[planner-init-probe] {json.dumps(payload, sort_keys=True)}", flush=True)

    faulthandler.enable(all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    if args.traceback_after_s > 0:
        faulthandler.dump_traceback_later(args.traceback_after_s, repeat=True)

    robotwin_root = args.robotwin_root.resolve()
    project_root = args.project_root.resolve()
    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(project_root / "src"))
    event("paths-ready")

    from curobo.wrap.reacher.motion_gen import MotionGen
    from envs.robot.planner import CuroboPlanner, MplibPlanner
    from script.collect_data import prepare_task_and_args

    originals: list[tuple[object, str, object]] = []


    def patch_method(owner: object, name: str, replacement: object) -> None:
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    original_curobo_init = CuroboPlanner.__init__
    original_mplib_init = MplibPlanner.__init__
    original_warmup = MotionGen.warmup

    def curobo_init(self: object, *init_args: Any, **init_kwargs: Any) -> None:
        event("CuroboPlanner.__init__:start")
        original_curobo_init(self, *init_args, **init_kwargs)
        event("CuroboPlanner.__init__:done")

    def mplib_init(self: object, *init_args: Any, **init_kwargs: Any) -> None:
        event("MplibPlanner.__init__:start")
        original_mplib_init(self, *init_args, **init_kwargs)
        event("MplibPlanner.__init__:done")

    def warmup(self: object, *warmup_args: Any, **warmup_kwargs: Any) -> Any:
        event(
            "MotionGen.warmup:start",
            args_repr=repr(warmup_args),
            kwargs_repr=repr(warmup_kwargs),
        )
        result = original_warmup(self, *warmup_args, **warmup_kwargs)
        event("MotionGen.warmup:done")
        return result

    patch_method(CuroboPlanner, "__init__", curobo_init)
    patch_method(MplibPlanner, "__init__", mplib_init)
    patch_method(MotionGen, "warmup", warmup)

    task = None
    try:
        event("prepare-task:start")
        task, task_args = prepare_task_and_args(args.task, args.task_config)
        event("prepare-task:done")

        task_args["render_freq"] = 0
        task_args["save_data"] = False
        task_args["collect_data"] = False
        task_args["domain_randomization"]["cluttered_table"] = False
        task_args["domain_randomization"]["random_background"] = False
        task_args["domain_randomization"]["random_light"] = False

        original_load_robot = task.load_robot

        def load_robot_then_stop(**kwargs: Any) -> None:
            event("Base_Task.load_robot:start")
            original_load_robot(**kwargs)
            event("Base_Task.load_robot:done")
            raise _PlannerInitComplete

        task.load_robot = load_robot_then_stop
        event("setup-demo:start")
        try:
            task.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
        except _PlannerInitComplete:
            event("planner-init-complete")
        else:
            raise RuntimeError("planner-init probe did not stop after load_robot")

        payload = {
            "schema": "panthera-planner-init-probe-v1",
            "task": args.task,
            "task_config": args.task_config,
            "seed": args.seed,
            "status": "PASS",
            "events": events,
            "duration_s": round(time.monotonic() - started_at, 6),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return 0

    finally:
        faulthandler.cancel_dump_traceback_later()
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


if __name__ == "__main__":
    raise SystemExit(main())
