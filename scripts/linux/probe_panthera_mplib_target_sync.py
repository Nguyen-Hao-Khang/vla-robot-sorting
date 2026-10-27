#!/usr/bin/env python3
"""Native preflight for target-aware MPlib world injection and pose resync."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from time import perf_counter
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="move_pillbottle_pad")
    parser.add_argument("--task-config", default="move_pillbottle_pad_panthera_single")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--translation-m", type=float, default=0.025)
    return parser.parse_args()


def load_helpers(project_root: Path) -> Any:
    path = project_root / "scripts/linux/probe_panthera_grasp_recovery.py"
    spec = importlib.util.spec_from_file_location("b57_target_sync_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pose_payload(value: Any) -> dict[str, list[float]]:
    return {
        "position_xyz_m": [float(item) for item in value.p],
        "orientation_wxyz": [float(item) for item in value.q],
    }


def fcl_pose_payload(planning_world: Any, target_name: str) -> dict[str, list[float]]:
    obj = planning_world.get_object(target_name)
    if obj is None:
        raise RuntimeError(f"MPlib target object {target_name!r} missing after injection")
    return pose_payload(obj.pose)


def main() -> int:
    args = parse_args()
    robotwin_root = args.robotwin_root.resolve()
    project_root = args.project_root.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(project_root / "src"))

    from script.collect_data import prepare_task_and_args

    from vla_robot_sorting.simulation.render_profile import (
        apply_fast_preview_render_profile,
    )

    helpers = load_helpers(project_root)
    payload: dict[str, Any] = {
        "schema": "panthera-native-mplib-target-sync-v1",
        "task": args.task,
        "task_config": args.task_config,
        "seed": args.seed,
        "passed": False,
    }
    task = None
    try:
        task, task_args = prepare_task_and_args(args.task, args.task_config)
        task_args["render_freq"] = 0
        task_args["save_data"] = False
        task_args["collect_data"] = False
        task_args["domain_randomization"]["cluttered_table"] = False
        task_args["domain_randomization"]["random_background"] = False
        task_args["domain_randomization"]["random_light"] = False
        apply_fast_preview_render_profile(task_args)

        started = perf_counter()
        task.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
        payload["setup_duration_s"] = perf_counter() - started

        world = task.robot.left_mplib_planner.planner.planning_world
        actor_before = pose_payload(task.pillbottle.actor.pose)

        initial_sync = helpers.sync_scaled_target_into_mplib_world(task)
        target_name = str(initial_sync["target_name"])
        fcl_before = fcl_pose_payload(world, target_name)
        initial_collisions = world.check_collision_between(
            task.robot.left_entity,
            task.pillbottle.actor,
        )

        perturb = helpers.perturb_target(task, args.translation_m)
        actor_after = pose_payload(task.pillbottle.actor.pose)
        moved_sync = helpers.sync_scaled_target_into_mplib_world(task)
        fcl_after = fcl_pose_payload(world, target_name)
        moved_collisions = world.check_collision_between(
            task.robot.left_entity,
            task.pillbottle.actor,
        )

        fcl_delta = sum(
            (after - before) ** 2
            for before, after in zip(
                fcl_before["position_xyz_m"],
                fcl_after["position_xyz_m"],
                strict=True,
            )
        ) ** 0.5
        actor_delta = sum(
            (after - before) ** 2
            for before, after in zip(
                actor_before["position_xyz_m"],
                actor_after["position_xyz_m"],
                strict=True,
            )
        ) ** 0.5

        passed = (
            initial_sync.get("target_present") is True
            and moved_sync.get("target_present") is True
            and initial_sync.get("strategy") == "scaled_target_fcl_injection"
            and moved_sync.get("strategy") == "scaled_target_fcl_injection"
            and int(initial_sync.get("scaled_mesh_shape_count", 0)) >= 1
            and int(initial_sync.get("non_unit_scale_shape_count", 0)) >= 1
            and abs(actor_delta - abs(args.translation_m)) <= 1e-5
            and abs(fcl_delta - actor_delta) <= 1e-5
        )
        payload.update(
            {
                "initial_sync": initial_sync,
                "moved_sync": moved_sync,
                "actor_pose_before": actor_before,
                "actor_pose_after": actor_after,
                "fcl_pose_before": fcl_before,
                "fcl_pose_after": fcl_after,
                "actor_translation_m": actor_delta,
                "fcl_translation_m": fcl_delta,
                "perturbation": perturb,
                "initial_robot_target_collision_count": len(initial_collisions),
                "moved_robot_target_collision_count": len(moved_collisions),
                "passed": passed,
            }
        )
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if task is not None:
            with suppress(Exception):
                task.close_env()

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
