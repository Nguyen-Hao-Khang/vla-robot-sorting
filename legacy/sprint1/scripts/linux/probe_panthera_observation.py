#!/usr/bin/env python3
"""Read Panthera cameras and actual qpos without executing a task trajectory.

The observation gate intentionally bypasses motion-planner initialization. Planner
bring-up is a separate native gate; coupling it here previously allowed a CuRobo/MPLib
warmup stall to hide camera/proprioception evidence.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="move_pillbottle_pad")
    parser.add_argument("--task-config", default="move_pillbottle_pad_panthera_single")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traceback-after-s", type=float, default=120.0)
    parser.add_argument(
        "--capture-native-depth-dir",
        type=Path,
        default=None,
        help="Opt-in read-only native Color/Position capture after the one get_obs call.",
    )
    return parser.parse_args()


def _progress(label: str, started_at: float) -> None:
    elapsed = time.monotonic() - started_at
    print(f"[observation-probe] {label} elapsed_s={elapsed:.3f}", flush=True)


class _ObservationOnlyPlanner:
    """Minimum gripper interpolation API needed during RoboTwin scene setup."""

    planner_id = "observation-only-no-motion-planner"

    @staticmethod
    def plan_grippers(now_val: float, target_val: float) -> dict[str, object]:
        import numpy as np

        num_step = 200
        values = np.linspace(now_val, target_val, num_step)
        return {
            "num_step": num_step,
            "per_step": (target_val - now_val) / num_step,
            "result": values,
        }


def _install_observation_only_planner(robot: object, scene: object | None = None) -> None:
    del scene
    planner = _ObservationOnlyPlanner()
    robot.communication_flag = False
    robot.left_planner = planner
    robot.right_planner = planner
    robot.left_mplib_planner = planner
    robot.right_mplib_planner = planner


def _select_depth_camera_handles(cameras: object) -> dict[str, object]:
    """Return only the source-audited single-arm head/wrist SAPIEN handles."""

    if not getattr(cameras, "single_arm_mode", False):
        raise RuntimeError("native depth probe requires single-arm camera configuration")
    if not getattr(cameras, "collect_head_camera", False):
        raise RuntimeError("native depth probe requires collected head camera")
    if not getattr(cameras, "collect_wrist_camera", False):
        raise RuntimeError("native depth probe requires collected wrist camera")
    names = getattr(cameras, "static_camera_name", ())
    handles = getattr(cameras, "static_camera_list", ())
    if len(names) != len(handles):
        raise RuntimeError("static camera name/handle lengths differ")
    head = [camera for name, camera in zip(names, handles, strict=True) if name == "head_camera"]
    if len(head) != 1 or head[0] is None:
        raise RuntimeError("exactly one head_camera SAPIEN handle is required")
    wrist = getattr(cameras, "wrist_camera", None)
    if wrist is None:
        raise RuntimeError("wrist_camera SAPIEN handle is missing")
    return {"head_camera": head[0], "wrist_camera": wrist}


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    faulthandler.enable(all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    if args.traceback_after_s > 0:
        faulthandler.dump_traceback_later(args.traceback_after_s, repeat=True)

    robotwin_root = args.robotwin_root.resolve()
    project_root = args.project_root.resolve()
    if not robotwin_root.is_dir():
        raise RuntimeError(f"RoboTwin root does not exist: {robotwin_root}")
    if not (project_root / "src").is_dir():
        raise RuntimeError(f"Project src does not exist: {project_root / 'src'}")

    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(project_root / "src"))
    _progress("paths-ready", started_at)

    from envs.robot.robot import Robot
    from script.collect_data import prepare_task_and_args

    from vla_robot_sorting.simulation.panthera_observation import PantheraObservationMapper
    from vla_robot_sorting.simulation.render_profile import (
        FAST_PREVIEW_RENDER_PROFILE,
        apply_fast_preview_render_profile,
    )

    original_set_planner = Robot.set_planner
    Robot.set_planner = _install_observation_only_planner

    initialized = False
    task = None
    try:
        _progress("prepare-task:start", started_at)
        task, task_args = prepare_task_and_args(args.task, args.task_config)
        _progress("prepare-task:done", started_at)

        task_args["render_freq"] = 0
        task_args["save_data"] = False
        task_args["collect_data"] = False
        task_args["domain_randomization"]["cluttered_table"] = False
        task_args["domain_randomization"]["random_background"] = False
        task_args["domain_randomization"]["random_light"] = False
        apply_fast_preview_render_profile(task_args)
        _progress(
            f"render-profile:fast-preview settings={FAST_PREVIEW_RENDER_PROFILE}",
            started_at,
        )

        _progress("setup-demo:start:planner-bypassed", started_at)
        task.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
        initialized = True
        _progress("setup-demo:done", started_at)

        cameras = task.cameras
        original_update_picture = cameras.update_picture
        refreshed_camera_ids: list[str] = []

        def instrumented_update_picture() -> None:
            if cameras.collect_wrist_camera:
                if cameras.single_arm_mode:
                    _progress("camera:wrist:take-picture:start", started_at)
                    cameras.wrist_camera.take_picture()
                    refreshed_camera_ids.append("wrist_camera")
                    _progress("camera:wrist:take-picture:done", started_at)
                else:
                    _progress("camera:left-wrist:take-picture:start", started_at)
                    cameras.left_camera.take_picture()
                    _progress("camera:left-wrist:take-picture:done", started_at)
                    _progress("camera:right-wrist:take-picture:start", started_at)
                    cameras.right_camera.take_picture()
                    _progress("camera:right-wrist:take-picture:done", started_at)

            for camera, camera_name in zip(
                cameras.static_camera_list,
                cameras.static_camera_name,
                strict=True,
            ):
                _progress(f"camera:{camera_name}:take-picture:start", started_at)
                camera.take_picture()
                if camera_name == "head_camera":
                    refreshed_camera_ids.append("head_camera")
                _progress(f"camera:{camera_name}:take-picture:done", started_at)

        cameras.update_picture = instrumented_update_picture
        try:
            timestamp_s = time.monotonic()
            _progress("get-obs:start", started_at)
            vendor_observation = task.get_obs()
        finally:
            cameras.update_picture = original_update_picture
        _progress("get-obs:done", started_at)
        native_depth_manifest_records: list[dict[str, object]] = []
        if args.capture_native_depth_dir is not None:
            # Exactly one synchronous get_obs() refresh: no extra physics steps.
            # Source evidence is a caller-observed refresh, not a sensor timestamp.
            from vla_robot_sorting.perception.native_depth_capture import (
                capture_sapien_depth_frame,
            )

            try:
                handles = _select_depth_camera_handles(cameras)
                for camera_id, handle in handles.items():
                    observed_once = refreshed_camera_ids.count(camera_id) == 1
                    manifest_path = capture_sapien_depth_frame(
                        handle,
                        camera_id=camera_id,
                        frame_index=0,
                        frame_association_source="probe.get_obs single-frame camera refresh",
                        output_dir=args.capture_native_depth_dir,
                        frame_index_verified=observed_once,
                        scene_revision=None,
                    )
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    native_depth_manifest_records.append({
                        "camera_id": camera_id,
                        "status": manifest.get("status"),
                        "manifest_path": str(manifest_path),
                        "refresh_count": refreshed_camera_ids.count(camera_id),
                    })
            except Exception as exc:
                native_depth_manifest_records.append({
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

        actual_state = task.robot.get_left_arm_real_jointState()
        commanded_state = task.robot.get_left_arm_jointState()

        mapped = PantheraObservationMapper().from_robotwin(
            vendor_observation,
            actual_state,
            sequence_id=0,
            timestamp_s=timestamp_s,
            scene_revision=f"probe-seed-{args.seed}",
            transform_revision="robotwin-static-v1",
        )


        payload = {
            "schema": "panthera-observation-probe-v2",
            "task": args.task,
            "task_config": args.task_config,
            "seed": args.seed,
            "planner_initialization": "bypassed-observation-only",
            "render_profile": dict(FAST_PREVIEW_RENDER_PROFILE),
            "joint_names": list(mapped.joint_state.joint_names),
            "actual_joint_positions_rad": list(mapped.joint_state.positions),
            "actual_gripper_normalized": mapped.gripper_state.normalized_position,
            "vendor_drive_target_vector": [float(value) for value in commanded_state],
            "proprioception_source": mapped.metadata["proprioception_source"],
            "cameras": [
                {
                    "camera_id": frame.camera_id,
                    "role": frame.role.value,
                    "frame_id": frame.frame_id,
                    "width_px": frame.width_px,
                    "height_px": frame.height_px,
                }
                for frame in mapped.cameras
            ],
            "single_arm_setup": getattr(task, "single_arm_setup", None),
            "native_depth_capture": {
                "requested": args.capture_native_depth_dir is not None,
                "results": native_depth_manifest_records,
                "all_cameras_captured": (
                    args.capture_native_depth_dir is not None
                    and len(native_depth_manifest_records) == 2
                    and all(
                        item["status"] == "capture_written"
                        for item in native_depth_manifest_records
                    )
                ),
                "frame_index": 0 if args.capture_native_depth_dir is not None else None,
                "timestamp_source": None,
                "controller_eligible": False,
            },
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _progress("payload-written", started_at)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        if (
            args.capture_native_depth_dir is not None
            and not payload["native_depth_capture"]["all_cameras_captured"]
        ):
            return 6
        return 0
    finally:
        Robot.set_planner = original_set_planner
        faulthandler.cancel_dump_traceback_later()
        if initialized and task is not None:
            _progress("close-env:start", started_at)
            task.close_env()
            _progress("close-env:done", started_at)


if __name__ == "__main__":
    raise SystemExit(main())
