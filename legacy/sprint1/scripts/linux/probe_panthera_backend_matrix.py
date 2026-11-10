#!/usr/bin/env python3
"""Native repeated backend/coexistence probe for S1.2 B1/B2/B3.

The probe never commands robot joints. It plans from measured qpos and, for B2,
alternates real head/wrist camera observations with planning in the same process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from math import cos, pi, sin
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
    parser.add_argument("--curobo-runs", type=int, default=10)
    parser.add_argument("--coexistence-runs", type=int, default=30)
    parser.add_argument("--mplib-runs", type=int, default=10)
    parser.add_argument("--translation-m", type=float, default=0.01)
    parser.add_argument("--rotation-deg", type=float, default=5.0)
    return parser.parse_args()


def quaternion_multiply_wxyz(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def xyzw_from_wxyz(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    qw, qx, qy, qz = quaternion
    return (qx, qy, qz, qw)


def gpu_snapshot() -> dict[str, float | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()[0]
        used, total, utilization = (float(item.strip()) for item in raw.split(",")[:3])
        return {
            "used_mib": used,
            "total_mib": total,
            "utilization_percent": utilization,
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"used_mib": None, "total_mib": None, "utilization_percent": None}


def main() -> int:
    args = parse_args()
    if min(args.curobo_runs, args.coexistence_runs, args.mplib_runs) < 1:
        raise ValueError("all run counts must be at least 1")

    robotwin_root = args.robotwin_root.resolve()
    project_root = args.project_root.resolve()
    output_path = args.output.resolve()
    timing_gates_path = project_root / "configs/s1.2/timing_gates.json"
    timing_gates = json.loads(timing_gates_path.read_text(encoding="utf-8"))
    physics_period_s = float(timing_gates["physics_period_s"])
    max_dilation_factor = int(timing_gates["time_dilation"]["max_factor"])
    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(project_root / "src"))

    import numpy as np
    import yaml
    from script.collect_data import prepare_task_and_args

    from vla_robot_sorting.contracts import JointState, MotionGoal, PlanningContext, Pose6D
    from vla_robot_sorting.evaluation.backend_stability import latency_summary, memory_summary
    from vla_robot_sorting.planning import (
        PantheraRoboTwinMotionPlanner,
        PantheraRoboTwinMPlibPlanner,
        PhysicsAlignedTimeDilationParameterizer,
        RoboTwinVendorTrajectoryParameterizer,
        load_joint_limits_from_urdf,
    )
    from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits
    from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES
    from vla_robot_sorting.simulation.render_profile import (
        FAST_PREVIEW_RENDER_PROFILE,
        apply_fast_preview_render_profile,
    )

    task, task_args = prepare_task_and_args(args.task, args.task_config)
    task_args["render_freq"] = 0
    task_args["save_data"] = False
    task_args["collect_data"] = False
    task_args["domain_randomization"]["cluttered_table"] = False
    task_args["domain_randomization"]["random_background"] = False
    task_args["domain_randomization"]["random_light"] = False
    apply_fast_preview_render_profile(task_args)

    initialized = False
    payload: dict[str, Any] = {
        "schema": "panthera-backend-matrix-v1",
        "seed": args.seed,
        "task": args.task,
        "task_config": args.task_config,
        "render_profile": dict(FAST_PREVIEW_RENDER_PROFILE),
        "actuation": "none",
    }

    try:
        setup_started = perf_counter()
        task.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
        initialized = True
        payload["setup_duration_s"] = perf_counter() - setup_started

        actual = tuple(float(value) for value in task.robot.get_left_arm_real_jointState())
        if len(actual) != 7:
            raise RuntimeError("expected six Panthera joints plus normalized gripper")
        start_state = JointState(
            joint_names=PANTHERA_JOINT_NAMES,
            positions=actual[:6],
        )
        context = PlanningContext(
            scene_revision=f"probe-seed-{args.seed}",
            transform_revision="robotwin-static-v1",
        )

        current_pose = tuple(float(value) for value in task.robot.get_left_ee_pose())
        position = current_pose[:3]
        orientation_wxyz = current_pose[3:]
        if len(position) != 3 or len(orientation_wxyz) != 4:
            raise RuntimeError("RoboTwin end-effector pose must be xyz + wxyz")

        half_angle = args.rotation_deg * pi / 180.0 / 2.0
        rotated_wxyz = quaternion_multiply_wxyz(
            (cos(half_angle), 0.0, 0.0, sin(half_angle)),
            orientation_wxyz,
        )
        goals = (
            MotionGoal(
                pose=Pose6D(
                    position_xyz=(position[0], position[1], position[2] + args.translation_m),
                    orientation_xyzw=xyzw_from_wxyz(orientation_wxyz),
                    frame_id="world",
                )
            ),
            MotionGoal(
                pose=Pose6D(
                    position_xyz=position,
                    orientation_xyzw=xyzw_from_wxyz(rotated_wxyz),
                    frame_id="world",
                )
            ),
        )

        curobo_config = robotwin_root / "assets/embodiments/panthera-6dof/curobo.yml"
        with curobo_config.open(encoding="utf-8") as handle:
            cspace = yaml.safe_load(handle)["robot_cfg"]["kinematics"]["cspace"]
        urdf_path = Path(task.robot.left_urdf_path)
        if not urdf_path.is_absolute():
            urdf_path = (robotwin_root / urdf_path).resolve()
        limits = load_joint_limits_from_urdf(
            urdf_path,
            PANTHERA_JOINT_NAMES,
            max_acceleration_rad_s2=cspace["max_acceleration"],
            max_jerk_rad_s3=cspace["max_jerk"],
        )

        curobo = PantheraRoboTwinMotionPlanner(task.robot)
        mplib = PantheraRoboTwinMPlibPlanner(task.robot)
        baseline_timer = RoboTwinVendorTrajectoryParameterizer()
        timer = PhysicsAlignedTimeDilationParameterizer(
            baseline_timer,
            physics_period_s=physics_period_s,
            max_dilation_factor=max_dilation_factor,
        )

        def recheck_mplib_path(path: Any) -> dict[str, Any]:
            active_names = tuple(
                joint.get_name() for joint in task.robot.left_entity.get_active_joints()
            )
            indices = tuple(active_names.index(name) for name in PANTHERA_JOINT_NAMES)
            base_qpos = np.asarray(task.robot.left_entity.get_qpos(), dtype=float)
            mplib_planner = task.robot.left_mplib_planner.planner
            collisions: list[dict[str, Any]] = []

            for waypoint_index, position in enumerate(path.positions):
                state = base_qpos.copy()
                for joint_index, value in zip(indices, position, strict=True):
                    state[joint_index] = value
                self_hits = mplib_planner.check_for_self_collision(state)
                env_hits = mplib_planner.check_for_env_collision(state)
                if self_hits or env_hits:
                    collisions.append(
                        {
                            "waypoint_index": waypoint_index,
                            "self_collision_count": len(self_hits),
                            "environment_collision_count": len(env_hits),
                        }
                    )
            return {
                "waypoints_checked": len(path.positions),
                "collision_free": not collisions,
                "collisions": collisions,
            }

        def evaluate(planner: Any, goal: MotionGoal, label: str) -> dict[str, Any]:
            before = gpu_snapshot()
            result = planner.plan(start_state, goal, context, scene={})
            after = gpu_snapshot()
            record: dict[str, Any] = {
                "goal": label,
                "planner_id": result.planner_id,
                "planning_time_s": result.planning_time_s,
                "succeeded": result.succeeded,
                "gpu_before": before,
                "gpu_after": after,
            }
            if not result.succeeded:
                failure = result.failure
                record["failure"] = {
                    "code": failure.code.value if failure else "missing",
                    "message": failure.message if failure else "missing",
                }
                return record

            assert result.path is not None
            baseline = baseline_timer.parameterize(result.path, limits)
            baseline_safety = validate_joint_trajectory_limits(baseline, limits)
            baseline_histogram: dict[str, int] = {}
            for violation in baseline_safety.violations:
                baseline_histogram[violation.code] = (
                    baseline_histogram.get(violation.code, 0) + 1
                )

            trajectory = timer.parameterize(result.path, limits)
            safety = validate_joint_trajectory_limits(trajectory, limits)
            if result.planner_id == "robotwin-mplib-rrt":
                try:
                    record["collision_recheck"] = recheck_mplib_path(result.path)
                except Exception as exc:
                    record["collision_recheck"] = {
                        "waypoints_checked": 0,
                        "collision_free": False,
                        "collisions": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            record.update(
                {
                    "path_samples": len(result.path.positions),
                    "baseline_trajectory_duration_s": baseline.duration_s,
                    "baseline_joint_limit_safe": baseline_safety.is_safe,
                    "baseline_violation_count": len(baseline_safety.violations),
                    "baseline_violation_histogram": baseline_histogram,
                    "trajectory_samples": len(trajectory.waypoints),
                    "trajectory_duration_s": trajectory.duration_s,
                    "required_continuous_time_scale": trajectory.metadata[
                        "required_continuous_time_scale"
                    ],
                    "applied_integer_time_dilation_factor": trajectory.metadata[
                        "applied_integer_time_dilation_factor"
                    ],
                    "controller_period_s": trajectory.metadata["controller_period_s"],
                    "joint_limit_safe": safety.is_safe,
                    "violations": [
                        {
                            "code": violation.code,
                            "message": violation.message,
                            "waypoint_index": violation.waypoint_index,
                        }
                        for violation in safety.violations
                    ],
                }
            )
            return record

        def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
            success = [item for item in records if item.get("succeeded")]
            latencies = [float(item["planning_time_s"]) for item in records]
            memory = [
                float(item["gpu_after"]["used_mib"])
                for item in records
                if item.get("gpu_after", {}).get("used_mib") is not None
            ]
            dilation_factors = [
                int(item["applied_integer_time_dilation_factor"])
                for item in success
                if "applied_integer_time_dilation_factor" in item
            ]
            return {
                "runs": len(records),
                "success_count": len(success),
                "failure_count": len(records) - len(success),
                "baseline_unsafe_success_count": sum(
                    item.get("baseline_joint_limit_safe") is False for item in success
                ),
                "all_successful_paths_joint_safe": all(
                    item.get("joint_limit_safe") is True for item in success
                ),
                "max_applied_time_dilation_factor": (
                    max(dilation_factors) if dilation_factors else None
                ),
                "latency": latency_summary(latencies),
                "gpu_memory": memory_summary(memory),
                "records": records,
            }

        curobo_records = [
            evaluate(
                curobo,
                goals[index % len(goals)],
                "translation" if index % 2 == 0 else "orientation",
            )
            for index in range(args.curobo_runs)
        ]
        b1 = summarize_records(curobo_records)

        coexistence_records: list[dict[str, Any]] = []
        camera_failures = 0
        for index in range(args.coexistence_runs):
            camera_started = perf_counter()
            try:
                observation = task.get_obs()
                camera_elapsed = perf_counter() - camera_started
                obs = observation.get("observation", {})
                head = obs.get("head_camera", {}).get("rgb")
                wrist = obs.get("wrist_camera", {}).get("rgb")
                camera_ok = (
                    getattr(head, "shape", None) is not None
                    and getattr(wrist, "shape", None) is not None
                )
                camera_error = None
            except Exception as exc:
                camera_elapsed = perf_counter() - camera_started
                camera_ok = False
                camera_error = f"{type(exc).__name__}: {exc}"
                camera_failures += 1

            record = evaluate(
                curobo,
                goals[index % len(goals)],
                "translation" if index % 2 == 0 else "orientation",
            )
            record["camera_capture_s"] = camera_elapsed
            record["camera_ok"] = camera_ok
            if camera_error is not None:
                record["camera_error"] = camera_error
            coexistence_records.append(record)
            if not camera_ok:
                break

        b2 = summarize_records(coexistence_records)
        b2["camera_failure_count"] = camera_failures
        b2["camera_latency"] = latency_summary(
            [float(item["camera_capture_s"]) for item in coexistence_records]
        )

        mplib_records = [
            evaluate(
                mplib,
                goals[index % len(goals)],
                "translation" if index % 2 == 0 else "orientation",
            )
            for index in range(args.mplib_runs)
        ]
        b3 = summarize_records(mplib_records)
        b3_successes = [item for item in mplib_records if item.get("succeeded")]
        b3["all_successful_paths_collision_free"] = bool(b3_successes) and all(
            item.get("collision_recheck", {}).get("collision_free") is True
            for item in b3_successes
        )

        payload.update(
            {
                "planner_classes": {
                    "curobo": type(task.robot.left_planner).__name__,
                    "mplib": type(task.robot.left_mplib_planner).__name__,
                },
                "actual_joint_positions_rad": actual[:6],
                "current_pose_robotwin_wxyz": current_pose,
                "joint_limits_source": "runtime URDF + pinned curobo.yml cspace",
                "timing_gates": timing_gates,
                "timing_gates_path": str(timing_gates_path),
                "B1_curobo_repeated": b1,
                "B2_renderer_camera_curobo_coexistence": b2,
                "B3_mplib_fallback": b3,
            }
        )

        b1_pass = (
            b1["runs"] >= args.curobo_runs
            and b1["success_count"] >= 1
            and not b1["gpu_memory"]["monotonic_growth_after_warmup"]
        )
        b2_peak = b2["gpu_memory"]["max_mib"]
        b2_total = next(
            (
                item["gpu_after"]["total_mib"]
                for item in coexistence_records
                if item.get("gpu_after", {}).get("total_mib") is not None
            ),
            None,
        )
        b2_pass = (
            b2["runs"] == args.coexistence_runs
            and b2["camera_failure_count"] == 0
            and b2["success_count"] >= 1
            and b2_peak is not None
            and b2_total is not None
            and float(b2_peak) < float(b2_total)
            and not b2["gpu_memory"]["monotonic_growth_after_warmup"]
        )
        b3_pass = (
            b3["runs"] >= args.mplib_runs
            and b3["success_count"] >= 1
            and b3["all_successful_paths_joint_safe"]
            and b3["all_successful_paths_collision_free"]
        )
        payload["gates"] = {
            "B1_pass": b1_pass,
            "B2_pass": b2_pass,
            "B3_pass": b3_pass,
            "retimed_curobo_paths_safe": (
                b1["all_successful_paths_joint_safe"]
                and b2["all_successful_paths_joint_safe"]
            ),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return 0 if b1_pass and b2_pass and b3_pass else 2
    finally:
        if initialized:
            task.close_env()


if __name__ == "__main__":
    raise SystemExit(main())
