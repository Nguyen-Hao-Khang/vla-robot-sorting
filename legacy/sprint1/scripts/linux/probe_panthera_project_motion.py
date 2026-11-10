#!/usr/bin/env python3
"""Plan small full-SE(3) Panthera goals through project-owned S1.2 adapters.

This probe is planning/timing only: it does not command the simulator joints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import cos, pi, sin
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="move_pillbottle_pad")
    parser.add_argument("--task-config", default="move_pillbottle_pad_panthera_single")
    parser.add_argument("--seed", type=int, default=0)
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


def resolve_runtime_path(robotwin_root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (robotwin_root / path).resolve()


def main() -> int:
    args = parse_args()
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

    import yaml
    from script.collect_data import prepare_task_and_args

    from vla_robot_sorting.contracts import JointState, MotionGoal, PlanningContext, Pose6D
    from vla_robot_sorting.planning import (
        PantheraRoboTwinMotionPlanner,
        PhysicsAlignedTimeDilationParameterizer,
        RoboTwinVendorTrajectoryParameterizer,
        load_joint_limits_from_urdf,
    )
    from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits
    from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES

    task, task_args = prepare_task_and_args(args.task, args.task_config)
    task_args["render_freq"] = 0
    task_args["save_data"] = False
    task_args["collect_data"] = False
    task_args["domain_randomization"]["cluttered_table"] = False
    task_args["domain_randomization"]["random_background"] = False
    task_args["domain_randomization"]["random_light"] = False


    initialized = False
    try:
        task.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
        initialized = True

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
        if len(current_pose) != 7:
            raise RuntimeError("RoboTwin get_left_ee_pose must return xyz + wxyz")
        position = current_pose[:3]
        orientation_wxyz = current_pose[3:]

        curobo_config = robotwin_root / "assets/embodiments/panthera-6dof/curobo.yml"
        with curobo_config.open(encoding="utf-8") as handle:
            cspace = yaml.safe_load(handle)["robot_cfg"]["kinematics"]["cspace"]


        urdf_path = resolve_runtime_path(robotwin_root, task.robot.left_urdf_path)
        limits = load_joint_limits_from_urdf(
            urdf_path,
            PANTHERA_JOINT_NAMES,
            max_acceleration_rad_s2=cspace["max_acceleration"],
            max_jerk_rad_s3=cspace["max_jerk"],
        )

        planner = PantheraRoboTwinMotionPlanner(task.robot)
        baseline_timer = RoboTwinVendorTrajectoryParameterizer()
        timer = PhysicsAlignedTimeDilationParameterizer(
            baseline_timer,
            physics_period_s=physics_period_s,
            max_dilation_factor=max_dilation_factor,
        )

        translation_goal = MotionGoal(
            pose=Pose6D(
                position_xyz=(position[0], position[1], position[2] + args.translation_m),
                orientation_xyzw=xyzw_from_wxyz(orientation_wxyz),
                frame_id="world",
            )
        )

        half_angle = args.rotation_deg * pi / 180.0 / 2.0
        world_z_delta = (cos(half_angle), 0.0, 0.0, sin(half_angle))
        rotated_wxyz = quaternion_multiply_wxyz(world_z_delta, orientation_wxyz)
        rotation_goal = MotionGoal(
            pose=Pose6D(
                position_xyz=position,
                orientation_xyzw=xyzw_from_wxyz(rotated_wxyz),
                frame_id="world",
            )
        )


        def evaluate(label: str, goal: MotionGoal) -> dict[str, Any]:
            result = planner.plan(start_state, goal, context, scene={})
            record: dict[str, Any] = {
                "label": label,
                "planner_id": result.planner_id,
                "planning_time_s": result.planning_time_s,
                "succeeded": result.succeeded,
            }
            if not result.succeeded:
                assert result.failure is not None
                record["failure"] = {
                    "code": result.failure.code.value,
                    "message": result.failure.message,
                    "recoverable": result.failure.recoverable,
                    "details": result.failure.details,
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
            record.update(
                {
                    "path_samples": len(result.path.positions),
                    "baseline_trajectory_duration_s": baseline.duration_s,
                    "baseline_joint_limit_safe": baseline_safety.is_safe,
                    "baseline_violation_count": len(baseline_safety.violations),
                    "baseline_violation_histogram": baseline_histogram,
                    "trajectory_duration_s": trajectory.duration_s,
                    "trajectory_samples": len(trajectory.waypoints),
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


        probes = [
            evaluate("translation", translation_goal),
            evaluate("orientation", rotation_goal),
        ]
        payload = {
            "schema": "panthera-project-motion-probe-v1",
            "seed": args.seed,
            "planner_class": type(task.robot.left_planner).__name__,
            "current_pose_robotwin_wxyz": current_pose,
            "actual_joint_positions_rad": actual[:6],
            "joint_limits": {
                "lower_position_rad": limits.lower_position_rad,
                "upper_position_rad": limits.upper_position_rad,
                "max_velocity_rad_s": limits.max_velocity_rad_s,
                "max_acceleration_rad_s2": limits.max_acceleration_rad_s2,
                "max_jerk_rad_s3": limits.max_jerk_rad_s3,
                "source": "runtime URDF + pinned curobo.yml cspace",
            },
            "requested_translation_m": args.translation_m,
            "requested_rotation_deg": args.rotation_deg,
            "timing_gates": timing_gates,
            "timing_gates_path": str(timing_gates_path),
            "probes": probes,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        all_safe = all(
            item["succeeded"] and item.get("joint_limit_safe")
            for item in probes
        )
        return 0 if all_safe else 2
    finally:
        if initialized:
            task.close_env()


if __name__ == "__main__":
    raise SystemExit(main())
