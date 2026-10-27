#!/usr/bin/env python3
"""Native S1.2 B4 timing/splice validation without robot actuation."""

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
    parser.add_argument("--runs", type=int, default=20)
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


def rebased_slice(
    trajectory: Any,
    start: int,
    stop: int,
    *,
    source_suffix: str,
) -> Any:
    from vla_robot_sorting.contracts import Trajectory, TrajectoryWaypoint

    selected = trajectory.waypoints[start:stop]
    if len(selected) < 2:
        raise ValueError("trajectory slice requires at least two waypoints")
    origin = selected[0].time_from_start_s
    waypoints = tuple(
        TrajectoryWaypoint(
            time_from_start_s=point.time_from_start_s - origin,
            positions=point.positions,
            velocities=point.velocities,
            accelerations=point.accelerations,
        )
        for point in selected
    )
    return Trajectory(
        joint_names=trajectory.joint_names,
        waypoints=waypoints,
        source=f"{trajectory.source}:{source_suffix}",
        frame_id=trajectory.frame_id,
        limits_used=trajectory.limits_used,
        interpolation=trajectory.interpolation,
        scene_revision=trajectory.scene_revision,
        transform_revision=trajectory.transform_revision,
        metadata=dict(trajectory.metadata),
    )


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("runs must be at least 1")

    robotwin_root = args.robotwin_root.resolve()
    project_root = args.project_root.resolve()
    output_path = args.output.resolve()
    timing_gates_path = project_root / "configs/s1.2/timing_gates.json"
    timing_gates = json.loads(timing_gates_path.read_text(encoding="utf-8"))
    physics_period_s = float(timing_gates["physics_period_s"])
    max_dilation_factor = int(timing_gates["time_dilation"]["max_factor"])
    splice_policy = timing_gates["splice"]
    position_tolerance_rad = float(splice_policy["position_tolerance_rad"])
    velocity_tolerance_rad_s = float(splice_policy["velocity_tolerance_rad_s"])
    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(project_root / "src"))

    import yaml
    from script.collect_data import prepare_task_and_args

    from vla_robot_sorting.contracts import JointState, MotionGoal, PlanningContext, Pose6D
    from vla_robot_sorting.evaluation.backend_stability import latency_summary
    from vla_robot_sorting.planning import (
        PantheraRoboTwinMotionPlanner,
        PhysicsAlignedTimeDilationParameterizer,
        RoboTwinVendorTrajectoryParameterizer,
        load_joint_limits_from_urdf,
    )
    from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits
    from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES
    from vla_robot_sorting.telemetry import (
        trajectory_kinematics,
        trajectory_splice_discontinuity,
    )

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
            scene_revision=f"b4-seed-{args.seed}",
            transform_revision="robotwin-static-v1",
        )

        current_pose = tuple(float(value) for value in task.robot.get_left_ee_pose())
        if len(current_pose) != 7:
            raise RuntimeError("RoboTwin end-effector pose must be xyz + wxyz")
        position = current_pose[:3]
        orientation_wxyz = current_pose[3:]
        half_angle = args.rotation_deg * pi / 180.0 / 2.0
        rotated_wxyz = quaternion_multiply_wxyz(
            (cos(half_angle), 0.0, 0.0, sin(half_angle)),
            orientation_wxyz,
        )
        goals = (
            (
                "translation",
                MotionGoal(
                    pose=Pose6D(
                        position_xyz=(
                            position[0],
                            position[1],
                            position[2] + args.translation_m,
                        ),
                        orientation_xyzw=xyzw_from_wxyz(orientation_wxyz),
                        frame_id="world",
                    )
                ),
            ),
            (
                "orientation",
                MotionGoal(
                    pose=Pose6D(
                        position_xyz=position,
                        orientation_xyzw=xyzw_from_wxyz(rotated_wxyz),
                        frame_id="world",
                    )
                ),
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

        planner = PantheraRoboTwinMotionPlanner(task.robot)
        baseline_timer = RoboTwinVendorTrajectoryParameterizer()
        timer = PhysicsAlignedTimeDilationParameterizer(
            baseline_timer,
            physics_period_s=physics_period_s,
            max_dilation_factor=max_dilation_factor,
        )

        records: list[dict[str, Any]] = []
        for index in range(args.runs):
            label, goal = goals[index % len(goals)]
            result = planner.plan(start_state, goal, context, scene={})
            record: dict[str, Any] = {
                "run": index,
                "goal": label,
                "planning_time_s": result.planning_time_s,
                "succeeded": result.succeeded,
            }
            if not result.succeeded:
                failure = result.failure
                record["failure"] = {
                    "code": failure.code.value if failure else "missing",
                    "message": failure.message if failure else "missing",
                }
                records.append(record)
                continue

            assert result.path is not None
            baseline = baseline_timer.parameterize(result.path, limits)
            baseline_safety = validate_joint_trajectory_limits(baseline, limits)
            safe = timer.parameterize(result.path, limits)
            safety = validate_joint_trajectory_limits(safe, limits)
            metrics = trajectory_kinematics(safe)

            midpoint = max(1, min(len(safe.waypoints) - 2, len(safe.waypoints) // 2))
            previous = rebased_slice(
                safe,
                0,
                midpoint + 1,
                source_suffix="splice-previous",
            )
            following = rebased_slice(
                safe,
                midpoint,
                len(safe.waypoints),
                source_suffix="splice-following",
            )
            splice = trajectory_splice_discontinuity(previous, following)

            controller_period_s = float(safe.metadata["controller_period_s"])
            position_gate_rad = position_tolerance_rad
            velocity_gate_rad_s = (
                min(limits.max_acceleration_rad_s2) * controller_period_s
                + velocity_tolerance_rad_s
            )
            splice_pass = (
                splice.position_rad_max <= position_gate_rad
                and splice.velocity_rad_s_max is not None
                and splice.velocity_rad_s_max <= velocity_gate_rad_s
            )

            record.update(
                {
                    "path_samples": len(result.path.positions),
                    "baseline_duration_s": baseline.duration_s,
                    "baseline_joint_limit_safe": baseline_safety.is_safe,
                    "baseline_violation_count": len(baseline_safety.violations),
                    "retimed_duration_s": safe.duration_s,
                    "required_continuous_time_scale": safe.metadata[
                        "required_continuous_time_scale"
                    ],
                    "applied_integer_time_dilation_factor": safe.metadata[
                        "applied_integer_time_dilation_factor"
                    ],
                    "controller_period_s": controller_period_s,
                    "joint_limit_safe": safety.is_safe,
                    "violation_count": len(safety.violations),
                    "kinematics": {
                        "peak_velocity_rad_s": metrics.peak_velocity_rad_s,
                        "peak_acceleration_rad_s2": metrics.peak_acceleration_rad_s2,
                        "peak_jerk_rad_s3": metrics.peak_jerk_rad_s3,
                        "rms_velocity_rad_s": metrics.rms_velocity_rad_s,
                        "rms_acceleration_rad_s2": metrics.rms_acceleration_rad_s2,
                        "rms_jerk_rad_s3": metrics.rms_jerk_rad_s3,
                    },
                    "splice": {
                        "position_rad_max": splice.position_rad_max,
                        "velocity_rad_s_max": splice.velocity_rad_s_max,
                        "acceleration_rad_s2_max": splice.acceleration_rad_s2_max,
                        "position_gate_rad": position_gate_rad,
                        "velocity_gate_rad_s": velocity_gate_rad_s,
                        "passed": splice_pass,
                    },
                }
            )
            records.append(record)

        successes = [record for record in records if record.get("succeeded")]
        safe_successes = [
            record
            for record in successes
            if record.get("joint_limit_safe") is True
            and record.get("splice", {}).get("passed") is True
        ]
        factors = [
            int(record["applied_integer_time_dilation_factor"])
            for record in successes
            if "applied_integer_time_dilation_factor" in record
        ]
        payload = {
            "schema": "panthera-timing-splice-probe-v1",
            "task": args.task,
            "task_config": args.task_config,
            "seed": args.seed,
            "runs": args.runs,
            "success_count": len(successes),
            "safe_splice_count": len(safe_successes),
            "baseline_unsafe_success_count": sum(
                record.get("baseline_joint_limit_safe") is False
                for record in successes
            ),
            "planning_latency": latency_summary(
                [float(record["planning_time_s"]) for record in records]
            ),
            "max_applied_time_dilation_factor": max(factors) if factors else None,
            "timing_gates": timing_gates,
            "timing_gates_path": str(timing_gates_path),
            "joint_limits": {
                "max_velocity_rad_s": limits.max_velocity_rad_s,
                "max_acceleration_rad_s2": limits.max_acceleration_rad_s2,
                "max_jerk_rad_s3": limits.max_jerk_rad_s3,
                "source": "runtime URDF + pinned curobo.yml cspace",
            },
            "records": records,
        }
        payload["B4_pass"] = (
            len(records) >= args.runs
            and len(successes) == args.runs
            and len(safe_successes) == args.runs
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return 0 if payload["B4_pass"] else 2
    finally:
        if initialized:
            task.close_env()


if __name__ == "__main__":
    raise SystemExit(main())
