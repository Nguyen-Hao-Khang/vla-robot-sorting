#!/usr/bin/env python3
"""Native integrated grounding + bounded pregrasp execution + destination recovery probe."""

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
    parser.add_argument("--requested-horizon-s", type=float, default=0.2)
    parser.add_argument("--destination-translation-m", type=float, default=0.025)
    return parser.parse_args()


def load_b57_helpers(project_root: Path) -> Any:
    path = project_root / "scripts/linux/probe_panthera_grasp_recovery.py"
    spec = importlib.util.spec_from_file_location("b57_native_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load B5/B7 helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PhysicalContactCounter:
    """Count only physical contacts between the active robot and target/pad."""

    IMPULSE_EPS = 1e-8

    def __init__(self, task: Any) -> None:
        self.object_entity = task.pillbottle.actor
        self.pad_entity = task.pad.actor
        self.robot_entity_ids = {
            int(link.entity.get_global_id())
            for link in task.robot.left_entity.get_links()
        }
        self.object_robot_frames = 0
        self.object_robot_points = 0
        self.pad_robot_frames = 0
        self.pad_robot_points = 0
        self.physics_steps = 0

    @staticmethod
    def _physical(contact: Any) -> bool:
        import numpy as np

        separations = [float(point.separation) for point in contact.points]
        impulses = [float(np.linalg.norm(point.impulse)) for point in contact.points]
        return (
            min(separations, default=float("inf")) <= 0.0
            or max(impulses, default=0.0) > PhysicalContactCounter.IMPULSE_EPS
        )

    def on_physics_step(self, task: Any) -> None:
        self.physics_steps += 1
        object_id = int(self.object_entity.get_global_id())
        pad_id = int(self.pad_entity.get_global_id())
        object_contact_this_step = False
        pad_contact_this_step = False
        for contact in task.scene.get_contacts():
            if len(contact.bodies) != 2 or not contact.points or not self._physical(contact):
                continue
            entities = [getattr(body, "entity", None) for body in contact.bodies]
            if any(entity is None for entity in entities):
                continue
            ids = {int(entity.get_global_id()) for entity in entities}
            robot_contact = bool(ids.intersection(self.robot_entity_ids))
            if not robot_contact:
                continue
            if object_id in ids:
                object_contact_this_step = True
                self.object_robot_points += len(contact.points)
            if pad_id in ids:
                pad_contact_this_step = True
                self.pad_robot_points += len(contact.points)
        self.object_robot_frames += int(object_contact_this_step)
        self.pad_robot_frames += int(pad_contact_this_step)


def pose_payload(actor: Any) -> dict[str, list[float]]:
    pose = actor.get_pose()
    return {
        "position_xyz_m": [float(value) for value in pose.p],
        "orientation_wxyz": [float(value) for value in pose.q],
    }


def max_abs_delta(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def main() -> int:
    args = parse_args()
    robotwin_root = args.robotwin_root.resolve()
    project_root = args.project_root.resolve()
    output_path = args.output.resolve()
    gate_path = project_root / "configs/s1.3/grasp_recovery_gates.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    timing_path = project_root / "configs/s1.2/timing_gates.json"
    timing_gate = json.loads(timing_path.read_text(encoding="utf-8"))

    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(project_root / "src"))

    import sapien
    import yaml
    from script.collect_data import prepare_task_and_args

    from vla_robot_sorting.contracts import MotionGoal, PlanningContext
    from vla_robot_sorting.contracts.motion import ExecutionStatus
    from vla_robot_sorting.contracts.recovery import (
        RecoveryAction,
        RecoveryEvent,
        RecoveryState,
        RecoveryTrigger,
        SceneChangeKind,
    )
    from vla_robot_sorting.control import (
        PantheraRoboTwinTrajectoryExecutor,
        build_execution_segment,
    )
    from vla_robot_sorting.orchestration.recovery import ClosedLoopRecoveryPolicy
    from vla_robot_sorting.perception.change_detection import SceneChangeDetector
    from vla_robot_sorting.planning import (
        PhysicsAlignedTimeDilationParameterizer,
        RoboTwinVendorTrajectoryParameterizer,
        load_joint_limits_from_urdf,
    )
    from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits
    from vla_robot_sorting.semantics.grounding import (
        CatalogDestinationGrounder,
        CatalogGrounder,
        GroundingRequest,
    )
    from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES
    from vla_robot_sorting.simulation.render_profile import apply_fast_preview_render_profile

    helpers = load_b57_helpers(project_root)
    task = None
    payload: dict[str, Any] = {
        "schema": "panthera-native-integrated-pregrasp-v1",
        "task": args.task,
        "task_config": args.task_config,
        "seed": args.seed,
        "passed": False,
    }

    try:
        task, task_args = prepare_task_and_args(args.task, args.task_config)
        task_args["render_freq"] = 0
        task_args["save_data"] = False
        task_args["collect_data"] = False
        task_args["domain_randomization"]["cluttered_table"] = False
        task_args["domain_randomization"]["random_background"] = False
        task_args["domain_randomization"]["random_light"] = False
        apply_fast_preview_render_profile(task_args)

        setup_started = perf_counter()
        task.setup_demo(now_ep_num=0, seed=args.seed, **task_args)
        payload["setup_duration_s"] = perf_counter() - setup_started
        payload["mplib_world_initial"] = helpers.sync_scaled_target_into_mplib_world(
            task
        )

        geometry = helpers.actor_box_from_collision_mesh(task, robotwin_root)
        gripper, gripper_provenance = helpers.load_gripper_proxy(robotwin_root, task_args)
        revision = f"integrated-seed-{args.seed}-initial"
        transform_revision = "robotwin-static-v1"
        scene = helpers.make_scene_snapshot(
            task,
            revision=revision,
            geometry_info=geometry,
            transform_revision=transform_revision,
        )
        context = PlanningContext(revision, transform_revision)

        grounding_started = perf_counter()
        target_grounding = CatalogGrounder().ground(
            GroundingRequest("pill bottle", revision, language="en"),
            scene,
        )
        destination_grounding = CatalogDestinationGrounder().ground(
            GroundingRequest("blue pad", revision, language="en"),
            scene,
        )
        grounding_latency_s = perf_counter() - grounding_started
        grounding_pass = (
            target_grounding.succeeded
            and target_grounding.object_id == "pillbottle"
            and destination_grounding.succeeded
            and destination_grounding.destination_id == "pad"
        )
        payload["grounding"] = {
            "passed": grounding_pass,
            "target_object_id": target_grounding.object_id,
            "destination_id": destination_grounding.destination_id,
            "target_confidence": target_grounding.confidence,
            "destination_confidence": destination_grounding.confidence,
            "scene_revision": revision,
            "latency_s": grounding_latency_s,
            "scene_source": scene.metadata.get("source"),
        }
        if not grounding_pass:
            raise RuntimeError("live-scene target/destination grounding failed")

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
        timer = PhysicsAlignedTimeDilationParameterizer(
            RoboTwinVendorTrajectoryParameterizer(),
            physics_period_s=float(timing_gate["physics_period_s"]),
            max_dilation_factor=int(timing_gate["time_dilation"]["max_factor"]),
        )

        b5 = helpers.evaluate_candidates(
            task,
            scene,
            gripper=gripper,
            gate=gate,
            context=context,
            limits=limits,
            timer=timer,
        )
        selected = b5.pop("_selected_object", None)
        payload["grasp"] = {
            "passed": b5.get("passed") is True and selected is not None,
            "candidate_count": b5.get("candidate_count"),
            "non_top_down_count": b5.get("non_top_down_count"),
            "selected": b5.get("selected"),
            "gripper": gripper_provenance,
            "geometry": geometry,
        }
        if selected is None or b5.get("passed") is not True:
            raise RuntimeError("no safe selected B5 pregrasp candidate")

        start_state = helpers.current_arm_state(task)
        plan, planning = helpers.plan_panthera_primary_fallback(
            task,
            start_state,
            MotionGoal(selected.pregrasp_pose),
            context,
            scene=scene,
        )
        if not plan.succeeded or plan.path is None:
            raise RuntimeError("primary/fallback pregrasp planning failed")

        collision = helpers.collision_recheck(task, plan.path)
        trajectory = timer.parameterize(plan.path, limits)
        dynamic = validate_joint_trajectory_limits(trajectory, limits)
        preconditions_pass = collision["collision_free"] and dynamic.is_safe
        payload["preconditions"] = {
            "passed": preconditions_pass,
            "planning": planning,
            "path_samples": len(plan.path.positions),
            "collision_recheck": collision,
            "trajectory_duration_s": trajectory.duration_s,
            "dynamic_safe": dynamic.is_safe,
            "dynamic_violation_count": len(dynamic.violations),
            "controller_period_s": trajectory.metadata["controller_period_s"],
            "time_dilation_factor": trajectory.metadata[
                "applied_integer_time_dilation_factor"
            ],
        }
        if not preconditions_pass:
            raise RuntimeError("pregrasp execution safety preconditions failed")

        controller_period_s = float(trajectory.metadata["controller_period_s"])
        segment = build_execution_segment(
            trajectory,
            context,
            segment_id=f"integrated-pregrasp-{args.seed}",
            requested_horizon_s=args.requested_horizon_s,
            controller_period_s=controller_period_s,
        )
        qpos_before = tuple(float(v) for v in task.robot.get_left_arm_real_jointState()[:6])
        object_before = pose_payload(task.pillbottle)
        pad_before = pose_payload(task.pad)
        contacts = PhysicalContactCounter(task)
        task.add_physics_step_observer(contacts)

        executor = PantheraRoboTwinTrajectoryExecutor(
            task,
            scene_revision=lambda: revision,
            transform_revision=lambda: transform_revision,
            tracking_tolerance_rad=0.08,
            physics_period_s=float(timing_gate["physics_period_s"]),
        )
        execution_started = perf_counter()
        feedback = executor.execute(segment)
        execution_latency_s = perf_counter() - execution_started
        task.clear_physics_step_observers()

        qpos_after = tuple(float(v) for v in task.robot.get_left_arm_real_jointState()[:6])
        object_after = pose_payload(task.pillbottle)
        pad_after = pose_payload(task.pad)
        expected = segment.predicted_stop_state
        actual = executor.last_actual_state
        tracking_error = None
        if expected is not None and actual is not None:
            tracking_error = max_abs_delta(expected.positions, actual.positions)
        qpos_delta = max_abs_delta(qpos_before, qpos_after)
        execution_pass = (
            feedback.status is ExecutionStatus.COMPLETED
            and not feedback.replan_required
            and contacts.physics_steps > 0
            and contacts.object_robot_frames == 0
            and contacts.pad_robot_frames == 0
            and qpos_delta > 1e-6
            and tracking_error is not None
            and tracking_error <= 0.08
        )
        payload["execution"] = {
            "passed": execution_pass,
            "feedback_status": feedback.status.value,
            "feedback_reason": feedback.reason,
            "replan_required": feedback.replan_required,
            "executed_until_s": feedback.executed_until_s,
            "segment_horizon_s": segment.horizon_end_s,
            "controller_period_s": segment.controller_period_s,
            "physics_steps": contacts.physics_steps,
            "object_robot_contact_frames": contacts.object_robot_frames,
            "object_robot_contact_points": contacts.object_robot_points,
            "pad_robot_contact_frames": contacts.pad_robot_frames,
            "pad_robot_contact_points": contacts.pad_robot_points,
            "qpos_before_rad": qpos_before,
            "qpos_after_rad": qpos_after,
            "max_qpos_delta_rad": qpos_delta,
            "max_tracking_error_rad": tracking_error,
            "execution_latency_s": execution_latency_s,
            "object_pose_before": object_before,
            "object_pose_after": object_after,
            "pad_pose_before": pad_before,
            "pad_pose_after": pad_after,
        }
        if not execution_pass:
            raise RuntimeError("bounded pregrasp execution did not meet safety/tracking gates")

        destination_old_scene = helpers.make_scene_snapshot(
            task,
            revision=f"integrated-seed-{args.seed}-before-pad-move",
            geometry_info=helpers.actor_box_from_collision_mesh(task, robotwin_root),
            transform_revision=transform_revision,
        )
        pad_pose = task.pad.get_pose()
        pad_before_xyz = [float(v) for v in pad_pose.p]
        delta = (
            args.destination_translation_m
            if pad_before_xyz[1] <= 0.05
            else -args.destination_translation_m
        )
        pad_after_xyz = [pad_before_xyz[0], pad_before_xyz[1] + delta, pad_before_xyz[2]]
        task.pad.actor.set_pose(sapien.Pose(pad_after_xyz, pad_pose.q))
        destination_new_scene = helpers.make_scene_snapshot(
            task,
            revision=f"integrated-seed-{args.seed}-after-pad-move",
            geometry_info=helpers.actor_box_from_collision_mesh(task, robotwin_root),
            transform_revision=transform_revision,
        )

        detection_started = perf_counter()
        change = SceneChangeDetector().detect(
            destination_old_scene,
            destination_new_scene,
            target_object_id="pillbottle",
            destination_id="pad",
        )
        detection_latency = perf_counter() - detection_started
        decision_started = perf_counter()
        decision = ClosedLoopRecoveryPolicy().decide(
            RecoveryEvent(
                RecoveryTrigger.SCENE_CHANGE,
                observed_at_s=detection_started,
                scene_change=change,
            ),
            RecoveryState(),
        )
        decision_latency = perf_counter() - decision_started
        destination_recovery_pass = (
            change.kind is SceneChangeKind.DESTINATION_MOVED
            and decision.action is RecoveryAction.REPLAN
        )
        payload["destination_recovery"] = {
            "passed": destination_recovery_pass,
            "change_kind": change.kind.value,
            "position_delta_m": change.position_delta_m,
            "decision_action": decision.action.value,
            "detection_latency_s": detection_latency,
            "decision_latency_s": decision_latency,
            "pad_before_xyz_m": pad_before_xyz,
            "pad_after_xyz_m": pad_after_xyz,
            "robot_execution_after_destination_move": False,
        }

        payload["passed"] = (
            grounding_pass
            and preconditions_pass
            and execution_pass
            and destination_recovery_pass
        )
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["passed"] = False
    finally:
        if task is not None:
            with suppress(Exception):
                task.clear_physics_step_observers()
            with suppress(Exception):
                task.close_env()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
