#!/usr/bin/env python3
"""Native project-owned Panthera pick/lift/place manipulation probe.

This probe intentionally avoids RoboTwin's play_once()/move() manipulation macro.
RoboTwin supplies the simulator, robot I/O and task success oracle; project-owned
components supply grasp selection, planning policy, retiming, bounded execution,
gripper execution, carried-object collision geometry and manipulation phase logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import traceback
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
    parser.add_argument("--operator-command", default=None)
    parser.add_argument("--operator-language", choices=("vi", "en"), default=None)
    parser.add_argument("--operator-rgbd-dir", type=Path, default=None)
    parser.add_argument("--operator-capture-only", action="store_true")
    parser.add_argument("--lift-m", type=float, default=0.05)
    parser.add_argument("--preplace-clearance-m", type=float, default=0.05)
    parser.add_argument("--release-clearance-m", type=float, default=0.01)
    parser.add_argument(
        "--render-freq",
        type=int,
        default=0,
        help="display only: open the SAPIEN viewer and redraw every N physics steps (0 = headless)",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="also print the full evidence JSON to stdout",
    )
    parser.add_argument(
        "--b5-evaluate-all-height-groups",
        action="store_true",
        help="diagnostic mode: do not stop B5 evaluation after the first feasible height group",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="diagnostic mode: require a specific B5-eligible candidate",
    )
    parser.add_argument(
        "--preferred-candidate-id",
        default=None,
        help="prefer this B5-eligible candidate; otherwise allow ranked safe fallback",
    )
    parser.add_argument(
        "--additional-diagnostic-dir",
        type=Path,
        help="opt-in bounded raw IK and command/physics evidence; no recovery execution",
    )
    parser.add_argument(
        "--plan-diagnostic-dir",
        type=Path,
        default=None,
        help="opt-in FCL/FK plan-only records before each motion; API failure aborts",
    )
    parser.add_argument(
        "--diagnose-native-ik-blockers",
        action="store_true",
        help="planning-only IK/collision diagnostics after failed canonical place or retreat",
    )
    return parser.parse_args()


def print_probe_summary(payload: dict[str, Any]) -> None:
    print(
        "MANIPULATION_SUMMARY "
        f"seed={payload.get('seed')} "
        f"passed={payload.get('passed')} "
        f"error={payload.get('error')} "
        f"failure_phase={payload.get('failure_phase')} "
        f"final_phase={payload.get('final_phase')}",
        flush=True,
    )
    for stage in payload.get("stages", []):
        if not isinstance(stage, dict):
            continue
        contacts = stage.get("contacts")
        contact_text = ""
        if isinstance(contacts, dict):
            contact_text = (
                f" robot_target={contacts.get('robot_target_contact_frames')}"
                f" target_pad={contacts.get('target_pad_contact_frames')}"
                f" target_table={contacts.get('target_table_contact_frames')}"
                f" robot_table={contacts.get('robot_table_contact_frames')}"
            )
        failure = stage.get("failure")
        failure_text = ""
        if isinstance(failure, dict):
            failure_text = (
                f" failure_code={failure.get('code')} failure_message={failure.get('message')}"
            )
        collision = stage.get("collision_validation")
        collision_text = ""
        if isinstance(collision, dict):
            collision_text = (
                f" collision_free={collision.get('collision_free')}"
                f" self_collision_free={collision.get('self_collision_free')}"
            )
        print(
            "MANIPULATION_STAGE "
            f"name={stage.get('stage')} "
            f"passed={stage.get('passed')} "
            f"planner={stage.get('planner_id')} "
            f"plan={stage.get('plan_success')} "
            f"dynamic={stage.get('dynamic_safe')} "
            f"execution={stage.get('execution_status')} "
            f"tracking={stage.get('max_tracking_error_rad')}"
            f"{collision_text}{contact_text}{failure_text}",
            flush=True,
        )


def load_b57_helpers(project_root: Path) -> Any:
    path = project_root / "scripts/linux/probe_panthera_grasp_recovery.py"
    spec = importlib.util.spec_from_file_location("b57_project_manip_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load B5/B7 helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pose_payload(actor: Any) -> dict[str, list[float]]:
    pose = actor.get_pose()
    return {
        "position_xyz_m": [float(value) for value in pose.p],
        "orientation_wxyz": [float(value) for value in pose.q],
    }


def robotwin_pose6d(raw: Any) -> Any:
    from vla_robot_sorting.contracts import Pose6D

    if hasattr(raw, "p") and hasattr(raw, "q"):
        position = tuple(float(value) for value in raw.p)
        qw, qx, qy, qz = (float(value) for value in raw.q)
    else:
        values = tuple(float(value) for value in raw)
        if len(values) != 7:
            raise ValueError("RoboTwin pose must contain seven values")
        position = values[:3]
        qw, qx, qy, qz = values[3:]
    return Pose6D(
        position_xyz=position,
        orientation_xyzw=(qx, qy, qz, qw),
        frame_id="world",
    )


def actor_pose6d(actor: Any) -> Any:
    return robotwin_pose6d(actor.get_pose())


def offset_pose_z(pose: Any, delta_z: float) -> Any:
    from vla_robot_sorting.contracts import Pose6D

    x, y, z = pose.position_xyz
    return Pose6D(
        position_xyz=(x, y, z + float(delta_z)),
        orientation_xyzw=pose.orientation_xyzw,
        frame_id=pose.frame_id,
    )


def verify_upstream_task_source(robotwin_root: Path) -> dict[str, Any]:
    from hashlib import sha256

    provenance = {
        "path": "RoboTwin/envs/move_pillbottle_pad.py",
        "commit": "2aee31ad75cb7b0cd779540441c6025853a5b60d",
        "sha256": "20e7b9d954fd42c3b78e4a9431aee68b3ff10f70f20104e7dc7e41c83fc8cffe",
        "check_success_orientation_constrained": False,
    }
    upstream = robotwin_root / "envs" / "move_pillbottle_pad.py"
    if not upstream.is_file():
        raise RuntimeError("pinned upstream task source missing; fail closed")
    raw = upstream.read_bytes()
    # The pinned Git blob is LF. Windows may materialize the identical text
    # with CRLF; normalize only that transport difference before attestation.
    canonical_bytes = raw.replace(b"\r\n", b"\n")
    actual_sha256 = sha256(canonical_bytes).hexdigest()
    if actual_sha256 != provenance["sha256"]:
        raise RuntimeError(
            "pinned upstream task source hash mismatch; fail closed: "
            f"expected={provenance['sha256']} actual={actual_sha256}"
        )
    provenance["raw_sha256"] = sha256(raw).hexdigest()
    provenance["line_ending_normalization"] = "CRLF_to_LF" if canonical_bytes != raw else "none"
    return provenance


def yaw_rotated_pose(pose: Any, yaw_rad: float) -> Any:
    from vla_robot_sorting.contracts import Pose6D

    x, y, z, w = (float(value) for value in pose.orientation_xyzw)
    half = 0.5 * float(yaw_rad)
    s = math.sin(half)
    c = math.cos(half)
    rotated = (
        c * x - s * y,
        c * y + s * x,
        c * z + s * w,
        c * w - s * z,
    )
    norm = math.sqrt(sum(value * value for value in rotated))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("yaw-rotated quaternion is invalid")
    orientation = tuple(value / norm for value in rotated)
    return Pose6D(
        position_xyz=pose.position_xyz,
        orientation_xyzw=orientation,
        frame_id=pose.frame_id,
    )


def carried_ik_plan_failure(stage: dict[str, Any]) -> bool:
    failure = stage.get("failure")
    details = failure.get("details") if isinstance(failure, dict) else None
    return (
        stage.get("plan_success") is False
        and stage.get("execution_status") is None
        and isinstance(failure, dict)
        and failure.get("code") == "optimization_failed"
        and isinstance(details, dict)
        and str(details.get("native_status")) == "IK Fail"
    )


def max_abs_delta(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def _entity_id(value: Any) -> int | None:
    entity = getattr(value, "entity", value)
    getter = getattr(entity, "get_global_id", None)
    if not callable(getter):
        return None
    with suppress(Exception):
        return int(getter())
    return None


def _physical_contact(contact: Any) -> bool:
    import numpy as np

    if not getattr(contact, "points", None):
        return False
    separation = min(
        (float(point.separation) for point in contact.points),
        default=float("inf"),
    )
    impulse = max(
        (float(np.linalg.norm(point.impulse)) for point in contact.points),
        default=0.0,
    )
    return separation <= 0.0 or impulse > 1e-8


def target_finger_contacts(task: Any) -> tuple[str, ...]:
    target_id = _entity_id(task.pillbottle.actor)
    if target_id is None:
        return ()
    finger_ids: dict[int, str] = {}
    for name in ("gripper_L_link", "gripper_R_link"):
        link = task.robot.left_entity.find_link_by_name(name)
        identifier = _entity_id(link)
        if identifier is not None:
            finger_ids[identifier] = name

    contacts: set[str] = set()
    for contact in task.scene.get_contacts():
        if len(contact.bodies) != 2 or not _physical_contact(contact):
            continue
        identifiers = {
            identifier for body in contact.bodies if (identifier := _entity_id(body)) is not None
        }
        if target_id not in identifiers:
            continue
        for identifier, name in finger_ids.items():
            if identifier in identifiers:
                contacts.add(name)
    return tuple(sorted(contacts))


class PrimaryFallbackPlanner:
    """Expose the audited CuRobo-primary/MPlib-fallback policy as a planner."""

    def __init__(self, task: Any, helpers: Any) -> None:
        self._task = task
        self._helpers = helpers
        self.last_provenance: dict[str, Any] | None = None

    def plan(self, start_state: Any, goal: Any, context: Any, scene: Any) -> Any:
        plan, provenance = self._helpers.plan_panthera_primary_fallback(
            self._task,
            start_state,
            goal,
            context,
            scene=scene,
        )
        self.last_provenance = provenance
        return plan


class ReleasedTargetCollisionScreenedPlanner:
    """Reject stale CuRobo world paths; consider target-aware MPlib fallback.

    The target must have been freshly injected into MPlib after release
    (10mm separation) or after separation (40mm retreat). The caller still
    performs the full collision recheck and strict runtime contact evaluation.
    """

    def __init__(self, task: Any, helpers: Any, *, phase: str = "retreat") -> None:
        if phase not in ("separation", "retreat"):
            raise ValueError("collision-screened post-release phase must be supported")
        self._task = task
        self._helpers = helpers
        self._phase = phase
        self.last_provenance: dict[str, Any] | None = None

    def plan(self, start_state: Any, goal: Any, context: Any, scene: Any) -> Any:
        from vla_robot_sorting.planning import (
            PantheraRoboTwinMotionPlanner,
            PantheraRoboTwinMPlibPlanner,
        )

        primary = PantheraRoboTwinMotionPlanner(self._task.robot).plan(
            start_state, goal, context, scene=scene
        )
        screen: dict[str, Any] | None = None
        if primary.succeeded and primary.path is not None:
            screen = self._helpers.collision_recheck(self._task, primary.path)
        primary_safe = (
            primary.succeeded
            and primary.path is not None
            and isinstance(screen, dict)
            and screen.get("collision_free") is True
        )
        fallback = None
        selected = primary
        if not primary_safe:
            fallback = PantheraRoboTwinMPlibPlanner(self._task.robot).plan(
                start_state, goal, context, scene=scene
            )
            selected = fallback
        self.last_provenance = {
            "strategy": "released_target_collision_screen_v1",
            "primary_planner_id": primary.planner_id,
            # For the existing strict policy, success means an FCL-eligible
            # primary path, not merely a vendor Success on a colliding path.
            # Preserve the unfiltered native outcome separately for audit.
            "primary_success": bool(primary_safe),
            "primary_native_success": bool(primary.succeeded),
            "primary_collision_check": screen,
            "primary_collision_safe": bool(primary_safe),
            "fallback_attempted": fallback is not None,
            "fallback_planner_id": (fallback.planner_id if fallback is not None else None),
            "fallback_success": (bool(fallback.succeeded) if fallback is not None else None),
            "selected_planner_id": selected.planner_id,
            "plan_success": bool(selected.succeeded),
            "planning_time_s": float(primary.planning_time_s)
            + (float(fallback.planning_time_s) if fallback else 0.0),
            "target_world_refreshed_after_separation": self._phase == "retreat",
            "target_world_refreshed_before_motion": True,
            "phase": self._phase,
        }
        return selected


class PreparedPlanPlanner:
    """Expose one already-built PlannerResult through the normal execution pipeline."""

    def __init__(self, plan: Any, provenance: dict[str, Any]) -> None:
        self._plan = plan
        self.last_provenance = dict(provenance)

    def plan(self, start_state: Any, goal: Any, context: Any, scene: Any) -> Any:
        del start_state, goal, context, scene
        return self._plan


class PreparedPregraspInteriorPlanner(PreparedPlanPlanner):
    """Revalidate B5's exact chosen native trajectory against live start/goal."""

    def __init__(self, plan: Any, provenance: dict[str, Any], candidate_id: str) -> None:
        super().__init__(plan, provenance)
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("selected B5 pregrasp candidate ID is required")
        self.last_provenance.update(
            {"prepared_from_b5": True, "prepared_candidate_id": candidate_id}
        )

    def plan(self, start_state: Any, goal: Any, context: Any, scene: Any) -> Any:
        from vla_robot_sorting.contracts.motion import (
            MotionFailure,
            MotionFailureCode,
            PlannerResult,
        )
        from vla_robot_sorting.planning.panthera_robotwin import pose6d_to_robotwin_target

        result = self._plan
        path = getattr(result, "path", None)
        metadata = getattr(path, "metadata", None)
        try:
            expected_goal = pose6d_to_robotwin_target(goal)
            if not result.succeeded or path is None or not path.positions:
                raise ValueError("B5 did not cache a successful pregrasp plan")
            if not isinstance(metadata, dict):
                raise ValueError("B5 cached plan has no provenance metadata")
            if metadata.get("scene_revision") != context.scene_revision:
                raise ValueError("B5 pregrasp plan scene revision changed")
            if metadata.get("transform_revision") != context.transform_revision:
                raise ValueError("B5 pregrasp plan transform revision changed")
            recorded_goal = metadata.get("target_pose_robotwin_wxyz")
            if (
                recorded_goal is None
                or len(recorded_goal) != 7
                or any(
                    not (math.isfinite(float(a)) and abs(float(a) - float(b)) <= 1e-9)
                    for a, b in zip(recorded_goal, expected_goal, strict=True)
                )
            ):
                raise ValueError("B5 pregrasp plan target pose mismatch")
            if path.joint_names != start_state.joint_names or any(
                not (math.isfinite(float(a)) and abs(float(a) - float(b)) <= 1e-5)
                for a, b in zip(path.positions[0], start_state.positions, strict=True)
            ):
                raise ValueError("B5 pregrasp plan start state changed")
        except (TypeError, ValueError, OverflowError) as exc:
            return PlannerResult(
                path=None,
                failure=MotionFailure(
                    code=MotionFailureCode.EXECUTION_STALE,
                    message=str(exc),
                    recoverable=False,
                    details={"prepared_pregrasp_rejected": True},
                ),
                planning_time_s=0.0,
                planner_id=result.planner_id,
            )
        return result



class PreparedBoundedReleasePlanner(PreparedPregraspInteriorPlanner):
    """Bind the screened released-object path to the original pose and qpos."""

    def __init__(self, plan: Any, provenance: dict[str, Any], direction_label: str) -> None:
        super().__init__(plan, provenance, direction_label)
        self.last_provenance.pop("prepared_from_b5", None)
        self.last_provenance.pop("prepared_candidate_id", None)
        self.last_provenance["prepared_postrelease_direction"] = direction_label
        self.last_provenance["prepared_postrelease_goal_verified"] = True


class PreparedCarriedPlacePlanner(PreparedPlanPlanner):
    """Execute only the exact pre-screened placement path and exact live goal."""

    def __init__(self, plan: Any, goal_pose: Any, yaw_offset_rad: float) -> None:
        super().__init__(
            plan,
            {
                "strategy": "joint5_placement_lookahead_v1",
                "selected_yaw_offset_rad": yaw_offset_rad,
                "prepared_placement_plan_verified": True,
            },
        )
        self._expected_goal = goal_pose

    def plan(self, start_state: Any, goal: Any, context: Any, scene: Any) -> Any:
        from vla_robot_sorting.contracts.motion import (
            MotionFailure,
            MotionFailureCode,
            PlannerResult,
        )
        from vla_robot_sorting.planning.panthera_robotwin import pose6d_to_robotwin_target
        from vla_robot_sorting.planning.placement_lookahead import fingerprint_joint_path

        result = self._plan
        path = getattr(result, "path", None)
        metadata = getattr(path, "metadata", None)
        try:
            if goal.pose != self._expected_goal or not result.succeeded or path is None:
                raise ValueError("pre-screened placement goal changed")
            if not isinstance(metadata, dict):
                raise ValueError("pre-screened placement metadata missing")
            if metadata.get("scene_revision") != context.scene_revision or (
                metadata.get("transform_revision") != context.transform_revision
            ):
                raise ValueError("placement planning revision changed")
            expected_pose = pose6d_to_robotwin_target(goal)
            actual_pose = metadata.get("lookahead_goal_robotwin_wxyz")
            if not isinstance(actual_pose, (tuple, list)) or len(actual_pose) != 7:
                raise ValueError("placement goal attestation missing")
            if any(
                not math.isfinite(float(a)) or abs(float(a) - float(b)) > 1e-9
                for a, b in zip(actual_pose, expected_pose, strict=True)
            ):
                raise ValueError("pre-screened placement target changed")
            if tuple(path.joint_names) != tuple(start_state.joint_names) or any(
                abs(float(a) - float(b)) > 1e-5
                for a, b in zip(path.positions[0], start_state.positions, strict=True)
            ):
                raise ValueError("pre-screened placement start joint state changed")
            if metadata.get("lookahead_plan_sha256") != fingerprint_joint_path(path):
                raise ValueError("pre-screened placement path digest changed")
        except (TypeError, ValueError, OverflowError) as exc:
            return PlannerResult(
                path=None,
                failure=MotionFailure(
                    code=MotionFailureCode.EXECUTION_STALE,
                    message=str(exc),
                    recoverable=False,
                ),
                planning_time_s=0.0,
                planner_id=result.planner_id,
            )
        return result


def support_escape_distance_m(
    support: dict[str, Any],
    *,
    minimum_escape_m: float = 0.005,
    clearance_margin_m: float = 0.001,
    maximum_escape_m: float = 0.01,
) -> float | None:
    if (
        support.get("diagnostic_only") is not True
        or support.get("conservative_overlap") is not True
        or support.get("within_surface_sphere_radius") is not True
    ):
        return None
    try:
        required = float(support["required_lift_to_sphere_clearance_m"])
    except (KeyError, TypeError, ValueError):
        return None
    if required < 0.0:
        return None
    requested = max(float(minimum_escape_m), required + float(clearance_margin_m))
    if requested > float(maximum_escape_m):
        return None
    return requested


def support_escape_trigger(
    initial_lift: dict[str, Any],
    support: dict[str, Any],
    table_diagnostic: dict[str, Any],
) -> bool:
    failure = initial_lift.get("failure")
    details = failure.get("details") if isinstance(failure, dict) else None
    native_status = details.get("native_status") if isinstance(details, dict) else None
    table_start = table_diagnostic.get("start_state")
    return (
        initial_lift.get("plan_success") is False
        and isinstance(native_status, str)
        and "colliding with world" in native_status.lower()
        and support_escape_distance_m(support) is not None
        and table_diagnostic.get("planning_only") is True
        and table_diagnostic.get("executed") is False
        and table_diagnostic.get("obstacle_name") == "table"
        and table_diagnostic.get("obstacle_disabled") is True
        and table_diagnostic.get("obstacle_reenabled") is True
        and isinstance(table_start, dict)
        and table_start.get("valid") is True
        and table_diagnostic.get("plan_success") is True
    )


class RuntimeSafetyStop(RuntimeError):
    """Stops the project executor at an observed violation; not recovery authorization."""


class ContactObserver:
    def __init__(
        self,
        task: Any,
        *,
        capture_contact_trace: bool = False,
        strict_policy: dict[str, bool] | None = None,
        limits: Any = None,
    ) -> None:
        self.strict_policy = strict_policy
        self.limits = limits
        self.runtime_checkpoint = {
            "joint_checks": 0,
            "minimum_margin_rad": None,
            "first_failure": None,
        }
        self.live_capture = None
        if getattr(task, "_additional_diagnostic_dir", None) is not None:
            import importlib.metadata

            from sapien import physx

            from vla_robot_sorting.planning.recovery_query_diagnostics import (
                MotionDiagnosticCapture,
            )

            if importlib.metadata.version("sapien") != "3.0.0b1" or any(
                not hasattr(physx.PhysxContactPoint, n)
                for n in ("position", "normal", "impulse", "separation")
            ):
                raise RuntimeSafetyStop("unverified native contact snapshot API")
            self.live_capture = MotionDiagnosticCapture(
                source_kind="native_privileged_simulator_state"
            )
            self.live_capture.observe(0, lambda: read_native_motion_state(task))
        self.target_id = _entity_id(task.pillbottle.actor)
        self.robot_ids = {
            identifier
            for link in task.robot.left_entity.get_links()
            if (identifier := _entity_id(link)) is not None
        }
        self.finger_ids = {
            identifier
            for name in ("gripper_L_link", "gripper_R_link")
            if (identifier := _entity_id(task.robot.left_entity.find_link_by_name(name)))
            is not None
        }
        self.finger_link_names = {
            identifier: name
            for name in ("gripper_L_link", "gripper_R_link")
            if (identifier := _entity_id(task.robot.left_entity.find_link_by_name(name)))
            is not None
        }
        self.capture_contact_trace = capture_contact_trace
        self.contact_trace: list[dict[str, Any]] = []
        self.contact_trace_truncated = False
        # Includes zero-contact frames: without them a successful retreat
        # records no poses, preventing a controlled PASS/FAIL comparison.
        self.motion_frame_trace: list[dict[str, Any]] = []
        self.motion_frame_trace_truncated = False
        self.pad_id = _entity_id(task.pad.actor)
        self.table_id = _entity_id(task.table)
        self.physics_steps = 0
        self.robot_target_frames = 0
        self.robot_target_finger_frames = 0
        self.robot_target_other_link_frames = 0
        self.first_robot_target_step: int | None = None
        self.last_robot_target_step: int | None = None
        self.target_pad_frames = 0
        self.target_table_frames = 0
        self.robot_table_frames = 0

    def on_physics_step(self, task: Any) -> None:
        self.physics_steps += 1
        target_robot = False
        target_robot_finger = False
        target_robot_other_link = False
        touched_finger_names: set[str] = set()
        target_pad = False
        target_table = False
        robot_table = False
        for contact in task.scene.get_contacts():
            if len(contact.bodies) != 2 or not _physical_contact(contact):
                continue
            identifiers = {
                identifier
                for body in contact.bodies
                if (identifier := _entity_id(body)) is not None
            }
            if (
                self.table_id is not None
                and self.table_id in identifiers
                and identifiers.intersection(self.robot_ids)
            ):
                robot_table = True
            if self.target_id not in identifiers:
                continue
            if identifiers.intersection(self.robot_ids):
                target_robot = True
                target_robot_finger |= bool(identifiers.intersection(self.finger_ids))
                touched_finger_names.update(
                    self.finger_link_names[identifier]
                    for identifier in identifiers.intersection(self.finger_ids)
                )
                target_robot_other_link |= bool(
                    identifiers.intersection(self.robot_ids - self.finger_ids)
                )
            if self.pad_id is not None and self.pad_id in identifiers:
                target_pad = True
            if self.table_id is not None and self.table_id in identifiers:
                target_table = True
        self.robot_target_frames += int(target_robot)
        self.robot_target_finger_frames += int(target_robot_finger)
        self.robot_target_other_link_frames += int(target_robot_other_link)
        if target_robot:
            if self.first_robot_target_step is None:
                self.first_robot_target_step = self.physics_steps
            self.last_robot_target_step = self.physics_steps
        if self.capture_contact_trace:
            self._capture_motion_frame(
                task, target_robot, touched_finger_names, target_robot_other_link
            )
        self.target_pad_frames += int(target_pad)
        self.target_table_frames += int(target_table)
        self.robot_table_frames += int(robot_table)
        try:
            self._runtime_check(task)
        except RuntimeSafetyStop:
            if self.live_capture is not None:
                self.live_capture.observe(
                    self.physics_steps, lambda: read_native_motion_state(task), violation=True
                )
            raise
        if self.live_capture is not None:
            self.live_capture.observe(self.physics_steps, lambda: read_native_motion_state(task))

    def on_command(self, index, position, velocity, time_s):
        if self.live_capture is not None:
            self.live_capture.on_command(index, position, velocity, time_s)

    def finish_live_capture(self, task):
        if self.live_capture is not None:
            self.live_capture.observe(
                self.physics_steps, lambda: read_native_motion_state(task), terminal=True
            )
            return self.live_capture.payload()
        return None

    def _runtime_check(self, task: Any) -> None:
        if self.strict_policy is None and self.limits is None:
            return
        from vla_robot_sorting.contracts import JointState
        from vla_robot_sorting.planning.manipulation_recovery import joint_bound_report

        if self.strict_policy is not None and not runtime_contact_policy_pass(
            self.payload(), **self.strict_policy
        ):
            self.runtime_checkpoint["first_failure"] = {
                "physics_step": self.physics_steps,
                "reason": "forbidden_contact",
                "contacts": self.payload(),
            }
            raise RuntimeSafetyStop("forbidden_contact; no further trajectory commands")
        if self.limits is None:
            raise RuntimeSafetyStop("missing measured joint limits")
        qpos = tuple(float(v) for v in task.robot.get_left_arm_real_jointState()[:6])
        report = joint_bound_report(JointState(self.limits.joint_names, qpos), self.limits)
        self.runtime_checkpoint["joint_checks"] += 1
        previous = self.runtime_checkpoint["minimum_margin_rad"]
        margin = report["minimum_margin_rad"]
        self.runtime_checkpoint["minimum_margin_rad"] = (
            margin if previous is None else min(previous, margin)
        )
        if not report["within_bounds"]:
            self.runtime_checkpoint["first_failure"] = {
                "physics_step": self.physics_steps,
                "reason": "measured_joint_limit",
                "bounds": report,
            }
            raise RuntimeSafetyStop("measured_joint_limit; no further trajectory commands")

    def _capture_motion_frame(
        self,
        task: Any,
        target_contact: bool,
        finger_names: set[str],
        other_link_contact: bool,
    ) -> None:
        """Record joint/object/hand state even when retreat has zero contacts."""
        if len(self.motion_frame_trace) < 256:
            frame: dict[str, Any] = {
                "physics_step": self.physics_steps,
                "robot_target_contact": target_contact,
                "finger_links": sorted(finger_names),
                "other_robot_link_contact": other_link_contact,
            }
            try:
                frame["target_xyz_m"] = list(actor_pose6d(task.pillbottle).position_xyz)
                frame["ee_xyz_m"] = list(
                    robotwin_pose6d(task.robot.get_left_ee_pose()).position_xyz
                )
                frame["arm_qpos_rad"] = [
                    float(value) for value in task.robot.get_left_arm_real_jointState()[:6]
                ]
            except Exception as exc:
                frame["snapshot_error"] = f"{type(exc).__name__}: {exc}"
            try:
                frame["finger_link_world_pose"] = {
                    name: pose_payload(task.robot.left_entity.find_link_by_name(name))
                    for name in ("gripper_L_link", "gripper_R_link")
                }
            except Exception as exc:
                frame["finger_pose_error"] = f"{type(exc).__name__}: {exc}"
            self.motion_frame_trace.append(frame)
        else:
            self.motion_frame_trace_truncated = True
        if target_contact:
            self._capture_contact_frame(task, finger_names, other_link_contact)

    def motion_frame_trace_payload(self) -> dict[str, Any]:
        return {
            "schema": "retreat-kinematic-frames-v1",
            "frames": list(self.motion_frame_trace),
            "frame_limit": 256,
            "truncated": self.motion_frame_trace_truncated,
            "not_contact_point_or_clearance_measurement": True,
        }

    def _capture_contact_frame(
        self, task: Any, finger_names: set[str], other_link_contact: bool
    ) -> None:
        """Bounded *observations*, not collision-clearance proof or contact points."""
        if len(self.contact_trace) >= 256:
            self.contact_trace_truncated = True
            return
        frame: dict[str, Any] = {
            "physics_step": self.physics_steps,
            "finger_links": sorted(finger_names),
            "other_robot_link_contact": other_link_contact,
        }
        try:
            frame["target_xyz_m"] = list(actor_pose6d(task.pillbottle).position_xyz)
            frame["ee_xyz_m"] = list(robotwin_pose6d(task.robot.get_left_ee_pose()).position_xyz)
            frame["arm_qpos_rad"] = [
                float(value) for value in task.robot.get_left_arm_real_jointState()[:6]
            ]
        except Exception as exc:
            frame["snapshot_error"] = f"{type(exc).__name__}: {exc}"
        self.contact_trace.append(frame)

    def contact_trace_payload(self) -> dict[str, Any]:
        return {
            "schema": "physx-contact-frame-observation-v1",
            "frames": list(self.contact_trace),
            "frame_limit": 256,
            "truncated": self.contact_trace_truncated,
            "not_contact_point_or_clearance_measurement": True,
        }

    def payload(self) -> dict[str, int | None]:
        return {
            "physics_steps": self.physics_steps,
            "robot_target_contact_frames": self.robot_target_frames,
            "robot_target_finger_contact_frames": self.robot_target_finger_frames,
            "robot_target_other_link_contact_frames": self.robot_target_other_link_frames,
            "first_robot_target_step": self.first_robot_target_step,
            "last_robot_target_step": self.last_robot_target_step,
            "target_pad_contact_frames": self.target_pad_frames,
            "target_table_contact_frames": self.target_table_frames,
            "robot_table_contact_frames": self.robot_table_frames,
        }


def runtime_contact_policy_pass(
    contacts: dict[str, int],
    *,
    allow_robot_target_contact: bool,
    allow_target_pad_contact: bool = False,
    allow_target_table_contact: bool = False,
    allow_robot_table_contact: bool = False,
) -> bool:
    return (
        (allow_robot_target_contact or contacts["robot_target_contact_frames"] == 0)
        and (allow_target_pad_contact or contacts["target_pad_contact_frames"] == 0)
        and (allow_target_table_contact or contacts["target_table_contact_frames"] == 0)
        and (allow_robot_table_contact or contacts.get("robot_table_contact_frames", 0) == 0)
    )


def execute_gripper_with_joint_checkpoint(
    task: Any, executor: Any, command: Any, limits: Any
) -> tuple[Any, dict[str, Any]]:
    """Observe strict arm bounds while retaining the existing gripper/contact policy."""
    from vla_robot_sorting.contracts import JointState
    from vla_robot_sorting.planning.manipulation_recovery import joint_bound_report

    def measured():
        return JointState(
            limits.joint_names,
            tuple(float(v) for v in task.robot.get_left_arm_real_jointState()[:6]),
        )

    start = joint_bound_report(measured(), limits)
    if not start["within_bounds"]:
        raise RuntimeSafetyStop("gripper start arm state outside declared joint bounds")
    observer = ContactObserver(task, limits=limits)  # bounds only; no new contact permission
    task.add_physics_step_observer(observer)
    try:
        result = executor.execute_gripper(command)
    finally:
        task.clear_physics_step_observers()
    end = measured()
    checkpoint = {
        "start_joint_bounds": start,
        "terminal_joint_bounds": joint_bound_report(end, limits),
        "qpos_after_rad": list(end.positions),
        "contacts": observer.payload(),
        "runtime_safety_checkpoint": observer.runtime_checkpoint,
        "live_command_observation": observer.finish_live_capture(task),
        "execution_status": "completed" if result.completed else "failed",
    }
    return result, checkpoint


def screen_executable_waypoints(
    task: Any, helpers: Any, trajectory: Any, require_environment_collision_free: bool
) -> dict[str, Any]:
    """Recheck the actual timed command positions; this is not swept-volume proof."""
    from types import SimpleNamespace

    positions = tuple(point.positions for point in trajectory.waypoints)
    if len(positions) < 2:
        raise ValueError("missing executable trajectory waypoints")
    path = SimpleNamespace(positions=positions)
    if require_environment_collision_free:
        return helpers.collision_recheck(task, path)
    return helpers.self_collision_recheck(task, path)


def collect_optional_ik_query(
    task: Any, helpers: Any, goal: Any, stage: str, context: Any, *, allow_rrt: bool = False
) -> dict[str, Any] | None:
    directory = getattr(task, "_additional_diagnostic_dir", None)
    if directory is None:
        return None
    from vla_robot_sorting.planning.manipulation_recovery import write_new_json
    from vla_robot_sorting.planning.native_plan_diagnostics import (
        NativeMPlibDiagnosticBackend,
        pose_vector,
    )
    from vla_robot_sorting.planning.native_recovery_ik import NativeRecoveryIKBackend
    from vla_robot_sorting.planning.recovery_query_diagnostics import collect_ik_diagnostics

    if not stage or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in stage):
        raise ValueError("unsafe diagnostic stage")
    sequence = getattr(task, "_additional_query_sequence", 0)
    task._additional_query_sequence = sequence + 1
    filename = f"{sequence:03d}-{stage}-ik.json"
    record = {
        "schema": "s1m.raw-ik-query.v1",
        "stage": stage,
        "executed": False,
        "controller_eligible": False,
        "native_acceptance": False,
    }
    try:
        geometry = NativeMPlibDiagnosticBackend(task, helpers)
        backend = NativeRecoveryIKBackend(geometry)
        gripper = goal.pose
        target = task.robot._trans_from_gripper_to_endlink(
            [*gripper.position_xyz, gripper.orientation_xyzw[3], *gripper.orientation_xyzw[:3]],
            arm_tag="left",
        )
        phase = "retreat" if stage.startswith("retreat") else stage
        record.update(
            provenance=backend.provenance,
            gripper_goal_world_xyzw=[*gripper.position_xyz, *gripper.orientation_xyzw],
            scene_revision=context.scene_revision,
            transform_revision=context.transform_revision,
        )
        record.update(
            collect_ik_diagnostics(backend, pose_vector(target), phase=phase, allow_rrt=allow_rrt)
        )
        if record.get("rejected_start"):
            raise RuntimeSafetyStop("raw IK diagnostic refuses invalid full measured start")
        record.update(
            stage=stage,
            provenance=backend.provenance,
            gripper_goal_world_xyzw=[*gripper.position_xyz, *gripper.orientation_xyzw],
            scene_revision=context.scene_revision,
            transform_revision=context.transform_revision,
            complete=True,
        )
    except Exception as exc:
        record.update(complete=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_new_json(directory / filename, record)
    return {
        "file": filename,
        "raw_ik_queries": len(record["queries"]),
        "rrt_query_requested": allow_rrt,
        "controller_eligible": False,
    }


def read_native_motion_state(task: Any) -> dict[str, Any]:
    from vla_robot_sorting.planning.recovery_query_diagnostics import read_motion_state

    row = read_motion_state(
        task,
        _entity_id,
        source_kind="native_privileged_simulator_state",
        # Read actual SAPIEN articulation links. MPlib's user_link_names
        # inventory is not an authority for physically observed links and
        # may differ between pinned backend versions.
        link_names=None,
    )
    pose = robotwin_pose6d(task.robot.get_left_ee_pose())
    row["gripper_pose_world_xyzw"] = [*pose.position_xyz, *pose.orientation_xyzw]
    row["target_pose_world"] = pose_payload(task.pillbottle)
    row["pad_pose_world"] = pose_payload(task.pad)
    return row


def collect_optional_plan_trace(
    task: Any, helpers: Any, plan: Any, start: Any, stage: str, context: Any, **metadata: Any
) -> dict[str, Any] | None:
    directory = getattr(task, "_plan_diagnostic_dir", None)
    if directory is None:
        return None
    from vla_robot_sorting.planning.native_plan_diagnostics import NativeMPlibDiagnosticBackend

    backend = NativeMPlibDiagnosticBackend(task, helpers)
    sequence = getattr(task, "_plan_diagnostic_sequence", 0)
    task._plan_diagnostic_sequence = sequence + 1
    rows = list(plan.path.positions) if plan.path is not None else [start.positions]
    filename = f"{sequence:03d}-{stage}.json"
    sample_times = metadata.pop("sample_times", None)
    semantics = metadata.pop("time_semantics", "nominal 250Hz path index; not measured sensor time")
    backend.collect(
        rows,
        sample_times if sample_times is not None else [i * 0.004 for i in range(len(rows))],
        directory / filename,
        stage=stage,
        planner_id=plan.planner_id,
        plan_success=plan.succeeded,
        time_semantics=semantics,
        scene_revision=context.scene_revision,
        transform_revision=context.transform_revision,
        **metadata,
    )
    return {"file": filename, "waypoint_count": len(rows)}


def execute_planned_motion(
    *,
    task: Any,
    helpers: Any,
    planner: Any,
    goal: Any,
    context: Any,
    scene: Any,
    limits: Any,
    timer: Any,
    timing_gate: dict[str, Any],
    stage: str,
    require_environment_collision_free: bool,
    allow_robot_target_contact: bool,
    allow_target_pad_contact: bool = False,
    allow_target_table_contact: bool = False,
) -> dict[str, Any]:
    from vla_robot_sorting.contracts import JointState
    from vla_robot_sorting.contracts.motion import ExecutionStatus
    from vla_robot_sorting.control import (
        PantheraRoboTwinTrajectoryExecutor,
        build_execution_segment,
    )
    from vla_robot_sorting.planning.manipulation_recovery import joint_bound_report
    from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits

    start = helpers.current_arm_state(task)
    start_bounds = joint_bound_report(start, limits)
    if not start_bounds["within_bounds"]:
        return {
            "stage": stage,
            "passed": False,
            "plan_success": False,
            "start_joint_bounds": start_bounds,
            "failure": {
                "code": "dynamic_limit_failed",
                "message": "measured start qpos outside declared bounds; no state correction",
                "details": start_bounds,
            },
        }
    plan = planner.plan(start, goal, context, scene=scene)
    result: dict[str, Any] = {
        "stage": stage,
        "start_joint_bounds": start_bounds,
        "planner_id": plan.planner_id,
        "planning_time_s": plan.planning_time_s,
        "plan_success": plan.succeeded,
        "passed": False,
    }
    planner_policy = getattr(planner, "last_provenance", None)
    if isinstance(planner_policy, dict):
        result["planner_policy"] = dict(planner_policy)
    result["plan_diagnostic"] = collect_optional_plan_trace(
        task,
        helpers,
        plan,
        start,
        stage,
        context,
        allow_robot_target_contact=allow_robot_target_contact,
        goal_world_xyzw=[*goal.pose.position_xyz, *goal.pose.orientation_xyzw]
        if goal is not None
        else None,
    )
    result["raw_ik_diagnostic"] = collect_optional_ik_query(
        task, helpers, goal, stage, context, allow_rrt=(stage == "retreat" and not plan.succeeded)
    )
    if not plan.succeeded or plan.path is None:
        result["failure"] = {
            "code": plan.failure.code.value if plan.failure else "missing",
            "message": plan.failure.message if plan.failure else "missing",
            "details": dict(plan.failure.details) if plan.failure else {},
        }
        return result

    trajectory = timer.parameterize(plan.path, limits)
    if getattr(task, "_additional_diagnostic_dir", None) is not None:
        from types import SimpleNamespace

        timed_plan = SimpleNamespace(
            path=SimpleNamespace(positions=tuple(p.positions for p in trajectory.waypoints)),
            planner_id=plan.planner_id,
            succeeded=plan.succeeded,
        )
        result["timed_plan_diagnostic"] = collect_optional_plan_trace(
            task,
            helpers,
            timed_plan,
            start,
            stage + "-timed",
            context,
            time_semantics="timed command schedule; not measured sensor timestamp",
            sample_times=[p.time_from_start_s for p in trajectory.waypoints],
        )
    dynamic = validate_joint_trajectory_limits(trajectory, limits)
    if require_environment_collision_free:
        collision = helpers.collision_recheck(task, plan.path)
        collision_safe = bool(collision["collision_free"])
    else:
        collision = helpers.self_collision_recheck(task, plan.path)
        collision_safe = bool(collision["self_collision_free"])

    timed_collision = screen_executable_waypoints(
        task, helpers, trajectory, require_environment_collision_free
    )
    timed_key = "collision_free" if require_environment_collision_free else "self_collision_free"
    collision_safe = collision_safe and timed_collision.get(timed_key) is True
    result.update(
        {
            "path_samples": len(plan.path.positions),
            "trajectory_duration_s": trajectory.duration_s,
            "dynamic_safe": dynamic.is_safe,
            "dynamic_violation_count": len(dynamic.violations),
            "collision_validation": collision,
            "timed_collision_validation": timed_collision,
            "collision_safe": collision_safe,
            "path_metadata": plan.path.metadata,
        }
    )
    if not dynamic.is_safe or not collision_safe:
        return result

    # B5 ranks only candidates with interior URDF margin. Revalidate the
    # newly planned executable trajectory before sending any physics commands:
    # the chosen planner may return a different IK/path after B5 evaluation.
    if stage == "pregrasp":
        from vla_robot_sorting.planning.pregrasp_joint_reserve import (
            audit_pregrasp_joint_reserve,
        )

        reserve = audit_pregrasp_joint_reserve(plan.path, trajectory, limits)
        result["pregrasp_joint_reserve"] = reserve
        if not reserve["passed"]:
            result["failure"] = {
                "code": "dynamic_limit_failed",
                "message": "planned pregrasp lacks interior joint reserve; no physics commands",
                "details": reserve,
            }
            return result

    segment = build_execution_segment(
        trajectory,
        context,
        segment_id=stage,
        requested_horizon_s=trajectory.duration_s,
        controller_period_s=float(trajectory.metadata["controller_period_s"]),
    )
    observer = ContactObserver(
        task,
        capture_contact_trace=(stage == "retreat"),
        limits=limits,
        strict_policy={
            "allow_robot_target_contact": allow_robot_target_contact,
            "allow_target_pad_contact": allow_target_pad_contact,
            "allow_target_table_contact": allow_target_table_contact,
        },
    )
    task.add_physics_step_observer(observer)
    qpos_before = tuple(float(value) for value in task.robot.get_left_arm_real_jointState()[:6])
    executor = PantheraRoboTwinTrajectoryExecutor(
        task,
        scene_revision=lambda: context.scene_revision,
        transform_revision=lambda: context.transform_revision,
        tracking_tolerance_rad=0.08,
        physics_period_s=float(timing_gate["physics_period_s"]),
        command_observer=getattr(observer, "on_command", None),
    )
    started = perf_counter()
    try:
        feedback = executor.execute(segment)
    finally:
        task.clear_physics_step_observers()
    latency = perf_counter() - started
    qpos_after = tuple(float(value) for value in task.robot.get_left_arm_real_jointState()[:6])
    terminal_bounds = joint_bound_report(JointState(start.joint_names, qpos_after), limits)

    expected = segment.predicted_stop_state
    actual = executor.last_actual_state
    tracking_error = None
    if expected is not None and actual is not None:
        tracking_error = max_abs_delta(expected.positions, actual.positions)

    contacts = observer.payload()
    runtime_contact_ok = runtime_contact_policy_pass(
        contacts,
        allow_robot_target_contact=allow_robot_target_contact,
        allow_target_pad_contact=allow_target_pad_contact,
        allow_target_table_contact=allow_target_table_contact,
    )
    passed = (
        feedback.status is ExecutionStatus.COMPLETED
        and terminal_bounds["within_bounds"]
        and not feedback.replan_required
        and contacts["physics_steps"] > 0
        and runtime_contact_ok
        and tracking_error is not None
        and tracking_error <= 0.08
    )
    result.update(
        {
            "execution_status": feedback.status.value,
            "execution_reason": feedback.reason,
            "replan_required": feedback.replan_required,
            "execution_latency_s": latency,
            "contacts": contacts,
            "contact_policy": {
                "allow_robot_target_contact": allow_robot_target_contact,
                "allow_target_pad_contact": allow_target_pad_contact,
                "allow_target_table_contact": allow_target_table_contact,
                "allow_robot_table_contact": False,
            },
            "runtime_contact_policy_pass": runtime_contact_ok,
            "qpos_before_rad": qpos_before,
            "qpos_after_rad": qpos_after,
            "terminal_joint_bounds": terminal_bounds,
            "runtime_safety_checkpoint": getattr(observer, "runtime_checkpoint", {}),
            "live_command_observation": observer.finish_live_capture(task)
            if hasattr(observer, "finish_live_capture")
            else None,
            "max_qpos_delta_rad": max_abs_delta(qpos_before, qpos_after),
            "max_tracking_error_rad": tracking_error,
            "passed": passed,
        }
    )
    if stage == "retreat":
        result["runtime_contact_trace"] = observer.contact_trace_payload()
        result["runtime_motion_frame_trace"] = observer.motion_frame_trace_payload()
    return result


def diagnose_place_vertical_reachability(
    *,
    task: Any,
    helpers: Any,
    carried_planner: Any,
    context: Any,
    scene: Any,
    desired_object_release: Any,
) -> dict[str, Any]:
    """Planning-only IK map; never executes or contributes to acceptance."""
    from vla_robot_sorting.contracts import MotionGoal
    from vla_robot_sorting.orchestration import gripper_pose_for_desired_object_pose

    results: list[dict[str, Any]] = []
    for extra_height_m in (0.01, 0.02, 0.03):
        desired_object = offset_pose_z(desired_object_release, extra_height_m)
        current_object = actor_pose6d(task.pillbottle)
        current_gripper = robotwin_pose6d(task.robot.get_left_ee_pose())
        goal = gripper_pose_for_desired_object_pose(current_object, current_gripper, desired_object)
        record: dict[str, Any] = {
            "extra_height_m": extra_height_m,
            "desired_object_xyz_m": list(desired_object.position_xyz),
            "desired_gripper_xyz_m": list(goal.position_xyz),
            "plan_success": False,
            "native_status": None,
        }
        try:
            plan = carried_planner.plan(
                helpers.current_arm_state(task),
                MotionGoal(goal),
                context,
                scene=scene,
            )
            record["plan_diagnostic"] = collect_optional_plan_trace(
                task,
                helpers,
                plan,
                helpers.current_arm_state(task),
                f"place-height-{int(extra_height_m * 1000)}mm",
                context,
            )
            record["plan_success"] = plan.succeeded is True
            if plan.failure is not None:
                record["failure_code"] = plan.failure.code.value
                details = plan.failure.details
                if isinstance(details, dict):
                    record["native_status"] = details.get("native_status")
        except Exception as exc:
            if getattr(task, "_plan_diagnostic_dir", None) is not None:
                raise
            record["diagnostic_exception"] = f"{type(exc).__name__}: {exc}"
        results.append(record)

    return {
        "schema": "planning_only_place_vertical_reachability_v1",
        "planning_only": True,
        "executed": False,
        "candidate_extra_heights_m": [0.01, 0.02, 0.03],
        "attempts": results,
        "acceptance_override": False,
    }


def execute_bounded_vertical_place_waypoint(
    *,
    task: Any,
    helpers: Any,
    carried_planner: Any,
    context: Any,
    scene: Any,
    limits: Any,
    timer: Any,
    timing_gate: dict[str, Any],
    desired_object_release: Any,
    extra_height_m: float = 0.02,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """After four exact IK failures, first move 20mm above canonical release.

    The final canonical release goal remains unchanged. A passed elevated
    waypoint alone cannot grant placement success or permit gripper release.
    """
    from vla_robot_sorting.contracts import MotionGoal
    from vla_robot_sorting.orchestration import gripper_pose_for_desired_object_pose

    if not math.isclose(extra_height_m, 0.02, abs_tol=1e-12):
        raise ValueError("only the audited +20mm staging waypoint is permitted")

    current_object = actor_pose6d(task.pillbottle)
    current_gripper = robotwin_pose6d(task.robot.get_left_ee_pose())
    raised_object = offset_pose_z(desired_object_release, extra_height_m)
    raised_gripper = gripper_pose_for_desired_object_pose(
        current_object, current_gripper, raised_object
    )
    elevated = execute_planned_motion(
        task=task,
        helpers=helpers,
        planner=carried_planner,
        goal=MotionGoal(raised_gripper),
        context=context,
        scene=scene,
        limits=limits,
        timer=timer,
        timing_gate=timing_gate,
        stage="place_elevated_waypoint",
        require_environment_collision_free=False,
        allow_robot_target_contact=True,
    )
    evidence = {
        "strategy": "bounded_plus_20mm_then_exact_canonical_v2",
        "extra_height_m": extra_height_m,
        "elevated_object_goal_xyz_m": list(raised_object.position_xyz),
        "canonical_object_goal_xyz_m": list(desired_object_release.position_xyz),
        "elevated_gripper_goal_xyz_m": list(raised_gripper.position_xyz),
        "elevated_motion": elevated,
        "canonical_attempted": False,
    }
    if elevated.get("passed") is not True:
        return elevated, None, evidence

    current_object = actor_pose6d(task.pillbottle)
    current_gripper = robotwin_pose6d(task.robot.get_left_ee_pose())
    exact_gripper = gripper_pose_for_desired_object_pose(
        current_object, current_gripper, desired_object_release
    )
    evidence["canonical_gripper_goal_xyz_m"] = list(exact_gripper.position_xyz)
    canonical = execute_planned_motion(
        task=task,
        helpers=helpers,
        planner=carried_planner,
        goal=MotionGoal(exact_gripper),
        context=context,
        scene=scene,
        limits=limits,
        timer=timer,
        timing_gate=timing_gate,
        stage="place_above_support",
        require_environment_collision_free=False,
        allow_robot_target_contact=True,
    )
    evidence["canonical_attempted"] = True
    evidence["canonical_plan_success"] = canonical.get("plan_success")
    evidence["canonical_motion_passed"] = canonical.get("passed")
    return elevated, canonical, evidence


def preflight_place_joint5_lookahead(
    *,
    task: Any,
    helpers: Any,
    carried_planner: Any,
    context: Any,
    scene: Any,
    limits: Any,
    timer: Any,
    desired_object_release: Any,
) -> tuple[PreparedCarriedPlacePlanner, Any, dict[str, Any]] | None:
    """Plan four exact-position placement yaws without any physical commands.

    Trigger only when the original yaw=0 candidate predicts joint5 near its
    declared boundary. Alternative must improve reserve AND pass all
    raw/retimed self-collision, interior joint and dynamic checks.
    The original greedy yaw fallback remains unchanged otherwise.
    """
    from dataclasses import replace

    from vla_robot_sorting.contracts import MotionGoal
    from vla_robot_sorting.orchestration import gripper_pose_for_desired_object_pose
    from vla_robot_sorting.planning.panthera_robotwin import pose6d_to_robotwin_target
    from vla_robot_sorting.planning.placement_lookahead import (
        NEAR_JOINT5_LIMIT_RAD,
        choose_safe_yaw,
        fingerprint_joint_path,
        joint5_margin,
    )
    from vla_robot_sorting.planning.pregrasp_joint_reserve import audit_pregrasp_joint_reserve
    from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits

    if any(value is None for value in (task, helpers, carried_planner, context, limits, timer)):
        return None
    start = helpers.current_arm_state(task)
    # The measured transport terminal state is the cheap pre-gate. Preserve
    # the exact original planner call count for well-interior seeds 0/2.
    measured_margin = joint5_margin(start.positions, limits)
    if measured_margin >= NEAR_JOINT5_LIMIT_RAD or measured_margin < 0.0:
        return None
    object_pose = actor_pose6d(task.pillbottle)
    ee_pose = robotwin_pose6d(task.robot.get_left_ee_pose())
    base_goal = gripper_pose_for_desired_object_pose(object_pose, ee_pose, desired_object_release)
    baseline = carried_planner.plan(start, MotionGoal(base_goal), context, scene=scene)
    if not baseline.succeeded or baseline.path is None:
        return None
    base_margin = joint5_margin(baseline.path.positions[-1], limits)
    if base_margin >= NEAR_JOINT5_LIMIT_RAD or base_margin < 0:
        return None

    yaw_offsets = (0.0, math.pi / 2.0, -math.pi / 2.0, math.pi)
    records: list[dict[str, Any]] = []
    saved: dict[float, tuple[Any, Any]] = {}
    for yaw in yaw_offsets:
        desired = yaw_rotated_pose(desired_object_release, yaw)
        goal_pose = gripper_pose_for_desired_object_pose(object_pose, ee_pose, desired)
        plan = (
            baseline
            if yaw == 0.0
            else carried_planner.plan(start, MotionGoal(goal_pose), context, scene=scene)
        )
        record: dict[str, Any] = {
            "yaw_offset_rad": yaw,
            "planner_id": plan.planner_id,
            "planner_success": bool(plan.succeeded and plan.path is not None),
            "dynamic_safe": False,
            "raw_self_collision_free": False,
            "timed_self_collision_free": False,
            "joint_interior_safe": False,
            "eligible": False,
        }
        records.append(record)
        if not record["planner_success"]:
            record["failure_code"] = (
                plan.failure.code.value if plan.failure is not None else "missing"
            )
            continue
        try:
            margin = joint5_margin(plan.path.positions[-1], limits)
            record["joint5_margin_rad"] = margin
            record["path_sha256"] = fingerprint_joint_path(plan.path)
            if margin < 0.0:
                continue
            traj = timer.parameterize(plan.path, limits)
            dyn = validate_joint_trajectory_limits(traj, limits)
            raw_col = helpers.self_collision_recheck(task, plan.path)
            timed_col = screen_executable_waypoints(task, helpers, traj, False)
            reserve = audit_pregrasp_joint_reserve(plan.path, traj, limits)
            record.update(
                dynamic_safe=dyn.is_safe and not dyn.violations,
                raw_self_collision_free=raw_col.get("self_collision_free") is True,
                timed_self_collision_free=timed_col.get("self_collision_free") is True,
                joint_interior_safe=reserve.get("passed") is True,
                minimum_all_joint_margin_rad=reserve.get("minimum_margin_rad"),
            )
            record["eligible"] = all(
                record[k] is True
                for k in (
                    "dynamic_safe",
                    "raw_self_collision_free",
                    "timed_self_collision_free",
                    "joint_interior_safe",
                )
            )
            if record["eligible"] and yaw != 0.0:
                saved[yaw] = (plan, goal_pose)
        except (TypeError, ValueError, OverflowError) as exc:
            record["screening_error"] = f"{type(exc).__name__}: {exc}"

    selected = choose_safe_yaw(records, baseline_margin_rad=base_margin)
    if selected is None:
        return None
    yaw = float(selected["yaw_offset_rad"])
    plan, goal_pose = saved[yaw]
    metadata = {
        **plan.path.metadata,
        "lookahead_plan_sha256": selected["path_sha256"],
        "lookahead_goal_robotwin_wxyz": list(pose6d_to_robotwin_target(MotionGoal(goal_pose))),
        "lookahead_selected_yaw_offset_rad": yaw,
    }
    prepared = replace(plan, path=replace(plan.path, metadata=metadata))
    report = {
        "schema": "s1m.joint5-placement-lookahead.v1",
        "planning_only": True,
        "candidate_yaw_offsets_rad": list(yaw_offsets),
        "baseline_yaw_rad": 0.0,
        "baseline_joint5_margin_rad": base_margin,
        "selected_yaw_offset_rad": yaw,
        "selected_plan_sha256": selected["path_sha256"],
        "selected_goal_robotwin_wxyz": metadata["lookahead_goal_robotwin_wxyz"],
        "evaluated": records,
        "max_candidate_plans": 4,
        "physics_executions": 0,
        "waivers": False,
    }
    return PreparedCarriedPlacePlanner(prepared, goal_pose, yaw), goal_pose, report


def execute_place_with_yaw_recovery(
    *,
    task: Any,
    helpers: Any,
    carried_planner: Any,
    context: Any,
    scene: Any,
    limits: Any,
    timer: Any,
    timing_gate: dict[str, Any],
    desired_object_release: Any,
) -> dict[str, Any]:
    from vla_robot_sorting.contracts import MotionGoal
    from vla_robot_sorting.orchestration import gripper_pose_for_desired_object_pose

    yaw_offsets = (0.0, math.pi / 2.0, -math.pi / 2.0, math.pi)

    # The baseline yaw remains unchanged for well-interior placements. When
    # its joint5 endpoint is near a bound, plan all four allowed yaws BEFORE
    # any physical command and execute at most one safe screened alternative.
    preflight = preflight_place_joint5_lookahead(
        task=task,
        helpers=helpers,
        carried_planner=carried_planner,
        context=context,
        scene=scene,
        limits=limits,
        timer=timer,
        desired_object_release=desired_object_release,
    )
    if preflight is not None:
        prepared, prepared_goal, lookahead = preflight
        result = execute_planned_motion(
            task=task,
            helpers=helpers,
            planner=prepared,
            goal=MotionGoal(prepared_goal),
            context=context,
            scene=scene,
            limits=limits,
            timer=timer,
            timing_gate=timing_gate,
            stage="place_above_support",
            require_environment_collision_free=False,
            allow_robot_target_contact=True,
        )
        yaw = lookahead["selected_yaw_offset_rad"]
        result["placement_yaw_offset_rad"] = yaw
        result["placement_recovery"] = {
            "strategy": "joint5_placement_lookahead_v1",
            "task_success_orientation_constrained": False,
            "upstream_task_source": {
                "path": "RoboTwin/envs/move_pillbottle_pad.py",
                "commit": "2aee31ad75cb7b0cd779540441c6025853a5b60d",
                "sha256": "20e7b9d954fd42c3b78e4a9431aee68b3ff10f70f20104e7dc7e41c83fc8cffe",
                "check_success_orientation_constrained": False,
            },
            "candidate_yaw_offsets_rad": list(yaw_offsets),
            "attempts": [
                {
                    "yaw_offset_rad": yaw,
                    "passed": result.get("passed"),
                    "plan_success": result.get("plan_success"),
                    "execution_status": result.get("execution_status"),
                    "failure_code": (
                        result.get("failure", {}).get("code")
                        if isinstance(result.get("failure"), dict)
                        else None
                    ),
                    "native_status": None,
                }
            ],
            "selected_yaw_offset_rad": yaw if result.get("passed") is True else None,
            "bounded_attempt_count": 1,
            "max_attempt_count": len(yaw_offsets),
            "vertical_reachability_diagnostic": None,
            "staged_descent": None,
            "lookahead": {
                **lookahead,
                "physics_executions": int(result.get("execution_status") is not None),
            },
        }
        return result

    attempts: list[dict[str, Any]] = []
    selected_yaw: float | None = None
    final: dict[str, Any] | None = None

    for yaw_offset in yaw_offsets:
        desired_object = yaw_rotated_pose(desired_object_release, yaw_offset)
        current_object = actor_pose6d(task.pillbottle)
        current_gripper = robotwin_pose6d(task.robot.get_left_ee_pose())
        release_goal = gripper_pose_for_desired_object_pose(
            current_object,
            current_gripper,
            desired_object,
        )
        attempt = execute_planned_motion(
            task=task,
            helpers=helpers,
            planner=carried_planner,
            goal=MotionGoal(release_goal),
            context=context,
            scene=scene,
            limits=limits,
            timer=timer,
            timing_gate=timing_gate,
            stage="place_above_support",
            require_environment_collision_free=False,
            allow_robot_target_contact=True,
        )
        attempt["placement_yaw_offset_rad"] = yaw_offset
        failure = attempt.get("failure")
        details = failure.get("details") if isinstance(failure, dict) else None
        attempts.append(
            {
                "yaw_offset_rad": yaw_offset,
                "passed": attempt.get("passed"),
                "plan_success": attempt.get("plan_success"),
                "execution_status": attempt.get("execution_status"),
                "failure_code": (failure.get("code") if isinstance(failure, dict) else None),
                "native_status": (
                    details.get("native_status") if isinstance(details, dict) else None
                ),
            }
        )
        final = attempt
        if attempt.get("passed") is True:
            selected_yaw = yaw_offset
            break
        if not carried_ik_plan_failure(attempt):
            break

    if final is None:
        raise RuntimeError("placement recovery produced no attempt")
    vertical_diagnostic = None
    if selected_yaw is None and carried_ik_plan_failure(final):
        vertical_diagnostic = diagnose_place_vertical_reachability(
            task=task,
            helpers=helpers,
            carried_planner=carried_planner,
            context=context,
            scene=scene,
            desired_object_release=desired_object_release,
        )
    staged_descent = None
    height_attempts = (
        vertical_diagnostic.get("attempts") if isinstance(vertical_diagnostic, dict) else None
    )
    # Only four exact native IK failures and a positive planning-only +20mm
    # waypoint may trigger an additional bounded move. This never bypasses the
    # exact canonical release goal or any runtime safety gate.
    exhausted_exact_ik = len(attempts) == len(yaw_offsets) and all(
        row.get("plan_success") is False
        and row.get("execution_status") is None
        and row.get("failure_code") == "optimization_failed"
        and row.get("native_status") == "IK Fail"
        for row in attempts
    )
    if (
        selected_yaw is None
        and exhausted_exact_ik
        and isinstance(height_attempts, list)
        and len(height_attempts) >= 2
        and height_attempts[1].get("extra_height_m") == 0.02
        and height_attempts[1].get("plan_success") is True
    ):
        elevated, canonical, staged_descent = execute_bounded_vertical_place_waypoint(
            task=task,
            helpers=helpers,
            carried_planner=carried_planner,
            context=context,
            scene=scene,
            limits=limits,
            timer=timer,
            timing_gate=timing_gate,
            desired_object_release=desired_object_release,
        )
        if canonical is not None:
            final = canonical
            if elevated.get("passed") is True and canonical.get("passed") is True:
                selected_yaw = 0.0
    final["placement_recovery"] = {
        "strategy": (
            "yaw_plus_bounded_staged_descent_v2"
            if staged_descent is not None
            else "yaw_only_task_unconstrained_orientation_v1"
        ),
        "task_success_orientation_constrained": False,
        "upstream_task_source": {
            "path": "RoboTwin/envs/move_pillbottle_pad.py",
            "commit": "2aee31ad75cb7b0cd779540441c6025853a5b60d",
            "sha256": "20e7b9d954fd42c3b78e4a9431aee68b3ff10f70f20104e7dc7e41c83fc8cffe",
            "check_success_orientation_constrained": False,
        },
        "candidate_yaw_offsets_rad": list(yaw_offsets),
        "attempts": attempts,
        "selected_yaw_offset_rad": selected_yaw,
        "bounded_attempt_count": len(attempts),
        "max_attempt_count": len(yaw_offsets),
        "vertical_reachability_diagnostic": vertical_diagnostic,
        "staged_descent": staged_descent,
    }
    return final


def bounded_postrelease_separation_goal(
    gripper_pose: Any, object_pose: Any, separation_m: float
) -> tuple[Any, dict[str, Any]]:
    """Move away from the released object's XY center; never approach it."""
    from vla_robot_sorting.contracts import Pose6D

    if not math.isfinite(separation_m) or not 0.0 < separation_m <= 0.01:
        raise ValueError("postrelease separation must remain within 10mm")
    ex, ey, ez = gripper_pose.position_xyz
    ox, oy, _ = object_pose.position_xyz
    dx, dy = ex - ox, ey - oy
    norm = math.hypot(dx, dy)
    if not math.isfinite(norm):
        raise ValueError("invalid gripper/object XY geometry")
    if norm < 0.005:
        return offset_pose_z(gripper_pose, separation_m), {
            "strategy": "vertical_fallback_when_radial_undefined_v1",
            "initial_robot_object_xy_distance_m": norm,
            "direction_xyz_unit": [0.0, 0.0, 1.0],
            "reference_object_xyz_m": list(object_pose.position_xyz),
            "start_gripper_xyz_m": list(gripper_pose.position_xyz),
        }
    ux, uy = dx / norm, dy / norm
    goal = Pose6D(
        position_xyz=(ex + separation_m * ux, ey + separation_m * uy, ez),
        orientation_xyzw=gripper_pose.orientation_xyzw,
        frame_id=gripper_pose.frame_id,
    )
    return goal, {
        "strategy": "radial_xy_away_from_released_object_v1",
        "initial_robot_object_xy_distance_m": norm,
        "direction_xyz_unit": [ux, uy, 0.0],
        "reference_object_xyz_m": list(object_pose.position_xyz),
        "start_gripper_xyz_m": list(gripper_pose.position_xyz),
    }


def bounded_contact_free_retreat_goal(
    gripper_pose: Any, object_pose: Any, distance_m: float = 0.04
) -> tuple[Any, dict[str, Any]]:
    """Plan the full strict-contact-free retreat away from the released object.

    This merely chooses a geometrically justified goal. Every normal runtime
    collision, dynamics, tracking and forbidden-contact gate is unchanged.
    """
    from vla_robot_sorting.contracts import Pose6D

    if not math.isclose(distance_m, 0.04, abs_tol=1e-12):
        raise ValueError("retreat remains exactly 40mm after 10mm separation")
    sx, sy, sz = gripper_pose.position_xyz
    ox, oy, _ = object_pose.position_xyz
    dx, dy = sx - ox, sy - oy
    radius = math.hypot(dx, dy)
    if not math.isfinite(radius):
        raise ValueError("invalid retreat reference geometry")
    if radius < 0.005:
        direction = (0.0, 0.0, 1.0)
        strategy = "vertical_fallback_when_radial_undefined_v1"
    else:
        direction = (dx / radius, dy / radius, 0.0)
        strategy = "radial_xy_away_from_released_object_v1"
    goal = Pose6D(
        position_xyz=(
            sx + distance_m * direction[0],
            sy + distance_m * direction[1],
            sz + distance_m * direction[2],
        ),
        orientation_xyzw=gripper_pose.orientation_xyzw,
        frame_id=gripper_pose.frame_id,
    )
    return goal, {
        "strategy": strategy,
        "requested_retreat_m": distance_m,
        "reference_object_xyz_m": list(object_pose.position_xyz),
        "start_gripper_xyz_m": list(gripper_pose.position_xyz),
        "direction_xyz_unit": list(direction),
        "requested_goal_xyz_m": list(goal.position_xyz),
        "initial_robot_object_xy_distance_m": radius,
        "no_contact_override": True,
    }


def bounded_retreat_diagnostic_goals(
    gripper_pose: Any, object_pose: Any, distance_m: float = 0.04
) -> tuple[tuple[str, Any], ...]:
    """Alternative strictly 40mm, non-inward retreat *planning-only* goals.

    These are not accepted paths. Physics/target collision and native motion
    screening must still pass before any direction could be executable.
    """
    from vla_robot_sorting.contracts import Pose6D

    if not math.isclose(distance_m, 0.04, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("diagnostic retreat distance must remain exactly 40mm")
    if gripper_pose.frame_id != object_pose.frame_id:
        raise ValueError("retreat poses must use the same reference frame")
    sx, sy, sz = gripper_pose.position_xyz
    ox, oy, _ = object_pose.position_xyz
    values = (sx, sy, sz, ox, oy)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("retreat poses must be finite")
    radius = math.hypot(sx - ox, sy - oy)
    if radius < 0.005:
        # The standard goal already uses vertical motion in this case.
        # No evidence-backed XY direction can be derived.
        return ()
    ux, uy = (sx - ox) / radius, (sy - oy) / radius
    a = math.sqrt(0.5)
    directions = (
        ("xy_plus_45deg", (a * (ux - uy), a * (ux + uy), 0.0)),
        ("xy_minus_45deg", (a * (ux + uy), a * (uy - ux), 0.0)),
        ("radial_plus_z_45deg", (a * ux, a * uy, a)),
        ("vertical_plus_z", (0.0, 0.0, 1.0)),
    )
    alternatives = []
    for label, direction in directions:
        dx, dy, dz = direction
        if dx * ux + dy * uy < -1e-12 or dz < -1e-12:
            raise ValueError("retreat diagnostic must never point inward/down")
        if not math.isclose(math.sqrt(dx * dx + dy * dy + dz * dz), 1.0, abs_tol=1e-12):
            raise ValueError("retreat diagnostic direction must be unit length")
        alternatives.append(
            (
                label,
                Pose6D(
                    position_xyz=(
                        sx + distance_m * dx,
                        sy + distance_m * dy,
                        sz + distance_m * dz,
                    ),
                    orientation_xyzw=gripper_pose.orientation_xyzw,
                    frame_id=gripper_pose.frame_id,
                ),
            )
        )
    return tuple(alternatives)


def diagnose_retreat_alternative_plans(
    *,
    task: Any,
    helpers: Any,
    context: Any,
    scene: Any,
    start_state: Any,
    gripper_pose: Any,
    object_pose: Any,
) -> dict[str, Any]:
    """Planning-only after baseline retreat fails; never execute any path."""
    from vla_robot_sorting.contracts import MotionGoal

    alternatives = bounded_retreat_diagnostic_goals(gripper_pose, object_pose)
    result: dict[str, Any] = {
        "schema": "s1m-retreat-direction-planning-only-v1",
        "planning_only": True,
        "executed": False,
        "acceptance_override": False,
        "source_failure_required": True,
        "max_alternative_count": 4,
        "retreat_distance_m": 0.04,
        "observation_scope": "native_planning_and_mplib_collision_recheck_not_physx",
        "attempts": [],
    }
    for label, goal in alternatives:
        row: dict[str, Any] = {
            "label": label,
            "goal_xyz_m": list(goal.position_xyz),
            "plan_success": False,
            "collision_recheck_pass": False,
            "eligible_for_execution": False,
        }
        try:
            planner = ReleasedTargetCollisionScreenedPlanner(task, helpers)
            plan = planner.plan(start_state, MotionGoal(goal), context, scene=scene)
            row["plan_diagnostic"] = collect_optional_plan_trace(
                task, helpers, plan, start_state, f"retreat-{label}", context
            )
            row["raw_ik_diagnostic"] = collect_optional_ik_query(
                task, helpers, MotionGoal(goal), "retreat-" + label, context, allow_rrt=True
            )
            row["planner_id"] = plan.planner_id
            row["plan_success"] = plan.succeeded is True and plan.path is not None
            provenance = planner.last_provenance or {}
            row["primary_collision_safe"] = provenance.get("primary_collision_safe")
            row["fallback_attempted"] = provenance.get("fallback_attempted")
            if plan.failure is not None:
                row["failure_code"] = plan.failure.code.value
                row["native_status"] = dict(plan.failure.details).get(
                    "native_status", dict(plan.failure.details).get("vendor_status")
                )
            if row["plan_success"]:
                screen = helpers.collision_recheck(task, plan.path)
                row["collision_recheck_pass"] = (
                    isinstance(screen, dict) and screen.get("collision_free") is True
                )
                if isinstance(screen, dict):
                    row["collision_waypoints_checked"] = screen.get("waypoints_checked")
                    row["collision_count"] = len(screen.get("collisions", []))
        except Exception as exc:
            if getattr(task, "_plan_diagnostic_dir", None) is not None:
                raise
            row["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
        result["attempts"].append(row)
    return result


def execute_bounded_postrelease_alternatives(
    *,
    task: Any,
    helpers: Any,
    context: Any,
    scene: Any,
    limits: Any,
    timer: Any,
    timing_gate: dict[str, Any],
    phase: str,
    baseline: dict[str, Any],
    gripper_pose: Any,
    object_pose: Any,
    distance_m: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Pre-screen bounded directions BEFORE any alternative physics commands.

    Called only after baseline planning failed without any command/physics
    execution. Forbid multiple physical retries or substitutions after contact.
    The chosen validated trajectory is passed to the same executor unchanged.
    """
    from vla_robot_sorting.contracts import MotionGoal
    from vla_robot_sorting.planning.postrelease_direction_search import alternative_goals
    from vla_robot_sorting.planning.pregrasp_joint_reserve import audit_pregrasp_joint_reserve
    from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits

    if phase not in ("separation", "retreat"):
        raise ValueError("unsupported bounded postrelease phase")
    if baseline.get("passed") is True:
        return baseline, None
    # Never move again after a failed physical attempt, including unsafe
    # contacts, tracking, measured bounds or task-oracle regression.
    if (
        baseline.get("execution_status") is not None
        or baseline.get("contacts") is not None
        or baseline.get("runtime_safety_checkpoint") is not None
    ):
        return baseline, None
    if not baseline.get("start_joint_bounds", {}).get("within_bounds"):
        return baseline, None
    failure = baseline.get("failure")
    if failure is not None and (
        not isinstance(failure, dict) or failure.get("code") not in ("optimization_failed",)
    ):
        # Only a normal IK/optimization failure or a collision/dynamic screen
        # rejection without a failure object may invoke bounded search.
        return baseline, None

    options = alternative_goals(gripper_pose, object_pose, distance_m=distance_m, phase=phase)
    report: dict[str, Any] = {
        "schema": "s1m.postrelease-outward-candidate-search.v1",
        "phase": phase,
        "baseline_failed_before_physics": True,
        "attempted": [],
        "max_candidates": 7 if phase == "separation" else 18,
        "selected_label": None,
        "physics_alternative_executions": 0,
        "contact_and_bounds_waivers": False,
    }
    if len(options) > report["max_candidates"]:
        raise RuntimeError("postrelease candidate budget exceeded")
    start = helpers.current_arm_state(task)

    for label, candidate_goal, geometry in options:
        record: dict[str, Any] = {
            "label": label,
            "goal_xyz_m": list(candidate_goal.position_xyz),
            "planner_success": False,
            "raw_collision_free": False,
            "timed_collision_free": False,
            "dynamic_safe": False,
            "joint_interior_safe": False,
            "eligible": False,
        }
        report["attempted"].append(record)
        planner = None
        plan = None
        if phase == "retreat":
            from vla_robot_sorting.planning import PantheraRoboTwinMPlibScrewPlanner

            screw = PantheraRoboTwinMPlibScrewPlanner(task.robot)
            screw_plan = screw.plan(start, MotionGoal(candidate_goal), context, scene=scene)
            screw_probe = {
                "attempted": True,
                "planner_id": screw_plan.planner_id,
                "native_success": bool(screw_plan.succeeded and screw_plan.path is not None),
                "collision_free": False,
                "vendor_status": (screw.last_provenance or {}).get("vendor_status"),
            }
            record["screw_probe"] = screw_probe
            if screw_probe["native_success"]:
                screw_collision = helpers.collision_recheck(task, screw_plan.path)
                screw_probe["collision_free"] = screw_collision.get("collision_free") is True
                if screw_probe["collision_free"]:
                    plan = screw_plan
                    from types import SimpleNamespace

                    planner = SimpleNamespace(
                        last_provenance={
                            "strategy": "released_target_screw_screen_v1",
                            "primary_planner_id": "robotwin-mplib-screw",
                            "primary_success": True,
                            "primary_collision_check": screw_collision,
                            "primary_collision_safe": True,
                            "fallback_attempted": False,
                            "fallback_planner_id": None,
                            "fallback_success": None,
                            "selected_planner_id": "robotwin-mplib-screw",
                            "plan_success": True,
                            "target_world_refreshed_after_separation": True,
                            "target_world_refreshed_before_motion": True,
                            "phase": phase,
                            "screw_planner_policy": dict(screw.last_provenance or {}),
                        }
                    )
        if plan is None:
            planner = ReleasedTargetCollisionScreenedPlanner(task, helpers, phase=phase)
            plan = planner.plan(start, MotionGoal(candidate_goal), context, scene=scene)
        record["planner_id"] = plan.planner_id
        record["planner_success"] = bool(plan.succeeded and plan.path is not None)
        if not record["planner_success"]:
            record["failure_code"] = (
                plan.failure.code.value if plan.failure is not None else "missing"
            )
            continue
        try:
            timed = timer.parameterize(plan.path, limits)
            dynamic = validate_joint_trajectory_limits(timed, limits)
            raw_collision = helpers.collision_recheck(task, plan.path)
            timed_collision = screen_executable_waypoints(task, helpers, timed, True)
            reserve = audit_pregrasp_joint_reserve(plan.path, timed, limits)
            record.update(
                raw_collision_free=raw_collision.get("collision_free") is True,
                timed_collision_free=timed_collision.get("collision_free") is True,
                dynamic_safe=dynamic.is_safe and not dynamic.violations,
                joint_interior_safe=reserve.get("passed") is True,
                minimum_joint_margin_rad=reserve.get("minimum_margin_rad"),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            record["screening_error"] = f"{type(exc).__name__}: {exc}"
            continue
        record["eligible"] = all(
            record[key] is True
            for key in (
                "raw_collision_free",
                "timed_collision_free",
                "dynamic_safe",
                "joint_interior_safe",
            )
        )
        if not record["eligible"]:
            continue

        report["selected_label"] = label
        cached = PreparedBoundedReleasePlanner(plan, dict(planner.last_provenance or {}), label)
        selected = execute_planned_motion(
            task=task,
            helpers=helpers,
            planner=cached,
            goal=MotionGoal(candidate_goal),
            context=context,
            scene=scene,
            limits=limits,
            timer=timer,
            timing_gate=timing_gate,
            stage="post_release_separation" if phase == "separation" else "retreat",
            require_environment_collision_free=True,
            allow_robot_target_contact=(phase == "separation"),
            allow_target_pad_contact=True,
            allow_target_table_contact=(phase == "retreat"),
        )
        report["physics_alternative_executions"] = int(selected.get("execution_status") is not None)
        selected["bounded_alternative_search"] = report
        return selected, geometry

    baseline["bounded_alternative_search"] = report
    return baseline, None


def execute_post_release_separation(
    *,
    task: Any,
    helpers: Any,
    context: Any,
    scene: Any,
    limits: Any,
    timer: Any,
    timing_gate: dict[str, Any],
    separation_m: float = 0.01,
) -> dict[str, Any]:
    from vla_robot_sorting.contracts import MotionGoal

    if not math.isfinite(separation_m) or separation_m <= 0.0 or separation_m > 0.01:
        return {
            "stage": "post_release_separation",
            "passed": False,
            "requested_separation_m": separation_m,
            "failure": {"code": "separation_distance_out_of_bounds"},
        }

    goal, separation_geometry = bounded_postrelease_separation_goal(
        robotwin_pose6d(task.robot.get_left_ee_pose()),
        actor_pose6d(task.pillbottle),
        separation_m,
    )
    result = execute_planned_motion(
        task=task,
        helpers=helpers,
        planner=ReleasedTargetCollisionScreenedPlanner(task, helpers, phase="separation"),
        goal=MotionGoal(goal),
        context=context,
        scene=scene,
        limits=limits,
        timer=timer,
        timing_gate=timing_gate,
        stage="post_release_separation",
        require_environment_collision_free=True,
        allow_robot_target_contact=True,
        allow_target_pad_contact=True,
        allow_target_table_contact=False,
    )
    if result.get("passed") is not True and result.get("execution_status") is None:
        alternative, alternative_geometry = execute_bounded_postrelease_alternatives(
            task=task,
            helpers=helpers,
            context=context,
            scene=scene,
            limits=limits,
            timer=timer,
            timing_gate=timing_gate,
            phase="separation",
            baseline=result,
            gripper_pose=robotwin_pose6d(task.robot.get_left_ee_pose()),
            object_pose=actor_pose6d(task.pillbottle),
            distance_m=separation_m,
        )
        result = alternative
        if alternative_geometry is not None:
            separation_geometry = alternative_geometry
            goal_xyz = result["bounded_alternative_search"]["attempted"][-1]["goal_xyz_m"]
        else:
            goal_xyz = list(goal.position_xyz)
    else:
        goal_xyz = list(goal.position_xyz)
    contacts = result.get("contacts")
    contacts = contacts if isinstance(contacts, dict) else {}
    fingers_after = target_finger_contacts(task)
    task_success_after = bool(task.check_success())
    result.update(
        {
            "requested_separation_m": separation_m,
            "maximum_separation_m": 0.01,
            "separation_geometry": separation_geometry,
            "requested_goal_xyz_m": goal_xyz,
            "task_success_after": task_success_after,
            "finger_contacts_after": list(fingers_after),
            "target_supported_during_separation": (
                int(contacts.get("physics_steps", 0)) > 0
                and int(contacts.get("target_pad_contact_frames", 0)) > 0
            ),
        }
    )
    result["passed"] = bool(
        result.get("passed") is True
        and task_success_after
        and not fingers_after
        and result["target_supported_during_separation"] is True
        and int(contacts.get("robot_table_contact_frames", 0)) == 0
        and int(contacts.get("target_table_contact_frames", 0)) == 0
    )
    return result


def execute_support_escape(
    *,
    task: Any,
    helpers: Any,
    carried_planner: Any,
    context: Any,
    scene: Any,
    limits: Any,
    timer: Any,
    timing_gate: dict[str, Any],
    support: dict[str, Any],
) -> dict[str, Any]:
    from vla_robot_sorting.contracts import MotionGoal

    requested_escape_m = support_escape_distance_m(support)
    result: dict[str, Any] = {
        "stage": "support_escape",
        "triggered": requested_escape_m is not None,
        "passed": False,
        "requested_escape_m": requested_escape_m,
        "maximum_escape_m": 0.01,
        "support_contact_geometry": dict(support),
    }
    if requested_escape_m is None:
        result["failure"] = {"code": "support_escape_not_bounded"}
        return result

    object_before = actor_pose6d(task.pillbottle)
    gripper_before = robotwin_pose6d(task.robot.get_left_ee_pose())
    escape_goal = MotionGoal(offset_pose_z(gripper_before, requested_escape_m))
    plan, planning = carried_planner.plan_support_escape(
        helpers.current_arm_state(task),
        escape_goal,
        context,
        scene,
        obstacle_name="table",
        requested_escape_m=requested_escape_m,
        max_escape_m=0.01,
    )
    result["planning"] = planning
    if not plan.succeeded or plan.path is None:
        result["failure"] = {
            "code": plan.failure.code.value if plan.failure else "missing",
            "message": plan.failure.message if plan.failure else "missing",
            "details": dict(plan.failure.details) if plan.failure else {},
        }
        return result

    prepared = PreparedPlanPlanner(
        plan,
        {"strategy": "bounded_support_escape", **planning},
    )
    try:
        execution = execute_planned_motion(
            task=task,
            helpers=helpers,
            planner=prepared,
            goal=escape_goal,
            context=context,
            scene=scene,
            limits=limits,
            timer=timer,
            timing_gate=timing_gate,
            stage="support_escape",
            require_environment_collision_free=False,
            allow_robot_target_contact=True,
            allow_target_pad_contact=False,
            allow_target_table_contact=True,
        )
    except Exception as exc:
        result["failure"] = {
            "code": "support_escape_execution_exception",
            "message": f"{type(exc).__name__}: {exc}",
        }
        return result
    result["execution"] = execution

    object_after = actor_pose6d(task.pillbottle)
    object_delta_z = object_after.position_xyz[2] - object_before.position_xyz[2]
    fingers_after = target_finger_contacts(task)
    post_start = carried_planner.diagnose_start_state(helpers.current_arm_state(task))
    contacts = execution.get("contacts") if isinstance(execution, dict) else None
    required_clearance = float(support["required_lift_to_sphere_clearance_m"])
    result.update(
        {
            "object_delta_z_m": object_delta_z,
            "required_clearance_m": required_clearance,
            "finger_contacts_after": list(fingers_after),
            "post_start_state": post_start,
            "contacts": contacts,
            "table_reenabled_before_execution": planning.get("obstacle_reenabled") is True,
        }
    )
    result["passed"] = bool(
        execution.get("passed") is True
        and planning.get("obstacle_reenabled") is True
        and isinstance(contacts, dict)
        and contacts.get("robot_table_contact_frames", 0) == 0
        and contacts.get("target_pad_contact_frames", 0) == 0
        and set(fingers_after) == {"gripper_L_link", "gripper_R_link"}
        and object_delta_z >= required_clearance
        and post_start.get("available") is True
        and post_start.get("valid") is True
    )
    return result


def validate_operator_entrypoint_args(args: argparse.Namespace) -> tuple[bool, bool]:
    """Validate mutually exclusive physical-operator and camera-only CLI modes."""
    operator_mode = args.operator_command is not None or args.operator_language is not None
    capture_only = args.operator_capture_only
    has_rgbd_dir = args.operator_rgbd_dir is not None
    if capture_only and operator_mode:
        raise ValueError("capture-only and physical operator mode are mutually exclusive")
    if capture_only and not has_rgbd_dir:
        raise ValueError("capture-only requires a new RGBD output directory")
    if not capture_only and operator_mode != has_rgbd_dir:
        raise ValueError("operator command/language and fresh RGBD output required together")
    if not capture_only and not operator_mode and has_rgbd_dir:
        raise ValueError("RGBD output is reserved for explicit operator/capture-only mode")
    if operator_mode and (args.operator_command is None or args.operator_language is None):
        raise ValueError("operator command and language must both be set")
    return operator_mode, capture_only


def install_display_viewer(task: Any, every: int) -> dict[str, Any]:
    """Redraw the SAPIEN viewer every ``every`` physics steps; never alters physics.

    Upstream rendering is switched off after setup (``task.render_freq = 0``) so the
    viewer is drawn exactly once per ``every`` scene steps by this wrapper only.
    """
    viewer = getattr(task, "viewer", None)
    if viewer is None:
        raise RuntimeError("display viewer requested but RoboTwin created no viewer")
    original = task._step_scene
    counter = {"steps": 0, "frames": 0}

    def step_and_draw(*step_args: Any, **step_kwargs: Any) -> Any:
        result = original(*step_args, **step_kwargs)
        counter["steps"] += 1
        if counter["steps"] % every == 0 and not getattr(viewer, "closed", False):
            if callable(getattr(task, "_update_render", None)):
                task._update_render()
            viewer.render()
            counter["frames"] += 1
        return result

    task.render_freq = 0
    task._step_scene = step_and_draw
    return {"render_every_physics_steps": every, "physics_affecting": False, "counter": counter}


def main() -> int:
    args = parse_args()
    robotwin_root = args.robotwin_root.resolve()
    project_root = args.project_root.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite native evidence: {output_path}")
    operator_mode, capture_only = validate_operator_entrypoint_args(args)
    if args.operator_rgbd_dir is not None and args.operator_rgbd_dir.resolve().exists():
        raise FileExistsError("operator RGBD output directory already exists")
    if args.additional_diagnostic_dir is not None:
        if args.plan_diagnostic_dir is None:
            raise ValueError("additional diagnostics require validated plan-diagnostic collector")
        args.additional_diagnostic_dir = args.additional_diagnostic_dir.resolve()
        args.additional_diagnostic_dir.mkdir(parents=True, exist_ok=False)
    if args.plan_diagnostic_dir is not None:
        args.plan_diagnostic_dir = args.plan_diagnostic_dir.resolve()
        args.plan_diagnostic_dir.mkdir(parents=True, exist_ok=False)
    gate = json.loads(
        (project_root / "configs/s1.3/grasp_recovery_gates.json").read_text(encoding="utf-8")
    )
    timing_gate = json.loads(
        (project_root / "configs/s1.2/timing_gates.json").read_text(encoding="utf-8")
    )

    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(project_root / "src"))

    import yaml
    from script.collect_data import prepare_task_and_args

    from vla_robot_sorting.contracts import (
        GripperCommand,
        ManipulationPhase,
        ManipulationSignal,
        MotionGoal,
        PlanningContext,
        Pose6D,
    )
    from vla_robot_sorting.control import PantheraRoboTwinGripperExecutor
    from vla_robot_sorting.orchestration import (
        ManipulationStateMachine,
        gripper_pose_for_desired_object_pose,
    )
    from vla_robot_sorting.planning import (
        PantheraRoboTwinCarriedObjectPlanner,
        PantheraRoboTwinMPlibAcquisitionPlanner,
        PhysicsAlignedTimeDilationParameterizer,
        RoboTwinVendorTrajectoryParameterizer,
        load_joint_limits_from_urdf,
    )
    from vla_robot_sorting.planning.native_curobo_interior import (
        native_curobo_interior_setup,
    )
    from vla_robot_sorting.planning.native_initial_home import (
        initialize_native_task_with_home,
    )
    from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES
    from vla_robot_sorting.simulation.render_profile import apply_fast_preview_render_profile

    helpers = load_b57_helpers(project_root)
    payload: dict[str, Any] = {
        "schema": "panthera-native-project-manipulation-v1",
        "recovery_contract_version": "bounded_yaw_postrelease_v4",
        "joint_bound_contract_version": "strict_measured_start_terminal_v1",
        "pregrasp_joint_reserve_contract_version": "raw_timed_5milliradian_v1",
        "pregrasp_plan_reuse_contract_version": "b5_cached_start_goal_scene_v1",
        "native_curobo_interior_contract_version": "scoped_native_cspace_clip_v1",
        "postrelease_direction_search_contract_version": "outward_discrete_prephysics_v2",
        "task": args.task,
        "task_config": args.task_config,
        "seed": args.seed,
        "passed": False,
        "stages": [],
        "upstream_manipulation_macro_used": False,
        "placement_goal_policy": "project_rigid_object_gripper_transform",
        "claim_boundary": (
            "project-owned planning/control/manipulation probe; native PASS requires every "
            "recorded phase plus RoboTwin task success oracle"
        ),
    }
    task = None
    carried_planner = None
    machine = None
    try:
        payload["upstream_task_source"] = verify_upstream_task_source(robotwin_root)
        task, task_args = prepare_task_and_args(args.task, args.task_config)
        if not 0 <= args.render_freq <= 60:
            raise ValueError("--render-freq must be between 0 and 60")
        # Display only: a viewer is created during setup when render_freq > 0.
        task_args["render_freq"] = args.render_freq
        payload["display_render_freq"] = args.render_freq
        task_args["save_data"] = False
        task_args["collect_data"] = False
        task_args["domain_randomization"]["cluttered_table"] = False
        task_args["domain_randomization"]["random_background"] = False
        task_args["domain_randomization"]["random_light"] = False
        apply_fast_preview_render_profile(task_args)

        setup_started = perf_counter()
        pinned_curobo_yaml = (
            robotwin_root / "assets/embodiments/panthera-6dof/curobo.yml"
        )
        with native_curobo_interior_setup(pinned_curobo_yaml) as curobo_policy:
            task_args, payload["initial_home"] = initialize_native_task_with_home(
                task, task_args, seed=args.seed
            )
        payload["native_curobo_interior_planner"] = curobo_policy
        task.robot._vla_native_curobo_search = True
        if args.render_freq:
            payload["display_viewer"] = install_display_viewer(task, args.render_freq)
        payload["setup_duration_s"] = perf_counter() - setup_started
        if payload["initial_home"]["measured"]["passed"] is not True:
            raise RuntimeSafetyStop("native initial joint bounds/reserve failed")
        if args.plan_diagnostic_dir is not None:
            task._plan_diagnostic_dir = args.plan_diagnostic_dir
        if args.additional_diagnostic_dir is not None:
            task._additional_diagnostic_dir = args.additional_diagnostic_dir
        payload["mplib_world_initial"] = helpers.sync_scaled_target_into_mplib_world(task)
        payload["initial_object_pose"] = pose_payload(task.pillbottle)
        payload["initial_pad_pose"] = pose_payload(task.pad)

        if operator_mode or capture_only:
            from vla_robot_sorting.perception.native_task_pair_capture import (
                capture_task_native_pair,
            )

            payload["operator_rgbd_observation"] = capture_task_native_pair(
                task, args.operator_rgbd_dir
            )
            if capture_only:
                payload.update({
                    "passed": False,
                    "final_phase": "observation_only",
                    "task_success_final": False,
                    "native_execution_attempted": False,
                    "claim_boundary": (
                        "RGBD CAPTURE AND METRIC PLAUSIBILITY ONLY; "
                        "NO MANIPULATION OR TASK ACCEPTANCE"
                    ),
                })
                from vla_robot_sorting.planning.manipulation_recovery import write_new_json

                output_path.parent.mkdir(parents=True, exist_ok=True)
                write_new_json(output_path, payload)
                print("NATIVE_RGBD_CAPTURE_ONLY=PASS_NON_CONTROL")
                print("PHYSICS_STEPS=0 MOTION_COMMANDS=0")
                print(f"RGBD_REPORT={args.operator_rgbd_dir}")
                print(f"NATIVE_EVIDENCE={output_path}")
                return 0
        geometry = helpers.actor_box_from_collision_mesh(task, robotwin_root)
        gripper, gripper_provenance = helpers.load_gripper_proxy(robotwin_root, task_args)
        revision = f"project-manip-seed-{args.seed}"
        transform_revision = "robotwin-static-v1"
        scene = helpers.make_scene_snapshot(
            task,
            revision=revision,
            geometry_info=geometry,
            transform_revision=transform_revision,
        )
        context = PlanningContext(revision, transform_revision)
        if args.operator_command is not None or args.operator_language is not None:
            if args.operator_command is None or args.operator_language is None:
                raise RuntimeSafetyStop("both operator command and language are required")
            import secrets
            from dataclasses import replace as dataclass_replace

            from vla_robot_sorting.orchestration.native_command_preflight import (
                compile_native_command,
            )
            from vla_robot_sorting.orchestration.unified_sprint1 import (
                NativeBinding,
                PipelineStatus,
                UnifiedSprint1Pipeline,
            )

            scene = dataclass_replace(
                scene,
                metadata={
                    **scene.metadata,
                    "observation_revision": revision,
                    "observation_source": "sim_ground_truth",
                    "identity_provenance": "simulator_native_actor_registry",
                    "destination_provenance": "simulator_native_actor_registry",
                    "evidence_session": f"native-run-{secrets.token_hex(16)}",
                },
            )
            semantic, command_sha = compile_native_command(
                args.operator_command, args.operator_language, revision
            )
            intent_binding = NativeBinding(
                task=args.task,
                seed=args.seed,
                scene_revision=revision,
                transform_revision=transform_revision,
                observation_revision=revision,
                evidence_session=scene.metadata["evidence_session"],
                target_id="pillbottle",
                destination_id="pad",
            )
            from vla_robot_sorting.grasping.geometry import GeometryFirstGraspGenerator

            native_candidate_generator = GeometryFirstGraspGenerator(
                cylinder_side_grasp_height_fractions=tuple(
                    float(v)
                    for v in gate.get("cylinder_side_grasp_height_fractions", (0.0,))
                ),
                box_side_grasp_height_fractions=tuple(
                    float(v)
                    for v in gate.get("box_side_grasp_height_fractions", (0.0,))
                ),
                box_parallel_jaw_roll_variants=bool(
                    gate.get("box_parallel_jaw_roll_variants", False)
                ),
            )
            preflight = UnifiedSprint1Pipeline(
                grasp_generator=native_candidate_generator,
                minimum_table_clearance_m=float(gate["minimum_table_clearance_m"]),
            ).run(
                command=args.operator_command,
                language=semantic.source_language,
                scene=scene,
                semantic=semantic,
                semantic_command_sha256=command_sha,
                gripper=gripper,
                binding=intent_binding,
                native_backend=None,
            )
            payload["operator_command_preflight"] = {
                "status": preflight.status.value,
                "stage": preflight.stage,
                "reason": preflight.reason,
                "target_id": preflight.target_id,
                "destination_id": preflight.destination_id,
                "grasp_candidate_count": preflight.candidate_count,
                "candidate_ids": list(preflight.candidate_ids),
                "source": "simulator_actor_registry_NOT_RGBD_detection",
                "semantic_provider": "explicit_fixed_task_compiler_NOT_GEMINI",
                "command_sha256": command_sha,
                "controller_eligible_from_semantic": False,
            }
            if (
                preflight.status is not PipelineStatus.NATIVE_PENDING
                or preflight.target_id != "pillbottle"
                or preflight.destination_id != "pad"
                or _entity_id(task.pillbottle.actor) is None
                or _entity_id(task.pad.actor) is None
            ):
                raise RuntimeSafetyStop("operator semantic/native actor preflight failed")

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
            stop_after_feasible_height_group=(not args.b5_evaluate_all_height_groups),
            selected_candidate_id=args.candidate_id,
            preferred_candidate_id=args.preferred_candidate_id,
        )
        selected = b5.pop("_selected_object", None)
        selected_pregrasp_plan = b5.pop("_selected_pregrasp_plan", None)
        payload["B5"] = b5
        payload["candidate_selection_diagnostic"] = {
            "evaluate_all_height_groups": args.b5_evaluate_all_height_groups,
            "candidate_id_override": args.candidate_id,
            "preferred_candidate_id": args.preferred_candidate_id,
        }
        payload["gripper_geometry"] = gripper_provenance
        payload["object_geometry"] = geometry
        if selected is None or b5.get("passed") is not True:
            raise RuntimeError("B5 did not produce a safe selected grasp")
        if operator_mode:
            allowed_ids = set(payload["operator_command_preflight"]["candidate_ids"])
            if selected.candidate_id not in allowed_ids:
                raise RuntimeSafetyStop(
                    "native B5 selected grasp not bound to semantic candidate set"
                )
            payload["operator_command_preflight"]["selected_native_candidate_id"] = (
                selected.candidate_id
            )
            payload["operator_command_preflight"]["candidate_geometry_binding_passed"] = True
        if selected_pregrasp_plan is None:
            raise RuntimeError("B5 selected grasp has no independently screened cached plan")

        machine = ManipulationStateMachine(revision, transform_revision)
        pregrasp_planner = PreparedPregraspInteriorPlanner(
            *selected_pregrasp_plan, selected.candidate_id
        )
        stage = execute_planned_motion(
            task=task,
            helpers=helpers,
            planner=pregrasp_planner,
            goal=MotionGoal(selected.pregrasp_pose),
            context=context,
            scene=scene,
            limits=limits,
            timer=timer,
            timing_gate=timing_gate,
            stage="pregrasp",
            require_environment_collision_free=True,
            allow_robot_target_contact=False,
            allow_target_table_contact=True,
        )
        stage["object_pose_after"] = pose_payload(task.pillbottle)
        payload["stages"].append(stage)
        if not stage["passed"]:
            machine.apply(ManipulationSignal.MOTION_FAILED)
            raise RuntimeError("pregrasp execution failed")
        machine.apply(ManipulationSignal.MOTION_COMPLETED)

        payload["mplib_world_before_approach"] = helpers.sync_scaled_target_into_mplib_world(task)
        approach = execute_planned_motion(
            task=task,
            helpers=helpers,
            planner=PantheraRoboTwinMPlibAcquisitionPlanner(task.robot),
            goal=MotionGoal(selected.grasp_pose),
            context=context,
            scene=scene,
            limits=limits,
            timer=timer,
            timing_gate=timing_gate,
            stage="approach",
            require_environment_collision_free=True,
            allow_robot_target_contact=False,
            allow_target_table_contact=True,
        )
        approach["object_pose_after"] = pose_payload(task.pillbottle)
        payload["stages"].append(approach)
        if not approach["passed"]:
            machine.apply(ManipulationSignal.MOTION_FAILED)
            raise RuntimeError("approach execution failed")
        machine.apply(ManipulationSignal.MOTION_COMPLETED)

        contact_before = target_finger_contacts(task)
        force_scale = float(selected.metadata.get("recommended_force_scale", 1.0))
        gripper_executor = PantheraRoboTwinGripperExecutor(
            task,
            scene_revision=lambda: revision,
            transform_revision=lambda: transform_revision,
            position_tolerance=0.08,
            contact_observed=lambda: bool(target_finger_contacts(task)),
            holding_object=lambda: len(target_finger_contacts(task)) == 2,
        )
        close_result, close_checkpoint = execute_gripper_with_joint_checkpoint(
            task,
            gripper_executor,
            GripperCommand(
                normalized_position=0.0,
                compliance_profile=selected.compliance_profile,
                force_scale=force_scale,
                scene_revision=revision,
                transform_revision=transform_revision,
            ),
            limits,
        )
        contact_after = target_finger_contacts(task)
        close_stage = {
            "stage": "close_gripper",
            **close_checkpoint,
            "passed": close_result.completed
            and len(contact_after) == 2
            and close_checkpoint["terminal_joint_bounds"]["within_bounds"],
            "result": {
                "completed": close_result.completed,
                "actual_normalized_position": close_result.actual_normalized_position,
                "contact_observed": close_result.contact_observed,
                "holding_object": close_result.holding_object,
                "reason": close_result.reason,
            },
            "contacts_before": contact_before,
            "contacts_after": contact_after,
            "compliance_profile": selected.compliance_profile,
            "force_scale": force_scale,
        }
        payload["stages"].append(close_stage)
        if not close_stage["passed"]:
            machine.apply(ManipulationSignal.GRASP_FAILED)
            raise RuntimeError("gripper close failed")
        machine.apply(ManipulationSignal.GRIPPER_CLOSED)
        if len(contact_after) != 2:
            machine.apply(ManipulationSignal.GRASP_FAILED)
            raise RuntimeError("dual-finger grasp confirmation failed")
        machine.apply(ManipulationSignal.GRASP_CONFIRMED)

        grasped_geometry = helpers.actor_box_from_collision_mesh(task, robotwin_root)
        grasped_scene = helpers.make_scene_snapshot(
            task,
            revision=revision,
            geometry_info=grasped_geometry,
            transform_revision=transform_revision,
        )
        carried_config_path = Path(task.robot.left_curobo_yml_path)
        if not carried_config_path.is_absolute():
            carried_config_path = (robotwin_root / carried_config_path).resolve()
        carried_urdf_path = Path(task.robot.left_urdf_path)
        if not carried_urdf_path.is_absolute():
            carried_urdf_path = (robotwin_root / carried_urdf_path).resolve()
        carried_collision_spheres_path = (
            carried_config_path.parent / "collision_panthera.yml"
        ).resolve()
        for label, path in (
            ("curobo_config", carried_config_path),
            ("urdf", carried_urdf_path),
            ("collision_spheres", carried_collision_spheres_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} asset is missing: {path}")
        payload["carried_asset_paths"] = {
            "curobo_config": str(carried_config_path),
            "urdf": str(carried_urdf_path),
            "collision_spheres": str(carried_collision_spheres_path),
        }
        carried_planner = PantheraRoboTwinCarriedObjectPlanner(
            task.robot,
            config_path=carried_config_path,
            object_id="pillbottle",
            attachment_parent_link=str(task.robot.left_move_group),
            asset_path_overrides={
                "urdf_path": carried_urdf_path,
                "collision_spheres": carried_collision_spheres_path,
            },
        )
        attachment = carried_planner.initialize_attachment(
            helpers.current_arm_state(task),
            grasped_scene,
        )
        payload["carried_object_attachment"] = attachment
        payload["carried_start_state_diagnostic"] = carried_planner.diagnose_start_state(
            helpers.current_arm_state(task)
        )

        object_before_lift = actor_pose6d(task.pillbottle)
        gripper_before_lift = robotwin_pose6d(task.robot.get_left_ee_pose())
        lift_goal = offset_pose_z(gripper_before_lift, args.lift_m)
        initial_lift = execute_planned_motion(
            task=task,
            helpers=helpers,
            planner=carried_planner,
            goal=MotionGoal(lift_goal),
            context=context,
            scene=grasped_scene,
            limits=limits,
            timer=timer,
            timing_gate=timing_gate,
            stage="lift",
            require_environment_collision_free=False,
            allow_robot_target_contact=True,
        )
        lift = initial_lift
        if initial_lift.get("plan_success") is False:
            table_diagnostic = carried_planner.diagnose_plan_with_obstacle_disabled(
                helpers.current_arm_state(task),
                MotionGoal(lift_goal),
                obstacle_name="table",
            )
            payload["lift_table_disabled_diagnostic"] = table_diagnostic
            support = attachment.get("support_contact_geometry")
            if isinstance(support, dict) and support_escape_trigger(
                initial_lift, support, table_diagnostic
            ):
                support_escape = execute_support_escape(
                    task=task,
                    helpers=helpers,
                    carried_planner=carried_planner,
                    context=context,
                    scene=grasped_scene,
                    limits=limits,
                    timer=timer,
                    timing_gate=timing_gate,
                    support=support,
                )
                payload["support_escape_diagnostic"] = support_escape
                if support_escape.get("passed") is True:
                    lift = execute_planned_motion(
                        task=task,
                        helpers=helpers,
                        planner=carried_planner,
                        goal=MotionGoal(lift_goal),
                        context=context,
                        scene=grasped_scene,
                        limits=limits,
                        timer=timer,
                        timing_gate=timing_gate,
                        stage="lift",
                        require_environment_collision_free=False,
                        allow_robot_target_contact=True,
                    )
                lift["initial_plan_failure"] = initial_lift.get("failure")
                lift["support_escape"] = support_escape
        object_after_lift = actor_pose6d(task.pillbottle)
        lift_delta_z = object_after_lift.position_xyz[2] - object_before_lift.position_xyz[2]
        lift["object_delta_z_m"] = lift_delta_z
        lift["object_pose_after"] = pose_payload(task.pillbottle)
        lift["passed"] = bool(lift["passed"] and lift_delta_z >= args.lift_m * 0.4)
        payload["stages"].append(lift)
        if not lift["passed"]:
            machine.apply(ManipulationSignal.GRASP_FAILED)
            raise RuntimeError("lift did not prove retained grasp")
        machine.apply(ManipulationSignal.MOTION_COMPLETED)

        current_object = actor_pose6d(task.pillbottle)
        current_gripper = robotwin_pose6d(task.robot.get_left_ee_pose())
        pad_pose = task.pad.get_pose()
        success_z = 0.741 + float(task.table_z_bias)
        desired_object_pre = Pose6D(
            position_xyz=(
                float(pad_pose.p[0]),
                float(pad_pose.p[1]),
                success_z + args.preplace_clearance_m,
            ),
            orientation_xyzw=current_object.orientation_xyzw,
            frame_id="world",
        )
        preplace_goal = gripper_pose_for_desired_object_pose(
            current_object,
            current_gripper,
            desired_object_pre,
        )
        transport = execute_planned_motion(
            task=task,
            helpers=helpers,
            planner=carried_planner,
            goal=MotionGoal(preplace_goal),
            context=context,
            scene=grasped_scene,
            limits=limits,
            timer=timer,
            timing_gate=timing_gate,
            stage="transport",
            require_environment_collision_free=False,
            allow_robot_target_contact=True,
        )
        transport["object_pose_after"] = pose_payload(task.pillbottle)
        payload["stages"].append(transport)
        if not transport["passed"]:
            machine.apply(ManipulationSignal.MOTION_FAILED)
            raise RuntimeError("transport failed")
        machine.apply(ManipulationSignal.MOTION_COMPLETED)

        current_object = actor_pose6d(task.pillbottle)
        desired_object_release = Pose6D(
            position_xyz=(
                float(pad_pose.p[0]),
                float(pad_pose.p[1]),
                success_z + args.release_clearance_m,
            ),
            orientation_xyzw=current_object.orientation_xyzw,
            frame_id="world",
        )
        place = execute_place_with_yaw_recovery(
            task=task,
            helpers=helpers,
            carried_planner=carried_planner,
            context=context,
            scene=grasped_scene,
            limits=limits,
            timer=timer,
            timing_gate=timing_gate,
            desired_object_release=desired_object_release,
        )
        payload["placement_recovery_diagnostic"] = place.get("placement_recovery")
        place["object_pose_after"] = pose_payload(task.pillbottle)
        payload["stages"].append(place)
        if not place["passed"]:
            if args.diagnose_native_ik_blockers:
                # The task already failed. One read-only planner test checks
                # whether table collision is the blocker at the *same*
                # canonical object goal; no substitute placement is accepted.
                diagnostic: dict[str, Any] = {
                    "schema": "s1m-canonical-place-ik-planning-only-v1",
                    "planning_only": True,
                    "executed": False,
                    "acceptance_override": False,
                    "canonical_object_goal_xyz_m": list(desired_object_release.position_xyz),
                }
                try:
                    state = helpers.current_arm_state(task)
                    diagnostic["start_state"] = carried_planner.diagnose_start_state(state)
                    canonical_goal = gripper_pose_for_desired_object_pose(
                        actor_pose6d(task.pillbottle),
                        robotwin_pose6d(task.robot.get_left_ee_pose()),
                        desired_object_release,
                    )
                    diagnostic["canonical_gripper_goal_xyz_m"] = list(canonical_goal.position_xyz)
                    diagnostic["table_disabled_test"] = (
                        carried_planner.diagnose_plan_with_obstacle_disabled(
                            state, MotionGoal(canonical_goal), obstacle_name="table"
                        )
                    )
                except Exception as exc:
                    diagnostic["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
                payload["place_ik_blocker_diagnostic"] = diagnostic
            machine.apply(ManipulationSignal.MOTION_FAILED)
            raise RuntimeError("place-above-support failed")
        machine.apply(ManipulationSignal.MOTION_COMPLETED)

        open_result, open_checkpoint = execute_gripper_with_joint_checkpoint(
            task,
            gripper_executor,
            GripperCommand(
                normalized_position=1.0,
                compliance_profile="standard",
                force_scale=1.0,
                scene_revision=revision,
                transform_revision=transform_revision,
            ),
            limits,
        )
        if carried_planner is not None:
            carried_planner.detach()
        release_stage = {
            "stage": "release",
            **open_checkpoint,
            "passed": open_result.completed
            and open_checkpoint["terminal_joint_bounds"]["within_bounds"],
            "result": {
                "completed": open_result.completed,
                "actual_normalized_position": open_result.actual_normalized_position,
                "contact_observed": open_result.contact_observed,
                "holding_object": open_result.holding_object,
                "reason": open_result.reason,
            },
            "object_pose_after_release": pose_payload(task.pillbottle),
        }
        payload["stages"].append(release_stage)
        if not release_stage["passed"]:
            machine.apply(ManipulationSignal.GRASP_FAILED)
            raise RuntimeError("gripper release failed")
        machine.apply(ManipulationSignal.GRIPPER_OPENED)

        success_after_release = bool(task.check_success())
        payload["task_success_after_release"] = success_after_release
        if not success_after_release:
            machine.apply(ManipulationSignal.MOTION_FAILED)
            raise RuntimeError("RoboTwin task success oracle failed after release")

        # The bottle moved from pickup to the pad. Re-inject its measured
        # post-release pose BEFORE 10 mm separation is collision-screened;
        # waiting until after separation leaves FCL screening against stale state.
        release_world_sync = helpers.sync_scaled_target_into_mplib_world(task)
        release_world_sync["target_pose_world_xyz_m"] = list(
            actor_pose6d(task.pillbottle).position_xyz
        )
        payload["mplib_world_after_release_before_separation"] = release_world_sync
        released_geometry = helpers.actor_box_from_collision_mesh(task, robotwin_root)
        released_scene = helpers.make_scene_snapshot(
            task,
            revision=revision,
            geometry_info=released_geometry,
            transform_revision=transform_revision,
        )
        separation = execute_post_release_separation(
            task=task,
            helpers=helpers,
            context=context,
            scene=released_scene,
            limits=limits,
            timer=timer,
            timing_gate=timing_gate,
            separation_m=0.01,
        )
        payload["post_release_separation_diagnostic"] = separation

        if separation.get("passed") is True:
            released_object_pose = actor_pose6d(task.pillbottle)
            # Late-spawned target was initially injected at acquisition.
            # Refresh actual post-release physics pose before collision
            # screening; a stale target obstacle must never certify retreat.
            released_world_sync = helpers.sync_scaled_target_into_mplib_world(task)
            released_world_sync["target_pose_world_xyz_m"] = list(released_object_pose.position_xyz)
            retreat_goal, retreat_geometry = bounded_contact_free_retreat_goal(
                robotwin_pose6d(task.robot.get_left_ee_pose()),
                released_object_pose,
                0.04,
            )
            retreat = execute_planned_motion(
                task=task,
                helpers=helpers,
                planner=ReleasedTargetCollisionScreenedPlanner(task, helpers),
                goal=MotionGoal(retreat_goal),
                context=context,
                scene=released_scene,
                limits=limits,
                timer=timer,
                timing_gate=timing_gate,
                stage="retreat",
                require_environment_collision_free=True,
                allow_robot_target_contact=False,
                allow_target_pad_contact=True,
                allow_target_table_contact=True,
            )
            if retreat.get("passed") is not True and retreat.get("execution_status") is None:
                alt, alt_geometry = execute_bounded_postrelease_alternatives(
                    task=task,
                    helpers=helpers,
                    context=context,
                    scene=released_scene,
                    limits=limits,
                    timer=timer,
                    timing_gate=timing_gate,
                    phase="retreat",
                    baseline=retreat,
                    gripper_pose=robotwin_pose6d(task.robot.get_left_ee_pose()),
                    object_pose=released_object_pose,
                    distance_m=0.04,
                )
                retreat = alt
                if alt_geometry is not None:
                    retreat_geometry = {
                        **alt_geometry,
                        "requested_retreat_m": 0.04,
                        "requested_goal_xyz_m": list(
                            retreat["bounded_alternative_search"]["attempted"][-1]["goal_xyz_m"]
                        ),
                        "no_contact_override": True,
                    }
            retreat["retreat_geometry"] = retreat_geometry
            retreat["mplib_world_after_separation"] = released_world_sync
        else:
            retreat = {
                "stage": "retreat",
                "passed": False,
                "plan_success": False,
                "failure": {
                    "code": "post_release_separation_failed",
                    "details": {},
                },
            }
        retreat["post_release_separation"] = separation
        retreat["retreat_total_m"] = 0.05
        retreat["retreat_remaining_m"] = 0.04
        if (
            args.diagnose_native_ik_blockers
            and separation.get("passed") is True
            and retreat.get("plan_success") is False
        ):
            # Evaluate four exact-length goal directions ONLY after the
            # original retreat has failed; never execute diagnostic plans.
            try:
                payload["retreat_direction_diagnostic"] = diagnose_retreat_alternative_plans(
                    task=task,
                    helpers=helpers,
                    context=context,
                    scene=released_scene,
                    start_state=helpers.current_arm_state(task),
                    gripper_pose=robotwin_pose6d(task.robot.get_left_ee_pose()),
                    object_pose=actor_pose6d(task.pillbottle),
                )
            except Exception as exc:
                payload["retreat_direction_diagnostic"] = {
                    "schema": "s1m-retreat-direction-planning-only-v1",
                    "planning_only": True,
                    "executed": False,
                    "acceptance_override": False,
                    "diagnostic_error": f"{type(exc).__name__}: {exc}",
                }
        payload["stages"].append(retreat)
        if not retreat["passed"]:
            machine.apply(ManipulationSignal.MOTION_FAILED)
            raise RuntimeError("retreat failed")
        machine.apply(ManipulationSignal.MOTION_COMPLETED)

        payload["final_phase"] = machine.phase.value
        payload["final_object_pose"] = pose_payload(task.pillbottle)
        payload["final_pad_pose"] = pose_payload(task.pad)
        payload["task_success_final"] = bool(task.check_success())
        payload["passed"] = (
            machine.phase is ManipulationPhase.COMPLETE
            and payload["task_success_final"] is True
            and all(stage.get("passed") is True for stage in payload["stages"])
        )
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["error_traceback"] = traceback.format_exc()
        if machine is not None:
            payload["failure_phase"] = machine.phase.value
            payload["final_phase"] = machine.phase.value
        payload["passed"] = False
    finally:
        if carried_planner is not None:
            with suppress(Exception):
                carried_planner.detach()
        if task is not None:
            with suppress(Exception):
                task.clear_physics_step_observers()
            with suppress(Exception):
                task.close_env()

    if operator_mode:
        preflight = payload.get("operator_command_preflight", {})
        rgbd = payload.get("operator_rgbd_observation", {})
        connected = (
            preflight.get("candidate_geometry_binding_passed") is True
            and preflight.get("status") == "native_pending"
            and rgbd.get("camera_count") == 2
            and rgbd.get("controller_eligible_from_rgbd") is False
        )
        if payload["passed"] and not connected:
            payload["passed"] = False
            payload["error"] = "operator pipeline missing source-bound preflight"
        payload["operator_pipeline_result"] = {
            "status": "native_passed" if payload["passed"] and connected else "native_failed",
            "physical_task_success": payload.get("task_success_final") is True,
            "executed_via_project_native_runner": bool(payload.get("stages")),
            "semantic_provider": "explicit_fixed_task_compiler_NOT_GEMINI",
            "identity_provider": "privileged_simulator_actor_registry_NOT_RGBD_detector",
            "captured_rgbd_same_task": rgbd.get("camera_count") == 2,
            "native_candidate_bound_to_semantic": (
                preflight.get("candidate_geometry_binding_passed") is True
            ),
            "visual_target_identity_verified": False,
            "general_vla_end_to_end_accepted": False,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    from vla_robot_sorting.planning.manipulation_recovery import write_new_json

    write_new_json(output_path, payload)
    print_probe_summary(payload)
    if args.print_json:
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
