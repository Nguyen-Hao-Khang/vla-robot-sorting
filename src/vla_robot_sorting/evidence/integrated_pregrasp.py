"""Offline audit for native integrated grounding/pregrasp/recovery evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class IntegratedPregraspGate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class IntegratedPregraspAudit:
    probe_path: Path
    gates: tuple[IntegratedPregraspGate, ...]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def failures(self) -> tuple[IntegratedPregraspGate, ...]:
        return tuple(gate for gate in self.gates if not gate.passed)


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(number) and number >= 0.0


def _finite(value: Any) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _positive(value: Any) -> bool:
    return _finite_nonnegative(value) and float(value) > 0.0


def _int_equals(value: Any, expected: int) -> bool:
    try:
        return int(value) == expected
    except (TypeError, ValueError):
        return False


def _int_at_least(value: Any, minimum: int) -> bool:
    try:
        return int(value) >= minimum
    except (TypeError, ValueError):
        return False


def _curobo_world_sync_ok(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("synchronized") is not True:
        return False
    pose = value.get("table_base_pose_wxyz")
    dims = value.get("table_dims_xyz_m")
    return (
        isinstance(pose, (list, tuple))
        and len(pose) == 7
        and all(_finite(item) for item in pose)
        and isinstance(dims, (list, tuple))
        and len(dims) == 3
        and all(_positive(item) for item in dims)
        and isinstance(value.get("source"), str)
        and bool(value["source"].strip())
    )


def _mplib_target_sync_ok(value: Any) -> bool:
    try:
        shape_count = int(value.get("collision_shape_count"))
        scaled_mesh_count = int(value.get("scaled_mesh_shape_count"))
        non_unit_scale_count = int(value.get("non_unit_scale_shape_count"))
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        isinstance(value, dict)
        and value.get("strategy") == "scaled_target_fcl_injection"
        and value.get("target_present") is True
        and isinstance(value.get("target_name"), str)
        and bool(value["target_name"].strip())
        and value.get("planner_class") == "SapienPlanner"
        and value.get("planning_world_class") == "SapienPlanningWorld"
        and isinstance(value.get("move_group"), str)
        and bool(value["move_group"].strip())
        and shape_count >= 1
        and scaled_mesh_count >= 1
        and non_unit_scale_count >= 1
        and isinstance(value.get("scale_vectors"), list)
        and bool(value["scale_vectors"])
        and value.get("planner_reused") is True
        and value.get("planning_world_reused") is True
    )


def _planner_policy_ok(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("primary_planner_id") != "robotwin-curobo":
        return False
    if value.get("plan_success") is not True:
        return False
    if value.get("primary_success") is True:
        return (
            value.get("fallback_attempted") is False
            and value.get("fallback_planner_id") is None
            and value.get("fallback_success") is None
            and value.get("selected_planner_id") == "robotwin-curobo"
            and _curobo_world_sync_ok(value.get("selected_curobo_world_sync"))
        )
    if value.get("primary_success") is False:
        return (
            value.get("fallback_attempted") is True
            and value.get("fallback_planner_id") == "robotwin-mplib-rrt"
            and value.get("fallback_success") is True
            and value.get("selected_planner_id") == "robotwin-mplib-rrt"
        )
    return False


def _collision_provenance_ok(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("provenance_schema") == "mplib-collision-v1"
        and isinstance(value.get("scene_entities"), dict)
        and isinstance(value.get("collisions"), list)
        and value.get("collision_free") is True
        and value.get("collisions") == []
    )


def _grounding_gate(data: dict[str, Any]) -> IntegratedPregraspGate:
    value = data.get("grounding")
    if not isinstance(value, dict):
        return IntegratedPregraspGate("grounding", False, "grounding object missing")
    good = (
        value.get("passed") is True
        and value.get("target_object_id") == "pillbottle"
        and value.get("destination_id") == "pad"
        and value.get("scene_source") == "canonical RoboTwin runtime ground truth"
        and _finite_nonnegative(value.get("latency_s"))
    )
    return IntegratedPregraspGate(
        "grounding",
        good,
        (
            f"target={value.get('target_object_id')!r}, "
            f"destination={value.get('destination_id')!r}"
        ),
    )


def _preconditions_gate(data: dict[str, Any]) -> IntegratedPregraspGate:
    value = data.get("preconditions")
    if not isinstance(value, dict):
        return IntegratedPregraspGate("preconditions", False, "preconditions missing")
    collision = value.get("collision_recheck")
    planning = value.get("planning")
    good = (
        value.get("passed") is True
        and _planner_policy_ok(planning)
        and value.get("dynamic_safe") is True
        and _int_equals(value.get("dynamic_violation_count"), 0)
        and _collision_provenance_ok(collision)
        and _int_at_least(collision.get("waypoints_checked"), 2)
        and _positive(value.get("trajectory_duration_s"))
        and _positive(value.get("controller_period_s"))
    )
    collision_free = (
        collision.get("collision_free") if isinstance(collision, dict) else None
    )
    return IntegratedPregraspGate(
        "preconditions",
        good,
        (
            f"path_samples={value.get('path_samples')!r}, "
            f"collision_free={collision_free!r}, "
            f"dynamic_safe={value.get('dynamic_safe')!r}"
        ),
    )


def _execution_gate(data: dict[str, Any]) -> IntegratedPregraspGate:
    value = data.get("execution")
    if not isinstance(value, dict):
        return IntegratedPregraspGate("execution", False, "execution object missing")
    tracking = value.get("max_tracking_error_rad")
    good = (
        value.get("passed") is True
        and value.get("feedback_status") == "completed"
        and value.get("replan_required") is False
        and _positive(value.get("executed_until_s"))
        and _positive(value.get("segment_horizon_s"))
        and _positive(value.get("physics_steps"))
        and _int_equals(value.get("object_robot_contact_frames"), 0)
        and _int_equals(value.get("object_robot_contact_points"), 0)
        and _int_equals(value.get("pad_robot_contact_frames"), 0)
        and _int_equals(value.get("pad_robot_contact_points"), 0)
        and _positive(value.get("max_qpos_delta_rad"))
        and _finite_nonnegative(tracking)
        and float(tracking) <= 0.08
        and _finite_nonnegative(value.get("execution_latency_s"))
    )
    return IntegratedPregraspGate(
        "execution",
        good,
        (
            f"status={value.get('feedback_status')!r}, "
            f"physics_steps={value.get('physics_steps')!r}, "
            f"tracking={tracking!r}"
        ),
    )


def _destination_recovery_gate(data: dict[str, Any]) -> IntegratedPregraspGate:
    value = data.get("destination_recovery")
    if not isinstance(value, dict):
        return IntegratedPregraspGate(
            "destination-recovery",
            False,
            "destination_recovery object missing",
        )
    good = (
        value.get("passed") is True
        and value.get("change_kind") == "destination_moved"
        and value.get("decision_action") == "replan"
        and value.get("robot_execution_after_destination_move") is False
        and _positive(value.get("position_delta_m"))
        and _finite_nonnegative(value.get("detection_latency_s"))
        and _finite_nonnegative(value.get("decision_latency_s"))
    )
    return IntegratedPregraspGate(
        "destination-recovery",
        good,
        (
            f"change={value.get('change_kind')!r}, "
            f"decision={value.get('decision_action')!r}"
        ),
    )


def audit_integrated_pregrasp_probe(
    probe_path: str | Path,
) -> IntegratedPregraspAudit:
    path = Path(probe_path).resolve()
    if not path.is_file():
        return IntegratedPregraspAudit(
            path,
            (IntegratedPregraspGate("file", False, "probe JSON missing"),),
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return IntegratedPregraspAudit(
            path,
            (IntegratedPregraspGate("json", False, f"invalid JSON: {exc}"),),
        )
    if not isinstance(raw, dict):
        return IntegratedPregraspAudit(
            path,
            (IntegratedPregraspGate("json-object", False, "top-level must be object"),),
        )

    gates = [
        IntegratedPregraspGate(
            "schema",
            raw.get("schema") == "panthera-native-integrated-pregrasp-v1",
            f"schema={raw.get('schema')!r}",
        ),
        _grounding_gate(raw),
        IntegratedPregraspGate(
            "mplib-world",
            _mplib_target_sync_ok(raw.get("mplib_world_initial")),
            f"world={raw.get('mplib_world_initial')!r}",
        ),
        IntegratedPregraspGate(
            "grasp",
            isinstance(raw.get("grasp"), dict)
            and raw["grasp"].get("passed") is True
            and _int_at_least(raw["grasp"].get("candidate_count"), 2)
            and _int_at_least(raw["grasp"].get("non_top_down_count"), 1),
            "requires selected non-top-down geometry-first grasp",
        ),
        _preconditions_gate(raw),
        _execution_gate(raw),
        _destination_recovery_gate(raw),
        IntegratedPregraspGate(
            "summary",
            raw.get("passed") is True and "error" not in raw,
            f"passed={raw.get('passed')!r}, error={raw.get('error')!r}",
        ),
    ]
    return IntegratedPregraspAudit(path, tuple(gates))
