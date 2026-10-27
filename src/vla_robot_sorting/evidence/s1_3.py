"""Offline audit helpers for the native S1.3 B5/B7 grasp-recovery probe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class S13ProbeGate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class S13GraspRecoveryAudit:
    probe_path: Path
    gates: tuple[S13ProbeGate, ...]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def failures(self) -> tuple[S13ProbeGate, ...]:
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


def _int_equals(value: Any, expected: int) -> bool:
    try:
        return int(value) == expected
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
        and all(_finite_nonnegative(item) and float(item) > 0.0 for item in dims)
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


def _planner_policy_ok(value: Any, *, prefix: str = "") -> bool:
    if not isinstance(value, dict):
        return False

    def field(name: str) -> Any:
        return value.get(f"{prefix}{name}")

    if field("primary_planner_id") != "robotwin-curobo":
        return False
    if field("plan_success") is not True:
        return False
    if field("primary_success") is True:
        return (
            field("fallback_attempted") is False
            and field("fallback_planner_id") is None
            and field("fallback_success") is None
            and field("selected_planner_id") == "robotwin-curobo"
            and _curobo_world_sync_ok(field("selected_curobo_world_sync"))
        )
    if field("primary_success") is False:
        return (
            field("fallback_attempted") is True
            and field("fallback_planner_id") == "robotwin-mplib-rrt"
            and field("fallback_success") is True
            and field("selected_planner_id") == "robotwin-mplib-rrt"
        )
    return False


def _collision_provenance_ok(value: Any, *, collision_free_key: str) -> bool:
    return (
        isinstance(value, dict)
        and value.get("provenance_schema") == "mplib-collision-v1"
        and isinstance(value.get("scene_entities"), dict)
        and isinstance(value.get("collisions"), list)
        and value.get(collision_free_key) is True
        and value.get("collisions") == []
    )


def _critical_policy_gate(gate_config: dict[str, Any]) -> S13ProbeGate:
    terminal = gate_config.get("terminal_reachability")
    perturbation = gate_config.get("perturbation")
    policy = gate_config.get("policy")
    try:
        runs = int(gate_config["runs"])
        max_evaluations = int(gate_config["max_candidate_evaluations"])
        non_top_limit = float(gate_config["non_top_down_abs_approach_z_max"])
        perturbation_m = float(perturbation["translation_m"])
    except (KeyError, TypeError, ValueError) as exc:
        return S13ProbeGate("critical-policy", False, f"invalid policy config: {exc}")

    good = (
        runs >= 20
        and max_evaluations >= 1
        and 0.0 <= non_top_limit <= 0.5
        and isinstance(terminal, dict)
        and terminal.get("backend") == "robotwin-curobo"
        and isinstance(perturbation, dict)
        and perturbation.get("axis") == "y"
        and perturbation_m >= 0.015
        and isinstance(policy, dict)
        and policy.get("allow_terminal_target_contact") is True
        and policy.get("require_pregrasp_environment_collision_free") is True
        and policy.get("require_planning_world_resync_after_perturbation") is True
        and policy.get("require_terminal_self_collision_free") is True
        and policy.get("require_dynamic_limits") is True
        and policy.get("require_all_runs_pass") is True
    )
    detail = (
        f"runs={runs}, max_eval={max_evaluations}, non_top_z={non_top_limit}, "
        f"perturbation_m={perturbation_m}"
    )
    return S13ProbeGate("critical-policy", good, detail)


def _unit_quaternion_xyzw(value: Any, *, tolerance: float = 1e-4) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    if not all(isfinite(item) for item in numbers):
        return False
    norm = sqrt(sum(item * item for item in numbers))
    return abs(norm - 1.0) <= tolerance


def _selected_evaluation(record: dict[str, Any]) -> dict[str, Any] | None:
    b5 = record.get("B5")
    if not isinstance(b5, dict):
        return None
    selected = b5.get("selected")
    evaluations = b5.get("evaluations")
    if not isinstance(selected, dict) or not isinstance(evaluations, list):
        return None
    candidate_id = selected.get("candidate_id")
    matches = [
        item
        for item in evaluations
        if isinstance(item, dict)
        and isinstance(item.get("candidate"), dict)
        and item["candidate"].get("candidate_id") == candidate_id
    ]
    return matches[0] if len(matches) == 1 else None


def _b5_run_gate(
    record: dict[str, Any],
    run_index: int,
    gate_config: dict[str, Any],
) -> S13ProbeGate:
    b5 = record.get("B5")
    if not isinstance(b5, dict):
        return S13ProbeGate(f"run-{run_index}-B5", False, "B5 object missing")

    try:
        candidate_count = int(b5["candidate_count"])
        non_top = int(b5["non_top_down_count"])
        provenance = int(b5["provenance_complete_count"])
    except (KeyError, TypeError, ValueError) as exc:
        return S13ProbeGate(f"run-{run_index}-B5", False, f"invalid counts: {exc}")

    selected = b5.get("selected")
    evaluation = _selected_evaluation(record)
    if not isinstance(selected, dict) or evaluation is None:
        return S13ProbeGate(
            f"run-{run_index}-B5",
            False,
            "selected candidate/evaluation missing or ambiguous",
        )

    approach = selected.get("approach_direction")
    orientation = selected.get("grasp_orientation_xyzw")
    provenance_ok = (
        selected.get("source") == "geometry_first"
        and selected.get("geometry_source") == "sim_ground_truth"
        and isinstance(selected.get("scene_revision"), str)
        and isinstance(selected.get("transform_revision"), str)
    )
    non_top_limit = float(gate_config["non_top_down_abs_approach_z_max"])
    approach_ok = (
        isinstance(approach, list)
        and len(approach) == 3
        and all(isfinite(float(value)) for value in approach)
        and abs(float(approach[2])) <= non_top_limit
    )

    terminal = evaluation.get("terminal_reachability")
    collision = evaluation.get("collision_recheck")
    terminal_collision = (
        terminal.get("self_collision_recheck") if isinstance(terminal, dict) else None
    )
    feasibility_ok = (
        isinstance(terminal, dict)
        and terminal.get("success") is True
        and terminal.get("dynamic_safe") is True
        and _curobo_world_sync_ok(terminal.get("curobo_world_sync"))
        and _collision_provenance_ok(
            terminal_collision,
            collision_free_key="self_collision_free",
        )
        and _planner_policy_ok(evaluation, prefix="pregrasp_")
        and evaluation.get("pregrasp_plan_success") is True
        and evaluation.get("dynamic_safe") is True
        and _collision_provenance_ok(
            collision,
            collision_free_key="collision_free",
        )
        and evaluation.get("reachable") is True
        and evaluation.get("collision_free") is True
    )

    good = (
        _mplib_target_sync_ok(record.get("mplib_world_initial"))
        and b5.get("passed") is True
        and candidate_count >= 2
        and non_top >= 1
        and provenance == candidate_count
        and provenance_ok
        and approach_ok
        and _unit_quaternion_xyzw(orientation)
        and feasibility_ok
    )
    detail = (
        f"candidates={candidate_count}, non_top={non_top}, "
        f"provenance={provenance}, candidate={selected.get('candidate_id')!r}"
    )
    return S13ProbeGate(f"run-{run_index}-B5", good, detail)


def _b7_run_gate(record: dict[str, Any], run_index: int) -> S13ProbeGate:
    b7 = record.get("B7")
    if not isinstance(b7, dict):
        return S13ProbeGate(f"run-{run_index}-B7", False, "B7 object missing")

    old_revision = b7.get("old_scene_revision")
    new_revision = b7.get("new_scene_revision")
    transform_revision = b7.get("transform_revision")
    replacement_collision = b7.get("replacement_collision_recheck")
    revision_lock_ok = (
        isinstance(old_revision, str)
        and isinstance(new_revision, str)
        and old_revision != new_revision
        and isinstance(transform_revision, str)
        and b7.get("stale_segment_scene_revision") == old_revision
        and b7.get("stale_segment_transform_revision") == transform_revision
        and b7.get("replacement_segment_scene_revision") == new_revision
        and b7.get("replacement_segment_transform_revision") == transform_revision
    )
    good = (
        b7.get("passed") is True
        and b7.get("scene_change_kind") == "target_moved"
        and b7.get("decision_action") == "replan"
        and _mplib_target_sync_ok(b7.get("mplib_world_after_perturbation"))
        and _planner_policy_ok(b7.get("initial_planning"))
        and b7.get("stale_feedback_status") == "cancelled"
        and b7.get("stale_replan_required") is True
        and _int_equals(b7.get("stale_physics_steps"), 0)
        and b7.get("qpos_unchanged") is True
        and b7.get("replacement_passed") is True
        and _planner_policy_ok(b7.get("replacement_planning"))
        and b7.get("replacement_segment_ready") is True
        and revision_lock_ok
        and _collision_provenance_ok(
            replacement_collision,
            collision_free_key="collision_free",
        )
        and b7.get("replacement_collision_free") is True
        and _finite_nonnegative(b7.get("position_delta_m"))
        and float(b7.get("position_delta_m", 0.0)) > 0.0
        and _finite_nonnegative(b7.get("detection_latency_s"))
        and _finite_nonnegative(b7.get("decision_latency_s"))
        and _finite_nonnegative(b7.get("replacement_latency_s"))
    )
    detail = (
        f"change={b7.get('scene_change_kind')!r}, "
        f"decision={b7.get('decision_action')!r}, "
        f"physics_steps={b7.get('stale_physics_steps')!r}"
    )
    return S13ProbeGate(f"run-{run_index}-B7", good, detail)


def audit_s1_3_grasp_recovery_probe(probe_path: str | Path) -> S13GraspRecoveryAudit:
    path = Path(probe_path).resolve()
    if not path.is_file():
        return S13GraspRecoveryAudit(
            path,
            (S13ProbeGate("file", False, "grasp-recovery JSON missing"),),
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return S13GraspRecoveryAudit(
            path,
            (S13ProbeGate("json", False, f"invalid JSON: {exc}"),),
        )
    if not isinstance(data, dict):
        return S13GraspRecoveryAudit(
            path,
            (S13ProbeGate("json-object", False, "top-level JSON must be an object"),),
        )

    gates: list[S13ProbeGate] = [
        S13ProbeGate(
            "schema",
            data.get("schema") == "panthera-native-b5-b7-v1",
            f"schema={data.get('schema')!r}",
        )
    ]
    gate_config = data.get("gate_config")
    records = data.get("records")
    if not isinstance(gate_config, dict) or not isinstance(records, list):
        gates.append(S13ProbeGate("structure", False, "gate_config/records missing"))
        return S13GraspRecoveryAudit(path, tuple(gates))

    try:
        runs = int(data["runs"])
        configured_runs = int(gate_config["runs"])
    except (KeyError, TypeError, ValueError) as exc:
        gates.append(S13ProbeGate("runs", False, f"invalid run count: {exc}"))
        return S13GraspRecoveryAudit(path, tuple(gates))

    gates.append(_critical_policy_gate(gate_config))
    gates.append(
        S13ProbeGate(
            "run-count",
            runs == configured_runs == len(records) and runs >= 20,
            f"runs={runs}, configured={configured_runs}, records={len(records)}",
        )
    )
    gates.append(
        S13ProbeGate(
            "summary",
            data.get("B5_pass") is True
            and data.get("B7_pass") is True
            and _int_equals(data.get("B5_pass_count"), runs)
            and _int_equals(data.get("B7_pass_count"), runs),
            (
                f"B5={data.get('B5_pass_count')}/{runs}, "
                f"B7={data.get('B7_pass_count')}/{runs}"
            ),
        )
    )

    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            gates.append(S13ProbeGate(f"run-{index}", False, "record must be an object"))
            continue
        gates.append(
            S13ProbeGate(
                f"run-{index}-no-error",
                "error" not in raw_record,
                f"error={raw_record.get('error')!r}",
            )
        )
        gates.append(_b5_run_gate(raw_record, index, gate_config))
        gates.append(_b7_run_gate(raw_record, index))

    return S13GraspRecoveryAudit(path, tuple(gates))
