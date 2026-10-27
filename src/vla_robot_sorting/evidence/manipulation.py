"""Independent audit for native project-owned Panthera manipulation evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import cos, isclose, isfinite, pi, sin
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ManipulationEvidenceGate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ManipulationEvidenceAudit:
    probe_path: Path
    gates: tuple[ManipulationEvidenceGate, ...]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def failures(self) -> tuple[ManipulationEvidenceGate, ...]:
        return tuple(gate for gate in self.gates if not gate.passed)


EXPECTED_STAGES = (
    "pregrasp",
    "approach",
    "close_gripper",
    "lift",
    "transport",
    "place_above_support",
    "release",
    "retreat",
)


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(number) and number >= 0.0


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
    if value.get("strategy") == "released_target_screw_screen_v1":
        screen = value.get("primary_collision_check")
        return (
            value.get("primary_planner_id") == "robotwin-mplib-screw"
            and value.get("selected_planner_id") == "robotwin-mplib-screw"
            and value.get("primary_success") is True
            and value.get("primary_collision_safe") is True
            and value.get("plan_success") is True
            and value.get("fallback_attempted") is False
            and value.get("target_world_refreshed_after_separation") is True
            and value.get("target_world_refreshed_before_motion") is True
            and isinstance(screen, dict)
            and screen.get("collision_free") is True
            and _screw_acquisition_policy_ok(value.get("screw_planner_policy"))
        )
    if value.get("primary_planner_id") != "robotwin-curobo":
        return False
    if value.get("plan_success") is not True:
        return False
    if value.get("strategy") == "pregrasp_interior_mplib_rescue_v1":
        rescue = value.get("interior_rescue")
        before = rescue.get("primary_reserve") if isinstance(rescue, dict) else None
        after = rescue.get("fallback_reserve") if isinstance(rescue, dict) else None
        return (
            value.get("primary_planner_id") == "robotwin-curobo"
            and value.get("primary_success") is True
            and value.get("primary_native_success") is True
            and value.get("plan_success") is True
            and value.get("fallback_attempted") is True
            and value.get("fallback_planner_id") == "robotwin-mplib-rrt"
            and value.get("fallback_success") is True
            and value.get("selected_planner_id") == "robotwin-mplib-rrt"
            and isinstance(rescue, dict)
            and rescue.get("strategy") == value["strategy"]
            and rescue.get("attempted") is True
            and rescue.get("accepted") is True
            and rescue.get("primary_native_success") is True
            and rescue.get("fallback_native_success") is True
            and rescue.get("fallback_failure") is None
            and isinstance(before, dict)
            and before.get("passed") is False
            and isinstance(after, dict)
            and after.get("passed") is True
            and value.get("primary_reserve") == before
        )
    if value.get("primary_success") is True:
        return (
            value.get("fallback_attempted") is False
            and value.get("fallback_planner_id") is None
            and value.get("fallback_success") is None
            and value.get("selected_planner_id") == "robotwin-curobo"
        )
    if value.get("primary_success") is False:
        return (
            value.get("fallback_attempted") is True
            and value.get("fallback_planner_id") == "robotwin-mplib-rrt"
            and value.get("fallback_success") is True
            and value.get("selected_planner_id") == "robotwin-mplib-rrt"
        )
    return False


def _screw_acquisition_policy_ok(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("strategy") == "mplib_screw_acquisition"
        and value.get("wrapper_method") == "left_mplib_planner.plan_screw"
        and value.get("target_aware_planning_world_required") is True
        and value.get("plan_success") is True
        and value.get("vendor_status") == "Success"
        and value.get("selected_planner_id") == "robotwin-mplib-screw"
        and value.get("straight_screw_motion") is True
        and value.get("independent_collision_recheck_required") is True
    )


def _target_aware_acquisition_policy_ok(value: Any, planner_id: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if (
        value.get("strategy") != "mplib_target_aware_acquisition"
        or value.get("target_aware_planning_world_required") is not True
        or value.get("independent_collision_recheck_required") is not True
        or value.get("primary_planner_id") != "robotwin-mplib-screw"
        or value.get("plan_success") is not True
    ):
        return False

    primary_policy = value.get("primary_policy")
    if value.get("primary_success") is True:
        return (
            planner_id == "robotwin-mplib-screw"
            and value.get("primary_failure_code") is None
            and value.get("fallback_allowed") is False
            and value.get("fallback_attempted") is False
            and value.get("fallback_planner_id") is None
            and value.get("fallback_success") is None
            and value.get("selected_planner_id") == "robotwin-mplib-screw"
            and _screw_acquisition_policy_ok(primary_policy)
        )

    return (
        planner_id == "robotwin-mplib-rrt"
        and value.get("primary_success") is False
        and value.get("primary_failure_code") == "optimization_failed"
        and isinstance(value.get("primary_vendor_status"), str)
        and bool(value["primary_vendor_status"].strip())
        and isinstance(primary_policy, dict)
        and primary_policy.get("strategy") == "mplib_screw_acquisition"
        and primary_policy.get("wrapper_method") == "left_mplib_planner.plan_screw"
        and primary_policy.get("target_aware_planning_world_required") is True
        and primary_policy.get("plan_success") is False
        and value.get("fallback_allowed") is True
        and value.get("fallback_attempted") is True
        and value.get("fallback_planner_id") == "robotwin-mplib-rrt"
        and value.get("fallback_wrapper_method") == "left_mplib_planner.plan_pose"
        and value.get("fallback_target_aware_planning_world_required") is True
        and value.get("fallback_success") is True
        and value.get("selected_planner_id") == "robotwin-mplib-rrt"
    )


def _released_target_retreat_policy_ok(value: Any, planner_id: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if (
        value.get("strategy") == "released_target_screw_screen_v1"
        and planner_id == "robotwin-mplib-screw"
    ):
        return _planner_policy_ok(value)
    if (
        value.get("strategy") != "released_target_collision_screen_v1"
        or value.get("target_world_refreshed_after_separation") is not True
        or value.get("primary_planner_id") != "robotwin-curobo"
        or value.get("plan_success") is not True
        or value.get("selected_planner_id") != planner_id
    ):
        return False
    if planner_id == "robotwin-curobo":
        screen = value.get("primary_collision_check")
        return (
            value.get("primary_success") is True
            and value.get("primary_collision_safe") is True
            and isinstance(screen, dict)
            and screen.get("collision_free") is True
            and value.get("fallback_attempted") is False
        )
    if planner_id != "robotwin-mplib-rrt":
        return False
    screen = value.get("primary_collision_check")
    primary_rejected = value.get("primary_success") is False or (
        isinstance(screen, dict) and screen.get("collision_free") is False
    )
    return (
        primary_rejected
        and value.get("primary_collision_safe") is False
        and value.get("fallback_attempted") is True
        and value.get("fallback_planner_id") == "robotwin-mplib-rrt"
        and value.get("fallback_success") is True
    )


def _released_target_world_gate(
    raw: dict[str, Any], retreat: dict[str, Any] | None
) -> ManipulationEvidenceGate:
    if raw.get("recovery_contract_version") != "bounded_yaw_postrelease_v4":
        return ManipulationEvidenceGate(
            "postrelease-mplib-target-world", True, "historical contract"
        )
    if not isinstance(retreat, dict):
        return ManipulationEvidenceGate(
            "postrelease-mplib-target-world", False, "retreat missing"
        )
    world = retreat.get("mplib_world_after_separation")
    geometry = retreat.get("retreat_geometry")
    pose = world.get("target_pose_world_xyz_m") if isinstance(world, dict) else None
    ref = geometry.get("reference_object_xyz_m") if isinstance(geometry, dict) else None
    try:
        aligned = (
            isinstance(pose, list) and isinstance(ref, list)
            and len(pose) == len(ref) == 3
            and all(
                isfinite(float(pose[i]))
                and isclose(float(pose[i]), float(ref[i]), abs_tol=1e-9)
                for i in range(3)
            )
        )
    except (TypeError, ValueError, OverflowError):
        aligned = False
    return ManipulationEvidenceGate(
        "postrelease-mplib-target-world",
        _mplib_target_sync_ok(world) and aligned,
        f"synced={_mplib_target_sync_ok(world)}, aligned={aligned}",
    )


def _stage_map(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stages = raw.get("stages")
    if not isinstance(stages, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        name = stage.get("stage")
        if isinstance(name, str) and name not in result:
            result[name] = stage
    return result


def _motion_stage_gate(
    stages: dict[str, dict[str, Any]],
    name: str,
    *,
    carried: bool,
    forbid_target_contact: bool,
    allow_target_pad_contact: bool = False,
    allow_target_table_contact: bool = False,
    planner_mode: str = "primary_fallback",
) -> ManipulationEvidenceGate:
    stage = stages.get(name)
    if not isinstance(stage, dict):
        return ManipulationEvidenceGate(name, False, "stage missing")

    if carried:
        planner_ok = stage.get("planner_id") == "robotwin-curobo-carried-object"
    elif planner_mode == "mplib_acquisition":
        planner_ok = _target_aware_acquisition_policy_ok(
            stage.get("planner_policy"),
            stage.get("planner_id"),
        )
    elif planner_mode == "released_target_collision_screen":
        planner_ok = _released_target_retreat_policy_ok(
            stage.get("planner_policy"), stage.get("planner_id")
        )
    else:
        planner_ok = _planner_policy_ok(stage.get("planner_policy"))
    contacts = stage.get("contacts")
    environment_contact_ok = (
        isinstance(contacts, dict)
        and (
            allow_target_pad_contact
            or contacts.get("target_pad_contact_frames") == 0
        )
        and (
            allow_target_table_contact
            or contacts.get("target_table_contact_frames") == 0
        )
        and contacts.get("robot_table_contact_frames", 0) == 0
    )
    contact_ok = environment_contact_ok
    if forbid_target_contact:
        contact_ok = (
            contact_ok
            and isinstance(contacts, dict)
            and contacts.get("robot_target_contact_frames") == 0
        )
    contact_policy = stage.get("contact_policy")
    contact_policy_ok = (
        isinstance(contact_policy, dict)
        and contact_policy.get("allow_robot_target_contact")
        is (not forbid_target_contact)
        and contact_policy.get("allow_target_pad_contact")
        is allow_target_pad_contact
        and contact_policy.get("allow_target_table_contact")
        is allow_target_table_contact
        and contact_policy.get("allow_robot_table_contact", False) is False
    )
    collision = stage.get("collision_validation")
    collision_ok = False
    if isinstance(collision, dict):
        collision_ok = (
            collision.get("self_collision_free") is True
            if carried
            else collision.get("collision_free") is True
        )
    path_metadata = stage.get("path_metadata")
    carried_metadata_ok = True
    if carried:
        carried_metadata = (
            path_metadata.get("carried_object")
            if isinstance(path_metadata, dict)
            else None
        )
        asset_overrides = (
            carried_metadata.get("asset_path_overrides")
            if isinstance(carried_metadata, dict)
            else None
        )
        contact_links = (
            carried_metadata.get("attachment_contact_links")
            if isinstance(carried_metadata, dict)
            else None
        )
        carried_metadata_ok = (
            isinstance(carried_metadata, dict)
            and carried_metadata.get("object_id") == "pillbottle"
            and carried_metadata.get("attached_object_link") == "attached_object"
            and carried_metadata.get("attachment_parent_link") == "tool_link"
            and carried_metadata.get("attachment_joint_name")
            == "attached_object_joint"
            and isinstance(contact_links, list)
            and {"gripper_L_link", "gripper_R_link"}.issubset(set(contact_links))
            and carried_metadata.get("world_obstacle_removed_on_attach") is True
            and carried_metadata.get("pinned_config_mutated_on_disk") is False
            and int(carried_metadata.get("reserved_sphere_count", 0)) > 0
            and isinstance(asset_overrides, dict)
            and isinstance(asset_overrides.get("urdf_path"), str)
            and bool(asset_overrides["urdf_path"].strip())
            and isinstance(asset_overrides.get("collision_spheres"), str)
            and bool(asset_overrides["collision_spheres"].strip())
            and asset_overrides["collision_spheres"].endswith(
                "collision_panthera.yml"
            )
        )

    good = (
        stage.get("passed") is True
        and stage.get("plan_success") is True
        and planner_ok
        and stage.get("dynamic_safe") is True
        and stage.get("dynamic_violation_count") == 0
        and collision_ok
        and contact_ok
        and contact_policy_ok
        and carried_metadata_ok
        and stage.get("execution_status") == "completed"
        and stage.get("replan_required") is False
        and _finite_nonnegative(stage.get("max_tracking_error_rad"))
        and float(stage.get("max_tracking_error_rad", 1.0)) <= 0.08
    )
    return ManipulationEvidenceGate(
        name,
        good,
        (
            f"planner={stage.get('planner_id')!r}, "
            f"dynamic={stage.get('dynamic_safe')!r}, "
            f"execution={stage.get('execution_status')!r}"
        ),
    )


def _support_escape_gate(
    raw: dict[str, Any],
    lift: dict[str, Any] | None,
) -> ManipulationEvidenceGate:
    if not isinstance(lift, dict):
        return ManipulationEvidenceGate(
            "support-escape",
            False,
            "lift stage missing",
        )
    escape = lift.get("support_escape")
    if escape is None:
        return ManipulationEvidenceGate(
            "support-escape",
            True,
            "not used",
        )
    if not isinstance(escape, dict):
        return ManipulationEvidenceGate(
            "support-escape",
            False,
            "support_escape must be an object",
        )

    planning = escape.get("planning")
    execution = escape.get("execution")
    contacts = escape.get("contacts")
    post_start = escape.get("post_start_state")
    support = escape.get("support_contact_geometry")
    initial_failure = lift.get("initial_plan_failure")
    initial_details = (
        initial_failure.get("details")
        if isinstance(initial_failure, dict)
        else None
    )
    initial_status = (
        initial_details.get("native_status")
        if isinstance(initial_details, dict)
        else None
    )
    contact_policy = (
        execution.get("contact_policy")
        if isinstance(execution, dict)
        else None
    )
    collision = (
        execution.get("collision_validation")
        if isinstance(execution, dict)
        else None
    )

    requested_ok = (
        _finite_nonnegative(escape.get("requested_escape_m"))
        and 0.0 < float(escape["requested_escape_m"]) <= 0.01
        and _finite_nonnegative(escape.get("maximum_escape_m"))
        and float(escape["maximum_escape_m"]) <= 0.01
    )
    clearance_ok = (
        _finite_nonnegative(escape.get("required_clearance_m"))
        and _finite_nonnegative(escape.get("object_delta_z_m"))
        and float(escape["object_delta_z_m"])
        >= float(escape["required_clearance_m"])
    )
    planning_ok = (
        isinstance(planning, dict)
        and planning.get("planning_only") is True
        and planning.get("executed") is False
        and planning.get("obstacle_name") == "table"
        and planning.get("obstacle_disabled") is True
        and planning.get("obstacle_reenabled") is True
        and planning.get("bounded") is True
        and planning.get("plan_success") is True
        and _finite_nonnegative(planning.get("requested_escape_m"))
        and float(planning["requested_escape_m"]) <= 0.01
    )
    execution_ok = (
        isinstance(execution, dict)
        and execution.get("passed") is True
        and execution.get("plan_success") is True
        and execution.get("dynamic_safe") is True
        and execution.get("dynamic_violation_count") == 0
        and isinstance(collision, dict)
        and collision.get("self_collision_free") is True
        and execution.get("execution_status") == "completed"
        and execution.get("replan_required") is False
        and _finite_nonnegative(execution.get("max_tracking_error_rad"))
        and float(execution["max_tracking_error_rad"]) <= 0.08
    )
    contacts_ok = (
        isinstance(contacts, dict)
        and int(contacts.get("physics_steps", 0)) > 0
        and contacts.get("robot_table_contact_frames", 0) == 0
        and contacts.get("target_pad_contact_frames", 0) == 0
    )
    policy_ok = (
        isinstance(contact_policy, dict)
        and contact_policy.get("allow_robot_target_contact") is True
        and contact_policy.get("allow_target_pad_contact") is False
        and contact_policy.get("allow_target_table_contact") is True
        and contact_policy.get("allow_robot_table_contact", False) is False
    )
    post_ok = (
        isinstance(post_start, dict)
        and post_start.get("available") is True
        and post_start.get("valid") is True
        and escape.get("table_reenabled_before_execution") is True
        and escape.get("finger_contacts_after")
        == ["gripper_L_link", "gripper_R_link"]
    )
    support_ok = (
        isinstance(support, dict)
        and support.get("diagnostic_only") is True
        and support.get("conservative_overlap") is True
        and support.get("within_surface_sphere_radius") is True
    )
    initial_ok = (
        isinstance(initial_status, str)
        and "colliding with world" in initial_status.lower()
    )
    mirrored = raw.get("support_escape_diagnostic")
    mirror_ok = isinstance(mirrored, dict) and mirrored == escape

    good = (
        escape.get("triggered") is True
        and escape.get("passed") is True
        and requested_ok
        and clearance_ok
        and planning_ok
        and execution_ok
        and contacts_ok
        and policy_ok
        and post_ok
        and support_ok
        and initial_ok
        and mirror_ok
    )
    return ManipulationEvidenceGate(
        "support-escape",
        good,
        (
            f"triggered={escape.get('triggered')!r}, "
            f"requested={escape.get('requested_escape_m')!r}, "
            f"object_delta={escape.get('object_delta_z_m')!r}, "
            f"post_start={post_start!r}"
        ),
    )


def _staged_placement_waypoint_ok(
    raw: dict[str, Any],
    recovery: dict[str, Any],
    attempts: Any,
    selected: Any,
) -> bool:
    """Independently validate the bounded +10mm waypoint, not just its label."""
    version = raw.get("recovery_contract_version")
    contract = {
        "bounded_yaw_postrelease_v2": (
            "bounded_plus_10mm_then_exact_canonical_v1", 0.01, 0
        ),
        "bounded_yaw_postrelease_v3": (
            "bounded_plus_20mm_then_exact_canonical_v2", 0.02, 1
        ),
        "bounded_yaw_postrelease_v4": (
            "bounded_plus_20mm_then_exact_canonical_v2", 0.02, 1
        ),
    }.get(version)
    if contract is None:
        return False
    expected_strategy, expected_height_m, height_index = contract
    staged = recovery.get("staged_descent")
    vertical = recovery.get("vertical_reachability_diagnostic")
    if (
        not isinstance(staged, dict)
        or not isinstance(vertical, dict)
        or vertical.get("planning_only") is not True
        or vertical.get("executed") is not False
        or vertical.get("acceptance_override") is not False
        or not isinstance(attempts, list)
        or len(attempts) != 4
        or selected != 0.0
        or not all(
            item.get("passed") is False
            and item.get("plan_success") is False
            and item.get("execution_status") is None
            and item.get("failure_code") == "optimization_failed"
            and item.get("native_status") == "IK Fail"
            for item in attempts
        )
        or staged.get("strategy") != expected_strategy
        or not isclose(
            float(staged.get("extra_height_m", -1)),
            expected_height_m,
            abs_tol=1e-12,
        )
        or staged.get("canonical_attempted") is not True
        or staged.get("canonical_plan_success") is not True
        or staged.get("canonical_motion_passed") is not True
    ):
        return False
    height_attempts = vertical.get("attempts")
    if (
        not isinstance(height_attempts, list)
        or len(height_attempts) <= height_index
        or height_attempts[height_index].get("extra_height_m") != expected_height_m
        or height_attempts[height_index].get("plan_success") is not True
    ):
        return False
    raised = staged.get("elevated_object_goal_xyz_m")
    canonical = staged.get("canonical_object_goal_xyz_m")
    if not (
        isinstance(raised, list)
        and isinstance(canonical, list)
        and len(raised) == len(canonical) == 3
        and all(_finite_nonnegative(abs(float(v))) for v in raised + canonical)
        and isclose(float(raised[0]), float(canonical[0]), abs_tol=1e-9)
        and isclose(float(raised[1]), float(canonical[1]), abs_tol=1e-9)
        and isclose(
            float(raised[2]) - float(canonical[2]),
            expected_height_m,
            abs_tol=1e-9,
        )
    ):
        return False
    elevated = staged.get("elevated_motion")
    return (
        isinstance(elevated, dict)
        and elevated.get("stage") == "place_elevated_waypoint"
        and _motion_stage_gate(
            {"place_elevated_waypoint": elevated},
            "place_elevated_waypoint",
            carried=True,
            forbid_target_contact=False,
        ).passed
    )


def _upstream_source_attestation_mirror_ok(top: Any, nested: Any) -> bool:
    """Match the pinned source and verify optional raw-byte provenance exactly.

    Older valid evidence used only the four canonical source fields. Current
    native probes add a raw hash and the LF/CRLF materialization policy to the
    top-level record; neither may override the canonical pinned Git blob.
    """
    if not isinstance(top, dict) or not isinstance(nested, dict):
        return False
    if {key: top.get(key) for key in nested} != nested:
        return False
    extra = set(top) - set(nested)
    if not extra:
        return True
    if extra != {"raw_sha256", "line_ending_normalization"}:
        return False
    expected_raw_sha256 = {
        "none": "20e7b9d954fd42c3b78e4a9431aee68b3ff10f70f20104e7dc7e41c83fc8cffe",
        "CRLF_to_LF": "f0c56ea8f29f5ae3851eb8ea22c44c7a1f106a4ab8005a810d457ed4e6299de0",
    }.get(top.get("line_ending_normalization"))
    return isinstance(top.get("raw_sha256"), str) and top["raw_sha256"] == expected_raw_sha256


def _native_curobo_interior_planner_gate(raw: dict[str, Any]) -> ManipulationEvidenceGate:
    """Independently require the native CuRobo in-memory joint-clip contract."""
    version = raw.get("native_curobo_interior_contract_version")
    if version is None:
        return ManipulationEvidenceGate("native-curobo-interior-planner", True, "historical source")
    if version != "scoped_native_cspace_clip_v1":
        return ManipulationEvidenceGate(
            "native-curobo-interior-planner", False, f"unsupported contract={version!r}"
        )
    policy = raw.get("native_curobo_interior_planner")
    good = (
        isinstance(policy, dict)
        and policy.get("schema") == "s1m.native-curobo-interior-planner.v1"
        and policy.get("method") == "scoped_in_memory_native_config_loader"
        and policy.get("pinned_yaml_sha256")
        == "5e73d20566ca49fe818f1cf759baa08c13eb118881a6ab807f63dc86039f0bc7"
        and policy.get("position_limit_clip_rad") == 0.0075
        and policy.get("interior_retract_config_rad") == [0.0, 0.05, 0.0625, 0.0, 0.0, 0.0]
        and policy.get("config_intercepts") == 2
        and policy.get("factory_calls") == 2
        and policy.get("single_arm_search_ik_seeds") == 64
        and policy.get("single_arm_search_trajopt_seeds") == 2
        and policy.get("original_urdf_and_vendor_yaml_modified") is False
        and policy.get("runtime_joint_limits_waived") is False
        and policy.get("controller_eligible") is False
    )
    return ManipulationEvidenceGate(
        "native-curobo-interior-planner",
        good,
        "requires pinned YAML, both scoped configurations, 7.5mrad interior and no upstream edits",
    )


def _pregrasp_interior_reserve_gate(
    raw: dict[str, Any], pregrasp: dict[str, Any] | None
) -> ManipulationEvidenceGate:
    """Independently require the bounded B5 selection and final pregrasp proof."""
    version = raw.get("pregrasp_joint_reserve_contract_version")
    if version is None:
        return ManipulationEvidenceGate(
            "pregrasp-interior-joint-reserve", True, "historical contract absent"
        )
    if version != "raw_timed_5milliradian_v1":
        return ManipulationEvidenceGate(
            "pregrasp-interior-joint-reserve", False, f"unsupported contract={version!r}"
        )

    def valid(record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        try:
            minimum = float(record.get("minimum_margin_rad"))
            required = float(record.get("required_margin_rad"))
            raw_count = record.get("raw_waypoint_count")
            timed_count = record.get("retimed_waypoint_count")
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            record.get("schema") == "s1m.pregrasp-interior-joint-reserve.v1"
            and record.get("passed") is True
            and isfinite(minimum)
            and minimum >= 0.005
            and required == 0.005
            and record.get("violating_joint_samples") == 0
            and type(raw_count) is int
            and raw_count >= 2
            and type(timed_count) is int
            and timed_count >= 2
            and record.get("limiting_joint") in tuple(f"joint{i}" for i in range(1, 7))
            and isinstance(record.get("limiting_waypoint"), str)
            and record["limiting_waypoint"].split(":")[0] in ("raw", "retimed")
            and record.get("joint_bounds_unchanged") is True
            and record.get("command_waypoints_modified") is False
            and record.get("physical_execution_performed") is False
        )

    b5 = raw.get("B5")
    selected = b5.get("selected") if isinstance(b5, dict) else None
    evaluations = b5.get("evaluations") if isinstance(b5, dict) else None
    candidate_id = selected.get("candidate_id") if isinstance(selected, dict) else None
    matches = (
        [
            entry
            for entry in evaluations
            if isinstance(entry, dict)
            and isinstance(entry.get("candidate"), dict)
            and entry["candidate"].get("candidate_id") == candidate_id
        ]
        if isinstance(evaluations, list) and isinstance(candidate_id, str)
        else []
    )
    stage_report = pregrasp.get("pregrasp_joint_reserve") if isinstance(pregrasp, dict) else None
    b5_eligible = len(matches) == 1 and valid(matches[0].get("pregrasp_joint_reserve"))
    stage_eligible = valid(stage_report)
    reuse_version = raw.get("pregrasp_plan_reuse_contract_version")
    reuse_ok = reuse_version is None
    if reuse_version == "b5_cached_start_goal_scene_v1":
        stage_policy = pregrasp.get("planner_policy") if isinstance(pregrasp, dict) else None
        reuse_ok = (
            isinstance(stage_policy, dict)
            and stage_policy.get("prepared_from_b5") is True
            and stage_policy.get("prepared_candidate_id") == candidate_id
            and stage_policy.get("selected_planner_id") == pregrasp.get("planner_id")
            and len(matches) == 1
            and matches[0].get("pregrasp_selected_planner_id") == pregrasp.get("planner_id")
            and matches[0].get("pregrasp_strategy") == stage_policy.get("strategy")
            and matches[0].get("pregrasp_interior_rescue") == stage_policy.get("interior_rescue")
        )
    good = (
        isinstance(b5, dict)
        and b5.get("passed") is True
        and isinstance(pregrasp, dict)
        and pregrasp.get("passed") is True
        and b5_eligible
        and stage_eligible
        and reuse_ok
    )
    return ManipulationEvidenceGate(
        "pregrasp-interior-joint-reserve",
        good,
        f"selected={candidate_id!r}, B5_eligible={b5_eligible}, stage_eligible={stage_eligible}",
    )


def _joint5_placement_lookahead_attested(place: dict[str, Any], recovery: dict[str, Any]) -> bool:
    """Independent, fail-closed audit of screened pre-execution yaw selection."""
    from math import isfinite

    look = recovery.get("lookahead")
    policy = place.get("planner_policy")
    metadata = place.get("path_metadata")
    attempts = recovery.get("attempts")
    if not all(isinstance(x, dict) for x in (look, policy, metadata)):
        return False
    offsets = (0.0, pi / 2.0, -pi / 2.0, pi)
    records = look.get("evaluated")
    selected = recovery.get("selected_yaw_offset_rad")
    baseline = look.get("baseline_joint5_margin_rad")
    if not (
        look.get("schema") == "s1m.joint5-placement-lookahead.v1"
        and look.get("planning_only") is True
        and look.get("waivers") is False
        and look.get("physics_executions") == 1
        and look.get("max_candidate_plans") == 4
        and look.get("baseline_yaw_rad") == 0.0
        and look.get("candidate_yaw_offsets_rad") == list(offsets)
        and isinstance(records, list)
        and len(records) == 4
        and isinstance(attempts, list)
        and len(attempts) == 1
        and place.get("passed") is True
        and place.get("execution_status") == "completed"
        and isinstance(selected, (int, float))
        and isfinite(float(selected))
        and selected in offsets[1:]
        and look.get("selected_yaw_offset_rad") == selected
        and place.get("placement_yaw_offset_rad") == selected
        and policy.get("strategy") == "joint5_placement_lookahead_v1"
        and policy.get("prepared_placement_plan_verified") is True
        and policy.get("selected_yaw_offset_rad") == selected
        and isinstance(baseline, (float, int))
        and isfinite(float(baseline))
        and 0.0 <= baseline < 0.10
        and isinstance(look.get("selected_plan_sha256"), str)
        and len(look["selected_plan_sha256"]) == 64
        and all(x in "0123456789abcdef" for x in look["selected_plan_sha256"])
        and metadata.get("lookahead_plan_sha256") == look["selected_plan_sha256"]
        and metadata.get("lookahead_selected_yaw_offset_rad") == selected
        and metadata.get("lookahead_goal_robotwin_wxyz") == look.get("selected_goal_robotwin_wxyz")
        and isinstance(metadata.get("lookahead_goal_robotwin_wxyz"), list)
        and len(metadata["lookahead_goal_robotwin_wxyz"]) == 7
        and all(isfinite(float(v)) for v in metadata["lookahead_goal_robotwin_wxyz"])
        and attempts[0].get("yaw_offset_rad") == selected
        and attempts[0].get("passed") is True
        and attempts[0].get("plan_success") is True
        and attempts[0].get("execution_status") == "completed"
    ):
        return False

    valid = []
    for idx, row in enumerate(records):
        if not isinstance(row, dict) or row.get("yaw_offset_rad") != offsets[idx]:
            return False
        if row.get("planner_id") != "robotwin-curobo-carried-object":
            return False
        if row.get("eligible") is True:
            if not all(
                row.get(k) is True
                for k in (
                    "planner_success",
                    "dynamic_safe",
                    "raw_self_collision_free",
                    "timed_self_collision_free",
                    "joint_interior_safe",
                )
            ):
                return False
            margin = row.get("joint5_margin_rad")
            joint = row.get("minimum_all_joint_margin_rad")
            sha = row.get("path_sha256")
            if not (
                isinstance(margin, (float, int))
                and isfinite(float(margin))
                and isinstance(joint, (float, int))
                and isfinite(float(joint))
                and joint >= 0.005
                and isinstance(sha, str)
                and len(sha) == 64
            ):
                return False
            if idx > 0 and margin >= 0.125 and margin - baseline >= 0.10:
                valid.append(row)
    if not valid:
        return False
    chosen = max(valid, key=lambda row: row["joint5_margin_rad"])
    return (
        chosen.get("yaw_offset_rad") == selected
        and chosen.get("path_sha256") == look["selected_plan_sha256"]
    )


def _placement_recovery_gate(
    raw: dict[str, Any],
    place: dict[str, Any] | None,
) -> ManipulationEvidenceGate:
    if not isinstance(place, dict):
        return ManipulationEvidenceGate(
            "placement-yaw-recovery",
            False,
            "place stage missing",
        )
    recovery = place.get("placement_recovery")
    if recovery is None:
        historical = (
            raw.get("recovery_contract_version") is None
            and "placement_recovery_diagnostic" not in raw
            and "upstream_task_source" not in raw
        )
        return ManipulationEvidenceGate(
            "placement-yaw-recovery",
            historical,
            "historical evidence" if historical else "required recovery evidence missing",
        )
    if not isinstance(recovery, dict):
        return ManipulationEvidenceGate(
            "placement-yaw-recovery",
            False,
            "placement_recovery must be an object",
        )

    expected_offsets = (0.0, pi / 2.0, -pi / 2.0, pi)
    offsets = recovery.get("candidate_yaw_offsets_rad")
    attempts = recovery.get("attempts")
    selected = recovery.get("selected_yaw_offset_rad")
    offsets_ok = (
        isinstance(offsets, list)
        and len(offsets) == len(expected_offsets)
        and all(
            _finite_nonnegative(abs(actual - expected))
            and isclose(float(actual), expected, abs_tol=1e-9)
            for actual, expected in zip(offsets, expected_offsets, strict=True)
        )
    )
    attempts_ok = isinstance(attempts, list) and 1 <= len(attempts) <= 4
    if attempts_ok:
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                attempts_ok = False
                break
            if not isclose(
                float(attempt.get("yaw_offset_rad", 999.0)),
                expected_offsets[index],
                abs_tol=1e-9,
            ):
                attempts_ok = False
                break
            if index < len(attempts) - 1 and not (
                attempt.get("passed") is False
                and attempt.get("plan_success") is False
                and attempt.get("execution_status") is None
                and attempt.get("failure_code") == "optimization_failed"
                and attempt.get("native_status") == "IK Fail"
            ):
                attempts_ok = False
                break

    lookahead_strategy = recovery.get("strategy") == "joint5_placement_lookahead_v1"
    if lookahead_strategy:
        # Planning-only yaw candidates are NOT motion attempts. Exactly one
        # selected physical move is allowed; independently attest its path.
        attempts_ok = (
            isinstance(attempts, list)
            and len(attempts) == 1
            and isinstance(attempts[0], dict)
            and attempts[0].get("yaw_offset_rad") == selected
        )
    selected_ok = (
        place.get("passed") is True
        and selected is not None
        and isinstance(attempts, list)
        and bool(attempts)
        and attempts[-1].get("passed") is True
        and isclose(
            float(attempts[-1].get("yaw_offset_rad", 999.0)),
            float(selected),
            abs_tol=1e-9,
        )
        and any(isclose(float(selected), expected, abs_tol=1e-9) for expected in expected_offsets)
    )
    staged_selected_ok = (
        place.get("passed") is True
        and recovery.get("strategy") == "yaw_plus_bounded_staged_descent_v2"
        and _staged_placement_waypoint_ok(raw, recovery, attempts, selected)
    )
    if recovery.get("strategy") == "yaw_only_task_unconstrained_orientation_v1":
        strategy_ok = recovery.get("staged_descent") is None
    elif lookahead_strategy:
        strategy_ok = (
            recovery.get("staged_descent") is None
            and recovery.get("vertical_reachability_diagnostic") is None
            and _joint5_placement_lookahead_attested(place, recovery)
        )
    else:
        strategy_ok = staged_selected_ok
    mirrored = raw.get("placement_recovery_diagnostic")
    good = (
        strategy_ok
        and recovery.get("task_success_orientation_constrained") is False
        and recovery.get("upstream_task_source")
        == {
            "path": "RoboTwin/envs/move_pillbottle_pad.py",
            "commit": "2aee31ad75cb7b0cd779540441c6025853a5b60d",
            "sha256": "20e7b9d954fd42c3b78e4a9431aee68b3ff10f70f20104e7dc7e41c83fc8cffe",
            "check_success_orientation_constrained": False,
        }
        and _upstream_source_attestation_mirror_ok(
            raw.get("upstream_task_source"), recovery.get("upstream_task_source")
        )
        and recovery.get("bounded_attempt_count") == len(attempts or [])
        and recovery.get("max_attempt_count") == 4
        and offsets_ok
        and attempts_ok
        and (selected_ok or staged_selected_ok)
        and isinstance(mirrored, dict)
        and mirrored == recovery
    )
    return ManipulationEvidenceGate(
        "placement-yaw-recovery",
        good,
        (
            f"selected={selected!r}, attempts="
            f"{len(attempts) if isinstance(attempts, list) else None!r}"
        ),
    )


def _postrelease_direction_geometry_ok(separation: dict[str, Any]) -> bool:
    """Verify bounded separation moves outward, never toward the bottle."""
    geometry = separation.get("separation_geometry")
    goal = separation.get("requested_goal_xyz_m")
    if not isinstance(geometry, dict) or not isinstance(goal, list):
        return False
    start = geometry.get("start_gripper_xyz_m")
    obj = geometry.get("reference_object_xyz_m")
    direction = geometry.get("direction_xyz_unit")
    if not all(
        isinstance(value, list)
        and len(value) == 3
        and all(isfinite(float(entry)) for entry in value)
        for value in (start, obj, goal, direction)
    ):
        return False
    distance = float(separation.get("requested_separation_m", 0))
    dx = float(start[0]) - float(obj[0])
    dy = float(start[1]) - float(obj[1])
    r = (dx * dx + dy * dy) ** 0.5
    if not isclose(
        float(geometry.get("initial_robot_object_xy_distance_m", -1)),
        r,
        abs_tol=1e-9,
    ):
        return False
    if geometry.get("strategy") == "radial_xy_away_from_released_object_v1":
        if r < 0.005:
            return False
        expected_unit = (dx / r, dy / r, 0.0)
    elif geometry.get("strategy") == "vertical_fallback_when_radial_undefined_v1":
        if r >= 0.005:
            return False
        expected_unit = (0.0, 0.0, 1.0)
    elif geometry.get("strategy") == "bounded_outward_direction_search_v1":
        if r < 0.005:
            return False
        expected_angles = {
            "radial_up_60": (0.0, 60.0),
            "radial_up_45": (0.0, 45.0),
            "radial_up_75": (0.0, 75.0),
            "radial_up_30": (0.0, 30.0),
            "radial_yaw_pos_30": (30.0, 0.0),
            "radial_yaw_neg_30": (-30.0, 0.0),
            "vertical_up_separation_only": (0.0, 90.0),
            "radial_pitch_neg_20": (0.0, 0.0),
            "radial_pitch_pos_20": (0.0, 0.0),
            "radial_up45_pitch_neg_20": (0.0, 45.0),
            "radial_up45_pitch_pos_20": (0.0, 45.0),
            "radial_pitch_neg_10": (0.0, 0.0),
            "radial_pitch_pos_10": (0.0, 0.0),
            "radial_up45_pitch_neg_10": (0.0, 45.0),
            "radial_up45_pitch_pos_10": (0.0, 45.0),
            "radial_pitch_neg_30": (0.0, 0.0),
            "radial_pitch_pos_30": (0.0, 0.0),
            "radial_up60_pitch_neg_20": (0.0, 60.0),
            "radial_up60_pitch_pos_20": (0.0, 60.0),
        }
        label = geometry.get("direction_label")
        if label not in expected_angles:
            return False
        if distance > 0.01 and label == "vertical_up_separation_only":
            return False
        yaw_deg, elev_deg = expected_angles[label]
        yaw, elev = yaw_deg * pi / 180, elev_deg * pi / 180
        ux, uy = dx / r, dy / r
        expected_unit = (
            cos(elev) * (ux * cos(yaw) - uy * sin(yaw)),
            cos(elev) * (ux * sin(yaw) + uy * cos(yaw)),
            sin(elev),
        )
    else:
        return False
    position_ok = all(
        isclose(float(direction[i]), expected_unit[i], abs_tol=1e-9)
        and isclose(
            float(goal[i]),
            float(start[i]) + distance * expected_unit[i],
            abs_tol=1e-9,
        )
        for i in range(3)
    )
    if not position_ok:
        return False
    if geometry.get("strategy") != "bounded_outward_direction_search_v1":
        return True
    original = geometry.get("start_gripper_orientation_xyzw")
    actual = geometry.get("goal_gripper_orientation_xyzw")
    if not isinstance(original, list) or not isinstance(actual, list):
        # Compatible with historical v1 bounded-direction evidence.
        return original is None and actual is None
    if len(original) != 4 or len(actual) != 4:
        return False
    pitch_by_label = {
        "radial_pitch_neg_20": -20.0,
        "radial_pitch_pos_20": 20.0,
        "radial_up45_pitch_neg_20": -20.0,
        "radial_up45_pitch_pos_20": 20.0,
        "radial_pitch_neg_10": -10.0,
        "radial_pitch_pos_10": 10.0,
        "radial_up45_pitch_neg_10": -10.0,
        "radial_up45_pitch_pos_10": 10.0,
        "radial_pitch_neg_30": -30.0,
        "radial_pitch_pos_30": 30.0,
        "radial_up60_pitch_neg_20": -20.0,
        "radial_up60_pitch_pos_20": 20.0,
    }
    pitch = pitch_by_label.get(geometry.get("direction_label"), 0.0)
    try:
        q = [float(v) for v in original]
        rotated = [float(v) for v in actual]
        angle = pitch * pi / 360.0
        c, s = cos(angle), sin(angle)
        x, y, z, w = q
        expected_q = (c * x - s * z, c * y + s * w, c * z + s * x, c * w - s * y)
        return (
            all(isfinite(v) for v in q + rotated)
            and isclose(sum(v * v for v in q), 1.0, abs_tol=1e-4)
            and isclose(float(geometry.get("wrist_local_y_pitch_deg")), pitch, abs_tol=1e-9)
            and all(isclose(rotated[i], expected_q[i], abs_tol=1e-9) for i in range(4))
            and (
                distance <= 0.01 or geometry.get("direction_label") != "vertical_up_separation_only"
            )
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _bounded_postrelease_search_attested(
    raw: dict[str, Any], stage: dict[str, Any], geometry: dict[str, Any], phase: str
) -> bool:
    """Independently bind a selected pre-screened goal to the executed plan."""
    if geometry.get("strategy") != "bounded_outward_direction_search_v1":
        # No alternative was selected; old and new source preserve baseline.
        existing = stage.get("bounded_alternative_search")
        return existing is None or (
            isinstance(existing, dict) and existing.get("selected_label") is None
        )
    contract = raw.get("postrelease_direction_search_contract_version")
    if contract not in ("outward_discrete_prephysics_v1", "outward_discrete_prephysics_v2"):
        return False
    search = stage.get("bounded_alternative_search")
    policy = stage.get("planner_policy")
    metadata = stage.get("path_metadata")
    if not all(isinstance(x, dict) for x in (search, policy, metadata)):
        return False
    attempts = search.get("attempted")
    expected_max = (
        7 if phase == "separation" else (18 if contract == "outward_discrete_prephysics_v2" else 6)
    )
    label = geometry.get("direction_label")
    goal = (
        stage.get("requested_goal_xyz_m")
        if phase == "separation"
        else geometry.get("requested_goal_xyz_m")
    )
    if (
        search.get("schema") != "s1m.postrelease-outward-candidate-search.v1"
        or search.get("phase") != phase
        or search.get("baseline_failed_before_physics") is not True
        or search.get("max_candidates") != expected_max
        or search.get("contact_and_bounds_waivers") is not False
        or search.get("physics_alternative_executions") != 1
        or not isinstance(attempts, list)
        or not 1 <= len(attempts) <= expected_max
        or search.get("selected_label") != label
        or policy.get("prepared_postrelease_direction") != label
        or policy.get("prepared_postrelease_goal_verified") is not True
        or not isinstance(goal, list)
        or len(goal) != 3
        or not isinstance(metadata.get("target_pose_robotwin_wxyz"), (tuple, list))
        or len(metadata["target_pose_robotwin_wxyz"]) != 7
    ):
        return False
    for record in attempts[:-1]:
        if not isinstance(record, dict) or record.get("eligible") is not False:
            return False
    selected = attempts[-1]
    if not isinstance(selected, dict) or selected.get("label") != label:
        return False
    if not all(
        selected.get(key) is True
        for key in (
            "planner_success",
            "raw_collision_free",
            "timed_collision_free",
            "dynamic_safe",
            "joint_interior_safe",
            "eligible",
        )
    ):
        return False
    if stage.get("planner_id") == "robotwin-mplib-screw":
        screw = selected.get("screw_probe")
        if not (
            contract == "outward_discrete_prephysics_v2"
            and isinstance(screw, dict)
            and screw.get("attempted") is True
            and screw.get("planner_id") == "robotwin-mplib-screw"
            and screw.get("native_success") is True
            and screw.get("collision_free") is True
            and _planner_policy_ok(policy)
        ):
            return False
    try:
        position_matches = all(
            isfinite(float(goal[i]))
            and isclose(float(selected["goal_xyz_m"][i]), float(goal[i]), abs_tol=1e-9)
            and isclose(
                float(metadata["target_pose_robotwin_wxyz"][i]),
                float(goal[i]),
                abs_tol=1e-9,
            )
            for i in range(3)
        )
        if not position_matches:
            return False
        if contract == "outward_discrete_prephysics_v1":
            return True
        quaternion = geometry.get("goal_gripper_orientation_xyzw")
        if not isinstance(quaternion, list) or len(quaternion) != 4:
            return False
        observed = metadata["target_pose_robotwin_wxyz"]
        expected = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        return all(
            isfinite(float(expected[i]))
            and isclose(float(observed[i + 3]), float(expected[i]), abs_tol=1e-9)
            for i in range(4)
        )
    except (ValueError, TypeError, IndexError, KeyError):
        return False


def _retreat_goal_geometry_gate(
    raw: dict[str, Any], retreat: dict[str, Any] | None
) -> ManipulationEvidenceGate:
    if raw.get("recovery_contract_version") not in (
        "bounded_yaw_postrelease_v3",
        "bounded_yaw_postrelease_v4",
    ):
        return ManipulationEvidenceGate("retreat-outward-goal", True, "historical contract")
    if not isinstance(retreat, dict):
        return ManipulationEvidenceGate("retreat-outward-goal", False, "retreat stage missing")
    geometry = retreat.get("retreat_geometry")
    if not isinstance(geometry, dict):
        return ManipulationEvidenceGate("retreat-outward-goal", False, "missing retreat geometry")
    requested = geometry.get("requested_retreat_m")
    if (
        not isinstance(requested, (int, float))
        or not isclose(float(requested), 0.04, abs_tol=1e-12)
        or geometry.get("no_contact_override") is not True
        or geometry.get("strategy")
        not in (
            "radial_xy_away_from_released_object_v1",
            "vertical_fallback_when_radial_undefined_v1",
            "bounded_outward_direction_search_v1",
        )
    ):
        return ManipulationEvidenceGate(
            "retreat-outward-goal", False, "bad bounded retreat policy"
        )
    # Reuse the independently verified exact signed-vector geometry, but with
    # the 40mm retreat displacement rather than the 10mm separation displacement.
    synthetic = {
        "requested_separation_m": 0.04,
        "separation_geometry": geometry,
        "requested_goal_xyz_m": geometry.get("requested_goal_xyz_m"),
    }
    try:
        valid = _postrelease_direction_geometry_ok(synthetic) and (
            _bounded_postrelease_search_attested(raw, retreat, geometry, "retreat")
        )
    except (TypeError, ValueError, OverflowError, KeyError):
        valid = False
    return ManipulationEvidenceGate(
        "retreat-outward-goal", valid,
        f"strategy={geometry.get('strategy')!r}, distance={requested!r}",
    )


def _post_release_separation_gate(
    raw: dict[str, Any],
    retreat: dict[str, Any] | None,
) -> ManipulationEvidenceGate:
    if not isinstance(retreat, dict):
        return ManipulationEvidenceGate(
            "post-release-separation",
            False,
            "retreat stage missing",
        )
    separation = retreat.get("post_release_separation")
    if separation is None:
        historical = (
            raw.get("recovery_contract_version") is None
            and "post_release_separation_diagnostic" not in raw
            and "upstream_task_source" not in raw
        )
        return ManipulationEvidenceGate(
            "post-release-separation",
            historical,
            "historical evidence" if historical else "required separation evidence missing",
        )
    if not isinstance(separation, dict):
        return ManipulationEvidenceGate(
            "post-release-separation",
            False,
            "post_release_separation must be an object",
        )

    contacts = separation.get("contacts")
    policy = separation.get("contact_policy")
    collision = separation.get("collision_validation")
    good = (
        separation.get("stage") == "post_release_separation"
        and separation.get("passed") is True
        and separation.get("plan_success") is True
        and _planner_policy_ok(separation.get("planner_policy"))
        and separation.get("dynamic_safe") is True
        and separation.get("dynamic_violation_count") == 0
        and isinstance(collision, dict)
        and collision.get("collision_free") is True
        and separation.get("execution_status") == "completed"
        and separation.get("replan_required") is False
        and _finite_nonnegative(separation.get("max_tracking_error_rad"))
        and float(separation.get("max_tracking_error_rad", 1.0)) <= 0.08
        and isinstance(contacts, dict)
        and int(contacts.get("physics_steps", 0)) > 0
        and int(contacts.get("target_pad_contact_frames", 0)) > 0
        and int(contacts.get("robot_table_contact_frames", 0)) == 0
        and int(contacts.get("target_table_contact_frames", 0)) == 0
        and isinstance(policy, dict)
        and policy.get("allow_robot_target_contact") is True
        and policy.get("allow_target_pad_contact") is True
        and policy.get("allow_target_table_contact") is False
        and policy.get("allow_robot_table_contact", False) is False
        and _finite_nonnegative(separation.get("requested_separation_m"))
        and 0.0 < float(separation["requested_separation_m"]) <= 0.01
        and isclose(
            float(separation.get("maximum_separation_m", -1.0)),
            0.01,
            abs_tol=1e-12,
        )
        and (
            raw.get("recovery_contract_version")
            not in (
                "bounded_yaw_postrelease_v2",
                "bounded_yaw_postrelease_v3",
                "bounded_yaw_postrelease_v4",
            )
            or _postrelease_direction_geometry_ok(separation)
        )
        and _bounded_postrelease_search_attested(
            raw, separation, separation.get("separation_geometry") or {}, "separation"
        )
        and separation.get("task_success_after") is True
        and separation.get("finger_contacts_after") == []
        and separation.get("target_supported_during_separation") is True
        and isclose(float(retreat.get("retreat_total_m", -1.0)), 0.05, abs_tol=1e-12)
        and isclose(
            float(retreat.get("retreat_remaining_m", -1.0)),
            0.04,
            abs_tol=1e-12,
        )
        and isinstance(raw.get("post_release_separation_diagnostic"), dict)
        and raw["post_release_separation_diagnostic"] == separation
    )
    return ManipulationEvidenceGate(
        "post-release-separation",
        good,
        (
            f"requested={separation.get('requested_separation_m')!r}, "
            f"success={separation.get('task_success_after')!r}, "
            f"fingers={separation.get('finger_contacts_after')!r}"
        ),
    )


def audit_project_manipulation_probe(
    probe_path: str | Path,
) -> ManipulationEvidenceAudit:
    path = Path(probe_path).resolve()
    if not path.is_file():
        return ManipulationEvidenceAudit(
            path,
            (ManipulationEvidenceGate("file", False, "probe JSON missing"),),
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return ManipulationEvidenceAudit(
            path,
            (ManipulationEvidenceGate("json", False, f"invalid JSON: {exc}"),),
        )
    if not isinstance(raw, dict):
        return ManipulationEvidenceAudit(
            path,
            (ManipulationEvidenceGate("json-object", False, "top-level must be object"),),
        )

    stages = _stage_map(raw)
    gates: list[ManipulationEvidenceGate] = [
        ManipulationEvidenceGate(
            "schema",
            raw.get("schema") == "panthera-native-project-manipulation-v1",
            f"schema={raw.get('schema')!r}",
        ),
        ManipulationEvidenceGate(
            "recovery-contract-version",
            raw.get("recovery_contract_version")
            in (
                None,
                "bounded_yaw_postrelease_v1",
                "bounded_yaw_postrelease_v2",
                "bounded_yaw_postrelease_v3",
                "bounded_yaw_postrelease_v4",
            ),
            f"version={raw.get('recovery_contract_version')!r}",
        ),
        ManipulationEvidenceGate(
            "mplib-target-world",
            _mplib_target_sync_ok(raw.get("mplib_world_initial")),
            f"world={raw.get('mplib_world_initial')!r}",
        ),
        ManipulationEvidenceGate(
            "mplib-target-world-before-approach",
            _mplib_target_sync_ok(raw.get("mplib_world_before_approach")),
            f"world={raw.get('mplib_world_before_approach')!r}",
        ),
        ManipulationEvidenceGate(
            "project-owned-boundary",
            raw.get("upstream_manipulation_macro_used") is False
            and raw.get("placement_goal_policy")
            == "project_rigid_object_gripper_transform",
            (
                f"macro={raw.get('upstream_manipulation_macro_used')!r}, "
                f"placement={raw.get('placement_goal_policy')!r}"
            ),
        ),
        ManipulationEvidenceGate(
            "stage-order",
            tuple(
                stage.get("stage")
                for stage in raw.get("stages", [])
                if isinstance(stage, dict)
            )
            == EXPECTED_STAGES,
            f"stages={tuple(stages)}",
        ),
        ManipulationEvidenceGate(
            "B5",
            isinstance(raw.get("B5"), dict)
            and raw["B5"].get("passed") is True
            and isinstance(raw["B5"].get("selected"), dict),
            "requires native selected safe B5 grasp",
        ),
    ]

    gates.append(
        _motion_stage_gate(
            stages,
            "pregrasp",
            carried=False,
            forbid_target_contact=True,
            allow_target_table_contact=True,
        )
    )
    gates.append(_native_curobo_interior_planner_gate(raw))
    gates.append(_pregrasp_interior_reserve_gate(raw, stages.get("pregrasp")))
    gates.append(
        _motion_stage_gate(
            stages,
            "approach",
            carried=False,
            forbid_target_contact=True,
            allow_target_table_contact=True,
            planner_mode="mplib_acquisition",
        )
    )

    close = stages.get("close_gripper")
    close_good = (
        isinstance(close, dict)
        and close.get("passed") is True
        and close.get("contacts_after")
        == ["gripper_L_link", "gripper_R_link"]
        and close.get("force_scale") == 1.0
        and isinstance(close.get("result"), dict)
        and close["result"].get("completed") is True
        and close["result"].get("holding_object") is True
    )
    gates.append(
        ManipulationEvidenceGate(
            "close_gripper",
            close_good,
            f"contacts={close.get('contacts_after') if isinstance(close, dict) else None!r}",
        )
    )

    attachment = raw.get("carried_object_attachment")
    gates.append(
        ManipulationEvidenceGate(
            "carried-object-attachment",
            isinstance(attachment, dict)
            and attachment.get("object_id") == "pillbottle"
            and attachment.get("attached_object_link") == "attached_object"
            and attachment.get("attachment_parent_link") == "tool_link"
            and attachment.get("attachment_joint_name") == "attached_object_joint"
            and isinstance(attachment.get("attachment_contact_links"), list)
            and {"gripper_L_link", "gripper_R_link"}.issubset(
                set(attachment["attachment_contact_links"])
            )
            and attachment.get("world_obstacle_removed_on_attach") is True
            and attachment.get("pinned_config_mutated_on_disk") is False
            and int(attachment.get("reserved_sphere_count", 0)) > 0,
            f"attachment={attachment!r}",
        )
    )

    gates.append(
        _motion_stage_gate(
            stages,
            "lift",
            carried=True,
            forbid_target_contact=False,
        )
    )
    lift = stages.get("lift")
    gates.append(_support_escape_gate(raw, lift))
    gates.append(
        ManipulationEvidenceGate(
            "lift-retained-grasp",
            isinstance(lift, dict)
            and _finite_nonnegative(lift.get("object_delta_z_m"))
            and float(lift.get("object_delta_z_m", 0.0)) >= 0.02,
            (
                "object_delta_z_m="
                f"{lift.get('object_delta_z_m') if isinstance(lift, dict) else None!r}"
            ),
        )
    )
    gates.append(
        _motion_stage_gate(
            stages,
            "transport",
            carried=True,
            forbid_target_contact=False,
        )
    )
    gates.append(
        _motion_stage_gate(
            stages,
            "place_above_support",
            carried=True,
            forbid_target_contact=False,
        )
    )
    place = stages.get("place_above_support")
    try:
        gates.append(_placement_recovery_gate(raw, place))
    except (TypeError, ValueError, OverflowError, AttributeError, IndexError) as exc:
        gates.append(
            ManipulationEvidenceGate(
                "placement-yaw-recovery", False, f"malformed evidence: {exc}"
            )
        )

    release = stages.get("release")
    gates.append(
        ManipulationEvidenceGate(
            "release",
            isinstance(release, dict)
            and release.get("passed") is True
            and isinstance(release.get("result"), dict)
            and release["result"].get("completed") is True
            and float(release["result"].get("actual_normalized_position", -1.0)) >= 0.9,
            f"release={release!r}",
        )
    )
    gates.append(
        ManipulationEvidenceGate(
            "task-success-after-release",
            raw.get("task_success_after_release") is True,
            f"value={raw.get('task_success_after_release')!r}",
        )
    )
    gates.append(
        _motion_stage_gate(
            stages,
            "retreat",
            carried=False,
            forbid_target_contact=True,
            allow_target_pad_contact=True,
            allow_target_table_contact=True,
            planner_mode=(
                "released_target_collision_screen"
                if raw.get("recovery_contract_version") == "bounded_yaw_postrelease_v4"
                else "primary_fallback"
            ),
        )
    )
    retreat = stages.get("retreat")
    gates.append(_released_target_world_gate(raw, retreat))
    gates.append(_retreat_goal_geometry_gate(raw, retreat))
    try:
        gates.append(_post_release_separation_gate(raw, retreat))
    except (TypeError, ValueError, OverflowError, AttributeError, IndexError) as exc:
        gates.append(
            ManipulationEvidenceGate(
                "post-release-separation", False, f"malformed evidence: {exc}"
            )
        )
    gates.append(
        ManipulationEvidenceGate(
            "summary",
            raw.get("passed") is True
            and raw.get("final_phase") == "complete"
            and raw.get("task_success_final") is True
            and "error" not in raw,
            (
                f"passed={raw.get('passed')!r}, phase={raw.get('final_phase')!r}, "
                f"task_success={raw.get('task_success_final')!r}, "
                f"error={raw.get('error')!r}"
            ),
        )
    )
    return ManipulationEvidenceAudit(path, tuple(gates))
