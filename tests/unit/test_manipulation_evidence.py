from __future__ import annotations

import json
from pathlib import Path

from vla_robot_sorting.evidence import audit_project_manipulation_probe


def _policy() -> dict[str, object]:
    return {
        "primary_planner_id": "robotwin-curobo",
        "primary_success": True,
        "fallback_attempted": False,
        "fallback_planner_id": None,
        "fallback_success": None,
        "selected_planner_id": "robotwin-curobo",
        "plan_success": True,
        "planning_time_s": 0.05,
    }


def _screw_policy(*, success: bool = True) -> dict[str, object]:
    return {
        "strategy": "mplib_screw_acquisition",
        "wrapper_method": "left_mplib_planner.plan_screw",
        "target_aware_planning_world_required": True,
        "plan_success": success,
        "vendor_status": "Success" if success else "screw plan failed",
        "selected_planner_id": "robotwin-mplib-screw" if success else None,
        "straight_screw_motion": success,
        "independent_collision_recheck_required": True,
    }


def _acquisition_policy(*, fallback: bool = False) -> dict[str, object]:
    if not fallback:
        return {
            "strategy": "mplib_target_aware_acquisition",
            "target_aware_planning_world_required": True,
            "independent_collision_recheck_required": True,
            "primary_planner_id": "robotwin-mplib-screw",
            "primary_success": True,
            "primary_failure_code": None,
            "primary_vendor_status": None,
            "primary_policy": _screw_policy(),
            "fallback_allowed": False,
            "fallback_attempted": False,
            "fallback_planner_id": None,
            "fallback_wrapper_method": "left_mplib_planner.plan_pose",
            "fallback_target_aware_planning_world_required": True,
            "fallback_success": None,
            "selected_planner_id": "robotwin-mplib-screw",
            "plan_success": True,
        }
    return {
        "strategy": "mplib_target_aware_acquisition",
        "target_aware_planning_world_required": True,
        "independent_collision_recheck_required": True,
        "primary_planner_id": "robotwin-mplib-screw",
        "primary_success": False,
        "primary_failure_code": "optimization_failed",
        "primary_vendor_status": "screw plan failed",
        "primary_policy": _screw_policy(success=False),
        "fallback_allowed": True,
        "fallback_attempted": True,
        "fallback_planner_id": "robotwin-mplib-rrt",
        "fallback_wrapper_method": "left_mplib_planner.plan_pose",
        "fallback_target_aware_planning_world_required": True,
        "fallback_success": True,
        "selected_planner_id": "robotwin-mplib-rrt",
        "plan_success": True,
    }


def _motion(name: str, *, carried: bool) -> dict[str, object]:
    stage = {
        "stage": name,
        "planner_id": (
            "robotwin-curobo-carried-object"
            if carried
            else ("robotwin-mplib-screw" if name == "approach" else "robotwin-curobo")
        ),
        "planning_time_s": 0.05,
        "plan_success": True,
        "passed": True,
        "path_samples": 20,
        "trajectory_duration_s": 0.8,
        "dynamic_safe": True,
        "dynamic_violation_count": 0,
        "execution_status": "completed",
        "replan_required": False,
        "max_tracking_error_rad": 0.02,
        "contacts": {
            "physics_steps": 50,
            "robot_target_contact_frames": 2 if carried else 0,
            "target_pad_contact_frames": 0,
            "target_table_contact_frames": 0,
        },
        "contact_policy": {
            "allow_robot_target_contact": carried,
            "allow_target_pad_contact": name == "retreat",
            "allow_target_table_contact": name in {"pregrasp", "approach", "retreat"},
        },
    }
    if carried:
        stage["collision_validation"] = {
            "self_collision_free": True,
            "collisions": [],
        }
        stage["path_metadata"] = {
            "carried_object": {
                "object_id": "pillbottle",
                "attached_object_link": "attached_object",
                "attachment_parent_link": "tool_link",
                "attachment_contact_links": ["gripper_L_link", "gripper_R_link"],
                "attachment_joint_name": "attached_object_joint",
                "world_obstacle_removed_on_attach": True,
                "reserved_sphere_count": 100,
                "pinned_config_mutated_on_disk": False,
                "asset_path_overrides": {
                    "urdf_path": "/robotwin/assets/embodiments/panthera-6dof/panthera_6dof.urdf",
                    "collision_spheres": (
                        "/robotwin/assets/embodiments/panthera-6dof/"
                        "collision_panthera.yml"
                    ),
                },
            }
        }
    else:
        stage["collision_validation"] = {
            "collision_free": True,
            "collisions": [],
        }
        stage["planner_policy"] = (
            _acquisition_policy() if name == "approach" else _policy()
        )
        stage["path_metadata"] = (
            {
                "acquisition_strategy": "mplib_target_aware_acquisition",
                "target_aware_planning_world_required": True,
                "independent_collision_recheck_required": True,
            }
            if name == "approach"
            else {}
        )
    return stage


def _payload() -> dict[str, object]:
    pregrasp = _motion("pregrasp", carried=False)
    pregrasp["contacts"]["target_table_contact_frames"] = 50
    approach = _motion("approach", carried=False)
    approach["contacts"]["target_table_contact_frames"] = 50
    lift = _motion("lift", carried=True)
    lift["object_delta_z_m"] = 0.05
    retreat = _motion("retreat", carried=False)
    retreat["contacts"]["target_pad_contact_frames"] = 50
    return {
        "schema": "panthera-native-project-manipulation-v1",
        "task": "move_pillbottle_pad",
        "task_config": "move_pillbottle_pad_panthera_single",
        "seed": 0,
        "mplib_world_initial": {
            "strategy": "scaled_target_fcl_injection",
            "planner_class": "SapienPlanner",
            "planning_world_class": "SapienPlanningWorld",
            "move_group": "tool_link",
            "target_name": "080_pillbottle_23",
            "target_present": True,
            "collision_shape_count": 1,
            "scaled_mesh_shape_count": 1,
            "non_unit_scale_shape_count": 1,
            "scale_vectors": [[0.08, 0.08, 0.08]],
            "planner_reused": True,
            "planning_world_reused": True,
        },
        "mplib_world_before_approach": {
            "strategy": "scaled_target_fcl_injection",
            "planner_class": "SapienPlanner",
            "planning_world_class": "SapienPlanningWorld",
            "move_group": "tool_link",
            "target_name": "080_pillbottle_23",
            "target_present": True,
            "collision_shape_count": 1,
            "scaled_mesh_shape_count": 1,
            "non_unit_scale_shape_count": 1,
            "scale_vectors": [[0.08, 0.08, 0.08]],
            "planner_reused": True,
            "planning_world_reused": True,
        },
        "passed": True,
        "upstream_manipulation_macro_used": False,
        "placement_goal_policy": "project_rigid_object_gripper_transform",
        "B5": {
            "passed": True,
            "selected": {"candidate_id": "pillbottle:box:c0:a2:s-1"},
        },
        "carried_object_attachment": {
            "object_id": "pillbottle",
            "attached_object_link": "attached_object",
            "attachment_parent_link": "tool_link",
            "attachment_contact_links": ["gripper_L_link", "gripper_R_link"],
            "attachment_joint_name": "attached_object_joint",
            "world_obstacle_removed_on_attach": True,
            "reserved_sphere_count": 100,
            "pinned_config_mutated_on_disk": False,
        },
        "stages": [
            pregrasp,
            approach,
            {
                "stage": "close_gripper",
                "passed": True,
                "contacts_after": ["gripper_L_link", "gripper_R_link"],
                "force_scale": 1.0,
                "result": {
                    "completed": True,
                    "actual_normalized_position": 0.0,
                    "contact_observed": True,
                    "holding_object": True,
                    "reason": None,
                },
            },
            lift,
            _motion("transport", carried=True),
            _motion("place_above_support", carried=True),
            {
                "stage": "release",
                "passed": True,
                "result": {
                    "completed": True,
                    "actual_normalized_position": 1.0,
                    "contact_observed": False,
                    "holding_object": False,
                    "reason": None,
                },
            },
            retreat,
        ],
        "task_success_after_release": True,
        "final_phase": "complete",
        "task_success_final": True,
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_valid_project_manipulation_probe_passes(tmp_path: Path) -> None:
    probe = tmp_path / "manipulation.json"
    _write(probe, _payload())

    result = audit_project_manipulation_probe(probe)

    assert result.passed
    assert result.failures == ()


def test_macro_claim_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["upstream_manipulation_macro_used"] = True
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "project-owned-boundary" for gate in result.failures)


def test_missing_dual_finger_contact_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    stages = payload["stages"]
    assert isinstance(stages, list)
    close = stages[2]
    assert isinstance(close, dict)
    close["contacts_after"] = ["gripper_L_link"]
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "close_gripper" for gate in result.failures)


def test_carried_stage_requires_attachment_provenance(tmp_path: Path) -> None:
    payload = _payload()
    stages = payload["stages"]
    assert isinstance(stages, list)
    transport = stages[4]
    assert isinstance(transport, dict)
    metadata = transport["path_metadata"]
    assert isinstance(metadata, dict)
    metadata["carried_object"] = {}
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "transport" for gate in result.failures)


def test_task_oracle_cannot_be_overridden_by_summary_flag(tmp_path: Path) -> None:
    payload = _payload()
    payload["task_success_final"] = False
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "summary" for gate in result.failures)


def test_lift_requires_measured_object_motion(tmp_path: Path) -> None:
    payload = _payload()
    stages = payload["stages"]
    assert isinstance(stages, list)
    lift = stages[3]
    assert isinstance(lift, dict)
    lift["object_delta_z_m"] = 0.005
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "lift-retained-grasp" for gate in result.failures)


def test_carried_motion_rejects_target_table_contact(tmp_path: Path) -> None:
    payload = _payload()
    stages = payload["stages"]
    assert isinstance(stages, list)
    lift = stages[3]
    assert isinstance(lift, dict)
    contacts = lift["contacts"]
    assert isinstance(contacts, dict)
    contacts["target_table_contact_frames"] = 1
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "lift" for gate in result.failures)


def test_transport_rejects_target_pad_contact_before_release(tmp_path: Path) -> None:
    payload = _payload()
    stages = payload["stages"]
    assert isinstance(stages, list)
    transport = stages[4]
    assert isinstance(transport, dict)
    contacts = transport["contacts"]
    assert isinstance(contacts, dict)
    contacts["target_pad_contact_frames"] = 1
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "transport" for gate in result.failures)


def test_project_manipulation_requires_target_aware_mplib_world(tmp_path: Path) -> None:
    payload = _payload()
    world = payload["mplib_world_initial"]
    assert isinstance(world, dict)
    world["target_present"] = False
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "mplib-target-world" for gate in result.failures)


def test_approach_requires_fresh_target_world_sync(tmp_path: Path) -> None:
    payload = _payload()
    payload["mplib_world_before_approach"] = None
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(
        gate.name == "mplib-target-world-before-approach"
        for gate in result.failures
    )


def test_approach_requires_mplib_screw_acquisition_provenance(tmp_path: Path) -> None:
    payload = _payload()
    stages = payload["stages"]
    assert isinstance(stages, list)
    approach = stages[1]
    assert isinstance(approach, dict)
    approach["planner_id"] = "robotwin-curobo"
    approach["planner_policy"] = _policy()
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "approach" for gate in result.failures)


def test_carried_stage_requires_asset_override_provenance(tmp_path: Path) -> None:
    payload = _payload()
    stages = payload["stages"]
    assert isinstance(stages, list)
    lift = stages[3]
    assert isinstance(lift, dict)
    metadata = lift["path_metadata"]
    assert isinstance(metadata, dict)
    carried = metadata["carried_object"]
    assert isinstance(carried, dict)
    carried.pop("asset_path_overrides")
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "lift" for gate in result.failures)


def test_target_aware_mplib_rrt_acquisition_fallback_is_accepted(tmp_path: Path) -> None:
    payload = _payload()
    stages = payload["stages"]
    assert isinstance(stages, list)
    approach = stages[1]
    assert isinstance(approach, dict)
    approach["planner_id"] = "robotwin-mplib-rrt"
    approach["planner_policy"] = _acquisition_policy(fallback=True)
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert result.passed


def test_approach_rejects_curobo_selected_as_acquisition_fallback(tmp_path: Path) -> None:
    payload = _payload()
    stages = payload["stages"]
    assert isinstance(stages, list)
    approach = stages[1]
    assert isinstance(approach, dict)
    policy = _acquisition_policy(fallback=True)
    policy["fallback_planner_id"] = "robotwin-curobo"
    policy["selected_planner_id"] = "robotwin-curobo"
    approach["planner_id"] = "robotwin-curobo"
    approach["planner_policy"] = policy
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "approach" for gate in result.failures)


def test_approach_rejects_rrt_fallback_without_target_aware_provenance(
    tmp_path: Path,
) -> None:
    payload = _payload()
    stages = payload["stages"]
    assert isinstance(stages, list)
    approach = stages[1]
    assert isinstance(approach, dict)
    policy = _acquisition_policy(fallback=True)
    policy["fallback_target_aware_planning_world_required"] = False
    approach["planner_id"] = "robotwin-mplib-rrt"
    approach["planner_policy"] = policy
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "approach" for gate in result.failures)


def test_attachment_requires_virtual_link_provenance(tmp_path: Path) -> None:
    payload = _payload()
    attachment = payload["carried_object_attachment"]
    assert isinstance(attachment, dict)
    attachment.pop("attachment_parent_link")
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "carried-object-attachment" for gate in result.failures)


def _with_valid_support_escape(payload: dict[str, object]) -> dict[str, object]:
    stages = payload["stages"]
    assert isinstance(stages, list)
    lift = stages[3]
    assert isinstance(lift, dict)
    support = {
        "diagnostic_only": True,
        "conservative_overlap": True,
        "within_surface_sphere_radius": True,
        "required_lift_to_sphere_clearance_m": 0.004,
    }
    planning = {
        "planning_only": True,
        "executed": False,
        "obstacle_name": "table",
        "obstacle_disabled": True,
        "obstacle_reenabled": True,
        "requested_escape_m": 0.005,
        "max_escape_m": 0.01,
        "bounded": True,
        "plan_success": True,
        "start_state": {"available": True, "valid": True, "status": None},
        "path_samples": 12,
    }
    contacts = {
        "physics_steps": 25,
        "robot_target_contact_frames": 20,
        "target_pad_contact_frames": 0,
        "target_table_contact_frames": 12,
        "robot_table_contact_frames": 0,
    }
    execution = {
        "stage": "support_escape",
        "planner_id": "robotwin-curobo-carried-object",
        "plan_success": True,
        "passed": True,
        "path_samples": 12,
        "trajectory_duration_s": 0.2,
        "dynamic_safe": True,
        "dynamic_violation_count": 0,
        "collision_validation": {
            "self_collision_free": True,
            "collisions": [],
        },
        "collision_safe": True,
        "execution_status": "completed",
        "replan_required": False,
        "max_tracking_error_rad": 0.01,
        "contacts": contacts,
        "contact_policy": {
            "allow_robot_target_contact": True,
            "allow_target_pad_contact": False,
            "allow_target_table_contact": True,
            "allow_robot_table_contact": False,
        },
    }
    escape = {
        "stage": "support_escape",
        "triggered": True,
        "passed": True,
        "requested_escape_m": 0.005,
        "maximum_escape_m": 0.01,
        "support_contact_geometry": support,
        "planning": planning,
        "execution": execution,
        "object_delta_z_m": 0.005,
        "required_clearance_m": 0.004,
        "finger_contacts_after": ["gripper_L_link", "gripper_R_link"],
        "post_start_state": {"available": True, "valid": True, "status": None},
        "contacts": contacts,
        "table_reenabled_before_execution": True,
    }
    lift["initial_plan_failure"] = {
        "code": "optimization_failed",
        "details": {
            "native_status": "Start state is colliding with world",
        },
    }
    lift["support_escape"] = escape
    payload["support_escape_diagnostic"] = escape
    return payload


def test_valid_bounded_support_escape_is_independently_accepted(
    tmp_path: Path,
) -> None:
    payload = _with_valid_support_escape(_payload())
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert result.passed
    assert not any(gate.name == "support-escape" for gate in result.failures)


def test_support_escape_rejects_robot_table_contact(tmp_path: Path) -> None:
    payload = _with_valid_support_escape(_payload())
    escape = payload["support_escape_diagnostic"]
    assert isinstance(escape, dict)
    contacts = escape["contacts"]
    assert isinstance(contacts, dict)
    contacts["robot_table_contact_frames"] = 1
    execution = escape["execution"]
    assert isinstance(execution, dict)
    execution_contacts = execution["contacts"]
    assert isinstance(execution_contacts, dict)
    execution_contacts["robot_table_contact_frames"] = 1
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "support-escape" for gate in result.failures)


def test_support_escape_rejects_invalid_post_escape_start(tmp_path: Path) -> None:
    payload = _with_valid_support_escape(_payload())
    escape = payload["support_escape_diagnostic"]
    assert isinstance(escape, dict)
    post_start = escape["post_start_state"]
    assert isinstance(post_start, dict)
    post_start["valid"] = False
    probe = tmp_path / "manipulation.json"
    _write(probe, payload)

    result = audit_project_manipulation_probe(probe)

    assert not result.passed
    assert any(gate.name == "support-escape" for gate in result.failures)


def _payload_with_bounded_recoveries() -> dict[str, object]:
    from copy import deepcopy
    from math import pi

    payload = _payload()
    payload["recovery_contract_version"] = "bounded_yaw_postrelease_v1"
    stages = payload["stages"]
    assert isinstance(stages, list)
    place = stages[5]
    retreat = stages[7]
    assert isinstance(place, dict) and isinstance(retreat, dict)
    recovery = {
        "strategy": "yaw_only_task_unconstrained_orientation_v1",
        "task_success_orientation_constrained": False,
        "upstream_task_source": {
            "path": "RoboTwin/envs/move_pillbottle_pad.py",
            "commit": "2aee31ad75cb7b0cd779540441c6025853a5b60d",
            "sha256": "20e7b9d954fd42c3b78e4a9431aee68b3ff10f70f20104e7dc7e41c83fc8cffe",
            "check_success_orientation_constrained": False,
        },
        "candidate_yaw_offsets_rad": [0.0, pi / 2, -pi / 2, pi],
        "attempts": [
            {
                "yaw_offset_rad": 0.0,
                "passed": False,
                "plan_success": False,
                "execution_status": None,
                "failure_code": "optimization_failed",
                "native_status": "IK Fail",
            },
            {
                "yaw_offset_rad": pi / 2,
                "passed": True,
                "plan_success": True,
                "execution_status": "completed",
                "failure_code": None,
                "native_status": None,
            },
        ],
        "selected_yaw_offset_rad": pi / 2,
        "bounded_attempt_count": 2,
        "max_attempt_count": 4,
    }
    place["placement_recovery"] = recovery
    payload["upstream_task_source"] = deepcopy(recovery["upstream_task_source"])
    payload["placement_recovery_diagnostic"] = deepcopy(recovery)
    separation = _motion("post_release_separation", carried=False)
    separation.update(
        {
            "requested_separation_m": 0.01,
            "maximum_separation_m": 0.01,
            "task_success_after": True,
            "finger_contacts_after": [],
            "target_supported_during_separation": True,
            "contact_policy": {
                "allow_robot_target_contact": True,
                "allow_target_pad_contact": True,
                "allow_target_table_contact": False,
                "allow_robot_table_contact": False,
            },
            "contacts": {
                "physics_steps": 30,
                "target_pad_contact_frames": 30,
                "target_table_contact_frames": 0,
                "robot_table_contact_frames": 0,
            },
        }
    )
    retreat["post_release_separation"] = separation
    retreat["retreat_total_m"] = 0.05
    retreat["retreat_remaining_m"] = 0.04
    payload["post_release_separation_diagnostic"] = deepcopy(separation)
    return payload


def _recovery_gates(payload: dict[str, object], tmp_path: Path) -> dict[str, bool]:
    probe = tmp_path / "manipulation-recovery.json"
    _write(probe, payload)
    audit = audit_project_manipulation_probe(probe)
    return {gate.name: gate.passed for gate in audit.gates}


def test_bounded_recovery_valid_evidence_is_accepted(tmp_path: Path) -> None:
    gates = _recovery_gates(_payload_with_bounded_recoveries(), tmp_path)
    assert gates["placement-yaw-recovery"]
    assert gates["post-release-separation"]
    assert all(gates.values())


def test_yaw_recovery_rejects_wrong_candidate_order_and_non_ik_retry(
    tmp_path: Path,
) -> None:
    for field, value in [
        ("candidate_yaw_offsets_rad", [0.0, -1.0, 1.0, 3.14]),
        ("selected_yaw_offset_rad", 3.14),
        ("bounded_attempt_count", 5),
    ]:
        payload = _payload_with_bounded_recoveries()
        stages = payload["stages"]
        recovery = stages[5]["placement_recovery"]
        recovery[field] = value
        payload["placement_recovery_diagnostic"] = recovery
        assert not _recovery_gates(payload, tmp_path)["placement-yaw-recovery"]

    payload = _payload_with_bounded_recoveries()
    recovery = payload["stages"][5]["placement_recovery"]
    recovery["attempts"][0]["native_status"] = "Collision"
    payload["placement_recovery_diagnostic"] = recovery
    assert not _recovery_gates(payload, tmp_path)["placement-yaw-recovery"]


def test_yaw_recovery_rejects_forged_provenance_and_mirror(tmp_path: Path) -> None:
    payload = _payload_with_bounded_recoveries()
    payload["stages"][5]["placement_recovery"]["upstream_task_source"]["sha256"] = "forged"
    payload["placement_recovery_diagnostic"] = payload["stages"][5]["placement_recovery"]
    assert not _recovery_gates(payload, tmp_path)["placement-yaw-recovery"]

    payload = _payload_with_bounded_recoveries()
    payload["placement_recovery_diagnostic"]["selected_yaw_offset_rad"] = 0.0
    assert not _recovery_gates(payload, tmp_path)["placement-yaw-recovery"]


def test_yaw_recovery_malformed_json_values_fail_closed(tmp_path: Path) -> None:
    for field, value in [
        ("candidate_yaw_offsets_rad", ["not-a-number", 1.0, -1.0, 3.0]),
        ("attempts", [{"yaw_offset_rad": {"bad": True}}]),
        ("selected_yaw_offset_rad", {"bad": True}),
    ]:
        payload = _payload_with_bounded_recoveries()
        recovery = payload["stages"][5]["placement_recovery"]
        recovery[field] = value
        payload["placement_recovery_diagnostic"] = recovery
        assert not _recovery_gates(payload, tmp_path)["placement-yaw-recovery"]


def test_separation_unsafe_contacts_and_loss_of_support_fail_closed(
    tmp_path: Path,
) -> None:
    for key, value in [
        ("target_pad_contact_frames", 0),
        ("target_table_contact_frames", 1),
        ("robot_table_contact_frames", 1),
        ("physics_steps", 0),
    ]:
        payload = _payload_with_bounded_recoveries()
        separation = payload["stages"][7]["post_release_separation"]
        separation["contacts"][key] = value
        payload["post_release_separation_diagnostic"] = separation
        assert not _recovery_gates(payload, tmp_path)["post-release-separation"]


def test_separation_requires_release_task_success_and_strict_bounds(
    tmp_path: Path,
) -> None:
    for key, value in [
        ("requested_separation_m", 0.0101),
        ("requested_separation_m", 0.0),
        ("task_success_after", False),
        ("finger_contacts_after", ["gripper_L_link"]),
        ("target_supported_during_separation", False),
        ("dynamic_safe", False),
        ("execution_status", None),
    ]:
        payload = _payload_with_bounded_recoveries()
        separation = payload["stages"][7]["post_release_separation"]
        separation[key] = value
        payload["post_release_separation_diagnostic"] = separation
        assert not _recovery_gates(payload, tmp_path)["post-release-separation"]


def test_separation_malformed_and_inconsistent_mirror_fail_closed(
    tmp_path: Path,
) -> None:
    payload = _payload_with_bounded_recoveries()
    payload["stages"][7]["post_release_separation"]["contacts"]["physics_steps"] = {}
    payload["post_release_separation_diagnostic"] = payload["stages"][7]["post_release_separation"]
    assert not _recovery_gates(payload, tmp_path)["post-release-separation"]

    payload = _payload_with_bounded_recoveries()
    payload["post_release_separation_diagnostic"]["finger_contacts_after"] = ["forged"]
    assert not _recovery_gates(payload, tmp_path)["post-release-separation"]


def test_new_recovery_contract_cannot_silently_omit_gates(tmp_path: Path) -> None:
    payload = _payload()
    payload["recovery_contract_version"] = "bounded_yaw_postrelease_v1"
    gates = _recovery_gates(payload, tmp_path)
    assert not gates["placement-yaw-recovery"]
    assert not gates["post-release-separation"]


def test_unknown_recovery_contract_version_is_rejected(tmp_path: Path) -> None:
    payload = _payload_with_bounded_recoveries()
    payload["recovery_contract_version"] = "unsupported"
    gates = _recovery_gates(payload, tmp_path)
    assert not gates["recovery-contract-version"]


def _v2_staged_recovery_payload() -> dict[str, object]:
    from copy import deepcopy
    from math import pi

    payload = _payload_with_bounded_recoveries()
    payload["recovery_contract_version"] = "bounded_yaw_postrelease_v2"
    recovery = payload["stages"][5]["placement_recovery"]
    recovery["strategy"] = "yaw_plus_bounded_staged_descent_v2"
    recovery["attempts"] = [
        {
            "yaw_offset_rad": yaw,
            "passed": False,
            "plan_success": False,
            "execution_status": None,
            "failure_code": "optimization_failed",
            "native_status": "IK Fail",
        }
        for yaw in (0.0, pi / 2, -pi / 2, pi)
    ]
    recovery["selected_yaw_offset_rad"] = 0.0
    recovery["bounded_attempt_count"] = 4
    recovery["vertical_reachability_diagnostic"] = {
        "planning_only": True,
        "executed": False,
        "acceptance_override": False,
        "attempts": [{"extra_height_m": 0.01, "plan_success": True}],
    }
    recovery["staged_descent"] = {
        "strategy": "bounded_plus_10mm_then_exact_canonical_v1",
        "extra_height_m": 0.01,
        "elevated_object_goal_xyz_m": [0.1, -0.04, 0.761],
        "canonical_object_goal_xyz_m": [0.1, -0.04, 0.751],
        "elevated_motion": _motion("place_elevated_waypoint", carried=True),
        "canonical_attempted": True,
        "canonical_plan_success": True,
        "canonical_motion_passed": True,
    }
    payload["placement_recovery_diagnostic"] = deepcopy(recovery)

    separation = payload["stages"][7]["post_release_separation"]
    separation["separation_geometry"] = {
        "strategy": "radial_xy_away_from_released_object_v1",
        "initial_robot_object_xy_distance_m": 0.1,
        "direction_xyz_unit": [1.0, 0.0, 0.0],
        "reference_object_xyz_m": [0.0, 0.0, 0.741],
        "start_gripper_xyz_m": [0.1, 0.0, 0.82],
    }
    separation["requested_goal_xyz_m"] = [0.11, 0.0, 0.82]
    payload["post_release_separation_diagnostic"] = deepcopy(separation)
    return payload


def test_v2_staged_placement_and_radial_separation_have_full_audit(
    tmp_path: Path,
) -> None:
    payload = _v2_staged_recovery_payload()
    gates = _recovery_gates(payload, tmp_path)
    assert gates["placement-yaw-recovery"]
    assert gates["post-release-separation"]
    assert all(gates.values())


def test_v2_elevated_waypoint_safety_cannot_be_forged(tmp_path: Path) -> None:
    from copy import deepcopy

    for field, value in (
        ("dynamic_safe", False),
        ("collision_validation", {"self_collision_free": False}),
        ("execution_status", "failed"),
        ("contact_policy", {"allow_robot_target_contact": False}),
    ):
        payload = _v2_staged_recovery_payload()
        stage = payload["stages"][5]
        recovery = stage["placement_recovery"]
        recovery["staged_descent"]["elevated_motion"][field] = value
        payload["placement_recovery_diagnostic"] = deepcopy(recovery)
        assert not _recovery_gates(payload, tmp_path)["placement-yaw-recovery"]


def test_v2_canonical_goal_and_four_ik_failures_are_required(tmp_path: Path) -> None:
    from copy import deepcopy

    for mutation in ("high_release", "one_yaw_success", "missing_final"):
        payload = _v2_staged_recovery_payload()
        recovery = payload["stages"][5]["placement_recovery"]
        if mutation == "high_release":
            recovery["staged_descent"]["canonical_object_goal_xyz_m"][2] = 0.76
        elif mutation == "one_yaw_success":
            recovery["attempts"][1]["plan_success"] = True
        else:
            recovery["staged_descent"]["canonical_attempted"] = False
        payload["placement_recovery_diagnostic"] = deepcopy(recovery)
        assert not _recovery_gates(payload, tmp_path)["placement-yaw-recovery"]


def test_v2_radial_separation_never_accepts_toward_object_or_missing_geometry(
    tmp_path: Path,
) -> None:
    from copy import deepcopy

    for mutation in ("toward_object", "overshoot", "missing", "unsupported_direction"):
        payload = _v2_staged_recovery_payload()
        separation = payload["stages"][7]["post_release_separation"]
        if mutation == "toward_object":
            separation["requested_goal_xyz_m"] = [0.09, 0.0, 0.82]
        elif mutation == "overshoot":
            separation["requested_goal_xyz_m"] = [0.12, 0.0, 0.82]
        elif mutation == "missing":
            separation.pop("separation_geometry")
        else:
            separation["separation_geometry"]["direction_xyz_unit"] = [-1.0, 0.0, 0.0]
        payload["post_release_separation_diagnostic"] = deepcopy(separation)
        assert not _recovery_gates(payload, tmp_path)["post-release-separation"]



def _v3_two_stage_payload() -> dict[str, object]:
    from copy import deepcopy

    payload = _v2_staged_recovery_payload()
    payload["recovery_contract_version"] = "bounded_yaw_postrelease_v3"
    recovery = payload["stages"][5]["placement_recovery"]
    staged = recovery["staged_descent"]
    staged["strategy"] = "bounded_plus_20mm_then_exact_canonical_v2"
    staged["extra_height_m"] = 0.02
    staged["elevated_object_goal_xyz_m"][2] = 0.771
    recovery["vertical_reachability_diagnostic"]["attempts"] = [
        {"extra_height_m": 0.01, "plan_success": True},
        {"extra_height_m": 0.02, "plan_success": True},
        {"extra_height_m": 0.03, "plan_success": True},
    ]
    payload["placement_recovery_diagnostic"] = deepcopy(recovery)
    retreat = payload["stages"][7]
    retreat["retreat_geometry"] = {
        "strategy": "radial_xy_away_from_released_object_v1",
        "initial_robot_object_xy_distance_m": 0.11,
        "direction_xyz_unit": [1.0, 0.0, 0.0],
        "reference_object_xyz_m": [0.0, 0.0, 0.741],
        "start_gripper_xyz_m": [0.11, 0.0, 0.82],
        "requested_goal_xyz_m": [0.15, 0.0, 0.82],
        "requested_retreat_m": 0.04,
        "no_contact_override": True,
    }
    return payload


def test_v3_full_recovery_uses_exact_20mm_then_40mm_outward(tmp_path: Path) -> None:
    gates = _recovery_gates(_v3_two_stage_payload(), tmp_path)
    assert gates["placement-yaw-recovery"]
    assert gates["post-release-separation"]
    assert gates["retreat-outward-goal"]
    assert all(gates.values())


def test_v3_fails_closed_on_invalid_waypoint_or_retreat(tmp_path: Path) -> None:
    from copy import deepcopy

    for bad in ("raise_goal", "missing_diagnostic", "toward_target", "invalid_distance",
                "forge_no_contact"):
        payload = _v3_two_stage_payload()
        staged = payload["stages"][5]["placement_recovery"]
        retreat = payload["stages"][7]["retreat_geometry"]
        if bad == "raise_goal":
            staged["staged_descent"]["elevated_object_goal_xyz_m"][2] += 0.001
        elif bad == "missing_diagnostic":
            staged["vertical_reachability_diagnostic"]["attempts"][1]["plan_success"] = False
        elif bad == "toward_target":
            retreat["requested_goal_xyz_m"][0] = 0.07
        elif bad == "invalid_distance":
            retreat["requested_retreat_m"] = 0.05
        else:
            retreat["no_contact_override"] = False
        payload["placement_recovery_diagnostic"] = deepcopy(staged)
        gates = _recovery_gates(payload, tmp_path)
        assert not all(gates.values()), bad
        if bad in ("raise_goal", "missing_diagnostic"):
            assert not gates["placement-yaw-recovery"]
        else:
            assert not gates["retreat-outward-goal"]


def test_v3_fails_existing_native_retreat_contacts_unchanged(tmp_path: Path) -> None:
    payload = _v3_two_stage_payload()
    payload["stages"][7]["contacts"]["robot_target_contact_frames"] = 1
    gates = _recovery_gates(payload, tmp_path)
    assert gates["retreat-outward-goal"]
    assert not gates["retreat"]



def _v4_collision_screen_payload(*, fallback: bool = False) -> dict[str, object]:
    payload = _v3_two_stage_payload()
    payload["recovery_contract_version"] = "bounded_yaw_postrelease_v4"
    retreat = payload["stages"][7]
    prior_world = payload["mplib_world_initial"]
    retreat["mplib_world_after_separation"] = {
        **prior_world,
        "target_pose_world_xyz_m": [0.0, 0.0, 0.741],
    }
    retreat["planner_id"] = "robotwin-mplib-rrt" if fallback else "robotwin-curobo"
    retreat["planner_policy"] = {
        "strategy": "released_target_collision_screen_v1",
        "target_world_refreshed_after_separation": True,
        "primary_planner_id": "robotwin-curobo",
        "primary_success": True,
        "primary_collision_check": {
            "collision_free": not fallback,
            "collisions": [{"waypoint_index": 1}] if fallback else [],
        },
        "primary_collision_safe": not fallback,
        "fallback_attempted": fallback,
        "fallback_planner_id": "robotwin-mplib-rrt" if fallback else None,
        "fallback_success": True if fallback else None,
        "selected_planner_id": retreat["planner_id"],
        "plan_success": True,
        "planning_time_s": 0.2,
    }
    return payload


def test_v4_postrelease_world_refresh_primary_and_fallback_pass(tmp_path: Path) -> None:
    for fallback in (False, True):
        gates = _recovery_gates(_v4_collision_screen_payload(fallback=fallback), tmp_path)
        assert all(gates.values()), [key for key, value in gates.items() if not value]
        assert gates["postrelease-mplib-target-world"]
        assert gates["retreat-outward-goal"]


def test_v4_postrelease_world_forgery_fails_closed(tmp_path: Path) -> None:

    for bad in ("missing", "stale_object_pose", "missing_target", "unscreened_primary",
                "fallback_unneeded", "fake_safe_fallback", "physical_contact"):
        payload = _v4_collision_screen_payload(fallback=(bad == "fake_safe_fallback"))
        retreat = payload["stages"][7]
        if bad == "missing":
            retreat.pop("mplib_world_after_separation")
        elif bad == "stale_object_pose":
            retreat["mplib_world_after_separation"]["target_pose_world_xyz_m"][0] += 0.01
        elif bad == "missing_target":
            retreat["mplib_world_after_separation"]["target_present"] = False
        elif bad == "unscreened_primary":
            retreat["planner_policy"]["primary_collision_safe"] = False
        elif bad == "fallback_unneeded":
            retreat["planner_policy"]["fallback_attempted"] = True
        elif bad == "fake_safe_fallback":
            retreat["planner_policy"]["primary_collision_check"]["collision_free"] = True
        else:
            retreat["contacts"]["robot_target_contact_frames"] = 1
        gates = _recovery_gates(payload, tmp_path)
        assert not all(gates.values()), bad
        if bad in ("missing", "stale_object_pose", "missing_target"):
            assert not gates["postrelease-mplib-target-world"]
        elif bad == "physical_contact":
            assert not gates["retreat"]
        else:
            assert not gates["retreat"]

def test_yaw_recovery_attests_exact_native_lf_and_crlf_source_bytes(tmp_path: Path) -> None:
    canonical_sha = "20e7b9d954fd42c3b78e4a9431aee68b3ff10f70f20104e7dc7e41c83fc8cffe"
    crlf_sha = "f0c56ea8f29f5ae3851eb8ea22c44c7a1f106a4ab8005a810d457ed4e6299de0"
    for normalization, raw_sha in (("none", canonical_sha), ("CRLF_to_LF", crlf_sha)):
        payload = _payload_with_bounded_recoveries()
        payload["upstream_task_source"].update(
            raw_sha256=raw_sha, line_ending_normalization=normalization
        )
        gates = _recovery_gates(payload, tmp_path)
        assert gates["placement-yaw-recovery"]
        assert all(gates.values())


def test_yaw_recovery_rejects_extra_source_attestation_tampering(tmp_path: Path) -> None:
    canonical_sha = "20e7b9d954fd42c3b78e4a9431aee68b3ff10f70f20104e7dc7e41c83fc8cffe"
    crlf_sha = "f0c56ea8f29f5ae3851eb8ea22c44c7a1f106a4ab8005a810d457ed4e6299de0"
    mutations = (
        {"raw_sha256": "0" * 64, "line_ending_normalization": "none"},
        {"raw_sha256": crlf_sha, "line_ending_normalization": "none"},
        {"raw_sha256": canonical_sha, "line_ending_normalization": "CRLF_to_LF"},
        {"raw_sha256": "0" * 64, "line_ending_normalization": "CRLF_to_LF"},
        {"raw_sha256": canonical_sha, "line_ending_normalization": "unknown"},
        {"raw_sha256": canonical_sha},
        {"line_ending_normalization": "none"},
        {"raw_sha256": 123, "line_ending_normalization": "none"},
        {"raw_sha256": canonical_sha, "line_ending_normalization": "none", "extra": True},
        {"raw_sha256": canonical_sha, "line_ending_normalization": "none", "commit": "forged"},
    )
    for changes in mutations:
        payload = _payload_with_bounded_recoveries()
        payload["upstream_task_source"].update(changes)
        assert not _recovery_gates(payload, tmp_path)["placement-yaw-recovery"]


def test_yaw_recovery_rejects_missing_nested_and_top_level_attestation(tmp_path: Path) -> None:
    for missing in ("upstream_task_source", "placement_recovery_diagnostic"):
        payload = _payload_with_bounded_recoveries()
        del payload[missing]
        assert not _recovery_gates(payload, tmp_path)["placement-yaw-recovery"]
    payload = _payload_with_bounded_recoveries()
    payload["upstream_task_source"]["check_success_orientation_constrained"] = True
    assert not _recovery_gates(payload, tmp_path)["placement-yaw-recovery"]


def _payload_with_pregrasp_reserve_contract() -> dict[str, object]:
    payload = _payload_with_bounded_recoveries()
    payload["pregrasp_joint_reserve_contract_version"] = "raw_timed_5milliradian_v1"
    reserved = {
        "schema": "s1m.pregrasp-interior-joint-reserve.v1",
        "passed": True,
        "minimum_margin_rad": 0.0497,
        "required_margin_rad": 0.005,
        "limiting_joint": "joint2",
        "limiting_waypoint": "raw:0",
        "violating_joint_samples": 0,
        "raw_waypoint_count": 724,
        "retimed_waypoint_count": 724,
        "joint_bounds_unchanged": True,
        "command_waypoints_modified": False,
        "physical_execution_performed": False,
    }
    from copy import deepcopy

    payload["stages"][0]["pregrasp_joint_reserve"] = deepcopy(reserved)
    cid = payload["B5"]["selected"]["candidate_id"]
    payload["B5"]["evaluations"] = [
        {"candidate": {"candidate_id": cid}, "pregrasp_joint_reserve": deepcopy(reserved)}
    ]
    return payload


def test_pregrasp_interior_reserve_contract_audit_passes(tmp_path: Path) -> None:
    payload = _payload_with_pregrasp_reserve_contract()
    gates = _recovery_gates(payload, tmp_path)
    assert gates["pregrasp-interior-joint-reserve"] is True
    assert all(gates.values())


def test_pregrasp_interior_reserve_attestation_fails_closed(tmp_path: Path) -> None:
    for key, value in (
        ("minimum_margin_rad", 0.00000019073486323684108),
        ("required_margin_rad", 0.0),
        ("command_waypoints_modified", True),
        ("physical_execution_performed", True),
        ("violating_joint_samples", 1),
        ("passed", False),
    ):
        for at in ("stage", "selection"):
            payload = _payload_with_pregrasp_reserve_contract()
            report = (
                payload["stages"][0]["pregrasp_joint_reserve"]
                if at == "stage"
                else payload["B5"]["evaluations"][0]["pregrasp_joint_reserve"]
            )
            report[key] = value
            assert _recovery_gates(payload, tmp_path)["pregrasp-interior-joint-reserve"] is False
    payload = _payload_with_pregrasp_reserve_contract()
    payload["B5"]["evaluations"] = []
    assert not _recovery_gates(payload, tmp_path)["pregrasp-interior-joint-reserve"]
    payload = _payload_with_pregrasp_reserve_contract()
    payload["pregrasp_joint_reserve_contract_version"] = "unknown"
    assert not _recovery_gates(payload, tmp_path)["pregrasp-interior-joint-reserve"]


def test_pregrasp_cached_plan_independent_acceptance_and_failclosed(tmp_path: Path) -> None:
    from copy import deepcopy

    payload = _payload_with_pregrasp_reserve_contract()
    payload["pregrasp_plan_reuse_contract_version"] = "b5_cached_start_goal_scene_v1"
    cid = payload["B5"]["selected"]["candidate_id"]
    stage = payload["stages"][0]
    policy = stage["planner_policy"]
    policy.update(prepared_from_b5=True, prepared_candidate_id=cid)
    payload["B5"]["evaluations"][0]["pregrasp_selected_planner_id"] = stage["planner_id"]
    assert _recovery_gates(payload, tmp_path)["pregrasp-interior-joint-reserve"]
    for mutation in (
        lambda x: x["stages"][0]["planner_policy"].update(prepared_from_b5=False),
        lambda x: x["stages"][0]["planner_policy"].update(prepared_candidate_id="forged"),
        lambda x: x["B5"]["evaluations"][0].update(pregrasp_selected_planner_id="forged"),
        lambda x: x.update(pregrasp_plan_reuse_contract_version="forged"),
    ):
        bad = deepcopy(payload)
        mutation(bad)
        assert not _recovery_gates(bad, tmp_path)["pregrasp-interior-joint-reserve"]


def test_pregrasp_mplib_rescue_requires_original_failure_to_be_verified(tmp_path: Path) -> None:
    from copy import deepcopy

    payload = _payload_with_pregrasp_reserve_contract()
    payload["pregrasp_plan_reuse_contract_version"] = "b5_cached_start_goal_scene_v1"
    cid = payload["B5"]["selected"]["candidate_id"]
    stage = payload["stages"][0]
    stage["planner_id"] = "robotwin-mplib-rrt"
    policy = stage["planner_policy"]
    primary = {
        **payload["stages"][0]["pregrasp_joint_reserve"],
        "passed": False,
        "minimum_margin_rad": 0.000000190734863,
        "violating_joint_samples": 2,
    }
    rescue = {
        "strategy": "pregrasp_interior_mplib_rescue_v1",
        "attempted": True,
        "accepted": True,
        "primary_native_success": True,
        "fallback_native_success": True,
        "fallback_failure": None,
        "primary_reserve": primary,
        "fallback_reserve": payload["stages"][0]["pregrasp_joint_reserve"],
    }
    policy.update(
        strategy="pregrasp_interior_mplib_rescue_v1",
        primary_success=True,
        primary_native_success=True,
        primary_reserve=primary,
        fallback_attempted=True,
        fallback_planner_id="robotwin-mplib-rrt",
        fallback_success=True,
        selected_planner_id="robotwin-mplib-rrt",
        interior_rescue=rescue,
        prepared_from_b5=True,
        prepared_candidate_id=cid,
    )
    payload["B5"]["evaluations"][0]["pregrasp_selected_planner_id"] = "robotwin-mplib-rrt"
    payload["B5"]["evaluations"][0]["pregrasp_strategy"] = policy["strategy"]
    payload["B5"]["evaluations"][0]["pregrasp_interior_rescue"] = rescue
    assert _recovery_gates(payload, tmp_path)["pregrasp-interior-joint-reserve"]
    assert _recovery_gates(payload, tmp_path)["pregrasp"]
    for edit in (
        lambda x: x["stages"][0]["planner_policy"]["interior_rescue"].update(accepted=False),
        lambda x: x["B5"]["evaluations"][0].update(pregrasp_strategy="forged"),
        lambda x: x["stages"][0]["planner_policy"]["primary_reserve"].update(passed=True),
        lambda x: x["stages"][0]["planner_policy"]["interior_rescue"]["fallback_reserve"].update(
            passed=False
        ),
    ):
        bad = deepcopy(payload)
        edit(bad)
        gates = _recovery_gates(bad, tmp_path)
        assert not (gates["pregrasp"] and gates["pregrasp-interior-joint-reserve"])
