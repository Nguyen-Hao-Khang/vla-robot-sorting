from __future__ import annotations

import json
from pathlib import Path

from vla_robot_sorting.evidence import audit_integrated_pregrasp_probe


def _mplib_world() -> dict[str, object]:
    return {
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
    }


def _world_sync() -> dict[str, object]:
    return {
        "source": "pinned RoboTwin create_table_and_wall tabletop collision",
        "table_world_center_xyz_m": [0.0, 0.0, 0.715],
        "table_world_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "table_dims_xyz_m": [1.2, 0.7, 0.05],
        "robot_base_world_xyz_m": [0.0, -0.45, 0.75],
        "robot_base_world_orientation_wxyz": [
            0.70710678,
            0.0,
            0.0,
            0.70710678,
        ],
        "table_base_pose_wxyz": [
            0.45,
            0.0,
            -0.035,
            0.70710678,
            0.0,
            0.0,
            -0.70710678,
        ],
        "synchronized": True,
        "cache_hit": False,
        "updated_motion_generators": ["motion_gen", "motion_gen_batch"],
    }


def _payload() -> dict[str, object]:
    return {
        "schema": "panthera-native-integrated-pregrasp-v1",
        "task": "move_pillbottle_pad",
        "task_config": "move_pillbottle_pad_panthera_single",
        "seed": 0,
        "mplib_world_initial": _mplib_world(),
        "grounding": {
            "passed": True,
            "target_object_id": "pillbottle",
            "destination_id": "pad",
            "target_confidence": 1.0,
            "destination_confidence": 1.0,
            "scene_revision": "scene-0",
            "latency_s": 0.001,
            "scene_source": "canonical RoboTwin runtime ground truth",
        },
        "grasp": {
            "passed": True,
            "candidate_count": 4,
            "non_top_down_count": 2,
            "selected": {"candidate_id": "c0"},
        },
        "preconditions": {
            "passed": True,
            "planning": {
                "primary_planner_id": "robotwin-curobo",
                "primary_success": True,
                "fallback_attempted": False,
                "fallback_planner_id": None,
                "fallback_success": None,
                "selected_planner_id": "robotwin-curobo",
                "selected_curobo_world_sync": _world_sync(),
                "plan_success": True,
                "planning_time_s": 0.05,
            },
            "path_samples": 20,
            "collision_recheck": {
                "provenance_schema": "mplib-collision-v1",
                "scene_entities": {},
                "waypoints_checked": 20,
                "collision_free": True,
                "collisions": [],
            },
            "trajectory_duration_s": 0.8,
            "dynamic_safe": True,
            "dynamic_violation_count": 0,
            "controller_period_s": 0.016,
            "time_dilation_factor": 4,
        },
        "execution": {
            "passed": True,
            "feedback_status": "completed",
            "feedback_reason": None,
            "replan_required": False,
            "executed_until_s": 0.192,
            "segment_horizon_s": 0.192,
            "controller_period_s": 0.016,
            "physics_steps": 48,
            "object_robot_contact_frames": 0,
            "object_robot_contact_points": 0,
            "pad_robot_contact_frames": 0,
            "pad_robot_contact_points": 0,
            "max_qpos_delta_rad": 0.04,
            "max_tracking_error_rad": 0.02,
            "execution_latency_s": 0.2,
        },
        "destination_recovery": {
            "passed": True,
            "change_kind": "destination_moved",
            "position_delta_m": 0.025,
            "decision_action": "replan",
            "detection_latency_s": 0.001,
            "decision_latency_s": 0.001,
            "robot_execution_after_destination_move": False,
        },
        "passed": True,
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_valid_integrated_pregrasp_probe_passes(tmp_path: Path) -> None:
    probe = tmp_path / "integrated.json"
    _write(probe, _payload())
    result = audit_integrated_pregrasp_probe(probe)
    assert result.passed


def test_runtime_object_contact_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    execution = payload["execution"]
    assert isinstance(execution, dict)
    execution["object_robot_contact_frames"] = 1
    probe = tmp_path / "integrated.json"
    _write(probe, payload)
    result = audit_integrated_pregrasp_probe(probe)
    assert not result.passed
    assert any(gate.name == "execution" for gate in result.failures)


def test_tracking_error_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    execution = payload["execution"]
    assert isinstance(execution, dict)
    execution["max_tracking_error_rad"] = 0.081
    probe = tmp_path / "integrated.json"
    _write(probe, payload)
    result = audit_integrated_pregrasp_probe(probe)
    assert not result.passed


def test_destination_recovery_must_not_execute_after_move(tmp_path: Path) -> None:
    payload = _payload()
    recovery = payload["destination_recovery"]
    assert isinstance(recovery, dict)
    recovery["robot_execution_after_destination_move"] = True
    probe = tmp_path / "integrated.json"
    _write(probe, payload)
    result = audit_integrated_pregrasp_probe(probe)
    assert not result.passed


def test_preconditions_require_primary_fallback_planner_provenance(tmp_path: Path) -> None:
    payload = _payload()
    preconditions = payload["preconditions"]
    assert isinstance(preconditions, dict)
    planning = preconditions["planning"]
    assert isinstance(planning, dict)
    planning["primary_planner_id"] = "robotwin-mplib-rrt"
    planning["selected_planner_id"] = "robotwin-mplib-rrt"
    probe = tmp_path / "integrated.json"
    _write(probe, payload)

    result = audit_integrated_pregrasp_probe(probe)

    assert not result.passed
    assert any(gate.name == "preconditions" for gate in result.failures)


def test_preconditions_require_collision_provenance(tmp_path: Path) -> None:
    payload = _payload()
    preconditions = payload["preconditions"]
    assert isinstance(preconditions, dict)
    collision = preconditions["collision_recheck"]
    assert isinstance(collision, dict)
    collision.pop("provenance_schema")
    probe = tmp_path / "integrated.json"
    _write(probe, payload)

    result = audit_integrated_pregrasp_probe(probe)

    assert not result.passed
    assert any(gate.name == "preconditions" for gate in result.failures)


def test_preconditions_require_curobo_world_sync_provenance(tmp_path: Path) -> None:
    payload = _payload()
    preconditions = payload["preconditions"]
    assert isinstance(preconditions, dict)
    planning = preconditions["planning"]
    assert isinstance(planning, dict)
    planning.pop("selected_curobo_world_sync")
    probe = tmp_path / "integrated.json"
    _write(probe, payload)

    result = audit_integrated_pregrasp_probe(probe)

    assert not result.passed
    assert any(gate.name == "preconditions" for gate in result.failures)


def test_integrated_requires_rebuilt_mplib_world_with_target(tmp_path: Path) -> None:
    payload = _payload()
    world = payload["mplib_world_initial"]
    assert isinstance(world, dict)
    world["target_present"] = False
    probe = tmp_path / "integrated.json"
    _write(probe, payload)

    result = audit_integrated_pregrasp_probe(probe)

    assert not result.passed
    assert any(gate.name == "mplib-world" for gate in result.failures)
