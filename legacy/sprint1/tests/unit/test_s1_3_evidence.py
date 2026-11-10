from __future__ import annotations

import json
from pathlib import Path

from vla_robot_sorting.evidence import audit_s1_3_grasp_recovery_probe


def _candidate() -> dict[str, object]:
    return {
        "candidate_id": "pillbottle:box:c0:a1:s+1",
        "object_id": "pillbottle",
        "grasp_position_xyz": [0.1, 0.0, 0.8],
        "grasp_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "pregrasp_position_xyz": [0.04, 0.0, 0.8],
        "pregrasp_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "approach_direction": [1.0, 0.0, 0.0],
        "closing_direction": [0.0, 1.0, 0.0],
        "jaw_width_m": 0.03,
        "source": "geometry_first",
        "geometry_source": "sim_ground_truth",
        "scene_revision": "scene-0",
        "transform_revision": "tf-1",
        "raw_quality": 0.8,
        "metadata": {"table_clearance_m": 0.02},
    }


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


def _record(index: int) -> dict[str, object]:
    candidate = _candidate()
    candidate["scene_revision"] = f"scene-{index}"
    evaluation = {
        "candidate": candidate,
        "terminal_reachability": {
            "success": True,
            "planner_id": "robotwin-curobo",
            "planning_time_s": 0.08,
            "dynamic_safe": True,
            "curobo_world_sync": _world_sync(),
            "self_collision_recheck": {
                "provenance_schema": "mplib-collision-v1",
                "scene_entities": {},
                "waypoints_checked": 20,
                "self_collision_free": True,
                "collisions": [],
            },
        },
        "pregrasp_primary_planner_id": "robotwin-curobo",
        "pregrasp_primary_success": True,
        "pregrasp_fallback_attempted": False,
        "pregrasp_fallback_planner_id": None,
        "pregrasp_fallback_success": None,
        "pregrasp_selected_planner_id": "robotwin-curobo",
        "pregrasp_selected_curobo_world_sync": _world_sync(),
        "pregrasp_plan_success": True,
        "pregrasp_planning_time_s": 0.05,
        "pregrasp_path_samples": 20,
        "retimed_duration_s": 0.8,
        "dynamic_safe": True,
        "dynamic_violation_count": 0,
        "collision_recheck": {
            "provenance_schema": "mplib-collision-v1",
            "scene_entities": {},
            "waypoints_checked": 20,
            "collision_free": True,
            "collisions": [],
        },
        "joint_path_length_rad": 0.5,
        "reachable": True,
        "collision_free": True,
    }
    return {
        "run": index,
        "seed": index,
        "mplib_world_initial": _mplib_world(),
        "B5": {
            "passed": True,
            "candidate_count": 4,
            "non_top_down_count": 2,
            "provenance_complete_count": 4,
            "candidates": [candidate],
            "evaluations": [evaluation],
            "ranking": [],
            "rejection_histogram": {},
            "selected": candidate,
        },
        "B7": {
            "passed": True,
            "perturbation": {
                "before_xyz_m": [0.1, 0.0, 0.8],
                "after_xyz_m": [0.1, 0.025, 0.8],
                "delta_y_m": 0.025,
            },
            "old_scene_revision": f"scene-{index}",
            "new_scene_revision": f"scene-{index}-moved",
            "transform_revision": "tf-1",
            "scene_change_kind": "target_moved",
            "position_delta_m": 0.025,
            "detection_latency_s": 0.001,
            "decision_action": "replan",
            "decision_latency_s": 0.001,
            "mplib_world_after_perturbation": _mplib_world(),
            "initial_planning": {
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
            "stale_segment_scene_revision": f"scene-{index}",
            "stale_segment_transform_revision": "tf-1",
            "stale_feedback_status": "cancelled",
            "stale_replan_required": True,
            "stale_physics_steps": 0,
            "qpos_unchanged": True,
            "replacement_latency_s": 0.2,
            "replacement_candidate_count": 4,
            "replacement_passed": True,
            "replacement_selected": candidate,
            "replacement_planning": {
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
            "replacement_collision_recheck": {
                "provenance_schema": "mplib-collision-v1",
                "scene_entities": {},
                "waypoints_checked": 20,
                "collision_free": True,
                "collisions": [],
            },
            "replacement_segment_ready": True,
            "replacement_segment_scene_revision": f"scene-{index}-moved",
            "replacement_segment_transform_revision": "tf-1",
            "replacement_collision_free": True,
        },
    }


def _payload() -> dict[str, object]:
    records = [_record(index) for index in range(20)]
    return {
        "schema": "panthera-native-b5-b7-v1",
        "task": "move_pillbottle_pad",
        "task_config": "move_pillbottle_pad_panthera_single",
        "gate_config": {
            "runs": 20,
            "max_candidate_evaluations": 8,
            "non_top_down_abs_approach_z_max": 0.5,
            "terminal_reachability": {
                "backend": "robotwin-curobo",
            },
            "perturbation": {
                "translation_m": 0.025,
                "axis": "y",
            },
            "policy": {
                "allow_terminal_target_contact": True,
                "require_pregrasp_environment_collision_free": True,
                "require_planning_world_resync_after_perturbation": True,
                "require_terminal_self_collision_free": True,
                "require_dynamic_limits": True,
                "require_all_runs_pass": True,
            },
        },
        "gate_config_path": "configs/s1.3/grasp_recovery_gates.json",
        "runs": 20,
        "B5_pass_count": 20,
        "B7_pass_count": 20,
        "B5_pass": True,
        "B7_pass": True,
        "records": records,
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_valid_s1_3_probe_passes(tmp_path: Path) -> None:
    probe = tmp_path / "grasp-recovery.json"
    _write(probe, _payload())

    result = audit_s1_3_grasp_recovery_probe(probe)

    assert result.passed
    assert result.failures == ()


def test_stale_segment_with_physics_step_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    record = records[3]
    assert isinstance(record, dict)
    b7 = record["B7"]
    assert isinstance(b7, dict)
    b7["stale_physics_steps"] = 1
    probe = tmp_path / "grasp-recovery.json"
    _write(probe, payload)

    result = audit_s1_3_grasp_recovery_probe(probe)

    assert not result.passed
    assert any(gate.name == "run-3-B7" for gate in result.failures)


def test_selected_candidate_must_have_complete_provenance(tmp_path: Path) -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    b5 = record["B5"]
    assert isinstance(b5, dict)
    selected = b5["selected"]
    assert isinstance(selected, dict)
    selected["geometry_source"] = "depth_estimate"
    probe = tmp_path / "grasp-recovery.json"
    _write(probe, payload)

    result = audit_s1_3_grasp_recovery_probe(probe)

    assert not result.passed
    assert any(gate.name == "run-0-B5" for gate in result.failures)


def test_summary_flags_cannot_override_bad_run(tmp_path: Path) -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    record = records[5]
    assert isinstance(record, dict)
    record["error"] = "synthetic runtime failure"
    probe = tmp_path / "grasp-recovery.json"
    _write(probe, payload)

    result = audit_s1_3_grasp_recovery_probe(probe)

    assert not result.passed
    assert any(gate.name == "run-5-no-error" for gate in result.failures)


def test_b7_rejects_mandatory_mplib_planner_provenance(tmp_path: Path) -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    b7 = record["B7"]
    assert isinstance(b7, dict)
    initial = b7["initial_planning"]
    assert isinstance(initial, dict)
    initial.update(
        {
            "primary_planner_id": "robotwin-mplib-rrt",
            "primary_success": True,
            "selected_planner_id": "robotwin-mplib-rrt",
        }
    )
    probe = tmp_path / "grasp-recovery.json"
    _write(probe, payload)

    result = audit_s1_3_grasp_recovery_probe(probe)

    assert not result.passed
    assert any(gate.name == "run-0-B7" for gate in result.failures)


def test_b7_replacement_segment_must_lock_new_revision(tmp_path: Path) -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    record = records[1]
    assert isinstance(record, dict)
    b7 = record["B7"]
    assert isinstance(b7, dict)
    b7["replacement_segment_scene_revision"] = b7["old_scene_revision"]
    probe = tmp_path / "grasp-recovery.json"
    _write(probe, payload)

    result = audit_s1_3_grasp_recovery_probe(probe)

    assert not result.passed
    assert any(gate.name == "run-1-B7" for gate in result.failures)


def test_b5_requires_collision_provenance_schema(tmp_path: Path) -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    record = records[2]
    assert isinstance(record, dict)
    b5 = record["B5"]
    assert isinstance(b5, dict)
    evaluations = b5["evaluations"]
    assert isinstance(evaluations, list)
    evaluation = evaluations[0]
    assert isinstance(evaluation, dict)
    collision = evaluation["collision_recheck"]
    assert isinstance(collision, dict)
    collision.pop("provenance_schema")
    probe = tmp_path / "grasp-recovery.json"
    _write(probe, payload)

    result = audit_s1_3_grasp_recovery_probe(probe)

    assert not result.passed
    assert any(gate.name == "run-2-B5" for gate in result.failures)


def test_b5_rejects_missing_curobo_world_sync_provenance(tmp_path: Path) -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    b5 = record["B5"]
    assert isinstance(b5, dict)
    evaluations = b5["evaluations"]
    assert isinstance(evaluations, list)
    evaluation = evaluations[0]
    assert isinstance(evaluation, dict)
    evaluation.pop("pregrasp_selected_curobo_world_sync")
    probe = tmp_path / "grasp-recovery.json"
    _write(probe, payload)

    result = audit_s1_3_grasp_recovery_probe(probe)

    assert not result.passed
    assert any(gate.name == "run-0-B5" for gate in result.failures)


def test_b5_requires_rebuilt_mplib_world_with_target_present(tmp_path: Path) -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    record.pop("mplib_world_initial")
    probe = tmp_path / "grasp-recovery.json"
    _write(probe, payload)

    result = audit_s1_3_grasp_recovery_probe(probe)

    assert not result.passed
    assert any(gate.name == "run-0-B5" for gate in result.failures)


def test_b7_requires_rebuilt_mplib_world_after_perturbation(tmp_path: Path) -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    b7 = record["B7"]
    assert isinstance(b7, dict)
    world = b7["mplib_world_after_perturbation"]
    assert isinstance(world, dict)
    world["target_present"] = False
    probe = tmp_path / "grasp-recovery.json"
    _write(probe, payload)

    result = audit_s1_3_grasp_recovery_probe(probe)

    assert not result.passed
    assert any(gate.name == "run-0-B7" for gate in result.failures)
