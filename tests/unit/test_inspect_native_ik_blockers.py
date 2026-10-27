"""Fail-closed Windows-only analysis of historical native evidence."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/inspect_native_ik_blockers.py"


def load():
    spec = importlib.util.spec_from_file_location("native_ik_inspection", SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed1():
    return {
        "schema": "panthera-native-project-manipulation-v1",
        "seed": 1, "passed": False,
        "B5": {"selected": {"candidate_id": "s1:up75"}},
        "stages": [
            {"stage": "retreat", "passed": False, "plan_success": False,
             "planner_id": "robotwin-mplib-rrt",
             "failure": {"code": "optimization_failed"},
             "post_release_separation": {"passed": True},
             "retreat_geometry": {
                 "start_gripper_xyz_m": [0.1, 0.0, 0.8],
                 "reference_object_xyz_m": [0.0, 0.0, 0.74],
                 "requested_goal_xyz_m": [0.14, 0.0, 0.8],
             }},
        ],
    }


def seed2():
    return {
        "schema": "panthera-native-project-manipulation-v1",
        "seed": 2, "passed": False,
        "B5": {"selected": {"candidate_id": "s2:side2"}},
        "stages": [
            {"stage": "place_above_support", "passed": False,
             "plan_success": False,
             "placement_recovery": {
                 "attempts": [
                     {"yaw_offset_rad": yaw, "native_status": "IK Fail",
                      "plan_success": False}
                     for yaw in (0.0, 1.57, -1.57, 3.14)
                 ],
                 "staged_descent": {
                     "canonical_attempted": True,
                     "canonical_motion_passed": False,
                     "canonical_object_goal_xyz_m": [0.1, 0.2, 0.751],
                     "elevated_object_goal_xyz_m": [0.1, 0.2, 0.771],
                     "canonical_gripper_goal_xyz_m": [0.05, 0.09, 0.85],
                     "elevated_motion": {"passed": True},
                 },
             }},
        ],
    }


def test_offline_inspection_requires_no_native_and_does_not_certify():
    result = load().inspect(seed1(), seed2())
    assert result["robot_commands"] == 0
    assert result["simulator_calls"] == 0
    assert "NOT_ACCEPTED" in result["classification"]
    assert len(result["seed1"]["alternative_position_only_goals"]) == 4
    assert result["seed1"]["alternative_native_ik_checked"] is False
    assert result["seed1"]["wrist_orientation_verified"] is False
    assert result["seed1"]["finger_clearance_measured"] is False
    assert result["seed2"]["canonical_object_xyz_m"] == [0.1, 0.2, 0.751]
    assert result["seed2"]["table_obstacle_caused_ik_failure"] == "UNKNOWN"


def test_rejects_shifted_canonical_xy_or_wrong_staging_height():
    module = load()
    data = seed2()
    data["stages"][0]["placement_recovery"]["staged_descent"][
        "elevated_object_goal_xyz_m"
    ][0] += 0.001
    with pytest.raises(ValueError, match="shifted"):
        module.inspect(seed1(), data)

    data = seed2()
    data["stages"][0]["placement_recovery"]["staged_descent"][
        "elevated_object_goal_xyz_m"
    ][2] = 0.773
    with pytest.raises(ValueError, match="20 mm"):
        module.inspect(seed1(), data)


def test_rejects_missing_native_retreat_evidence_and_modified_baseline():
    module = load()
    data = seed1()
    data["stages"][0]["retreat_geometry"]["requested_goal_xyz_m"][0] += 0.002
    with pytest.raises(ValueError, match="reproduce recorded baseline"):
        module.inspect(data, seed2())

    data = seed1()
    data["stages"][0]["post_release_separation"]["passed"] = False
    with pytest.raises(ValueError, match="passed separation"):
        module.inspect(data, seed2())


def test_read_evidence_validates_schema_seed_and_failed_state(tmp_path):
    module = load()
    input_file = tmp_path / "evidence.json"
    input_file.write_text(json.dumps(seed1()))
    data, digest = module.read_evidence(input_file, 1)
    assert data["seed"] == 1
    assert len(digest) == 64
    with pytest.raises(ValueError, match="expected seed 2"):
        module.read_evidence(input_file, 2)
    good = seed1()
    good["passed"] = True
    input_file.write_text(json.dumps(good))
    with pytest.raises(ValueError, match="historical failed"):
        module.read_evidence(input_file, 1)
