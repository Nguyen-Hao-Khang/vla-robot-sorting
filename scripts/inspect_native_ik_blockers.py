"""Read-only comparison of historical seed1 retreat and seed2 canonical IK.

No RoboTwin, native IK, simulator physics, or command execution occurs.
The seed1 alternative coordinates are position-only; wrist orientation and
finger clearance cannot be reconstructed from retreat_geometry alone.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from vla_robot_sorting.contracts import Pose6D

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts/linux/probe_panthera_project_manipulation.py"


def read_evidence(path: Path, seed: int) -> tuple[dict, str]:
    raw = path.read_bytes()
    record = json.loads(raw)
    if record.get("schema") != "panthera-native-project-manipulation-v1":
        raise ValueError("not a project-owned native manipulation evidence file")
    if record.get("seed") != seed:
        raise ValueError(f"expected seed {seed}")
    if record.get("passed") is not False:
        raise ValueError("diagnostic requires the historical failed attempt")
    return record, hashlib.sha256(raw).hexdigest()


def stage(record: dict, name: str) -> dict:
    matches = [item for item in record.get("stages", ()) if item.get("stage") == name]
    if len(matches) != 1:
        raise ValueError(f"evidence needs exactly one {name} stage")
    return matches[0]


def inspect(seed1: dict, seed2: dict) -> dict:
    ret = stage(seed1, "retreat")
    if ret.get("plan_success") is not False or (
        ret.get("post_release_separation", {}).get("passed") is not True
    ):
        raise ValueError("seed1 must have passed separation and failed retreat planning")
    geometry = ret.get("retreat_geometry")
    if not isinstance(geometry, dict):
        raise ValueError("seed1 retreat geometry not recorded")
    spec = importlib.util.spec_from_file_location("s1m_probe_geometry", PROBE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity = (0.0, 0.0, 0.0, 1.0)
    ee = Pose6D(tuple(geometry["start_gripper_xyz_m"]), identity, "world")
    obj = Pose6D(tuple(geometry["reference_object_xyz_m"]), identity, "world")
    baseline, _ = module.bounded_contact_free_retreat_goal(ee, obj, 0.04)
    recorded = tuple(geometry["requested_goal_xyz_m"])
    if any(abs(x - y) > 1e-7 for x, y in zip(baseline.position_xyz, recorded, strict=True)):
        raise ValueError("seed1 geometry cannot reproduce recorded baseline")
    targets = [
        {"direction": name, "position_xyz_m": list(goal.position_xyz)}
        for name, goal in module.bounded_retreat_diagnostic_goals(ee, obj)
    ]

    place = stage(seed2, "place_above_support")
    if place.get("plan_success") is not False:
        raise ValueError("seed2 canonical placement did not fail planning")
    recovery = place.get("placement_recovery", {})
    staged = recovery.get("staged_descent", {})
    if staged.get("canonical_attempted") is not True or (
        staged.get("canonical_motion_passed") is not False
    ):
        raise ValueError("seed2 expected elevated success and canonical failure")
    canonical_xyz = staged["canonical_object_goal_xyz_m"]
    elevated_xyz = staged["elevated_object_goal_xyz_m"]
    if any(abs(a - b) > 1e-6 for a, b in zip(canonical_xyz[:2], elevated_xyz[:2], strict=True)):
        raise ValueError("staged waypoint shifted exact placement XY")
    if abs(elevated_xyz[2] - canonical_xyz[2] - 0.02) > 1e-6:
        raise ValueError("staged waypoint was not exactly +20 mm")
    attempts = recovery.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 4:
        raise ValueError("seed2 four canonical yaw attempts not recorded")

    return {
        "classification": "OFFLINE_NATIVE_EVIDENCE_GEOMETRY_ONLY_NOT_ACCEPTED",
        "simulator_calls": 0,
        "robot_commands": 0,
        "seed1": {
            "selected_candidate_id": seed1["B5"]["selected"]["candidate_id"],
            "retreat_planner": ret.get("planner_id"),
            "failure": ret.get("failure"),
            "standard_goal_xyz_m": list(recorded),
            "standard_goal_distance_m": 0.04,
            "alternative_position_only_goals": targets,
            "wrist_orientation_verified": False,
            "finger_clearance_measured": False,
            "alternative_native_ik_checked": False,
        },
        "seed2": {
            "selected_candidate_id": seed2["B5"]["selected"]["candidate_id"],
            "canonical_object_xyz_m": canonical_xyz,
            "canonical_gripper_xyz_m": staged.get("canonical_gripper_goal_xyz_m"),
            "elevated_object_xyz_m": elevated_xyz,
            "height_above_canonical_m": 0.02,
            "elevated_motion_passed": staged.get("elevated_motion", {}).get("passed"),
            "canonical_motion_passed": False,
            "yaw_ik_failures": [
                {
                    "yaw_offset_rad": item.get("yaw_offset_rad"),
                    "native_status": item.get("native_status"),
                    "plan_success": item.get("plan_success"),
                }
                for item in attempts
            ],
            "table_obstacle_caused_ik_failure": "UNKNOWN",
            "alternative_native_ik_checked": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed1", required=True, type=Path)
    parser.add_argument("--seed2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    first, first_hash = read_evidence(args.seed1, 1)
    second, second_hash = read_evidence(args.seed2, 2)
    result = inspect(first, second)
    result["input_sha256"] = {"seed1": first_hash, "seed2": second_hash}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["classification"])
    print(
        "seed1 alternative retreat positions:",
        len(result["seed1"]["alternative_position_only_goals"]),
    )
    print("seed2 exact placement preserved; four yaw IK failures reviewed")
    print("output:", args.output)


if __name__ == "__main__":
    main()
