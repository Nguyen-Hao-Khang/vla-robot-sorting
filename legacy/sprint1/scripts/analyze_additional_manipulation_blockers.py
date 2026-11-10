#!/usr/bin/env python3
"""Immutable historical seed1/2 incident analysis, not a native experiment."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_robot_sorting.planning.approach_safety_audit import verify_checksum_inventory
from vla_robot_sorting.planning.manipulation_recovery import (
    CANONICAL_URDF_SHA256,
    UPSTREAM_SHA,
    RecoveryContractError,
    digest,
    write_new_json,
)


def stage(data, name):
    rows = [s for s in data["stages"] if s.get("stage") == name]
    if len(rows) != 1:
        raise RecoveryContractError("missing/duplicate authoritative stage")
    return rows[0]


def pairs(hits):
    result = set()
    for row in hits:
        for hit in row.get("environment_hits", []):
            payload = hit.get("attributes", hit)
            result.add(
                (
                    str(payload.get("link_name1")),
                    str(payload.get("object_name2")),
                    str(payload.get("link_name2")),
                )
            )
    return [list(x) for x in sorted(result)]


def analyze(seed12_dir, alternative_dir, urdf):
    roots = [Path(seed12_dir).resolve(), Path(alternative_dir).resolve()]
    hashes = [verify_checksum_inventory(root) for root in roots]
    expected = [
        {
            "seed1-up75.json": ("1e16de127796dc9ba5ec1e0ffc46cd848231c34090e04e114fd76823af1dd664"),
            "seed2-canonical.json": (
                "9ccf29229c329405ecf6f40a8cbf1cf8dbfdf37f00f4c498c9193295a2e25256"
            ),
        },
        {
            "seed2-alternative.json": (
                "214cd667cb48adb22010734516f96ea7b9468f736e6b20feb9a23b718f9eee71"
            )
        },
    ]
    for root, inventory, pins in zip(roots, hashes, expected, strict=True):
        if any(inventory.get(n) != sha for n, sha in pins.items()):
            raise RecoveryContractError("authoritative record checksum mismatch")
        if (root / "robotwin-upstream-pin.txt").read_text().strip() != UPSTREAM_SHA:
            raise RecoveryContractError("historical upstream mismatch")
    if digest(Path(urdf)) != CANONICAL_URDF_SHA256:
        raise RecoveryContractError("canonical URDF differs from pinned LFS payload")
    joint4 = next(
        j for j in ET.parse(urdf).getroot().findall("joint") if j.attrib["name"] == "joint4"
    )
    upper = float(joint4.find("limit").attrib["upper"])
    a = json.loads((roots[0] / "seed1-up75.json").read_text())
    b = json.loads((roots[0] / "seed2-canonical.json").read_text())
    c = json.loads((roots[1] / "seed2-alternative.json").read_text())
    retreat = stage(a, "retreat")
    screen = retreat["planner_policy"]["primary_collision_check"]
    indexes = sorted({int(x["waypoint_index"]) for x in screen["collisions"]})
    geom = retreat["retreat_geometry"]
    distance = math.dist(geom["start_gripper_xyz_m"], geom["requested_goal_xyz_m"])
    place = stage(b, "place_above_support")
    elevated = place["placement_recovery"]["staged_descent"]["elevated_motion"]
    measured = elevated["qpos_after_rad"][3]
    nearest = struct.unpack("<f", struct.pack("<f", upper))[0]
    approach = stage(c, "approach")
    alt_hits = approach["collision_validation"]["collisions"]
    alt_indexes = sorted({int(x["waypoint_index"]) for x in alt_hits})
    return {
        "schema": "s1m.additional-historical-blockers.v1",
        "sources": [
            {
                "directory": str(root),
                "hashes_verified": len(inventory),
                "project_sha": (root / "git-head.txt").read_text().strip(),
                "inventory_sha256": digest(root / "SHA256SUMS.txt"),
            }
            for root, inventory in zip(roots, hashes, strict=True)
        ],
        "urdf_sha256": CANONICAL_URDF_SHA256,
        "seed1": {
            "primary_fcl_collision_count": len(indexes),
            "primary_collision_range": [indexes[0], indexes[-1]],
            "pairs": pairs(screen["collisions"]),
            "fallback_failure": retreat["failure"],
            "requested_retreat_m": distance,
            "measured_separation_arm_qpos": retreat["post_release_separation"]["qpos_after_rad"],
            "proven": "CuRobo path screened out; MPlib did not return valid IK. No retreat motion.",
            "unproven": "Solver convergence/bounds/collision filtering; exact goal feasibility",
        },
        "seed2": {
            "joint4_upper_rad": upper,
            "elevated_measured_joint4_rad": measured,
            "strict_excess_rad": measured - upper,
            "float32_nearest_upper_rad": nearest,
            "float32_upper_rounding_excess_rad": nearest - upper,
            "ulp_at_upper_rad": 2**-23,
            "ulps_above_nearest_float32": (measured - nearest) / (2**-23),
            "elevated_start_arm_qpos": elevated["qpos_before_rad"],
            "elevated_terminal_arm_qpos": elevated["qpos_after_rad"],
            "max_tracking_error_rad": elevated["max_tracking_error_rad"],
            "next_query_failure": place["failure"],
            "missing_fields": [
                "timed_commanded_qpos",
                "per-step_full_qpos",
                "per-step_FK",
                "per-step_drive_association",
                "physical_contact_points",
            ],
            "proven": "Measured joint4 above declared limit; next CuRobo query rejected start",
            "unproven": "Missing first-crossing drive targets/tracking/contact forces",
        },
        "seed2_alternative": {
            "path_samples": approach["path_samples"],
            "collision_count": len(alt_indexes),
            "collision_range": [alt_indexes[0], alt_indexes[-1]],
            "pairs": pairs(alt_hits),
            "execution_status": approach.get("execution_status"),
            "planner_success": approach["plan_success"],
            "dynamic_safe": approach["dynamic_safe"],
        },
        "new_native_calls": 0,
        "native_acceptance": False,
        "controller_eligible": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed12-dir", type=Path, required=True)
    p.add_argument("--alternative-dir", type=Path, required=True)
    p.add_argument("--canonical-urdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if any(
        root.resolve() == output or root.resolve() in output.parents
        for root in (args.seed12_dir, args.alternative_dir)
    ):
        raise RecoveryContractError("refusing write within immutable evidence")
    if output.exists():
        raise FileExistsError("refusing analysis overwrite")
    record = analyze(args.seed12_dir, args.alternative_dir, args.canonical_urdf)
    write_new_json(output, record)
    print(
        json.dumps(
            {
                "hashes_verified": sum(s["hashes_verified"] for s in record["sources"]),
                "new_native_calls": 0,
                "native_acceptance": False,
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
