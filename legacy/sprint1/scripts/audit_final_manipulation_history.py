#!/usr/bin/env python3
"""Read-only historical evidence audit; no promotion or native execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_robot_sorting.evidence import audit_project_manipulation_probe
from vla_robot_sorting.planning.approach_safety_audit import verify_checksum_inventory
from vla_robot_sorting.planning.manipulation_recovery import (
    CANONICAL_LOWER_RAD,
    CANONICAL_UPPER_RAD,
    CANONICAL_URDF_SHA256,
    RecoveryContractError,
    digest,
    write_new_json,
)
from vla_robot_sorting.planning.native_plan_diagnostics import finite_vector


def summarize_motion_records(data: Any) -> list[dict[str, Any]]:
    records = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, f"{path}/{i}")
        elif isinstance(value, dict):
            if "stage" in value and ("plan_success" in value or "execution_status" in value):
                record = {
                    k: value.get(k)
                    for k in (
                        "stage",
                        "passed",
                        "plan_success",
                        "planner_id",
                        "execution_status",
                        "dynamic_safe",
                        "collision_safe",
                        "path_samples",
                        "max_tracking_error_rad",
                        "runtime_contact_policy_pass",
                        "contacts",
                        "failure",
                    )
                }
                record["json_pointer"] = path
                collision = value.get("collision_validation", {})
                if isinstance(collision, dict):
                    hits = collision.get("collisions", [])
                    record["collision_waypoint_indices"] = [h.get("waypoint_index") for h in hits]
                    record["collision_sample_count"] = collision.get("waypoints_checked")
                    record["environment_hit_pairs"] = sorted(
                        {
                            (
                                str(h.get("link_name1")),
                                str(h.get("object_name2")),
                                str(h.get("link_name2")),
                            )
                            for hit in hits
                            for h in hit.get("environment_hits", [])
                        }
                    )
                if value.get("execution_status") == "completed":
                    try:
                        q = finite_vector(value.get("qpos_after_rad"), 6)
                        margins = [
                            min(v - lo, hi - v)
                            for v, lo, hi in zip(
                                q, CANONICAL_LOWER_RAD, CANONICAL_UPPER_RAD, strict=True
                            )
                        ]
                        record.update(
                            qpos_after_rad=q,
                            pinned_urdf_joint_margins_rad=margins,
                            minimum_joint_margin_rad=min(margins),
                            terminal_bounds_recomputed=all(m >= 0 for m in margins),
                        )
                    except (ValueError, TypeError) as exc:
                        record["terminal_data_error"] = str(exc)
                records.append(record)
            for k, child in value.items():
                # Exclude enormous velocity, raw contacts and trace arrays;
                # nested stage dictionaries are retained.
                if k not in (
                    "path_metadata",
                    "runtime_motion_frame_trace",
                    "runtime_contact_trace",
                    "candidate_evaluations",
                ):
                    walk(child, f"{path}/{k}")

    walk(data, "")
    return records


def audit_directory(root: Path) -> dict[str, Any]:
    hashes = verify_checksum_inventory(root)
    result = {
        "path": str(root.resolve()),
        "checksum_file_count": len(hashes),
        "checksum_inventory_sha256": digest(root / "SHA256SUMS.txt"),
        "records": [],
    }
    for key in ("git-head.txt", "robotwin-upstream-pin.txt", "classification.txt"):
        if (root / key).exists():
            if key not in hashes:
                raise RecoveryContractError(f"unbound metadata: {key}")
            result[key] = (root / key).read_text(encoding="utf-8").strip()
    for name, sha in hashes.items():
        if not name.endswith(".json"):
            continue
        data = json.loads((root / name).read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("schema") != "panthera-native-project-manipulation-v1"
        ):
            continue
        audit = audit_project_manipulation_probe(root / name)
        result["records"].append(
            {
                "file": name,
                "sha256": sha,
                "seed": data.get("seed"),
                "task_config": data.get("task_config"),
                "reported_passed": data.get("passed"),
                "independent_historical_audit_passed": audit.passed,
                "independent_failures": [
                    {"gate": g.name, "detail": g.detail} for g in audit.failures
                ],
                "selected_candidate_id": data.get("selected_candidate_id"),
                "task_success_after_release": data.get("task_success_after_release"),
                "recovery_contract_version": data.get("recovery_contract_version"),
                "motion_records": summarize_motion_records(data),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, action="append", required=True)
    parser.add_argument("--canonical-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing historical audit output overwrite")
    if digest(args.canonical_urdf) != CANONICAL_URDF_SHA256:
        raise RecoveryContractError("canonical URDF differs from pinned LFS payload")
    record = {
        "schema": "s1m.final-historical-audit.v1",
        "urdf_sha256": CANONICAL_URDF_SHA256,
        "directories": [],
        "new_native_calls": 0,
        "fixed_3seed_acceptance_granted": False,
    }
    for directory in args.evidence_dir:
        record["directories"].append(audit_directory(directory))
    write_new_json(args.output, record)
    count = sum(len(d["records"]) for d in record["directories"])
    print(
        json.dumps(
            {
                "directories_verified": len(record["directories"]),
                "historical_records_audited": count,
                "new_native_calls": 0,
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
