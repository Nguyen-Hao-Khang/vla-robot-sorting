"""Offline, fail-closed audit contracts for S1-M approach planning.

This module distinguishes verified native plan evidence from geometry-only
candidate data and synthetic tests. It never approves execution or contact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_SOURCE_GIT_SHA = "7ac99081a7636e3fab0c8c0cb947da3310f0fe33"
EXPECTED_ROBOTWIN_GIT_SHA = "2aee31ad75cb7b0cd779540441c6025853a5b60d"
EXPECTED_CANDIDATE_ID = "pillbottle:cyl:side:3:s-1"
EXPECTED_TARGET_LINK = "080_pillbottle_23"
EXPECTED_ROBOT_LINK = "gripper_R_link"
EXPECTED_CONTACT_POLICY = {
    "allow_robot_table_contact": False,
    "allow_robot_target_contact": False,
    "allow_target_pad_contact": False,
    "allow_target_table_contact": True,
}
APPROACH_DIAGNOSTIC_SCHEMA = "s1m.approach-plan-only-diagnostic.v2"
EXPECTED_SEED2_EVIDENCE_SHA256 = "214cd667cb48adb22010734516f96ea7b9468f736e6b20feb9a23b718f9eee71"


class ApproachAuditError(ValueError):
    """Seed2 evidence or a diagnostic trial violates its audit contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(root: Path, name: str) -> str:
    try:
        return (root / name).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ApproachAuditError(f"missing evidence file: {name}") from exc


def verify_checksum_inventory(evidence_dir: str | Path) -> dict[str, str]:
    """Verify each SHA256SUMS entry without following paths outside the bundle."""
    root = Path(evidence_dir).resolve()
    sums_path = root / "SHA256SUMS.txt"
    try:
        lines = sums_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ApproachAuditError("missing SHA256SUMS.txt") from exc
    if not lines:
        raise ApproachAuditError("checksum inventory is empty")

    verified: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ApproachAuditError(f"malformed checksum line {line_number}")
        expected, encoded_path = match.groups()
        if not encoded_path.startswith("./"):
            raise ApproachAuditError(f"checksum path must be relative at line {line_number}")
        relative = encoded_path[2:]
        posix = PurePosixPath(relative)
        if (
            not relative
            or "\\" in relative
            or posix.is_absolute()
            or any(part in ("", ".", "..") for part in posix.parts)
        ):
            raise ApproachAuditError(f"unsafe checksum path at line {line_number}")
        key = posix.as_posix()
        if key in verified:
            raise ApproachAuditError(f"duplicate checksum path: {key}")
        candidate = (root / Path(*posix.parts)).resolve()
        if root not in candidate.parents:
            raise ApproachAuditError(f"checksum path escapes evidence directory: {key}")
        if not candidate.is_file():
            raise ApproachAuditError(f"checksum target is missing: {key}")
        actual = _sha256_file(candidate)
        if actual != expected:
            raise ApproachAuditError(f"checksum mismatch: {key}")
        verified[key] = actual
    return verified


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ApproachAuditError(f"{label} must contain {length} values")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ApproachAuditError(f"{label} must be numeric") from exc
    if not all(math.isfinite(item) for item in vector):
        raise ApproachAuditError(f"{label} must be finite")
    return vector


def _candidate_geometry(candidate: dict[str, Any], status: str, rank: int | None) -> dict[str, Any]:
    candidate_id = candidate.get("candidate_id")
    grasp = _finite_vector(candidate.get("grasp_position_xyz"), 3, "grasp_position_xyz")
    pregrasp = _finite_vector(candidate.get("pregrasp_position_xyz"), 3, "pregrasp_position_xyz")
    direction = _finite_vector(candidate.get("approach_direction"), 3, "approach_direction")
    displacement = tuple(grasp[i] - pregrasp[i] for i in range(3))
    standoff = math.sqrt(sum(value * value for value in displacement))
    direction_norm = math.sqrt(sum(value * value for value in direction))
    if standoff <= 0 or direction_norm <= 0:
        raise ApproachAuditError(f"candidate geometry is degenerate: {candidate_id}")
    alignment = sum(displacement[i] * direction[i] for i in range(3)) / (standoff * direction_norm)
    metadata = candidate.get("metadata")
    if not isinstance(metadata, dict):
        raise ApproachAuditError(f"candidate metadata missing: {candidate_id}")
    orientation = _finite_vector(candidate.get("grasp_orientation_xyzw"), 4, "grasp orientation")
    pre_orientation = _finite_vector(
        candidate.get("pregrasp_orientation_xyzw"), 4, "pregrasp orientation"
    )
    return {
        "candidate_id": candidate_id,
        "geometry_source": candidate.get("geometry_source"),
        "grasp_position_xyz_m": grasp,
        "pregrasp_position_xyz_m": pregrasp,
        "approach_direction": direction,
        "standoff_m": standoff,
        "approach_alignment_cosine": alignment,
        "grasp_height_fraction": metadata.get("grasp_height_fraction"),
        "table_clearance_m": metadata.get("table_clearance_m"),
        "jaw_width_m": candidate.get("jaw_width_m"),
        "orientation_unchanged_at_pregrasp": all(
            math.isclose(a, b, abs_tol=1e-9)
            for a, b in zip(orientation, pre_orientation, strict=True)
        ),
        "b5_pregrasp_rank_1_based": rank,
        "native_path_status": status,
        "controller_eligible": False,
    }


def _stage_collision_data(stage: dict[str, Any]) -> dict[str, Any]:
    validation = stage.get("collision_validation")
    if not isinstance(validation, dict):
        raise ApproachAuditError(f"{stage.get('stage')} collision validation missing")
    collision_items = validation.get("collisions")
    if not isinstance(collision_items, list):
        raise ApproachAuditError("collision records must be a list")
    waypoint_count = stage.get("path_samples")
    checked = validation.get("waypoints_checked")
    if type(waypoint_count) is not int or waypoint_count < 1:
        raise ApproachAuditError(f"{stage.get('stage')} path sample count missing")
    if type(checked) is not int or checked != waypoint_count:
        raise ApproachAuditError(f"{stage.get('stage')} waypoint audit is incomplete")

    indices: list[int] = []
    pairs: set[tuple[str, str]] = set()
    for item in collision_items:
        if not isinstance(item, dict) or type(item.get("waypoint_index")) is not int:
            raise ApproachAuditError("malformed waypoint collision record")
        index = item["waypoint_index"]
        if not 0 <= index < waypoint_count:
            raise ApproachAuditError("collision waypoint is out of range")
        indices.append(index)
        roles = item.get("matched_environment_roles")
        if roles != ["robot_link", "target"]:
            raise ApproachAuditError("collision roles do not match robot-target evidence")
        hits = item.get("environment_hits")
        if not isinstance(hits, list) or not hits:
            raise ApproachAuditError("collision record lacks environment hits")
        for hit in hits:
            attributes = hit.get("attributes") if isinstance(hit, dict) else None
            if not isinstance(attributes, dict):
                raise ApproachAuditError("collision hit attributes missing")
            pairs.add((str(attributes.get("link_name1")), str(attributes.get("link_name2"))))
    if len(indices) != len(set(indices)):
        raise ApproachAuditError("duplicate collision waypoint records")
    ordered = sorted(indices)
    if ordered and ordered != list(range(ordered[0], ordered[-1] + 1)):
        raise ApproachAuditError("recorded collision indices are not contiguous")
    fraction = len(indices) / waypoint_count
    return {
        "waypoint_count": waypoint_count,
        "waypoints_checked": checked,
        "collision_count": len(indices),
        "collision_waypoint_fraction": fraction,
        "collision_span_indices": [ordered[0], ordered[-1]] if ordered else None,
        "collision_pairs": [{"robot_link": pair[0], "target": pair[1]} for pair in sorted(pairs)],
        "collision_free": validation.get("collision_free") is True and not indices,
    }


def audit_seed2_approach_failure(evidence_dir: str | Path) -> dict[str, Any]:
    """Verify and summarize the immutable seed2 alternative diagnostic export."""
    root = Path(evidence_dir).resolve()
    checksums = verify_checksum_inventory(root)
    if "seed2-alternative.json" not in checksums:
        raise ApproachAuditError("seed2-alternative.json is not checksum-bound")
    source_git = _read_text(root, "git-head.txt")
    robotwin_git = _read_text(root, "robotwin-upstream-pin.txt")
    if source_git != EXPECTED_SOURCE_GIT_SHA:
        raise ApproachAuditError("unexpected source Git SHA")
    if robotwin_git != EXPECTED_ROBOTWIN_GIT_SHA:
        raise ApproachAuditError("unexpected RoboTwin upstream SHA")
    if _read_text(root, "classification.txt") != "DIAGNOSTIC_ONLY_NOT_ACCEPTED":
        raise ApproachAuditError("native evidence classification is not diagnostic-only")

    try:
        payload = json.loads((root / "seed2-alternative.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApproachAuditError("seed2 JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ApproachAuditError("seed2 root must be an object")
    if payload.get("schema") != "panthera-native-project-manipulation-v1":
        raise ApproachAuditError("unexpected native manipulation schema")
    if payload.get("seed") != 2 or payload.get("passed") is not False:
        raise ApproachAuditError("record is not the failed seed2 diagnostic")
    if payload.get("final_phase") != "recovery_required":
        raise ApproachAuditError("unexpected final phase")

    b5 = payload.get("B5")
    selected = b5.get("selected") if isinstance(b5, dict) else None
    diagnostic = payload.get("candidate_selection_diagnostic")
    if not isinstance(selected, dict) or not isinstance(diagnostic, dict):
        raise ApproachAuditError("selected B5 candidate provenance missing")
    if (
        selected.get("candidate_id") != EXPECTED_CANDIDATE_ID
        or diagnostic.get("candidate_id_override") != EXPECTED_CANDIDATE_ID
        or selected.get("geometry_source") != "sim_ground_truth"
    ):
        raise ApproachAuditError("selected candidate/source does not match seed2 evidence")
    if selected.get("object_id") != "pillbottle":
        raise ApproachAuditError("selected grasp does not target the expected object")

    stages = payload.get("stages")
    if not isinstance(stages, list) or len(stages) != 2:
        raise ApproachAuditError("expected exactly pregrasp and approach stages")
    pregrasp, approach = stages
    if pregrasp.get("stage") != "pregrasp" or approach.get("stage") != "approach":
        raise ApproachAuditError("pregrasp/approach stage order is invalid")
    pre = _stage_collision_data(pregrasp)
    app = _stage_collision_data(approach)
    if (
        pregrasp.get("passed") is not True
        or pregrasp.get("plan_success") is not True
        or pregrasp.get("dynamic_safe") is not True
        or pregrasp.get("collision_safe") is not True
        or pre["collision_free"] is not True
        or pregrasp.get("execution_status") != "completed"
    ):
        raise ApproachAuditError("recorded pregrasp does not satisfy its PASS claim")
    if (
        approach.get("plan_success") is not True
        or approach.get("dynamic_safe") is not True
        or approach.get("collision_safe") is not False
        or app["collision_free"] is not False
        or not app["collision_count"]
    ):
        raise ApproachAuditError("approach failure gates are inconsistent")
    if app["collision_pairs"] != [
        {"robot_link": EXPECTED_ROBOT_LINK, "target": EXPECTED_TARGET_LINK}
    ]:
        raise ApproachAuditError("unexpected robot-target collision pair")
    if approach.get("execution_status") not in (None, "not_started_due_collision_veto"):
        raise ApproachAuditError("approach execution was unexpectedly recorded")
    if payload.get("failure_phase") not in (None, "approach") and (
        "approach" not in str(payload.get("error", "")).lower()
    ):
        raise ApproachAuditError("failure did not occur at approach")
    if app["waypoint_count"] != 311 or app["collision_count"] != 59:
        raise ApproachAuditError("seed2 collision counts differ from the audited incident")
    if app["collision_span_indices"] != [222, 280]:
        raise ApproachAuditError("seed2 collision span differs from the audited incident")

    candidates = b5.get("candidates")
    ranking = b5.get("ranking")
    if not isinstance(candidates, list) or not isinstance(ranking, list):
        raise ApproachAuditError("B5 candidate geometry/ranking missing")
    rank_by_id = {
        item.get("candidate_id"): index + 1
        for index, item in enumerate(ranking)
        if isinstance(item, dict)
    }
    compare_ids = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and (
            candidate.get("candidate_id") == EXPECTED_CANDIDATE_ID
            or candidate.get("candidate_id", "").startswith(
                ("pillbottle:cyl:side:2:", "pillbottle:cyl:side:3:")
            )
        )
    ]
    if not compare_ids:
        raise ApproachAuditError("no recorded side-2/side-3 geometry comparisons")
    geometry_rows = []
    seen: set[str] = set()
    for candidate in compare_ids:
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in seen:
            raise ApproachAuditError("candidate IDs are missing or duplicated")
        seen.add(candidate_id)
        status = "COLLISION_REJECTED" if candidate_id == EXPECTED_CANDIDATE_ID else "NOT_EVALUATED"
        geometry_rows.append(_candidate_geometry(candidate, status, rank_by_id.get(candidate_id)))
    geometry_rows.sort(
        key=lambda row: (
            row["candidate_id"] != EXPECTED_CANDIDATE_ID,
            row["candidate_id"],
        )
    )

    target_geometry = payload.get("object_geometry")
    gripper_geometry = payload.get("gripper_geometry")
    if not isinstance(target_geometry, dict) or not isinstance(gripper_geometry, dict):
        raise ApproachAuditError("target/gripper geometry provenance missing")
    dimensions = _finite_vector(target_geometry.get("size_xyz_m"), 3, "target dimensions")
    if any(value <= 0 for value in dimensions):
        raise ApproachAuditError("target dimensions must be positive")
    jaw = gripper_geometry.get("max_width_m")
    finger = gripper_geometry.get("finger_proxy_diameter_m")
    if (
        not isinstance(jaw, (int, float))
        or not math.isfinite(jaw)
        or jaw <= 0
        or not isinstance(finger, (int, float))
        or not math.isfinite(finger)
        or finger <= 0
    ):
        raise ApproachAuditError("gripper geometry dimensions are invalid")

    return {
        "schema": "s1m.seed2-approach-failure-audit.v1",
        "classification": "APPROACH_COLLISION_REJECTED_DIAGNOSTIC_ONLY",
        "source_git_sha": source_git,
        "robotwin_git_sha": robotwin_git,
        "source_sha256_inventory_count": len(checksums),
        "evidence_file_sha256": checksums["seed2-alternative.json"],
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "candidate_geometry_source": selected["geometry_source"],
        "pregrasp": {
            "passed": True,
            "waypoint_count": pre["waypoint_count"],
            "collision_free": True,
        },
        "approach": {
            **app,
            "planner_success": True,
            "dynamic_safe": True,
            "executed": False,
            "execution_status_recorded": approach.get("execution_status"),
        },
        "geometry_comparison": {
            "target_dimensions_xyz_m": dimensions,
            "jaw_max_width_m": jaw,
            "finger_proxy_diameter_m": finger,
            "candidate_count": len(geometry_rows),
            "candidate_native_scores": "NOT_AVAILABLE",
            "candidates": geometry_rows,
        },
        "approach_contact_policy_explicit": isinstance(approach.get("contact_policy"), dict),
        "evidence_gaps": [
            "approach stage does not serialize an explicit contact_policy",
            "per-waypoint joint positions and forward-kinematic link poses are absent",
            "collision records contain link/object names but no contact points or signed distances",
            "non-selected B5 geometries have no recorded MPlib approach result",
        ],
        "controller_eligible": False,
        "native_acceptance": False,
    }


def _valid_sha(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _diagnostic_stage_data(
    stage: Any, label: str, scene_revision: str
) -> tuple[list[str], list[dict[str, Any]]]:
    """Check complete, per-waypoint planner, FK, and collision evidence."""
    reasons: list[str] = []
    if not isinstance(stage, dict):
        return [f"{label}_incomplete_waypoint_validation"], []
    count = stage.get("waypoint_count")
    checked = stage.get("waypoints_checked")
    waypoints = stage.get("waypoints")
    if (
        type(count) is not int
        or count <= 0
        or type(checked) is not int
        or checked != count
        or not isinstance(waypoints, list)
        or len(waypoints) != count
    ):
        return [f"{label}_incomplete_waypoint_validation"], []
    if stage.get("plan_success") is not True or stage.get("dynamic_safe") is not True:
        reasons.append(f"{label}_planner_or_dynamic_gate")
    if stage.get("collision_free") is not True or stage.get("collisions") != []:
        reasons.append(f"{label}_collision")
    if stage.get("contact_policy") != EXPECTED_CONTACT_POLICY:
        reasons.append(f"{label}_contact_policy")

    link_names: set[str] | None = None
    scope_hashes: set[str] = set()
    distances: list[float | None] = []
    previous_time: float | None = None
    for index, sample in enumerate(waypoints):
        if not isinstance(sample, dict) or sample.get("sample_index") != index:
            reasons.append(f"{label}_malformed_waypoint_records")
            continue
        time_value = sample.get("time_from_start_s")
        qpos = sample.get("qpos_rad")
        tool_pose = sample.get("tool_pose_world_xyzw")
        if (
            not isinstance(time_value, (int, float))
            or isinstance(time_value, bool)
            or not math.isfinite(time_value)
            or (previous_time is not None and time_value <= previous_time)
            or not isinstance(qpos, list)
            or len(qpos) != 6
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in qpos
            )
            or not isinstance(tool_pose, list)
            or len(tool_pose) != 7
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in tool_pose
            )
        ):
            reasons.append(f"{label}_malformed_waypoint_records")
        if isinstance(time_value, (int, float)) and not isinstance(time_value, bool):
            previous_time = float(time_value)

        fk = sample.get("fk_link_poses_world")
        if not isinstance(fk, dict) or not fk:
            reasons.append(f"{label}_missing_fk_or_collision_evidence")
        else:
            current_links = set(fk)
            if any(not isinstance(name, str) or not name.strip() for name in current_links):
                reasons.append(f"{label}_missing_fk_or_collision_evidence")
            if link_names is None:
                link_names = current_links
            elif current_links != link_names:
                reasons.append(f"{label}_inconsistent_fk_link_scope")
            for pose in fk.values():
                if (
                    not isinstance(pose, list)
                    or len(pose) != 7
                    or any(
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(value)
                        for value in pose
                    )
                ):
                    reasons.append(f"{label}_missing_fk_or_collision_evidence")

        check = sample.get("collision_check")
        if not isinstance(check, dict):
            reasons.append(f"{label}_missing_fk_or_collision_evidence")
            distances.append(None)
            continue
        scope_hash = check.get("scope_sha256")
        collision_pairs = check.get("collision_pairs")
        pair_count = check.get("checked_pair_count")
        if _valid_sha(scope_hash, 64):
            scope_hashes.add(scope_hash)
        if (
            check.get("checked") is not True
            or check.get("collision_free") is not True
            or collision_pairs != []
            or type(pair_count) is not int
            or pair_count <= 0
            or check.get("scene_revision") != scene_revision
            or not _valid_sha(scope_hash, 64)
        ):
            reasons.append(f"{label}_waypoint_collision_or_scope")
        distance = check.get("minimum_signed_clearance_m")
        if distance is None:
            distances.append(None)
        elif (
            not isinstance(distance, (int, float))
            or isinstance(distance, bool)
            or not math.isfinite(distance)
        ):
            reasons.append(f"{label}_invalid_signed_clearance")
            distances.append(None)
        else:
            value = float(distance)
            distances.append(value)
            if value <= 0:
                reasons.append(f"{label}_waypoint_collision_or_scope")
    if len(scope_hashes) != 1:
        reasons.append(f"{label}_inconsistent_collision_scope")
    if (
        distances
        and any(value is None for value in distances)
        and any(value is not None for value in distances)
    ):
        reasons.append(f"{label}_partial_clearance_evidence")
    return reasons, waypoints


def _trajectory_metrics(stages: list[dict[str, Any]]) -> tuple[float, float]:
    joint_path_length = 0.0
    duration = 0.0
    for stage in stages:
        waypoints = stage.get("waypoints") if isinstance(stage, dict) else None
        if not isinstance(waypoints, list) or not waypoints:
            continue
        try:
            for before, after in pairwise(waypoints):
                joint_path_length += sum(
                    abs(float(right) - float(left))
                    for left, right in zip(before["qpos_rad"], after["qpos_rad"], strict=True)
                )
            duration += float(waypoints[-1]["time_from_start_s"]) - float(
                waypoints[0]["time_from_start_s"]
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            # The stage validator will reject malformed records; scoring must not crash.
            return math.nan, math.nan
    return joint_path_length, duration


def validate_plan_only_trial(record: Any) -> dict[str, Any]:
    """Validate a future native plan-only trial; never authorizes motion."""
    reasons: list[str] = []
    if not isinstance(record, dict):
        return {
            "eligible_for_plan_only_review": False,
            "rejection_reasons": ["record_not_object"],
            "controller_eligible": False,
            "native_acceptance": False,
        }
    if record.get("schema") != APPROACH_DIAGNOSTIC_SCHEMA:
        reasons.append("schema")
    if record.get("classification") != "PLAN_ONLY_APPROACH_DIAGNOSTIC_NOT_ACCEPTANCE":
        reasons.append("classification")

    provenance = record.get("provenance")
    required = {
        "project_git_sha": 40,
        "robotwin_git_sha": 40,
        "source_evidence_sha256": 64,
        "baseline_evidence_sha256": 64,
        "start_state_sha256": 64,
        "geometry_sha256": 64,
        "canonical_placement_goal_sha256": 64,
    }
    if not isinstance(provenance, dict):
        reasons.append("provenance")
        provenance = {}
    for key, length in required.items():
        if not _valid_sha(provenance.get(key), length):
            reasons.append(f"provenance_{key}")
    for key in ("candidate_id", "scene_revision", "transform_revision"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            reasons.append(f"provenance_{key}")

    stages = record.get("stages")
    if not isinstance(stages, dict):
        reasons.append("stages")
        stages = {}
    pregrasp = stages.get("pregrasp")
    approach = stages.get("approach")
    for label, stage in (("pregrasp", pregrasp), ("approach", approach)):
        if not isinstance(stage, dict):
            reasons.append(f"{label}_incomplete_waypoint_validation")
            continue
        stage_reasons, _ = _diagnostic_stage_data(
            stage, label, provenance.get("scene_revision", "")
        )
        reasons.extend(stage_reasons)
    if not isinstance(pregrasp, dict):
        pregrasp = {}
    if not isinstance(approach, dict):
        approach = {}

    if pregrasp.get("passed") is not True:
        reasons.append("pregrasp_gate")
    if approach.get("planner_id") != "robotwin-mplib-screw":
        reasons.append("planner_id")
    for stage in (pregrasp, approach):
        if stage.get("transform_revision") != provenance.get("transform_revision"):
            reasons.append("stale_scene_or_transform")
        if stage.get("scene_revision") != provenance.get("scene_revision"):
            reasons.append("stale_scene_or_transform")
        if stage.get("execution_status") != "not_executed":
            reasons.append("approach_was_executed_or_unrecorded")

    runtime = record.get("runtime")
    if not isinstance(runtime, dict):
        reasons.append("runtime")
        runtime = {}
    if runtime.get("execution_performed") is not False:
        reasons.append("execution_performed")
    if type(runtime.get("physics_steps")) is not int or runtime.get("physics_steps") != 0:
        reasons.append("physics_steps")
    if runtime.get("policy_changed") is not False:
        reasons.append("policy_changed")
    if runtime.get("canonical_placement_changed") is not False:
        reasons.append("canonical_placement_changed")

    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        reasons.append("metrics")
        metrics = {}
    valid_stages = [stage for stage in (pregrasp, approach) if isinstance(stage, dict)]
    measured_clearances = [
        sample["collision_check"]["minimum_signed_clearance_m"]
        for stage in valid_stages
        for sample in stage.get("waypoints", [])
        if isinstance(sample, dict)
        and isinstance(sample.get("collision_check"), dict)
        and isinstance(sample["collision_check"].get("minimum_signed_clearance_m"), (int, float))
        and not isinstance(sample["collision_check"].get("minimum_signed_clearance_m"), bool)
        and math.isfinite(sample["collision_check"]["minimum_signed_clearance_m"])
    ]
    distances_missing = any(
        sample.get("collision_check", {}).get("minimum_signed_clearance_m") is None
        for stage in valid_stages
        for sample in stage.get("waypoints", [])
        if isinstance(sample, dict) and isinstance(sample.get("collision_check"), dict)
    )
    metric_clearance = metrics.get("minimum_clearance_m")
    clearance_source = metrics.get("minimum_clearance_source")
    if measured_clearances:
        if (
            distances_missing
            or not isinstance(metric_clearance, (int, float))
            or isinstance(metric_clearance, bool)
            or not math.isfinite(metric_clearance)
            or not math.isclose(
                float(metric_clearance),
                min(measured_clearances),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            reasons.append("clearance_summary_mismatch")
        clearance_provenance_invalid = not isinstance(clearance_source, dict) or (
            not isinstance(clearance_source.get("method_id"), str)
            or not clearance_source["method_id"].strip()
            or not isinstance(clearance_source.get("engine"), str)
            or not clearance_source["engine"].strip()
            or not _valid_sha(clearance_source.get("implementation_sha256"), 64)
            or clearance_source.get("unit") != "m"
            or clearance_source.get("sign_convention") != "positive_means_separated"
            or not _valid_sha(clearance_source.get("scope_sha256"), 64)
            or not isinstance(clearance_source.get("implementation_path"), str)
            or not clearance_source["implementation_path"].strip()
            or "\\" in clearance_source["implementation_path"]
            or PurePosixPath(clearance_source["implementation_path"]).is_absolute()
            or any(
                part in ("", ".", "..")
                for part in PurePosixPath(clearance_source["implementation_path"]).parts
            )
        )
        if clearance_provenance_invalid or (
            isinstance(clearance_source, dict)
            and any(
                sample["collision_check"].get("scope_sha256")
                != clearance_source.get("scope_sha256")
                for stage in valid_stages
                for sample in stage.get("waypoints", [])
                if isinstance(sample, dict) and isinstance(sample.get("collision_check"), dict)
            )
        ):
            reasons.append("clearance_provenance")
    elif metric_clearance is not None or clearance_source is not None:
        reasons.append("clearance_summary_mismatch")

    path_length, duration = _trajectory_metrics(valid_stages)
    for key, derived in (
        ("joint_path_length_rad", path_length),
        ("trajectory_duration_s", duration),
    ):
        supplied = metrics.get(key)
        if supplied is not None and (
            not isinstance(supplied, (int, float))
            or isinstance(supplied, bool)
            or not math.isfinite(supplied)
            or not math.isclose(float(supplied), derived, rel_tol=0.0, abs_tol=1e-9)
        ):
            reasons.append(f"{key}_summary_mismatch")

    return {
        "eligible_for_plan_only_review": not reasons,
        "rejection_reasons": sorted(set(reasons)),
        "evidence_grade": (
            "SAMPLED_COLLISION_FREE_CLEARANCE_UNVERIFIED"
            if not measured_clearances
            else "SAMPLED_COLLISION_FREE_WITH_CLEARANCE_EVIDENCE_PLAN_ONLY"
        ),
        "controller_eligible": False,
        "native_acceptance": False,
    }


def validate_plan_only_trial_set(records: Any) -> dict[str, Any]:
    """Require compared candidate trials to share task, source, scene, and start state."""
    if not isinstance(records, list) or len(records) < 2:
        return {
            "eligible_for_comparison": False,
            "rejection_reasons": ["at_least_two_trials_required"],
            "controller_eligible": False,
            "native_acceptance": False,
        }
    reasons: list[str] = []
    validations = [validate_plan_only_trial(record) for record in records]
    if any(not item["eligible_for_plan_only_review"] for item in validations):
        reasons.append("trial_set_contains_rejected_trial")
    shared_fields = (
        "project_git_sha",
        "robotwin_git_sha",
        "source_evidence_sha256",
        "baseline_evidence_sha256",
        "start_state_sha256",
        "scene_revision",
        "transform_revision",
        "canonical_placement_goal_sha256",
    )
    provenances = [
        record.get("provenance") if isinstance(record, dict) else None for record in records
    ]
    if any(not isinstance(item, dict) for item in provenances):
        reasons.append("trial_set_provenance_missing")
    else:
        for field in shared_fields:
            values = [item.get(field) for item in provenances if isinstance(item, dict)]
            if values and any(value != values[0] for value in values[1:]):
                reasons.append(f"trial_set_mismatched_{field}")
    candidate_ids = [item.get("candidate_id") for item in provenances if isinstance(item, dict)]
    if any(not isinstance(item, str) or not item for item in candidate_ids):
        reasons.append("trial_set_candidate_id_missing")
    elif len(candidate_ids) != len(set(candidate_ids)):
        reasons.append("trial_set_candidate_id_duplicate")
    return {
        "eligible_for_comparison": not reasons,
        "rejection_reasons": sorted(set(reasons)),
        "trial_count": len(records),
        "candidate_ids": candidate_ids,
        "canonical_placement_goal_sha256": (
            provenances[0].get("canonical_placement_goal_sha256")
            if provenances and isinstance(provenances[0], dict)
            else None
        ),
        "controller_eligible": False,
        "native_acceptance": False,
    }


def audit_plan_only_trial_bundle(evidence_dir: str | Path) -> dict[str, Any]:
    """Checksum-audit a future diagnostic export without treating it as acceptance."""
    root = Path(evidence_dir).resolve()
    checksums = verify_checksum_inventory(root)
    required_files = {
        "plan-only-trial.json",
        "git-head.txt",
        "robotwin-upstream-pin.txt",
        "classification.txt",
        "baseline-evidence-sha256.txt",
    }
    if not required_files.issubset(checksums):
        raise ApproachAuditError("plan-only bundle is missing required checksum-bound files")
    if _read_text(root, "classification.txt") != "PLAN_ONLY_DIAGNOSTIC_NOT_ACCEPTED":
        raise ApproachAuditError("plan-only export classification is not diagnostic-only")
    if _read_text(root, "baseline-evidence-sha256.txt") != EXPECTED_SEED2_EVIDENCE_SHA256:
        raise ApproachAuditError("baseline evidence SHA record does not match audited source")
    project_sha = _read_text(root, "git-head.txt")
    robotwin_sha = _read_text(root, "robotwin-upstream-pin.txt")
    if not _valid_sha(project_sha, 40) or not _valid_sha(robotwin_sha, 40):
        raise ApproachAuditError("plan-only bundle source revision is malformed")
    try:
        record = json.loads((root / "plan-only-trial.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApproachAuditError("plan-only trial JSON is invalid") from exc
    if not isinstance(record, dict):
        raise ApproachAuditError("plan-only trial root must be an object")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ApproachAuditError("plan-only trial provenance is missing")
    if provenance.get("project_git_sha") != project_sha:
        raise ApproachAuditError("project Git SHA does not match bundle lock")
    if provenance.get("robotwin_git_sha") != robotwin_sha:
        raise ApproachAuditError("RoboTwin SHA does not match bundle lock")
    if provenance.get("baseline_evidence_sha256") != EXPECTED_SEED2_EVIDENCE_SHA256:
        raise ApproachAuditError("baseline seed2 evidence SHA does not match audited source")
    for field in ("source_evidence_sha256", "start_state_sha256", "geometry_sha256"):
        if provenance.get(field) not in set(checksums.values()):
            raise ApproachAuditError(f"{field} is not present in the checksum inventory")

    validation = validate_plan_only_trial(record)
    if not validation["eligible_for_plan_only_review"]:
        return {
            "schema": "s1m.approach-plan-only-bundle-audit.v1",
            "classification": "PLAN_ONLY_DIAGNOSTIC_REJECTED",
            "source_sha256_inventory_count": len(checksums),
            "trial_sha256": checksums["plan-only-trial.json"],
            "trial_validation": validation,
            "controller_eligible": False,
            "native_acceptance": False,
        }

    metrics = record["metrics"]
    source = metrics.get("minimum_clearance_source")
    if source is not None:
        implementation_path = source["implementation_path"]
        if implementation_path not in checksums:
            raise ApproachAuditError("clearance implementation is not checksum-bound")
        if checksums[implementation_path] != source.get("implementation_sha256"):
            raise ApproachAuditError("clearance implementation hash mismatch")
    return {
        "schema": "s1m.approach-plan-only-bundle-audit.v1",
        "classification": "PLAN_ONLY_DIAGNOSTIC_STRUCTURALLY_VALID_NOT_ACCEPTANCE",
        "source_git_sha": project_sha,
        "robotwin_git_sha": robotwin_sha,
        "baseline_evidence_sha256": EXPECTED_SEED2_EVIDENCE_SHA256,
        "source_sha256_inventory_count": len(checksums),
        "trial_sha256": checksums["plan-only-trial.json"],
        "trial_validation": validation,
        "controller_eligible": False,
        "native_acceptance": False,
    }


def score_plan_only_trial(record: Any) -> dict[str, Any]:
    """Apply hard safety gates and score only values derived from waypoint records."""
    validation = validate_plan_only_trial(record)
    base = {
        **validation,
        "score_kind": "lexicographic_minimize",
        "controller_eligible": False,
        "native_acceptance": False,
    }
    if not validation["eligible_for_plan_only_review"]:
        return {**base, "status": "REJECTED_NO_SCORE", "score_vector": None}
    stages = record["stages"]
    validated_stages = [stages["pregrasp"], stages["approach"]]
    waypoints = [sample for stage in validated_stages for sample in stage["waypoints"]]
    clearances = [
        sample["collision_check"]["minimum_signed_clearance_m"]
        for sample in waypoints
        if sample["collision_check"].get("minimum_signed_clearance_m") is not None
    ]
    if not clearances:
        return {
            **base,
            "status": "UNSCORED_CLEARANCE_MISSING",
            "score_vector": None,
        }
    clearance = min(float(value) for value in clearances)
    if clearance <= 0:
        return {
            **base,
            "status": "REJECTED_NO_SCORE",
            "score_vector": None,
            "rejection_reasons": sorted(
                {*validation["rejection_reasons"], "nonpositive_clearance"}
            ),
        }
    path_length, duration = _trajectory_metrics(validated_stages)
    source = record["metrics"]["minimum_clearance_source"]
    score_vector = [
        -clearance,
        path_length,
        duration,
        record["provenance"]["candidate_id"],
    ]
    return {
        **base,
        "status": "PLAN_ONLY_DIAGNOSTIC_SCORED",
        "score_vector": score_vector,
        "score_metric_provenance": {
            "minimum_clearance_m": source,
            "joint_path_length_rad": "recomputed_sum_absolute_joint_delta_rad",
            "trajectory_duration_s": "recomputed_from_native_waypoint_timestamps",
        },
    }
