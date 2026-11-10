#!/usr/bin/env python3
"""Read-only comparison of independently exported native seed-0 diagnostics.

Reports observations and provenance, never grants project/campaign acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from math import dist, isfinite
from pathlib import Path, PurePosixPath
from typing import Any

from vla_robot_sorting.evidence import audit_project_manipulation_probe

_SHA_LINE = re.compile(r"([0-9a-f]{64})  (.+)")


def verify_export(folder: Path) -> int:
    manifest = folder / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise ValueError(f"Missing native export checksum manifest: {folder}")
    seen: set[str] = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        match = _SHA_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"Malformed SHA256SUMS entry in {folder}")
        expected, relative = match.groups()
        relative = relative.removeprefix("./")
        parts = PurePosixPath(relative).parts
        if (
            not parts or ".." in parts or relative.startswith("/")
            or "\\" in relative or relative in seen
            or relative == "SHA256SUMS.txt"
        ):
            raise ValueError(f"Unsafe or duplicate checksum filename: {relative}")
        seen.add(relative)
        source = folder / relative
        if not source.is_file():
            raise ValueError(f"Missing native evidence file: {source}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Native evidence checksum mismatch: {source}")
    if not seen or "seed0.json" not in seen or "git-head.txt" not in seen:
        raise ValueError(f"Incomplete evidence checksums: {folder}")
    return len(seen)


def _stage(payload: dict[str, Any], name: str) -> dict[str, Any]:
    return next(
        (row for row in payload.get("stages", [])
         if isinstance(row, dict) and row.get("stage") == name),
        {},
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _vector(frame: dict[str, Any], key: str, n: int) -> tuple[float, ...] | None:
    value = frame.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != n:
        return None
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return values if all(isfinite(v) for v in values) else None


def _trace_quality(retreat: dict[str, Any]) -> dict[str, Any]:
    record = _dict(retreat.get("runtime_motion_frame_trace"))
    if not record:
        return {"status": "not_recorded", "frame_count": 0}
    frames = record.get("frames")
    counts = _dict(retreat.get("contacts"))
    if record.get("schema") != "retreat-kinematic-frames-v1" or not isinstance(frames, list):
        return {"status": "invalid", "reason": "schema_or_frames_invalid"}
    if record.get("truncated") is True:
        return {"status": "partial", "frame_count": len(frames)}
    steps = counts.get("physics_steps")
    flags = [
        frame.get("robot_target_contact") for frame in frames
        if isinstance(frame, dict)
    ]
    if (
        not isinstance(steps, int) or len(frames) != steps
        or len(flags) != len(frames) or any(type(flag) is not bool for flag in flags)
        or sum(flags) != counts.get("robot_target_contact_frames")
        or any(frame.get("physics_step") != index
               for index, frame in enumerate(frames, start=1))
    ):
        return {"status": "invalid", "reason": "frame_step_or_contact_count_mismatch"}
    return {"status": "complete", "frame_count": len(frames)}


def inspect_run(folder: Path) -> dict[str, Any]:
    checked = verify_export(folder)
    payload = json.loads((folder / "seed0.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("seed") != 0:
        raise ValueError(f"Not a seed0 native probe: {folder}")
    head = (folder / "git-head.txt").read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError(f"Missing or invalid Linux Git HEAD: {folder}")
    classification = (folder / "classification.txt").read_text(
        encoding="utf-8"
    ).strip()
    if classification != "DIAGNOSTIC_ONLY_NOT_ACCEPTED":
        raise ValueError(f"Unexpected evidence classification: {folder}")
    audit = audit_project_manipulation_probe(folder / "seed0.json")
    retreat = _stage(payload, "retreat")
    contact = _dict(retreat.get("contacts"))
    motion = _dict(retreat.get("runtime_motion_frame_trace"))
    return {
        "name": folder.name,
        "folder": str(folder),
        "git_head": head,
        "checksums_verified": checked,
        "classification": classification,
        "probe_passed": payload.get("passed") is True,
        "audit_passed": audit.passed,
        "audit_failures": [gate.name for gate in audit.failures],
        "retreat_passed": retreat.get("passed") is True,
        "retreat_contacts": contact.get("robot_target_contact_frames"),
        "retreat_contact_physics_steps": contact.get("physics_steps"),
        "retreat_first_contact_step": contact.get("first_robot_target_step"),
        "retreat_last_contact_step": contact.get("last_robot_target_step"),
        "retreat_planner": retreat.get("planner_id"),
        "retreat_geometry": _dict(retreat.get("retreat_geometry")),
        "initial_object_pose": payload.get("initial_object_pose"),
        "initial_pad_pose": payload.get("initial_pad_pose"),
        "selected_grasp_id": _dict(_dict(payload.get("B5")).get("selected")).get(
            "candidate_id"
        ),
        "trace_quality": _trace_quality(retreat),
        "motion_frames": motion.get("frames", []) if isinstance(
            motion.get("frames", []), list
        ) else [],
    }


def compare_pair(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    left = {
        f["physics_step"]: f for f in first["motion_frames"]
        if isinstance(f, dict) and isinstance(f.get("physics_step"), int)
    }
    right = {
        f["physics_step"]: f for f in second["motion_frames"]
        if isinstance(f, dict) and isinstance(f.get("physics_step"), int)
    }
    steps = sorted(left.keys() & right.keys())
    joint_deltas: list[float] = []
    ee_deltas: list[float] = []
    target_deltas: list[float] = []
    finger_deltas: dict[str, list[float]] = {
        "gripper_L_link": [], "gripper_R_link": []
    }
    first_contact_difference = None
    for step in steps:
        a, b = left[step], right[step]
        if (
            first_contact_difference is None
            and a.get("robot_target_contact") != b.get("robot_target_contact")
        ):
            first_contact_difference = step
        for name, length, output in (
            ("arm_qpos_rad", 6, joint_deltas),
            ("ee_xyz_m", 3, ee_deltas),
            ("target_xyz_m", 3, target_deltas),
        ):
            x = _vector(a, name, length)
            y = _vector(b, name, length)
            if x is not None and y is not None:
                output.append(max(abs(i - j) for i, j in zip(x, y, strict=True))
                              if length == 6 else dist(x, y))
        for name, output in finger_deltas.items():
            x = _vector(_dict(_dict(a.get("finger_link_world_pose")).get(name)),
                        "position_xyz_m", 3)
            y = _vector(_dict(_dict(b.get("finger_link_world_pose")).get(name)),
                        "position_xyz_m", 3)
            if x is not None and y is not None:
                output.append(dist(x, y))
    return {
        "left": first["name"],
        "right": second["name"],
        "same_source_revision": first["git_head"] == second["git_head"],
        "same_initial_object_pose": (
            first["initial_object_pose"] == second["initial_object_pose"]
        ),
        "same_initial_pad_pose": first["initial_pad_pose"] == second["initial_pad_pose"],
        "same_grasp_candidate": first["selected_grasp_id"] == second["selected_grasp_id"],
        "forbidden_contact_difference": (
            second["retreat_contacts"] - first["retreat_contacts"]
            if isinstance(first["retreat_contacts"], int)
            and isinstance(second["retreat_contacts"], int) else None
        ),
        "frame_comparison_available": bool(steps),
        "matched_frame_count": len(steps),
        "first_contact_disagreement_step": first_contact_difference,
        "max_joint_angle_difference_rad": max(joint_deltas, default=None),
        "max_end_effector_position_difference_m": max(ee_deltas, default=None),
        "max_object_position_difference_m": max(target_deltas, default=None),
        "max_left_finger_position_difference_m": max(
            finger_deltas["gripper_L_link"], default=None
        ),
        "max_right_finger_position_difference_m": max(
            finger_deltas["gripper_R_link"], default=None
        ),
    }


def compare_runs(folders: list[Path]) -> dict[str, Any]:
    if not folders:
        raise ValueError("At least one native seed0 diagnostic export is required")
    if len({folder.resolve() for folder in folders}) != len(folders):
        raise ValueError("Duplicate export paths are not independent native runs")
    runs = [inspect_run(folder) for folder in folders]
    heads = {item["git_head"] for item in runs}
    comparisons = [compare_pair(a, b) for a, b in itertools.combinations(runs, 2)]
    return {
        "schema": "seed0-retreat-native-replay-comparison-v1",
        "claim_boundary": "DIAGNOSTIC_ONLY_NOT_ACCEPTED; never grants 3-seed campaign closure",
        "run_count": len(runs),
        "same_revision": len(heads) == 1,
        "repeatability_assessed": len(runs) >= 2 and len(heads) == 1,
        "runs": runs,
        "comparisons": comparisons,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Seed0 native repeatability comparison", "",
        "**Diagnostic only — not accepted campaign evidence.**", "",
        f"Runs: {result['run_count']}; Same Git revision: {result['same_revision']}; "
        f"Repeatability assessed: {result['repeatability_assessed']}", "",
        "| Export | Audit | Retreat contact frames | Trace | Git HEAD |",
        "|---|---|---:|---|---|",
    ]
    for run in result["runs"]:
        lines.append(
            f"| {run['name']} | {'PASS' if run['audit_passed'] else 'FAIL'} | "
            f"{run['retreat_contacts']} | {run['trace_quality']['status']} | "
            f"{run['git_head'][:12]} |"
        )
    lines.extend(["", "## Pairwise comparison", ""])
    for pair in result["comparisons"]:
        lines.append(
            f"- **{pair['left']} ↔ {pair['right']}**: source revision same="
            f"{pair['same_source_revision']}, compared physics frames="
            f"{pair['matched_frame_count']}, first different contact frame="
            f"{pair['first_contact_disagreement_step']}, max EE position delta="
            f"{pair['max_end_effector_position_difference_m']} m, max joint delta="
            f"{pair['max_joint_angle_difference_rad']} rad."
        )
    lines.extend(["", "Missing frame traces mean unknown, not equal motion. "
                  "Planner proxy clearance is not a physical contact measurement.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = compare_runs(args.exports)
        destination = args.output_dir.resolve()
        for folder in args.exports:
            if destination == folder.resolve() or destination.is_relative_to(
                folder.resolve()
            ):
                raise ValueError("Outputs must not mutate hashed native evidence folders")
        destination.mkdir(parents=True, exist_ok=True)
        # Keep full frames in comparison.json for exact reproducibility.
        (destination / "comparison.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (destination / "comparison.md").write_text(
            _markdown(result), encoding="utf-8"
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"DIAGNOSTIC_ONLY_NOT_ACCEPTED; analyzed {result['run_count']} seed0 runs")
    print(f"report={destination / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
