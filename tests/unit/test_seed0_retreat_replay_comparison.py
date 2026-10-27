"""Offline contract tests for a read-only native diagnostic comparator.

Synthetic fixtures in these tests are not native/accepted robot evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import compare_seed0_retreat_replays as tool


def diagnostic_export(
    folder: Path, *, revision: str = "a" * 40, contacts: tuple[bool, ...] = (False, False)
) -> Path:
    folder.mkdir()
    probe = {
        "seed": 0,
        "passed": not any(contacts),
        "initial_object_pose": {"position_xyz": [0.1, 0.2, 0.3]},
        "initial_pad_pose": {"position_xyz": [0.2, 0.3, 0.4]},
        "B5": {"selected": {"candidate_id": "side-grasp"}},
        "stages": [{
            "stage": "retreat",
            "passed": not any(contacts),
            "planner_id": "robotwin-mplib-rrt",
            "contacts": {
                "physics_steps": len(contacts),
                "robot_target_contact_frames": sum(contacts),
                "first_robot_target_step": next(
                    (i + 1 for i, flag in enumerate(contacts) if flag), None
                ),
                "last_robot_target_step": next(
                    (len(contacts) - i for i, flag in enumerate(reversed(contacts))
                     if flag), None
                ),
            },
            "runtime_motion_frame_trace": {
                "schema": "retreat-kinematic-frames-v1",
                "truncated": False,
                "frames": [
                    {
                        "physics_step": i + 1,
                        "robot_target_contact": flag,
                        "ee_xyz_m": [0.1 + i / 1000, 0.2, 0.3],
                        "target_xyz_m": [0.8, 0.0, 0.1],
                        "arm_qpos_rad": [0.1 + i / 100, 0.2, 0.3, 0.4, 0.5, 0.6],
                    }
                    for i, flag in enumerate(contacts)
                ],
            },
        }],
    }
    (folder / "seed0.json").write_text(json.dumps(probe), encoding="utf-8")
    (folder / "git-head.txt").write_text(revision, encoding="ascii")
    (folder / "classification.txt").write_text(
        "DIAGNOSTIC_ONLY_NOT_ACCEPTED", encoding="ascii"
    )
    checks = "".join(
        f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.name}\n"
        for file in sorted(folder.iterdir())
    )
    (folder / "SHA256SUMS.txt").write_text(checks, encoding="ascii")
    return folder


def test_one_native_labelled_export_is_not_repeatability_proof(
    monkeypatch, tmp_path: Path
) -> None:
    folder = diagnostic_export(tmp_path / "one")
    monkeypatch.setattr(
        tool, "audit_project_manipulation_probe",
        lambda _: SimpleNamespace(passed=True, failures=()),
    )
    result = tool.compare_runs([folder])
    assert result["run_count"] == 1
    assert result["repeatability_assessed"] is False
    assert result["runs"][0]["trace_quality"] == {
        "status": "complete", "frame_count": 2
    }
    assert result["runs"][0]["checksums_verified"] == 3
    assert "NOT_ACCEPTED" in result["claim_boundary"]


def test_two_runs_compare_physics_frames_and_contacts(
    monkeypatch, tmp_path: Path
) -> None:
    old = diagnostic_export(tmp_path / "old", contacts=(False, True, False))
    new = diagnostic_export(tmp_path / "new", contacts=(False, False, False))
    monkeypatch.setattr(
        tool, "audit_project_manipulation_probe",
        lambda _: SimpleNamespace(passed=False, failures=(SimpleNamespace(name="retreat"),)),
    )
    result = tool.compare_runs([old, new])
    pair = result["comparisons"][0]
    assert result["repeatability_assessed"] is True
    assert pair["same_initial_object_pose"] is True
    assert pair["same_grasp_candidate"] is True
    assert pair["forbidden_contact_difference"] == -1
    assert pair["matched_frame_count"] == 3
    assert pair["first_contact_disagreement_step"] == 2
    assert pair["max_joint_angle_difference_rad"] == 0
    assert pair["max_end_effector_position_difference_m"] == 0
    assert result["runs"][0]["audit_passed"] is False


def test_different_source_revision_is_not_repeatability(monkeypatch, tmp_path: Path) -> None:
    one = diagnostic_export(tmp_path / "one", revision="a" * 40)
    two = diagnostic_export(tmp_path / "two", revision="b" * 40)
    monkeypatch.setattr(
        tool, "audit_project_manipulation_probe",
        lambda _: SimpleNamespace(passed=False, failures=()),
    )
    result = tool.compare_runs([one, two])
    assert result["same_revision"] is False
    assert result["repeatability_assessed"] is False
    assert result["comparisons"][0]["same_source_revision"] is False


def test_missing_frames_never_means_identical_motion() -> None:
    basic = {
        "name": "one", "git_head": "0" * 40, "motion_frames": [],
        "initial_object_pose": None, "initial_pad_pose": None,
        "selected_grasp_id": None, "retreat_contacts": 0,
    }
    compared = tool.compare_pair(basic, {**basic, "name": "two"})
    assert compared["frame_comparison_available"] is False
    assert compared["matched_frame_count"] == 0
    assert compared["max_joint_angle_difference_rad"] is None


def test_tampered_native_evidence_fails_closed(tmp_path: Path) -> None:
    folder = diagnostic_export(tmp_path / "one")
    (folder / "seed0.json").write_text('{"seed": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        tool.inspect_run(folder)


def test_trace_mismatch_detected_without_changing_native_auditor() -> None:
    record = {
        "contacts": {"physics_steps": 2, "robot_target_contact_frames": 1},
        "runtime_motion_frame_trace": {
            "schema": "retreat-kinematic-frames-v1",
            "truncated": False,
            "frames": [
                {"physics_step": 1, "robot_target_contact": False},
                {"physics_step": 2, "robot_target_contact": False},
            ],
        },
    }
    assert tool._trace_quality(record)["status"] == "invalid"
    record["runtime_motion_frame_trace"]["truncated"] = True
    assert tool._trace_quality(record)["status"] == "partial"


def test_duplicate_export_cannot_count_as_repeatability(tmp_path: Path) -> None:
    folder = diagnostic_export(tmp_path / "same")
    with pytest.raises(ValueError, match="Duplicate export"):
        tool.compare_runs([folder, folder])
