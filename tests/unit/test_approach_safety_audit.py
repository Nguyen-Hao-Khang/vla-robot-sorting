from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vla_robot_sorting.planning.approach_safety_audit import (
    ApproachAuditError,
    audit_plan_only_trial_bundle,
    audit_seed2_approach_failure,
    score_plan_only_trial,
    validate_plan_only_trial,
    validate_plan_only_trial_set,
)

EXPECTED_SOURCE_HEAD = "7ac99081a7636e3fab0c8c0cb947da3310f0fe33"
EXPECTED_ROBOTWIN = "2aee31ad75cb7b0cd779540441c6025853a5b60d"
POLICY = {
    "allow_robot_table_contact": False,
    "allow_robot_target_contact": False,
    "allow_target_pad_contact": False,
    "allow_target_table_contact": True,
}


def _write_evidence(root: Path, payload: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "git-head.txt": EXPECTED_SOURCE_HEAD + "\n",
        "robotwin-upstream-pin.txt": EXPECTED_ROBOTWIN + "\n",
        "classification.txt": "DIAGNOSTIC_ONLY_NOT_ACCEPTED\n",
        "source-sha256.txt": "0" * 64 + "  source.py\n",
        "seed2-alternative.json": json.dumps(payload or _native_fixture()),
    }
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    sums = "".join(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  ./{name}\n" for name in files
    )
    (root / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    return root


def _collision(index: int) -> dict:
    return {
        "waypoint_index": index,
        "matched_environment_roles": ["robot_link", "target"],
        "environment_hits": [
            {
                "attributes": {
                    "link_name1": "gripper_R_link",
                    "link_name2": "080_pillbottle_23",
                }
            }
        ],
        "self_hits": [],
        "self_collision_count": 0,
    }


def _candidate(candidate_id: str, *, grasp, pregrasp, direction, height=0.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "object_id": "pillbottle",
        "geometry_source": "sim_ground_truth",
        "grasp_position_xyz": grasp,
        "pregrasp_position_xyz": pregrasp,
        "approach_direction": direction,
        "grasp_orientation_xyzw": [0, 0, 0, 1],
        "pregrasp_orientation_xyzw": [0, 0, 0, 1],
        "scene_revision": "project-manip-seed-2",
        "transform_revision": "robotwin-static-v1",
        "metadata": {
            "mode": "side",
            "shape": "cylinder",
            "sample_index": 3,
            "grasp_height_fraction": height,
            "table_clearance_m": 0.059,
            "gripper_frame_contact_offset_m": 0.10,
            "pregrasp_contact_center_position_xyz_m": [0, 0, 0],
            "contact_center_position_xyz_m": [0, 0, 0],
        },
    }


def _native_fixture() -> dict:
    chosen = _candidate(
        "pillbottle:cyl:side:3:s-1",
        grasp=[1.0, 0.0, 0.0],
        pregrasp=[0.92, 0.0, 0.0],
        direction=[1.0, 0.0, 0.0],
    )
    alternate = _candidate(
        "pillbottle:cyl:side:2:s-1",
        grasp=[0.0, 1.0, 0.0],
        pregrasp=[0.0, 0.92, 0.0],
        direction=[0.0, 1.0, 0.0],
    )
    approach_collisions = [_collision(i) for i in range(222, 281)]
    return {
        "schema": "panthera-native-project-manipulation-v1",
        "seed": 2,
        "task": "move_pillbottle_pad",
        "passed": False,
        "final_phase": "recovery_required",
        "candidate_selection_diagnostic": {"candidate_id_override": chosen["candidate_id"]},
        "B5": {
            "selected": chosen,
            "candidates": [chosen, alternate],
            "ranking": [
                {"candidate_id": alternate["candidate_id"], "total": 0.75},
                {"candidate_id": chosen["candidate_id"], "total": 0.71},
            ],
        },
        "object_geometry": {
            "size_xyz_m": [0.0751, 0.1548, 0.0750],
            "model_scale": [0.08, 0.08, 0.08],
            "collision_mesh_sha256": "a" * 64,
        },
        "gripper_geometry": {
            "max_width_m": 0.08,
            "finger_proxy_diameter_m": 0.036,
            "pregrasp_standoff_m": 0.08,
            "contact_center_offset_m": 0.10,
        },
        "stages": [
            {
                "stage": "pregrasp",
                "passed": True,
                "plan_success": True,
                "dynamic_safe": True,
                "collision_safe": True,
                "execution_status": "completed",
                "path_samples": 733,
                "collision_validation": {
                    "collision_free": True,
                    "waypoints_checked": 733,
                    "collisions": [],
                },
                "contact_policy": dict(POLICY),
            },
            {
                "stage": "approach",
                "passed": False,
                "plan_success": True,
                "dynamic_safe": True,
                "collision_safe": False,
                "path_samples": 311,
                "path_metadata": {
                    "vendor_status": "Success",
                    "vendor_sample_count": 311,
                    "vendor_interpolation_dt_s": 0.004,
                    "scene_revision": "project-manip-seed-2",
                    "transform_revision": "robotwin-static-v1",
                },
                "collision_validation": {
                    "collision_free": False,
                    "waypoints_checked": 311,
                    "collisions": approach_collisions,
                },
            },
        ],
    }


def _valid_trial() -> dict:
    return {
        "schema": "s1m.approach-plan-only-diagnostic.v2",
        "classification": "PLAN_ONLY_APPROACH_DIAGNOSTIC_NOT_ACCEPTANCE",
        "provenance": {
            "project_git_sha": "1" * 40,
            "robotwin_git_sha": EXPECTED_ROBOTWIN,
            "source_evidence_sha256": "2" * 64,
            "baseline_evidence_sha256": "3" * 64,
            "candidate_id": "pillbottle:cyl:side:2:s-1",
            "scene_revision": "project-manip-seed-2",
            "transform_revision": "robotwin-static-v1",
            "start_state_sha256": "4" * 64,
            "geometry_sha256": "5" * 64,
            "canonical_placement_goal_sha256": "6" * 64,
        },
        "stages": {
            "pregrasp": {
                "passed": True,
                "plan_success": True,
                "dynamic_safe": True,
                "scene_revision": "project-manip-seed-2",
                "transform_revision": "robotwin-static-v1",
                "execution_status": "not_executed",
                "contact_policy": dict(POLICY),
                "collision_free": True,
                "collisions": [],
                "waypoint_count": 100,
                "waypoints_checked": 100,
                "waypoints": [
                    {
                        "sample_index": index,
                        "time_from_start_s": index * 0.004,
                        "qpos_rad": [index / 1000.0] * 6,
                        "tool_pose_world_xyzw": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                        "fk_link_poses_world": {
                            "gripper_R_link": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
                        },
                        "collision_check": {
                            "checked": True,
                            "collision_free": True,
                            "collision_pairs": [],
                            "checked_pair_count": 4,
                            "scene_revision": "project-manip-seed-2",
                            "scope_sha256": "7" * 64,
                            "minimum_signed_clearance_m": (0.012 if index == 0 else 0.013),
                        },
                    }
                    for index in range(100)
                ],
            },
            "approach": {
                "planner_id": "robotwin-mplib-screw",
                "plan_success": True,
                "dynamic_safe": True,
                "scene_revision": "project-manip-seed-2",
                "transform_revision": "robotwin-static-v1",
                "waypoint_count": 200,
                "waypoints_checked": 200,
                "waypoints": [
                    {
                        "sample_index": index,
                        "time_from_start_s": index * 0.004,
                        "qpos_rad": [index / 1000.0] * 6,
                        "tool_pose_world_xyzw": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                        "fk_link_poses_world": {
                            "gripper_R_link": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
                        },
                        "collision_check": {
                            "checked": True,
                            "collision_free": True,
                            "collision_pairs": [],
                            "checked_pair_count": 4,
                            "scene_revision": "project-manip-seed-2",
                            "scope_sha256": "7" * 64,
                            "minimum_signed_clearance_m": (0.012 if index == 0 else 0.013),
                        },
                    }
                    for index in range(200)
                ],
                "collision_free": True,
                "collisions": [],
                "contact_policy": dict(POLICY),
                "execution_status": "not_executed",
            },
        },
        "runtime": {
            "execution_performed": False,
            "physics_steps": 0,
            "policy_changed": False,
            "canonical_placement_changed": False,
        },
        "metrics": {
            "minimum_clearance_m": 0.012,
            "minimum_clearance_source": {
                "method_id": "native-signed-distance-query",
                "engine": "source-verified-native-collision-engine",
                "implementation_sha256": "8" * 64,
                "unit": "m",
                "sign_convention": "positive_means_separated",
                "scope_sha256": "7" * 64,
                "implementation_path": "distance-source.txt",
            },
        },
    }


def _refresh_sums(root: Path) -> None:
    entries = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(f"{digest}  ./{path.name}\n")
    (root / "SHA256SUMS.txt").write_text("".join(entries), encoding="utf-8")


def _write_plan_bundle(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    trial = _valid_trial()
    lf = chr(10)
    files = {
        "git-head.txt": "1" * 40 + lf,
        "robotwin-upstream-pin.txt": EXPECTED_ROBOTWIN + lf,
        "classification.txt": "PLAN_ONLY_DIAGNOSTIC_NOT_ACCEPTED" + lf,
        "baseline-evidence-sha256.txt": (
            "214cd667cb48adb22010734516f96ea7b9468f736e6b20feb9a23b718f9eee71" + lf
        ),
        "source-input.json": "{}" + lf,
        "start-state.json": "{}" + lf,
        "candidate-geometry.json": "{}" + lf,
        "distance-source.txt": "synthetic test fixture; not native source" + lf,
    }
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")

    def digest(name: str) -> str:
        return hashlib.sha256((root / name).read_bytes()).hexdigest()

    trial["provenance"]["baseline_evidence_sha256"] = (
        "214cd667cb48adb22010734516f96ea7b9468f736e6b20feb9a23b718f9eee71"
    )
    trial["provenance"]["source_evidence_sha256"] = digest("source-input.json")
    trial["provenance"]["start_state_sha256"] = digest("start-state.json")
    trial["provenance"]["geometry_sha256"] = digest("candidate-geometry.json")
    trial["metrics"]["minimum_clearance_source"]["implementation_sha256"] = digest(
        "distance-source.txt"
    )
    (root / "plan-only-trial.json").write_text(json.dumps(trial, sort_keys=True), encoding="utf-8")
    _refresh_sums(root)
    return root


def test_audit_verifies_original_seed2_failure_and_geometry(tmp_path):
    evidence = _write_evidence(tmp_path / "evidence")

    report = audit_seed2_approach_failure(evidence)

    assert report["classification"] == "APPROACH_COLLISION_REJECTED_DIAGNOSTIC_ONLY"
    assert report["source_git_sha"] == EXPECTED_SOURCE_HEAD
    assert report["robotwin_git_sha"] == EXPECTED_ROBOTWIN
    assert report["pregrasp"]["passed"] is True
    assert report["approach"]["planner_success"] is True
    assert report["approach"]["executed"] is False
    assert report["approach"]["waypoint_count"] == 311
    assert report["approach"]["collision_count"] == 59
    assert report["approach"]["collision_span_indices"] == [222, 280]
    assert report["approach"]["collision_waypoint_fraction"] == pytest.approx(59 / 311)
    assert report["approach"]["collision_pairs"] == [
        {"robot_link": "gripper_R_link", "target": "080_pillbottle_23"}
    ]
    selected = report["geometry_comparison"]["candidates"][0]
    assert selected["candidate_id"] == "pillbottle:cyl:side:3:s-1"
    assert selected["standoff_m"] == pytest.approx(0.08)
    assert selected["approach_alignment_cosine"] == pytest.approx(1.0)
    assert selected["native_path_status"] == "COLLISION_REJECTED"
    assert report["geometry_comparison"]["candidates"][1]["native_path_status"] == ("NOT_EVALUATED")
    assert report["controller_eligible"] is False


def test_audit_rejects_tampered_native_json(tmp_path):
    evidence = _write_evidence(tmp_path / "evidence")
    path = evidence / "seed2-alternative.json"
    path.write_text(path.read_text() + " ", encoding="utf-8")

    with pytest.raises(ApproachAuditError, match="checksum"):
        audit_seed2_approach_failure(evidence)


def test_audit_rejects_checksum_path_traversal(tmp_path):
    evidence = _write_evidence(tmp_path / "evidence")
    (evidence / "SHA256SUMS.txt").write_text("0" * 64 + "  ./../outside.json\n", encoding="utf-8")

    with pytest.raises(ApproachAuditError, match="path"):
        audit_seed2_approach_failure(evidence)


def test_audit_rejects_noncontiguous_collision_record(tmp_path):
    payload = _native_fixture()
    payload["stages"][1]["collision_validation"]["collisions"][1]["waypoint_index"] = 8
    evidence = _write_evidence(tmp_path / "evidence", payload)
    _refresh_sums(evidence)

    with pytest.raises(ApproachAuditError, match="contiguous"):
        audit_seed2_approach_failure(evidence)


def test_audit_rejects_wrong_source_provenance(tmp_path):
    evidence = _write_evidence(tmp_path / "evidence")
    (evidence / "git-head.txt").write_text("0" * 40 + "\n", encoding="utf-8")
    _refresh_sums(evidence)

    with pytest.raises(ApproachAuditError, match="source Git"):
        audit_seed2_approach_failure(evidence)


def test_collision_rejects_plan_only_trial_and_suppresses_score():
    trial = _valid_trial()
    trial["stages"]["approach"]["collision_free"] = False
    trial["stages"]["approach"]["collisions"] = [_collision(33)]
    trial["stages"]["approach"]["waypoints_checked"] = 200

    audit = validate_plan_only_trial(trial)
    score = score_plan_only_trial(trial)

    assert audit["eligible_for_plan_only_review"] is False
    assert "approach_collision" in audit["rejection_reasons"]
    assert score["score_vector"] is None
    assert score["controller_eligible"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda trial: trial["runtime"].update(execution_performed=True),
        lambda trial: trial["runtime"].update(physics_steps=1),
        lambda trial: trial["runtime"].update(policy_changed=True),
        lambda trial: trial["runtime"].update(canonical_placement_changed=True),
        lambda trial: trial["stages"]["approach"]["contact_policy"].update(
            allow_robot_target_contact=True
        ),
        lambda trial: trial["stages"]["approach"].update(waypoints_checked=199),
        lambda trial: trial["stages"]["approach"].update(scene_revision="stale"),
        lambda trial: trial["provenance"].update(start_state_sha256="missing"),
        lambda trial: trial["metrics"].update(minimum_clearance_m=float("nan")),
    ],
)
def test_plan_only_contract_fails_closed(mutation):
    trial = _valid_trial()
    mutation(trial)

    assert validate_plan_only_trial(trial)["eligible_for_plan_only_review"] is False
    assert score_plan_only_trial(trial)["score_vector"] is None


def test_plan_only_contract_requires_per_waypoint_fk_and_collision_checks():
    trial = _valid_trial()
    trial["stages"]["approach"]["waypoints"][77].pop("fk_link_poses_world")
    trial["stages"]["approach"]["waypoints"][88].pop("collision_check")

    audit = validate_plan_only_trial(trial)

    assert audit["eligible_for_plan_only_review"] is False
    assert "approach_missing_fk_or_collision_evidence" in audit["rejection_reasons"]


def test_plan_only_contract_requires_pregrasp_waypoint_validation_too():
    trial = _valid_trial()
    trial["stages"]["pregrasp"]["waypoints"].pop()

    audit = validate_plan_only_trial(trial)

    assert audit["eligible_for_plan_only_review"] is False
    assert "pregrasp_incomplete_waypoint_validation" in audit["rejection_reasons"]


def test_plan_only_contract_rejects_per_waypoint_collision_or_stale_scope():
    trial = _valid_trial()
    trial["stages"]["approach"]["waypoints"][100]["collision_check"]["collision_free"] = False

    audit = validate_plan_only_trial(trial)

    assert audit["eligible_for_plan_only_review"] is False
    assert "approach_waypoint_collision_or_scope" in audit["rejection_reasons"]


def test_score_recomputes_clearance_from_waypoint_measurements():
    trial = _valid_trial()
    trial["stages"]["pregrasp"]["waypoints"][0]["collision_check"]["minimum_signed_clearance_m"] = (
        0.021
    )
    trial["stages"]["approach"]["waypoints"][0]["collision_check"]["minimum_signed_clearance_m"] = (
        0.021
    )

    result = score_plan_only_trial(trial)

    assert result["score_vector"] is None
    assert result["status"] == "REJECTED_NO_SCORE"
    assert "clearance_summary_mismatch" in result["rejection_reasons"]


def test_plan_only_score_is_lexicographic_and_never_controller_authority():
    first = _valid_trial()
    second = _valid_trial()
    second["provenance"]["candidate_id"] = "candidate-b"
    second["metrics"]["minimum_clearance_m"] = 0.018
    for stage in second["stages"].values():
        for waypoint in stage["waypoints"]:
            waypoint["collision_check"]["minimum_signed_clearance_m"] = 0.018

    a = score_plan_only_trial(first)
    b = score_plan_only_trial(second)

    assert a["eligible_for_plan_only_review"] is True
    assert b["eligible_for_plan_only_review"] is True
    assert a["score_vector"][:3] == pytest.approx([-0.012, 1.788, 1.192])
    assert a["score_vector"][3] == "pillbottle:cyl:side:2:s-1"
    assert b["score_vector"][:3] == pytest.approx([-0.018, 1.788, 1.192])
    assert b["score_vector"][3] == "candidate-b"
    assert b["score_vector"] < a["score_vector"]
    assert a["controller_eligible"] is False
    assert a["native_acceptance"] is False


def test_trial_set_requires_fixed_canonical_goal_and_same_start_state():
    first = _valid_trial()
    second = _valid_trial()
    second["provenance"]["candidate_id"] = "pillbottle:cyl:side:3:s-1"

    valid = validate_plan_only_trial_set([first, second])
    assert valid["eligible_for_comparison"] is True
    assert valid["canonical_placement_goal_sha256"] == "6" * 64

    second["provenance"]["canonical_placement_goal_sha256"] = "9" * 64
    changed_goal = validate_plan_only_trial_set([first, second])
    assert changed_goal["eligible_for_comparison"] is False
    assert (
        "trial_set_mismatched_canonical_placement_goal_sha256"
        in (changed_goal["rejection_reasons"])
    )

    second["provenance"]["canonical_placement_goal_sha256"] = ["malformed"]
    malformed = validate_plan_only_trial_set([first, second])
    assert malformed["eligible_for_comparison"] is False


def test_missing_clearance_prevents_score_even_when_sampled_path_is_clear():
    trial = _valid_trial()
    trial["metrics"]["minimum_clearance_m"] = None
    trial["metrics"]["minimum_clearance_source"] = None
    for stage in trial["stages"].values():
        for waypoint in stage["waypoints"]:
            waypoint["collision_check"]["minimum_signed_clearance_m"] = None

    audit = validate_plan_only_trial(trial)
    score = score_plan_only_trial(trial)

    assert audit["eligible_for_plan_only_review"] is True
    assert audit["evidence_grade"] == "SAMPLED_COLLISION_FREE_CLEARANCE_UNVERIFIED"
    assert score["score_vector"] is None
    assert score["status"] == "UNSCORED_CLEARANCE_MISSING"


def test_plan_only_bundle_binds_sources_and_hashes_without_acceptance(tmp_path):
    bundle = _write_plan_bundle(tmp_path / "plan-only")

    report = audit_plan_only_trial_bundle(bundle)

    assert report["classification"] == ("PLAN_ONLY_DIAGNOSTIC_STRUCTURALLY_VALID_NOT_ACCEPTANCE")
    assert report["source_sha256_inventory_count"] == 9
    assert report["trial_validation"]["eligible_for_plan_only_review"] is True
    assert report["controller_eligible"] is False
    assert report["native_acceptance"] is False


def test_plan_only_bundle_rejects_corrupt_trial_and_wrong_source_lock(tmp_path):
    bundle = _write_plan_bundle(tmp_path / "plan-only")
    trial_path = bundle / "plan-only-trial.json"
    trial_path.write_text(trial_path.read_text() + " ", encoding="utf-8")

    with pytest.raises(ApproachAuditError, match="checksum mismatch"):
        audit_plan_only_trial_bundle(bundle)

    bundle = _write_plan_bundle(tmp_path / "wrong-source")
    (bundle / "git-head.txt").write_text("0" * 40 + chr(10), encoding="utf-8")
    _refresh_sums(bundle)
    with pytest.raises(ApproachAuditError, match="Git SHA"):
        audit_plan_only_trial_bundle(bundle)


def test_plan_only_bundle_rejects_unhashed_clearance_implementation(tmp_path):
    bundle = _write_plan_bundle(tmp_path / "plan-only")
    trial_path = bundle / "plan-only-trial.json"
    record = json.loads(trial_path.read_text(encoding="utf-8"))
    record["metrics"]["minimum_clearance_source"]["implementation_sha256"] = "f" * 64
    trial_path.write_text(json.dumps(record), encoding="utf-8")
    _refresh_sums(bundle)

    with pytest.raises(ApproachAuditError, match="implementation hash"):
        audit_plan_only_trial_bundle(bundle)


def _run_cli(evidence: Path, *args: str, script_name: str = "audit_s1m_seed2_approach.py"):
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts" / script_name
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    return subprocess.run(
        [sys.executable, str(script), str(evidence), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_emits_audited_json_for_verified_native_export(tmp_path):
    evidence = _write_evidence(tmp_path / "evidence")
    result = _run_cli(evidence)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["classification"] == "APPROACH_COLLISION_REJECTED_DIAGNOSTIC_ONLY"
    assert report["native_acceptance"] is False


def test_cli_writes_only_new_report_outside_immutable_evidence(tmp_path):
    evidence = _write_evidence(tmp_path / "evidence")
    output = tmp_path / "analysis.json"
    result = _run_cli(evidence, "--output", str(output))

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["candidate_id"] == "pillbottle:cyl:side:3:s-1"

    overwrite = _run_cli(evidence, "--output", str(output))
    assert overwrite.returncode == 2
    assert "overwrite" in overwrite.stderr.lower()


def test_cli_refuses_output_inside_source_evidence(tmp_path):
    evidence = _write_evidence(tmp_path / "evidence")
    result = _run_cli(evidence, "--output", str(evidence / "audit.json"))

    assert result.returncode == 2
    assert "inside" in result.stderr.lower()
    assert not (evidence / "audit.json").exists()


def test_plan_only_bundle_cli_audits_fixture_only(tmp_path):
    bundle = _write_plan_bundle(tmp_path / "plan-only")
    result = _run_cli(
        bundle,
        script_name="audit_s1m_approach_diagnostic.py",
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["classification"] == ("PLAN_ONLY_DIAGNOSTIC_STRUCTURALLY_VALID_NOT_ACCEPTANCE")
    assert report["native_acceptance"] is False
