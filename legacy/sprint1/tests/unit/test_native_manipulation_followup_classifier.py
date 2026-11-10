from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "classify_native_manipulation_followup.py"
)
SPEC = importlib.util.spec_from_file_location("native_followup_classifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
classify_campaign = MODULE.classify_campaign


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_campaign(path: Path, **values: int) -> None:
    defaults = {
        "CAMPAIGN_ACCEPTED": 0,
        "RECOVERY_DIAG_COUNT": 0,
        "RECOVERY_PASS_COUNT": 0,
    }
    defaults.update(values)
    path.write_text(
        "\n".join(f"{key}={value}" for key, value in defaults.items()) + "\n",
        encoding="utf-8",
    )


def _lift_payload(
    *,
    start: dict,
    native_status: str,
    table_diag: dict | None = None,
    signed_gap_m: float = -0.0005,
) -> dict:
    payload = {
        "passed": False,
        "carried_start_state_diagnostic": start,
        "carried_object_attachment": {
            "support_contact_geometry": {
                "signed_gap_m": signed_gap_m,
                "diagnostic_only": True,
            }
        },
        "stages": [
            {
                "stage": "lift",
                "passed": False,
                "plan_success": False,
                "failure": {
                    "code": "optimization_failed",
                    "message": "native fail",
                    "details": {
                        "native_status": native_status,
                        "native_diagnostics": {
                            "start_state": start,
                        },
                    },
                },
            }
        ],
    }
    if table_diag is not None:
        payload["lift_table_disabled_diagnostic"] = table_diag
    return payload


def test_classifies_support_contact_start_collision(tmp_path: Path) -> None:
    _write_campaign(tmp_path / "CAMPAIGN_RESULT.txt")
    _write_json(
        tmp_path / "manipulation-seed-0.json",
        _lift_payload(
            start={
                "available": True,
                "valid": False,
                "status": "Start state is colliding with world",
            },
            native_status="Invalid query",
            table_diag={
                "obstacle_disabled": True,
                "obstacle_reenabled": True,
                "executed": False,
                "start_state": {"available": True, "valid": True, "status": None},
                "plan_success": True,
                "native_status": "Success",
                "path_samples": 123,
            },
        ),
    )

    result = classify_campaign(tmp_path)

    assert result["primary_decision"] == "SUPPORT_CONTACT_START_COLLISION"
    decision = result["decisions"][0]
    assert decision["seeds"] == [0]
    assert decision["signed_gap_m"] == -0.0005


def test_classifies_valid_start_ik_failure(tmp_path: Path) -> None:
    _write_campaign(tmp_path / "CAMPAIGN_RESULT.txt")
    _write_json(
        tmp_path / "manipulation-seed-0.json",
        _lift_payload(
            start={"available": True, "valid": True, "status": None},
            native_status="IK Fail",
        ),
    )

    result = classify_campaign(tmp_path)

    assert result["primary_decision"] == "IK_OR_TRAJOPT"
    assert result["decisions"][0]["seeds"] == [0]


def test_classifies_table_not_primary_when_failure_status_is_unchanged(
    tmp_path: Path,
) -> None:
    _write_campaign(tmp_path / "CAMPAIGN_RESULT.txt")
    _write_json(
        tmp_path / "manipulation-seed-0.json",
        _lift_payload(
            start={
                "available": True,
                "valid": False,
                "status": "Start state invalid",
            },
            native_status="TrajOpt Fail",
            table_diag={
                "obstacle_disabled": True,
                "obstacle_reenabled": True,
                "executed": False,
                "start_state": {
                    "available": True,
                    "valid": False,
                    "status": "Start state invalid",
                },
                "plan_success": False,
                "native_status": "TrajOpt Fail",
                "path_samples": 0,
            },
        ),
    )

    result = classify_campaign(tmp_path)

    assert result["primary_decision"] == "TABLE_NOT_PRIMARY"


def test_classifies_seed1_recovery_candidate(tmp_path: Path) -> None:
    _write_campaign(
        tmp_path / "CAMPAIGN_RESULT.txt",
        RECOVERY_DIAG_COUNT=2,
        RECOVERY_PASS_COUNT=1,
    )

    result = classify_campaign(tmp_path)

    assert result["primary_decision"] == "SEED1_RECOVERY_CANDIDATE"


def test_campaign_acceptance_has_closure_priority(tmp_path: Path) -> None:
    _write_campaign(
        tmp_path / "CAMPAIGN_RESULT.txt",
        CAMPAIGN_ACCEPTED=1,
        CLOSURE_CANDIDATE_RC=0,
        NATIVE_CLOSURE_RC=0,
    )

    result = classify_campaign(tmp_path)

    assert result["primary_decision"] == "READY_FOR_SPRINT1_CLOSURE"
    assert result["native_closure_ready"] is True


def test_manipulation_pass_without_closure_audit_is_bookkeeping_error(
    tmp_path: Path,
) -> None:
    _write_campaign(
        tmp_path / "CAMPAIGN_RESULT.txt",
        CAMPAIGN_ACCEPTED=1,
        CLOSURE_CANDIDATE_RC=2,
        NATIVE_CLOSURE_RC=98,
    )

    result = classify_campaign(tmp_path)

    assert result["primary_decision"] == "CLOSURE_PIPELINE_ERROR"
    assert result["campaign_accepted"] is True
    assert result["native_closure_ready"] is False


def test_old_campaign_without_new_diagnostics_stays_unresolved(tmp_path: Path) -> None:
    _write_campaign(tmp_path / "CAMPAIGN_RESULT.txt")
    _write_json(
        tmp_path / "manipulation-seed-0.json",
        {
            "passed": False,
            "stages": [
                {
                    "stage": "lift",
                    "passed": False,
                    "plan_success": False,
                    "failure": {
                        "code": "optimization_failed",
                        "message": "carried-object CuRobo status=Fail",
                    },
                }
            ],
        },
    )

    result = classify_campaign(tmp_path)

    assert result["primary_decision"] == "CARRIED_LIFT_UNRESOLVED"


def test_same_unresolved_lift_decision_merges_seed_list(tmp_path: Path) -> None:
    _write_campaign(tmp_path / "CAMPAIGN_RESULT.txt")
    payload = {
        "passed": False,
        "stages": [
            {
                "stage": "lift",
                "passed": False,
                "plan_success": False,
                "failure": {
                    "code": "optimization_failed",
                    "message": "carried-object CuRobo status=Fail",
                },
            }
        ],
    }
    _write_json(tmp_path / "manipulation-seed-0.json", payload)
    _write_json(tmp_path / "manipulation-seed-2.json", payload)

    result = classify_campaign(tmp_path)

    assert result["primary_decision"] == "CARRIED_LIFT_UNRESOLVED"
    assert result["decisions"] == [
        {
            "code": "CARRIED_LIFT_UNRESOLVED",
            "seeds": [0, 2],
            "reason": "carried lift planning remains failed but evidence is not decisive",
            "next_action": "inspect native MotionGen diagnostics and attachment geometry",
        }
    ]


def _seed1_execution_contact_payload() -> dict:
    return {
        "passed": False,
        "stages": [
            {
                "stage": "approach",
                "passed": False,
                "planner_id": "robotwin-mplib-rrt",
                "plan_success": True,
                "dynamic_safe": True,
                "collision_safe": True,
                "execution_status": "completed",
                "collision_validation": {
                    "collision_free": True,
                    "collisions": [],
                },
                "contacts": {
                    "physics_steps": 278,
                    "robot_target_contact_frames": 3,
                    "target_pad_contact_frames": 0,
                    "target_table_contact_frames": 278,
                },
                "contact_policy": {
                    "allow_robot_target_contact": False,
                    "allow_target_pad_contact": False,
                    "allow_target_table_contact": True,
                },
            }
        ],
    }


def test_classifies_seed1_execution_contact_rejection(tmp_path: Path) -> None:
    _write_campaign(tmp_path / "CAMPAIGN_RESULT.txt")
    _write_json(
        tmp_path / "manipulation-seed-1.json",
        _seed1_execution_contact_payload(),
    )

    result = classify_campaign(tmp_path)

    assert result["primary_decision"] == "SEED1_EXECUTION_CONTACT_REJECTED"
    decision = result["decisions"][0]
    assert decision["seeds"] == [1]
    assert decision["robot_target_contact_frames"] == 3


def test_seed1_recovery_candidate_has_priority_over_original_contact_rejection(
    tmp_path: Path,
) -> None:
    _write_campaign(
        tmp_path / "CAMPAIGN_RESULT.txt",
        RECOVERY_DIAG_COUNT=2,
        RECOVERY_PASS_COUNT=1,
    )
    _write_json(
        tmp_path / "manipulation-seed-1.json",
        _seed1_execution_contact_payload(),
    )

    result = classify_campaign(tmp_path)

    assert result["primary_decision"] == "SEED1_RECOVERY_CANDIDATE"


def test_classifies_released_seed0_residual_retreat_contact(tmp_path: Path) -> None:
    _write_campaign(tmp_path / "CAMPAIGN_RESULT.txt")
    payload = {
        "passed": False,
        "task_success_after_release": True,
        "stages": [
            {
                "stage": "lift",
                "passed": True,
                "plan_success": True,
                "support_escape": {"passed": True},
            },
            {"stage": "release", "passed": True},
            {
                "stage": "retreat",
                "passed": False,
                "contacts": {"robot_target_contact_frames": 307},
            },
        ],
    }
    _write_json(tmp_path / "manipulation-seed-0.json", payload)
    result = classify_campaign(tmp_path)
    assert result["primary_decision"] == "POST_RELEASE_CONTACT_SEPARATION"
    assert result["decisions"][0]["seeds"] == [0]


def test_classifies_seed2_place_ik_after_supported_lift(tmp_path: Path) -> None:
    _write_campaign(tmp_path / "CAMPAIGN_RESULT.txt")
    payload = {
        "passed": False,
        "stages": [
            {
                "stage": "lift",
                "passed": True,
                "plan_success": True,
                "support_escape": {"passed": True},
            },
            {
                "stage": "place_above_support",
                "passed": False,
                "plan_success": False,
                "execution_status": None,
                "failure": {
                    "code": "optimization_failed",
                    "details": {"native_status": "IK Fail"},
                },
            },
        ],
    }
    _write_json(tmp_path / "manipulation-seed-2.json", payload)
    result = classify_campaign(tmp_path)
    assert result["primary_decision"] == "PLACEMENT_YAW_IK_RECOVERY"


def test_seed1_target_collision_classifies_despite_dynamic_unsafe(
    tmp_path: Path,
) -> None:
    _write_campaign(tmp_path / "CAMPAIGN_RESULT.txt")
    _write_json(
        tmp_path / "manipulation-seed-1.json",
        {
            "passed": False,
            "stages": [
                {
                    "stage": "approach",
                    "passed": False,
                    "planner_id": "robotwin-mplib-rrt",
                    "plan_success": True,
                    "dynamic_safe": False,
                    "collision_safe": False,
                    "execution_status": None,
                    "collision_validation": {
                        "collisions": [
                            {"matched_environment_roles": ["target"]}
                        ],
                    },
                }
            ],
        },
    )
    result = classify_campaign(tmp_path)
    assert result["primary_decision"] == "SEED1_ACQUISITION_TARGET_COLLISION"
    assert result["decisions"][0]["seeds"] == [1]



def test_classifies_preflight_upstream_source_mismatch_across_seeds(
    tmp_path: Path,
) -> None:
    _write_campaign(tmp_path / "CAMPAIGN_RESULT.txt")
    for seed in (0, 1, 2):
        _write_json(
            tmp_path / f"manipulation-seed-{seed}.json",
            {
                "seed": seed,
                "passed": False,
                "stages": [],
                "error": "RuntimeError: pinned upstream task source hash mismatch; fail closed",
            },
        )

    result = classify_campaign(tmp_path)
    assert result["primary_decision"] == "UPSTREAM_TASK_SOURCE_ATTESTATION_MISMATCH"
    assert result["decisions"][0]["seeds"] == [0, 1, 2]
    assert result["campaign_accepted"] is False
