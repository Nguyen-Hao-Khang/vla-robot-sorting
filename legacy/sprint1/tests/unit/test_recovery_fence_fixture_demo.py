"""Verify honest classification of S1-R fixture replay."""

from __future__ import annotations

import json

from scripts.demo_recovery_fence_fixture import evaluate_fixture_matrix, main


def test_recovery_fixture_matrix_vetoes_all_uncertified_motion():
    result = evaluate_fixture_matrix()
    assert result["source_classification"] == "synthetic_fixture"
    assert result["case_count"] == 12
    assert result["vetoed_count"] == 9
    assert result["allowed_later_checks_count"] == 3
    assert result["native_simulation_calls"] == 0
    assert result["motor_commands_issued"] == 0
    assert result["native_evidence_accepted"] is False
    assert result["scene_recovery_action"] == "replan"
    assert result["invalid_occlusion_proof_result"] == "bad_recovery_proof"
    assert all(
        case["motor_command_authorized"] is False for case in result["scenarios"]
    )
    by_case = {row["scenario"]: row for row in result["scenarios"]}
    assert by_case["hard_contact_sticky_abort"]["gate_code"] == "aborted"
    assert by_case["moved_target_old_plan_veto"]["gate_code"] == "revision_mismatch"
    assert by_case["occluded_unreplanned_veto"]["gate_code"] == "recovery_pending"
    assert by_case["moved_target_new_plan_preflight"]["gate_code"] == (
        "downstream_checks_required"
    )


def test_recovery_fixture_cli_writes_source_typed_json(tmp_path, monkeypatch):
    out = tmp_path / "s1r-recovery.json"
    monkeypatch.setattr(
        "sys.argv", ["demo_recovery_fence_fixture.py", "--output", str(out)]
    )
    assert main() == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == evaluate_fixture_matrix()
