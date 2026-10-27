from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_sprint1_readiness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sprint1_readiness_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readiness_keeps_native_and_cloud_work_explicitly_pending(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    module = _load_module()
    monkeypatch.setattr(module, "S16_MANIPULATION", tmp_path / "not-promoted.json")
    report = module.build_report()

    assert report["schema"] == "sprint1-readiness-v1"
    assert report["accepted_count"] == report["accepted_total"] == 11
    assert all(report["accepted"].values())
    assert report["accepted"]["fixed_12_case_matrix_complete"] is True
    assert report["accepted"]["S1.3_target_sync_B5_B7"] is True
    assert report["accepted"]["integrated_runtime_grounding_bounded_pregrasp"] is True
    assert report["accepted"]["integrated_destination_moved_recovery"] is True
    assert report["accepted"]["integrated_resource_profile"] is True
    assert report["native_pending_count"] == 1
    assert set(report["native_pending"]) == {"project_owned_grasp_lift_place_sort"}
    assert "RUN_SPRINT1_MANIPULATION_CAMPAIGN.sh" in report["native_pending"][
        "project_owned_grasp_lift_place_sort"
    ]
    cloud = report["cloud_api_pending"]["gemini_live_semantic_evaluation"]
    assert cloud["key_present_in_current_environment"] is False
    assert "Do not reboot solely for it" in cloud["note"]


def test_readiness_reports_key_presence_without_treating_it_as_native(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-placeholder")
    module = _load_module()
    monkeypatch.setattr(module, "S16_MANIPULATION", tmp_path / "not-promoted.json")
    report = module.build_report()

    cloud = report["cloud_api_pending"]["gemini_live_semantic_evaluation"]
    assert cloud["key_present_in_current_environment"] is True
    assert report["native_pending_count"] == 1


def _candidate_payload() -> dict[str, object]:
    return {
        "schema": "native-manipulation-closure-candidate-v1",
        "gate_id": "S1.6-MANIPULATION",
        "classification": "PASS_AUDITED_CANDIDATE",
        "linux_git_head": "abc123",
        "result_dir": "native-result",
        "fixed_seeds": [0, 1, 2],
        "probe_pass_count": 3,
        "independent_audit_pass_count": 3,
        "seeds": [
            {
                "seed": seed,
                "probe_passed": True,
                "independent_audit_passed": True,
                "final_phase": "complete",
                "task_success_final": True,
            }
            for seed in (0, 1, 2)
        ],
        "promotion_required_on_windows": True,
    }


def test_readiness_clears_native_pending_for_valid_closure_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    candidate = tmp_path / "manipulation-closure-candidate.json"
    candidate.write_text(json.dumps(_candidate_payload()), encoding="utf-8")

    module = _load_module()
    monkeypatch.setattr(module, "S16_MANIPULATION", tmp_path / "not-promoted.json")
    report = module.build_report(candidate)

    assert report["accepted_count"] == report["accepted_total"] == 11
    assert all(report["accepted"].values())
    assert report["native_pending_count"] == 0
    assert report["native_pending"] == {}
    closure = report["native_closure_candidate"]
    assert closure["provided"] is True
    assert closure["ready"] is True
    assert closure["classification"] == "PASS_AUDITED_CANDIDATE"
    assert closure["promotion_required_on_windows"] is True
    assert closure["linux_git_head"] == "abc123"


def test_readiness_rejects_incomplete_manipulation_candidate(tmp_path: Path) -> None:
    payload = _candidate_payload()
    payload["probe_pass_count"] = 2
    candidate = tmp_path / "bad-candidate.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="probe count"):
        _load_module().build_report(candidate)


def _durable_manifest_payload() -> dict[str, object]:
    return {
        "schema": "accepted-native-gate-v1",
        "gate_id": "S1.6-MANIPULATION",
        "classification": "PASS_AUDITED",
        "accepted_at": "2026-10-08",
        "review": "docs/reviews/manipulation-pass.md",
        "source_bridge_relative": "incoming/session",
        "linux_git_head": "abc123",
        "candidate_sha256": "c" * 64,
        "fixed_seeds": [0, 1, 2],
        "probe_pass_count": 3,
        "independent_audit_pass_count": 3,
        "seeds": [
            {
                "seed": seed,
                "probe_sha256": "a" * 64,
                "audit_log_sha256": "b" * 64,
                "final_phase": "complete",
                "task_success_final": True,
            }
            for seed in (0, 1, 2)
        ],
    }


def test_readiness_reports_12_of_12_after_durable_manipulation_promotion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    manifest = tmp_path / "native_s1_6_manipulation.json"
    manifest.write_text(json.dumps(_durable_manifest_payload()), encoding="utf-8")
    monkeypatch.setattr(module, "S16_MANIPULATION", manifest)

    report = module.build_report()

    assert report["accepted_count"] == report["accepted_total"] == 12
    assert all(report["accepted"].values())
    assert report["accepted"]["project_owned_grasp_lift_place_sort"] is True
    assert report["native_pending_count"] == 0
    closure = report["native_closure_candidate"]
    assert closure["ready"] is True
    assert closure["source"] == "durable_manifest"
    assert closure["classification"] == "PASS_AUDITED"
    assert closure["promotion_required_on_windows"] is False
