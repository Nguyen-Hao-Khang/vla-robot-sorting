"""Final single-boot planning policy: safe stop coverage without false acceptance."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from vla_robot_sorting.evidence.manipulation import _native_curobo_interior_planner_gate
from vla_robot_sorting.planning.native_curobo_interior import (
    CUROBO_POSITION_LIMIT_CLIP_RAD,
    PINNED_CUROBO_YAML_LF_SHA256,
)

REPO = Path(__file__).resolve().parents[2]


def _runner():
    p = REPO / "scripts/run_final_manipulation_session.py"
    spec = importlib.util.spec_from_file_location("native_final_runner", p)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_stop(seed: int):
    return {
        "seed": seed,
        "error": "RuntimeError: B5 did not produce a safe selected grasp",
        "stages": [],
        "task_success_final": None,
        "B5": {
            "passed": False,
            "selected": None,
            "eligible_candidate_ids": [],
            "evaluations": [{"candidate": {"candidate_id": "native"}}],
        },
    }


def test_seed1_seed2_known_safe_planning_stop_continues_other_seed_without_retry():
    runner = _runner()
    assert runner.known_diagnostic_failure(_safe_stop(1), 1)
    assert runner.known_diagnostic_failure(_safe_stop(2), 2)
    assert not runner.known_diagnostic_failure(_safe_stop(1), 2)
    assert not runner.known_diagnostic_failure(_safe_stop(0), 0)


def test_planning_stop_is_never_accepted_when_claims_are_spoofed():
    runner = _runner()
    for field, value in (
        ("stages", [{"stage": "pregrasp"}]),
        ("task_success_final", True),
        ("error", "unknown error"),
    ):
        candidate = _safe_stop(1)
        candidate[field] = value
        assert not runner.known_diagnostic_failure(candidate, 1)
    bad = _safe_stop(1)
    bad["B5"]["eligible_candidate_ids"] = ["forged"]
    assert not runner.known_diagnostic_failure(bad, 1)
    bad = _safe_stop(1)
    bad["B5"]["evaluations"] = []
    assert not runner.known_diagnostic_failure(bad, 1)


def test_native_curobo_interpolation_policy_requires_exact_pinned_provenance():
    policy = {
        "schema": "s1m.native-curobo-interior-planner.v1",
        "method": "scoped_in_memory_native_config_loader",
        "pinned_yaml_sha256": PINNED_CUROBO_YAML_LF_SHA256,
        "position_limit_clip_rad": CUROBO_POSITION_LIMIT_CLIP_RAD,
        "interior_retract_config_rad": [0.0, 0.05, 0.0625, 0.0, 0.0, 0.0],
        "config_intercepts": 2,
        "factory_calls": 2,
        "single_arm_search_ik_seeds": 64,
        "single_arm_search_trajopt_seeds": 2,
        "original_urdf_and_vendor_yaml_modified": False,
        "runtime_joint_limits_waived": False,
        "controller_eligible": False,
    }
    payload = {
        "native_curobo_interior_contract_version": "scoped_native_cspace_clip_v1",
        "native_curobo_interior_planner": policy,
    }
    assert _native_curobo_interior_planner_gate(payload).passed
    for key, value in (
        ("position_limit_clip_rad", 0.0),
        ("config_intercepts", 1),
        ("factory_calls", 3),
        ("pinned_yaml_sha256", "forged"),
        ("original_urdf_and_vendor_yaml_modified", True),
        ("runtime_joint_limits_waived", True),
    ):
        altered = {
            "native_curobo_interior_contract_version": payload[
                "native_curobo_interior_contract_version"
            ],
            "native_curobo_interior_planner": {**policy, key: value},
        }
        assert not _native_curobo_interior_planner_gate(altered).passed
    assert not _native_curobo_interior_planner_gate(
        {"native_curobo_interior_contract_version": "unexpected"}
    ).passed


def test_runner_prefers_historical_grasp_but_no_longer_requires_it():
    runner = (REPO / "scripts/run_final_manipulation_session.py").read_text(encoding="utf8")
    helper = (REPO / "scripts/linux/probe_panthera_grasp_recovery.py").read_text(encoding="utf8")
    assert '"--preferred-candidate-id"' in runner
    assert '"--candidate-id"' not in runner
    assert "preferred_candidate_id in eligible_candidate_ids" in helper
    assert "selected_candidate_id is None" in helper
    assert "selected_candidate_id in eligible_candidate_ids" in helper
