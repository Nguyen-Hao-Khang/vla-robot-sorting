from __future__ import annotations

import json
from pathlib import Path

from vla_robot_sorting.evidence import audit_s1_a007_safe_failure_probe


def _case(fault: str) -> dict[str, object]:
    return {
        "fault": fault,
        "passed": True,
        "motion_cycle_status": "planning_failed",
        "planner_id": "a007-controlled-fault-injection",
        "planner_failure_code": fault,
        "trajectory_created": False,
        "execution_created": False,
        "parameterizer_calls": 0,
        "path_safety_calls": 0,
        "trajectory_safety_calls": 0,
        "executor_calls": 0,
        "physics_steps": 0,
        "qpos_unchanged": True,
    }


def _payload() -> dict[str, object]:
    return {
        "schema": "panthera-native-safe-failure-v1",
        "scenario_id": "S1-A-007",
        "task": "move_pillbottle_pad",
        "task_config": "move_pillbottle_pad_panthera_single",
        "seed": 0,
        "actual_joint_positions_rad": [0.0] * 6,
        "cases": [_case("timeout"), _case("collision")],
        "pass": True,
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_valid_a007_probe_passes(tmp_path: Path) -> None:
    probe = tmp_path / "safe-failure.json"
    _write(probe, _payload())

    result = audit_s1_a007_safe_failure_probe(probe)

    assert result.passed
    assert result.failures == ()


def test_executor_call_breaks_fail_closed_gate(tmp_path: Path) -> None:
    payload = _payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    timeout = cases[0]
    assert isinstance(timeout, dict)
    timeout["executor_calls"] = 1
    probe = tmp_path / "safe-failure.json"
    _write(probe, payload)

    result = audit_s1_a007_safe_failure_probe(probe)

    assert not result.passed
    assert any(gate.name == "fault-timeout" for gate in result.failures)


def test_physics_step_breaks_no_execution_gate(tmp_path: Path) -> None:
    payload = _payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    collision = cases[1]
    assert isinstance(collision, dict)
    collision["physics_steps"] = 1
    probe = tmp_path / "safe-failure.json"
    _write(probe, payload)

    result = audit_s1_a007_safe_failure_probe(probe)

    assert not result.passed
    assert any(gate.name == "fault-collision" for gate in result.failures)


def test_summary_true_cannot_hide_wrong_failure_code(tmp_path: Path) -> None:
    payload = _payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    timeout = cases[0]
    assert isinstance(timeout, dict)
    timeout["planner_failure_code"] = "collision"
    probe = tmp_path / "safe-failure.json"
    _write(probe, payload)

    result = audit_s1_a007_safe_failure_probe(probe)

    assert not result.passed
    assert any(gate.name == "fault-timeout" for gate in result.failures)
