from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MATRIX = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "scenarios"
    / "sprint1_matrix.json"
)


def test_sprint1_matrix_has_unique_ids_and_both_suites() -> None:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert data["schema"] == "sprint1-scenario-matrix-v1"
    scenarios = data["scenarios"]
    ids = [item["scenario_id"] for item in scenarios]

    assert len(ids) == len(set(ids))
    assert {"regression", "adversarial"} <= {item["suite"] for item in scenarios}
    assert all(item["expected_behavior"] for item in scenarios)
    assert all(item["required_evidence"] for item in scenarios)


def test_matrix_marks_native_only_cases_explicitly() -> None:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    scenarios = data["scenarios"]

    native = [item for item in scenarios if item["native_required"]]
    offline = [item for item in scenarios if not item["native_required"]]

    assert native
    assert offline
    assert any(item["expected_behavior"] == "episode_success" for item in native)
    assert any(item["expected_behavior"] == "reobserve" for item in offline)


def test_offline_runner_covers_every_non_native_matrix_scenario() -> None:
    runner_path = MATRIX.parents[2] / "scripts" / "run_sprint1_offline_scenarios.py"
    spec = importlib.util.spec_from_file_location("sprint1_offline_runner", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    expected = {
        item["scenario_id"]
        for item in data["scenarios"]
        if not item["native_required"]
    }

    records = module.run()
    actual = {record.scenario_id for record in records}

    assert actual == expected
    assert len(records) == len(expected)
    assert all(record.runtime.value == "offline_fake" for record in records)
    assert all(record.outcome.value == "pass" for record in records)
