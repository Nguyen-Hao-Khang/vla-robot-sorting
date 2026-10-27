"""Sprint-1 scenario-matrix coverage validation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vla_robot_sorting.contracts.evaluation import (
    EvaluationRuntime,
    ScenarioOutcome,
    ScenarioRecord,
)


def load_scenario_matrix(path: str | Path) -> tuple[dict[str, Any], ...]:
    matrix_path = Path(path)
    raw = json.loads(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != "sprint1-scenario-matrix-v1":
        raise ValueError("unsupported Sprint-1 scenario matrix")
    scenarios = raw.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Sprint-1 scenario matrix is empty")
    if not all(isinstance(item, dict) for item in scenarios):
        raise ValueError("Sprint-1 scenario matrix entries must be objects")
    return tuple(scenarios)


def validate_records_against_matrix(
    matrix: Iterable[dict[str, Any]],
    records: Iterable[ScenarioRecord],
) -> None:
    specs = tuple(matrix)
    items = tuple(records)
    spec_by_id = {str(item["scenario_id"]): item for item in specs}
    if len(spec_by_id) != len(specs):
        raise ValueError("duplicate scenario_id in Sprint-1 matrix")

    record_by_id = {item.scenario_id: item for item in items}
    if len(record_by_id) != len(items):
        raise ValueError("duplicate scenario_id in Sprint-1 records")

    expected_ids = set(spec_by_id)
    actual_ids = set(record_by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"scenario coverage mismatch: missing={missing}, extra={extra}")

    for scenario_id, spec in spec_by_id.items():
        record = record_by_id[scenario_id]
        if record.suite != spec["suite"]:
            raise ValueError(f"{scenario_id}: suite mismatch")
        if record.expected_behavior != spec["expected_behavior"]:
            raise ValueError(f"{scenario_id}: expected_behavior mismatch")
        native_required = bool(spec["native_required"])
        if native_required and record.runtime is not EvaluationRuntime.NATIVE_SIM:
            raise ValueError(f"{scenario_id}: native scenario cannot use {record.runtime.value}")
        if not native_required and record.runtime is not EvaluationRuntime.OFFLINE_FAKE:
            raise ValueError(f"{scenario_id}: offline scenario must use offline_fake runtime")


def scenario_coverage_report(
    matrix: Iterable[dict[str, Any]],
    records: Iterable[ScenarioRecord],
) -> dict[str, Any]:
    specs = tuple(matrix)
    items = tuple(records)
    validate_records_against_matrix(specs, items)

    outcomes = Counter(item.outcome.value for item in items)
    runtimes = Counter(item.runtime.value for item in items)
    suites = Counter(item.suite for item in items)
    pending = sorted(
        item.scenario_id
        for item in items
        if item.outcome is ScenarioOutcome.SKIPPED
    )
    failures = sorted(
        item.scenario_id
        for item in items
        if item.outcome is ScenarioOutcome.FAIL
    )
    native_ids = {
        str(spec["scenario_id"])
        for spec in specs
        if bool(spec["native_required"])
    }
    native_pass = sorted(
        item.scenario_id
        for item in items
        if item.scenario_id in native_ids and item.outcome is ScenarioOutcome.PASS
    )

    return {
        "matrix_total": len(specs),
        "record_total": len(items),
        "exact_matrix_coverage": True,
        "outcomes": dict(sorted(outcomes.items())),
        "runtimes": dict(sorted(runtimes.items())),
        "suites": dict(sorted(suites.items())),
        "native_required_total": len(native_ids),
        "native_pass_count": len(native_pass),
        "native_pass_ids": native_pass,
        "pending_ids": pending,
        "failure_ids": failures,
        "complete": not pending and not failures,
    }
