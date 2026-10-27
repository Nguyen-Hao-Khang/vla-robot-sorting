"""Aggregate actual scenario records without inventing missing measurements."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from vla_robot_sorting.contracts.evaluation import (
    EvaluationRuntime,
    EvaluationSummary,
    ResourceMetrics,
    ScenarioOutcome,
    ScenarioRecord,
)


def summarize(records: Iterable[ScenarioRecord]) -> EvaluationSummary:
    items = tuple(records)
    outcomes = Counter(item.outcome for item in items)
    failures = Counter(
        item.failure_code
        for item in items
        if item.outcome is ScenarioOutcome.FAIL and item.failure_code is not None
    )
    stages = Counter(
        item.failure_stage
        for item in items
        if item.outcome is ScenarioOutcome.FAIL and item.failure_stage is not None
    )
    detection = [
        item.detection_latency_s
        for item in items
        if item.detection_latency_s is not None
    ]
    recovery = [
        item.recovery_latency_s
        for item in items
        if item.recovery_latency_s is not None
    ]
    return EvaluationSummary(
        total=len(items),
        passed=outcomes[ScenarioOutcome.PASS],
        failed=outcomes[ScenarioOutcome.FAIL],
        skipped=outcomes[ScenarioOutcome.SKIPPED],
        native_sim_total=sum(item.runtime is EvaluationRuntime.NATIVE_SIM for item in items),
        offline_total=sum(item.runtime is EvaluationRuntime.OFFLINE_FAKE for item in items),
        failure_taxonomy=dict(sorted(failures.items())),
        stage_failures=dict(sorted(stages.items())),
        measured_detection_latency_count=len(detection),
        mean_detection_latency_s=(sum(detection) / len(detection) if detection else None),
        measured_recovery_latency_count=len(recovery),
        mean_recovery_latency_s=(sum(recovery) / len(recovery) if recovery else None),
        total_deadline_misses=sum(item.resources.deadline_misses for item in items),
    )


def record_from_dict(data: dict[str, object]) -> ScenarioRecord:
    resources_data = data.get("resources", {})
    if not isinstance(resources_data, dict):
        raise ValueError("resources must be an object")
    resources = ResourceMetrics(**resources_data)
    values = dict(data)
    values["runtime"] = EvaluationRuntime(str(values["runtime"]))
    values["outcome"] = ScenarioOutcome(str(values["outcome"]))
    values["resources"] = resources
    return ScenarioRecord(**values)  # type: ignore[arg-type]


def load_jsonl(path: Path) -> tuple[ScenarioRecord, ...]:
    records: list[ScenarioRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number}: record must be an object")
        records.append(record_from_dict(raw))
    return tuple(records)


def summary_to_dict(summary: EvaluationSummary) -> dict[str, object]:
    return {
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "skipped": summary.skipped,
        "native_sim_total": summary.native_sim_total,
        "offline_total": summary.offline_total,
        "failure_taxonomy": summary.failure_taxonomy,
        "stage_failures": summary.stage_failures,
        "measured_detection_latency_count": summary.measured_detection_latency_count,
        "mean_detection_latency_s": summary.mean_detection_latency_s,
        "measured_recovery_latency_count": summary.measured_recovery_latency_count,
        "mean_recovery_latency_s": summary.mean_recovery_latency_s,
        "total_deadline_misses": summary.total_deadline_misses,
    }


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def record_to_dict(record: ScenarioRecord) -> dict[str, object]:
    """Serialize one validated ScenarioRecord without leaking Enum reprs."""

    value = _jsonable(asdict(record))
    assert isinstance(value, dict)
    return value
