from __future__ import annotations

import pytest

from vla_robot_sorting.contracts.evaluation import (
    EvaluationRuntime,
    ResourceMetrics,
    ScenarioOutcome,
    ScenarioRecord,
)
from vla_robot_sorting.evaluation.aggregate import summarize


def record(
    scenario_id: str,
    *,
    outcome: ScenarioOutcome,
    runtime: EvaluationRuntime = EvaluationRuntime.OFFLINE_FAKE,
    failure_code: str | None = None,
    failure_stage: str | None = None,
    detection_latency_s: float | None = None,
    recovery_latency_s: float | None = None,
    deadline_misses: int = 0,
) -> ScenarioRecord:
    return ScenarioRecord(
        scenario_id=scenario_id,
        suite="test",
        runtime=runtime,
        outcome=outcome,
        expected_behavior="safe",
        observed_behavior="safe" if outcome is ScenarioOutcome.PASS else None,
        failure_stage=failure_stage,
        failure_code=failure_code,
        detection_latency_s=detection_latency_s,
        recovery_latency_s=recovery_latency_s,
        resources=ResourceMetrics(deadline_misses=deadline_misses),
    )


def test_summary_keeps_missing_metrics_out_of_denominator() -> None:
    summary = summarize(
        (
            record(
                "a",
                outcome=ScenarioOutcome.PASS,
                detection_latency_s=0.2,
                recovery_latency_s=0.8,
                deadline_misses=1,
            ),
            record("b", outcome=ScenarioOutcome.SKIPPED),
            record(
                "c",
                outcome=ScenarioOutcome.FAIL,
                runtime=EvaluationRuntime.NATIVE_SIM,
                failure_code="collision",
                failure_stage="planning",
            ),
        )
    )

    assert summary.total == 3
    assert (summary.passed, summary.failed, summary.skipped) == (1, 1, 1)
    assert summary.native_sim_total == 1
    assert summary.offline_total == 2
    assert summary.failure_taxonomy == {"collision": 1}
    assert summary.stage_failures == {"planning": 1}
    assert summary.measured_detection_latency_count == 1
    assert summary.mean_detection_latency_s == pytest.approx(0.2)
    assert summary.measured_recovery_latency_count == 1
    assert summary.mean_recovery_latency_s == pytest.approx(0.8)
    assert summary.total_deadline_misses == 1


def test_fail_record_requires_failure_stage() -> None:
    with pytest.raises(ValueError, match="failure_stage"):
        record("bad", outcome=ScenarioOutcome.FAIL)
