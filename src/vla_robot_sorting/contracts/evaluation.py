"""Machine-readable Sprint 1 evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any


class EvaluationRuntime(str, Enum):
    OFFLINE_FAKE = "offline_fake"
    NATIVE_SIM = "native_sim"
    PHYSICAL = "physical"


class ScenarioOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ResourceMetrics:
    wall_time_s: float | None = None
    peak_host_rss_mib: float | None = None
    peak_vram_mib: float | None = None
    peak_gpu_utilization_percent: float | None = None
    deadline_misses: int = 0

    def __post_init__(self) -> None:
        for label, value in (
            ("wall_time_s", self.wall_time_s),
            ("peak_host_rss_mib", self.peak_host_rss_mib),
            ("peak_vram_mib", self.peak_vram_mib),
            ("peak_gpu_utilization_percent", self.peak_gpu_utilization_percent),
        ):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{label} must be finite and non-negative")
        if self.deadline_misses < 0:
            raise ValueError("deadline_misses must be non-negative")


@dataclass(frozen=True, slots=True)
class ScenarioRecord:
    scenario_id: str
    suite: str
    runtime: EvaluationRuntime
    outcome: ScenarioOutcome
    expected_behavior: str
    observed_behavior: str | None
    failure_stage: str | None = None
    failure_code: str | None = None
    semantic_correct: bool | None = None
    grounding_correct: bool | None = None
    grasp_generated: bool | None = None
    motion_succeeded: bool | None = None
    recovery_succeeded: bool | None = None
    detection_latency_s: float | None = None
    recovery_latency_s: float | None = None
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("scenario_id", self.scenario_id),
            ("suite", self.suite),
            ("expected_behavior", self.expected_behavior),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be non-empty")
        for label, value in (
            ("observed_behavior", self.observed_behavior),
            ("failure_stage", self.failure_stage),
            ("failure_code", self.failure_code),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{label} must be non-empty when present")
        for label, value in (
            ("detection_latency_s", self.detection_latency_s),
            ("recovery_latency_s", self.recovery_latency_s),
        ):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{label} must be finite and non-negative")
        if self.outcome is ScenarioOutcome.PASS and self.observed_behavior is None:
            raise ValueError("PASS records require observed_behavior")
        if self.outcome is ScenarioOutcome.FAIL and self.failure_stage is None:
            raise ValueError("FAIL records require failure_stage")


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    total: int
    passed: int
    failed: int
    skipped: int
    native_sim_total: int
    offline_total: int
    failure_taxonomy: dict[str, int]
    stage_failures: dict[str, int]
    measured_detection_latency_count: int
    mean_detection_latency_s: float | None
    measured_recovery_latency_count: int
    mean_recovery_latency_s: float | None
    total_deadline_misses: int

    def __post_init__(self) -> None:
        counts = (
            self.total,
            self.passed,
            self.failed,
            self.skipped,
            self.native_sim_total,
            self.offline_total,
            self.measured_detection_latency_count,
            self.measured_recovery_latency_count,
            self.total_deadline_misses,
        )
        if any(value < 0 for value in counts):
            raise ValueError("evaluation summary counts must be non-negative")
        if self.passed + self.failed + self.skipped != self.total:
            raise ValueError("outcome counts must sum to total")
