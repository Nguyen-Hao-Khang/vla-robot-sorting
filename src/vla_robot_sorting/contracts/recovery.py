"""Closed-loop adaptation and recovery contracts for S1.5."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class SceneChangeKind(str, Enum):
    NONE = "none"
    TRANSFORM_CHANGED = "transform_changed"
    TARGET_MOVED = "target_moved"
    DESTINATION_MOVED = "destination_moved"
    TARGET_OCCLUDED = "target_occluded"
    TARGET_MISSING = "target_missing"
    DESTINATION_MISSING = "destination_missing"


class RecoveryAction(str, Enum):
    CONTINUE = "continue"
    REPLAN = "replan"
    REOBSERVE = "reobserve"
    REGRASP = "regrasp"
    REASON = "reason"
    ABORT = "abort"


class RecoveryTrigger(str, Enum):
    SCENE_CHANGE = "scene_change"
    GRASP_FAILURE = "grasp_failure"
    MOTION_FAILURE = "motion_failure"
    EXECUTION_REPLAN = "execution_replan"
    SEMANTIC_UNCERTAINTY = "semantic_uncertainty"


@dataclass(frozen=True, slots=True)
class SceneChange:
    kind: SceneChangeKind
    previous_scene_revision: str
    current_scene_revision: str
    target_object_id: str | None = None
    destination_id: str | None = None
    position_delta_m: float | None = None
    orientation_delta_rad: float | None = None

    def __post_init__(self) -> None:
        if not self.previous_scene_revision.strip() or not self.current_scene_revision.strip():
            raise ValueError("scene revisions must be non-empty")
        for label, value in (
            ("position_delta_m", self.position_delta_m),
            ("orientation_delta_rad", self.orientation_delta_rad),
        ):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{label} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RecoveryEvent:
    trigger: RecoveryTrigger
    observed_at_s: float
    scene_change: SceneChange | None = None
    recoverable: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.observed_at_s) or self.observed_at_s < 0:
            raise ValueError("observed_at_s must be finite and non-negative")
        if self.trigger is RecoveryTrigger.SCENE_CHANGE and self.scene_change is None:
            raise ValueError("SCENE_CHANGE event requires scene_change")
        if self.reason is not None and not self.reason.strip():
            raise ValueError("reason must be non-empty when present")


@dataclass(frozen=True, slots=True)
class RecoveryBudget:
    max_replans: int = 3
    max_reobserves: int = 2
    max_regrasps: int = 2
    max_reasoning_calls: int = 1

    def __post_init__(self) -> None:
        if min(
            self.max_replans,
            self.max_reobserves,
            self.max_regrasps,
            self.max_reasoning_calls,
        ) < 0:
            raise ValueError("recovery budgets must be non-negative")


@dataclass(frozen=True, slots=True)
class RecoveryState:
    replans: int = 0
    reobserves: int = 0
    regrasps: int = 0
    reasoning_calls: int = 0

    def __post_init__(self) -> None:
        if min(self.replans, self.reobserves, self.regrasps, self.reasoning_calls) < 0:
            raise ValueError("recovery counters must be non-negative")


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str
    next_state: RecoveryState

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("RecoveryDecision.reason must be non-empty")


@dataclass(frozen=True, slots=True)
class RecoveryTiming:
    observed_at_s: float
    detected_at_s: float
    decision_at_s: float
    resolved_at_s: float | None = None

    def __post_init__(self) -> None:
        values = [self.observed_at_s, self.detected_at_s, self.decision_at_s]
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("timing values must be finite and non-negative")
        if not self.observed_at_s <= self.detected_at_s <= self.decision_at_s:
            raise ValueError("timing must be monotonic through decision")
        if self.resolved_at_s is not None and (
            not isfinite(self.resolved_at_s) or self.resolved_at_s < self.decision_at_s
        ):
            raise ValueError("resolved_at_s must be finite and >= decision_at_s")

    @property
    def detection_latency_s(self) -> float:
        return self.detected_at_s - self.observed_at_s

    @property
    def decision_latency_s(self) -> float:
        return self.decision_at_s - self.detected_at_s

    @property
    def recovery_latency_s(self) -> float | None:
        if self.resolved_at_s is None:
            return None
        return self.resolved_at_s - self.detected_at_s
