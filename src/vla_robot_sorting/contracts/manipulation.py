"""Project-owned manipulation phase and gripper contracts for Sprint 1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ManipulationPhase(str, Enum):
    PREGRASP = "pregrasp"
    APPROACH = "approach"
    CLOSE = "close"
    VERIFY_GRASP = "verify_grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    RELEASE = "release"
    RETREAT = "retreat"
    COMPLETE = "complete"
    RECOVERY_REQUIRED = "recovery_required"


class ManipulationSignal(str, Enum):
    MOTION_COMPLETED = "motion_completed"
    GRIPPER_CLOSED = "gripper_closed"
    GRASP_CONFIRMED = "grasp_confirmed"
    GRIPPER_OPENED = "gripper_opened"
    MOTION_FAILED = "motion_failed"
    GRASP_FAILED = "grasp_failed"
    SCENE_STALE = "scene_stale"


@dataclass(frozen=True, slots=True)
class GripperCommand:
    """Revision-locked normalized gripper command.

    normalized_position follows RoboTwin semantics used by the project:
    0 = fully closed, 1 = fully open.
    """

    normalized_position: float
    compliance_profile: str
    force_scale: float
    scene_revision: str
    transform_revision: str

    def __post_init__(self) -> None:
        if not isfinite(self.normalized_position) or not 0.0 <= self.normalized_position <= 1.0:
            raise ValueError("normalized_position must be finite and in [0, 1]")
        if not self.compliance_profile.strip():
            raise ValueError("compliance_profile must be non-empty")
        if not isfinite(self.force_scale) or not 0.0 < self.force_scale <= 1.0:
            raise ValueError("force_scale must be finite and in (0, 1]")
        if not self.scene_revision.strip() or not self.transform_revision.strip():
            raise ValueError("gripper command revisions must be non-empty")


@dataclass(frozen=True, slots=True)
class GripperExecutionResult:
    completed: bool
    actual_normalized_position: float
    contact_observed: bool
    holding_object: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not isfinite(self.actual_normalized_position)
            or not 0.0 <= self.actual_normalized_position <= 1.0
        ):
            raise ValueError("actual_normalized_position must be finite and in [0, 1]")
        if not self.completed and not (self.reason and self.reason.strip()):
            raise ValueError("failed gripper execution requires a reason")


@dataclass(frozen=True, slots=True)
class ManipulationTransition:
    previous_phase: ManipulationPhase
    next_phase: ManipulationPhase
    signal: ManipulationSignal
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("transition reason must be non-empty")
