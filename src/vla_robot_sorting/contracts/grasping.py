"""Project-owned full-6-DoF grasp contracts for Sprint 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any

from .models import Pose6D
from .perception import SceneSnapshot


class GraspFailureCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    TARGET_NOT_FOUND = "target_not_found"
    SCENE_STALE = "scene_stale"
    TRANSFORM_STALE = "transform_stale"
    NO_GEOMETRIC_CANDIDATE = "no_geometric_candidate"
    NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"


@dataclass(frozen=True, slots=True)
class ParallelJawGripper:
    """Simplified geometry used by the first project grasp generator."""

    min_width_m: float
    max_width_m: float
    finger_thickness_m: float
    finger_height_m: float
    pregrasp_standoff_m: float
    jaw_clearance_m: float = 0.004
    contact_center_offset_m: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.min_width_m,
            self.max_width_m,
            self.finger_thickness_m,
            self.finger_height_m,
            self.pregrasp_standoff_m,
            self.jaw_clearance_m,
            self.contact_center_offset_m,
        )
        if not all(isfinite(value) and value >= 0 for value in values):
            raise ValueError("gripper dimensions must be finite and non-negative")
        if self.min_width_m >= self.max_width_m:
            raise ValueError("min_width_m must be < max_width_m")
        if self.finger_thickness_m <= 0 or self.finger_height_m <= 0:
            raise ValueError("finger dimensions must be positive")
        if self.pregrasp_standoff_m <= 0:
            raise ValueError("pregrasp_standoff_m must be positive")


@dataclass(frozen=True, slots=True)
class GraspRequest:
    """Version-locked request for one target object."""

    object_id: str
    scene_revision: str
    transform_revision: str
    gripper: ParallelJawGripper
    minimum_table_clearance_m: float = 0.003

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("GraspRequest.object_id must be non-empty")
        if not self.scene_revision.strip() or not self.transform_revision.strip():
            raise ValueError("GraspRequest revisions must be non-empty")
        if not isfinite(self.minimum_table_clearance_m) or self.minimum_table_clearance_m < 0:
            raise ValueError("minimum_table_clearance_m must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class GraspCandidate:
    """Object-specific full-SE(3) parallel-jaw candidate."""

    candidate_id: str
    object_id: str
    grasp_pose: Pose6D
    pregrasp_pose: Pose6D
    approach_direction: tuple[float, float, float]
    closing_direction: tuple[float, float, float]
    jaw_width_m: float
    source: str
    geometry_source: str
    scene_revision: str
    transform_revision: str
    compliance_profile: str
    raw_quality: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.object_id.strip():
            raise ValueError("candidate_id/object_id must be non-empty")
        if self.grasp_pose.frame_id != self.pregrasp_pose.frame_id:
            raise ValueError("grasp and pregrasp poses must use the same frame")
        for label, vector in (
            ("approach_direction", self.approach_direction),
            ("closing_direction", self.closing_direction),
        ):
            if len(vector) != 3 or not all(isfinite(value) for value in vector):
                raise ValueError(f"{label} must contain three finite values")
            norm_sq = sum(value * value for value in vector)
            if abs(norm_sq - 1.0) > 1e-3:
                raise ValueError(f"{label} must be unit length")
        if not isfinite(self.jaw_width_m) or self.jaw_width_m < 0:
            raise ValueError("jaw_width_m must be finite and non-negative")
        if not self.source.strip() or not self.geometry_source.strip():
            raise ValueError("source/geometry_source must be non-empty")
        if not self.scene_revision.strip() or not self.transform_revision.strip():
            raise ValueError("candidate revisions must be non-empty")
        if not self.compliance_profile.strip():
            raise ValueError("compliance_profile must be non-empty")
        if not 0.0 <= self.raw_quality <= 1.0:
            raise ValueError("raw_quality must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class GraspFailure:
    code: GraspFailureCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("GraspFailure.message must be non-empty")


@dataclass(frozen=True, slots=True)
class GraspGenerationResult:
    candidates: tuple[GraspCandidate, ...]
    failure: GraspFailure | None = None

    def __post_init__(self) -> None:
        if self.candidates and self.failure is not None:
            raise ValueError("successful grasp generation cannot also contain a failure")
        if not self.candidates and self.failure is None:
            raise ValueError("empty grasp generation requires a failure")

    @property
    def succeeded(self) -> bool:
        return bool(self.candidates)


@dataclass(frozen=True, slots=True)
class FeasibilityEvaluation:
    """Planner/collision-derived factors kept separate from geometric generation."""

    candidate_id: str
    reachable: bool
    collision_free: bool
    clearance_margin_m: float
    approach_clearance_m: float
    grasp_quality: float
    motion_cost: float
    object_identity_confidence: float

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        for label, value in (
            ("clearance_margin_m", self.clearance_margin_m),
            ("approach_clearance_m", self.approach_clearance_m),
            ("motion_cost", self.motion_cost),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        for label, value in (
            ("grasp_quality", self.grasp_quality),
            ("object_identity_confidence", self.object_identity_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate_id: str
    total: float
    factors: dict[str, float]
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if not 0.0 <= self.total <= 1.0:
            raise ValueError("CandidateScore.total must be in [0, 1]")
        if any(not 0.0 <= value <= 1.0 for value in self.factors.values()):
            raise ValueError("all score factors must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RankedGraspSet:
    scene_revision: str
    transform_revision: str
    scores: tuple[CandidateScore, ...]

    def __post_init__(self) -> None:
        if not self.scene_revision.strip() or not self.transform_revision.strip():
            raise ValueError("RankedGraspSet revisions must be non-empty")


def validate_request_against_scene(request: GraspRequest, scene: SceneSnapshot) -> None:
    if request.scene_revision != scene.scene_revision:
        raise ValueError("scene revision is stale")
    if request.transform_revision != scene.transform_revision:
        raise ValueError("transform revision is stale")
