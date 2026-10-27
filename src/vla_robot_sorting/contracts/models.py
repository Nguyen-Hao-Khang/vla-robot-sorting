"""Core cross-subsystem data contracts.

These contracts intentionally avoid simulator/model-specific dependencies so they can be tested on
Windows and implemented by Linux-only adapters later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from typing import Any


class SemanticSkill(str, Enum):
    """High-level actions allowed to cross the semantic/control boundary."""

    PICK = "pick"
    PLACE = "place"
    REGRASP = "regrasp"
    RETRACT = "retract"
    WAIT = "wait"
    ABORT = "abort"


@dataclass(frozen=True, slots=True)
class Pose6D:
    """Position in metres and orientation quaternion in xyzw order."""

    position_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    frame_id: str

    def __post_init__(self) -> None:
        if len(self.position_xyz) != 3:
            raise ValueError("Pose6D.position_xyz must contain exactly 3 values")
        if len(self.orientation_xyzw) != 4:
            raise ValueError("Pose6D.orientation_xyzw must contain exactly 4 values")
        values = (*self.position_xyz, *self.orientation_xyzw)
        if not all(isfinite(value) for value in values):
            raise ValueError("Pose6D values must be finite")
        if not self.frame_id.strip():
            raise ValueError("Pose6D.frame_id must be non-empty")

        q_norm = sqrt(sum(value * value for value in self.orientation_xyzw))
        if q_norm < 1e-8:
            raise ValueError("Pose6D quaternion must have non-zero norm")
        if abs(q_norm - 1.0) > 1e-3:
            raise ValueError("Pose6D quaternion must be normalized")


@dataclass(frozen=True, slots=True)
class SkillCommand:
    """Validated semantic intent; never a raw motor command."""

    skill: SemanticSkill
    target_object: str | None = None
    destination: str | None = None
    semantic_constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.skill in {SemanticSkill.PICK, SemanticSkill.REGRASP} and not self.target_object:
            raise ValueError(f"{self.skill.value} requires target_object")
        if self.skill is SemanticSkill.PLACE and not self.destination:
            raise ValueError("place requires destination")


@dataclass(frozen=True, slots=True)
class GraspTarget:
    """A ranked geometric grasp candidate produced downstream of semantic reasoning."""

    object_id: str
    pose: Pose6D
    approach_vector: tuple[float, float, float]
    gripper_width_m: float
    confidence: float
    compliance_profile: str | None = None

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("object_id must be non-empty")
        if len(self.approach_vector) != 3 or not all(isfinite(v) for v in self.approach_vector):
            raise ValueError("approach_vector must contain three finite values")
        if self.gripper_width_m < 0 or not isfinite(self.gripper_width_m):
            raise ValueError("gripper_width_m must be a finite non-negative value")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
