"""Simulator-facing observation contracts.

These types keep camera/proprioception data explicit without importing SAPIEN, NumPy, or RoboTwin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any

from .motion import JointState


class CameraRole(str, Enum):
    """Semantic camera roles used by the project."""

    HEAD = "head"
    WRIST = "wrist"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One RGB frame plus stable metadata; payload remains backend-owned."""

    camera_id: str
    role: CameraRole
    frame_id: str
    width_px: int
    height_px: int
    timestamp_s: float
    payload: Any

    def __post_init__(self) -> None:
        if not self.camera_id.strip() or not self.frame_id.strip():
            raise ValueError("camera_id and frame_id must be non-empty")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("camera dimensions must be positive")
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class GripperState:
    """Normalized RoboTwin-compatible gripper position in [0, 1]."""

    normalized_position: float

    def __post_init__(self) -> None:
        if not isfinite(self.normalized_position):
            raise ValueError("normalized_position must be finite")
        if not 0.0 <= self.normalized_position <= 1.0:
            raise ValueError("normalized_position must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SimulatorObservation:
    """Typed snapshot returned by a simulator adapter."""

    sequence_id: int
    timestamp_s: float
    joint_state: JointState
    gripper_state: GripperState
    cameras: tuple[CameraFrame, ...]
    scene_revision: str
    transform_revision: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if not self.scene_revision.strip() or not self.transform_revision.strip():
            raise ValueError("scene_revision and transform_revision must be non-empty")
        if (
            self.joint_state.timestamp_s is not None
            and abs(self.joint_state.timestamp_s - self.timestamp_s) > 1e-9
        ):
            raise ValueError("joint_state timestamp must match observation timestamp")
        camera_ids = tuple(frame.camera_id for frame in self.cameras)
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("camera ids must be unique")
        if any(abs(frame.timestamp_s - self.timestamp_s) > 1e-9 for frame in self.cameras):
            raise ValueError("camera timestamps must match observation timestamp")

    def camera(self, camera_id: str) -> CameraFrame:
        """Return a frame by id or raise KeyError."""

        for frame in self.cameras:
            if frame.camera_id == camera_id:
                return frame
        raise KeyError(camera_id)
