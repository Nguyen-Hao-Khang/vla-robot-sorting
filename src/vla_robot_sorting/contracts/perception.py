"""Scene/object contracts for S1.3 perception, grounding and grasping."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from typing import Any

from .models import Pose6D


class GeometrySource(str, Enum):
    """Provenance of object geometry used by grasp generation."""

    SIM_GROUND_TRUTH = "sim_ground_truth"
    RECONSTRUCTED_DEPTH = "reconstructed_depth"
    ESTIMATED_PROXY = "estimated_proxy"


class ObjectVisibility(str, Enum):
    VISIBLE = "visible"
    OCCLUDED = "occluded"
    NOT_PRESENT = "not_present"


@dataclass(frozen=True, slots=True)
class BoxGeometry:
    """Oriented box dimensions in the object's local x/y/z axes."""

    size_xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.size_xyz_m) != 3:
            raise ValueError("BoxGeometry.size_xyz_m must contain exactly 3 values")
        if not all(isfinite(value) and value > 0 for value in self.size_xyz_m):
            raise ValueError("BoxGeometry dimensions must be finite and positive")


@dataclass(frozen=True, slots=True)
class CylinderGeometry:
    """Cylinder with an explicit symmetry axis in the object's local frame."""

    radius_m: float
    height_m: float
    axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("CylinderGeometry.radius_m must be finite and positive")
        if not isfinite(self.height_m) or self.height_m <= 0:
            raise ValueError("CylinderGeometry.height_m must be finite and positive")
        if len(self.axis_local) != 3 or not all(
            isfinite(value) for value in self.axis_local
        ):
            raise ValueError("CylinderGeometry.axis_local must contain three finite values")
        norm = sqrt(sum(value * value for value in self.axis_local))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError("CylinderGeometry.axis_local must be a unit vector")


ObjectGeometry = BoxGeometry | CylinderGeometry


@dataclass(frozen=True, slots=True)
class SceneObject:
    """One grounded scene object with explicit geometry and provenance."""

    object_id: str
    pose: Pose6D
    geometry: ObjectGeometry
    geometry_source: GeometrySource
    semantic_labels: tuple[str, ...]
    identity_confidence: float
    fragile: bool = False
    visibility: ObjectVisibility = ObjectVisibility.VISIBLE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("SceneObject.object_id must be non-empty")
        if not self.semantic_labels or any(not label.strip() for label in self.semantic_labels):
            raise ValueError("SceneObject.semantic_labels must be non-empty")
        if not 0.0 <= self.identity_confidence <= 1.0:
            raise ValueError("SceneObject.identity_confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class DestinationRegion:
    """A semantic placement destination represented by a pose and tolerance."""

    destination_id: str
    pose: Pose6D
    tolerance_xyz_m: tuple[float, float, float]
    semantic_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.destination_id.strip():
            raise ValueError("DestinationRegion.destination_id must be non-empty")
        if not self.semantic_labels or any(not label.strip() for label in self.semantic_labels):
            raise ValueError("DestinationRegion.semantic_labels must be non-empty")
        if len(self.tolerance_xyz_m) != 3 or not all(
            isfinite(value) and value > 0 for value in self.tolerance_xyz_m
        ):
            raise ValueError("DestinationRegion tolerances must be three positive values")


@dataclass(frozen=True, slots=True)
class SceneSnapshot:
    """Versioned scene state consumed by grounding/grasp/motion layers."""

    scene_revision: str
    transform_revision: str
    frame_id: str
    objects: tuple[SceneObject, ...]
    destinations: tuple[DestinationRegion, ...] = ()
    table_height_m: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scene_revision.strip() or not self.transform_revision.strip():
            raise ValueError("SceneSnapshot revisions must be non-empty")
        if not self.frame_id.strip():
            raise ValueError("SceneSnapshot.frame_id must be non-empty")
        if not isfinite(self.table_height_m):
            raise ValueError("SceneSnapshot.table_height_m must be finite")
        object_ids = [item.object_id for item in self.objects]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("SceneSnapshot object ids must be unique")
        destination_ids = [item.destination_id for item in self.destinations]
        if len(set(destination_ids)) != len(destination_ids):
            raise ValueError("SceneSnapshot destination ids must be unique")
        if any(item.pose.frame_id != self.frame_id for item in self.objects):
            raise ValueError("Scene object poses must use SceneSnapshot.frame_id")
        if any(item.pose.frame_id != self.frame_id for item in self.destinations):
            raise ValueError("Destination poses must use SceneSnapshot.frame_id")

    def object(self, object_id: str) -> SceneObject:
        for item in self.objects:
            if item.object_id == object_id:
                return item
        raise KeyError(object_id)

    def destination(self, destination_id: str) -> DestinationRegion:
        for item in self.destinations:
            if item.destination_id == destination_id:
                return item
        raise KeyError(destination_id)
