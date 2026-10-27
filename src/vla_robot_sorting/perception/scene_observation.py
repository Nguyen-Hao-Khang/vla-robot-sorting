"""Provenance-locked camera observations converted into canonical scene state.

This module performs calibration validation and RGB-D geometric projection only.
It does not detect objects, synthesize missing depth, or certify robot motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt

from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.perception import (
    DestinationRegion,
    GeometrySource,
    ObjectGeometry,
    ObjectVisibility,
    SceneObject,
    SceneSnapshot,
)

Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
Quaternion = tuple[float, float, float, float]
Pixel = tuple[float, float]


class SceneEvidenceKind(str, Enum):
    """Where an object's identity/pose evidence came from."""

    SIM_GROUND_TRUTH = "sim_ground_truth"
    RECORDED_SIM_OBSERVATION = "recorded_sim_observation"
    IMAGE_ESTIMATE = "image_estimate"
    FIXTURE = "fixture"


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Camera intrinsics and camera-to-world rigid transform for one timestamp."""

    camera_id: str
    camera_frame_id: str
    world_frame_id: str
    width_px: int
    height_px: int
    intrinsic_matrix: Matrix3
    camera_to_world: Matrix4
    transform_revision: str
    timestamp_s: float
    calibration_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.camera_id.strip()
            or not self.camera_frame_id.strip()
            or not self.world_frame_id.strip()
        ):
            raise ValueError("camera and frame ids must be non-empty")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("camera dimensions must be positive")
        if not self.transform_revision.strip():
            raise ValueError("transform_revision must be non-empty")
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("calibration timestamp must be finite and non-negative")
        _validate_sha256(self.calibration_sha256, "calibration_sha256")
        _validate_matrix(self.intrinsic_matrix, 3, "intrinsic_matrix")
        _validate_matrix(self.camera_to_world, 4, "camera_to_world")
        k = self.intrinsic_matrix
        if k[0][0] <= 0 or k[1][1] <= 0 or k[2] != (0.0, 0.0, 1.0):
            raise ValueError(
                "intrinsic_matrix must have positive focal lengths and canonical last row"
            )
        if abs(k[0][1]) > 1e-9 or abs(k[1][0]) > 1e-9:
            raise ValueError(
                "intrinsic_matrix must have zero skew for this pixel back-projection"
            )
        transform = self.camera_to_world
        if any(abs(transform[3][i] - target) > 1e-6 for i, target in enumerate((0, 0, 0, 1))):
            raise ValueError("camera_to_world must be a homogeneous transform")
        rotation = tuple(tuple(transform[i][j] for j in range(3)) for i in range(3))
        _validate_rotation(rotation)


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")


def _validate_matrix(matrix: tuple[tuple[float, ...], ...], size: int, label: str) -> None:
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError(f"{label} must be {size}x{size}")
    if not all(isfinite(value) for row in matrix for value in row):
        raise ValueError(f"{label} must contain finite values")


def _validate_rotation(rotation: tuple[tuple[float, float, float], ...]) -> None:
    for row in rotation:
        if abs(sum(value * value for value in row) - 1.0) > 1e-4:
            raise ValueError("camera_to_world rotation must be orthonormal")
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(sum(rotation[i][k] * rotation[j][k] for k in range(3))) > 1e-4:
                raise ValueError("camera_to_world rotation must be orthonormal")
    det = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(det - 1.0) > 1e-4:
        raise ValueError("camera_to_world rotation must be right-handed")


@dataclass(frozen=True, slots=True)
class SceneObservationRecord:
    """One stored RGB (optionally RGB-D) frame with verifiable artifact lineage."""

    observation_id: str
    source: SceneEvidenceKind
    scene_revision: str
    timestamp_s: float
    rgb_sha256: str
    calibration: CameraCalibration
    artifact_sha256: str
    depth_sha256: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observation_id.strip() or not self.scene_revision.strip():
            raise ValueError("observation_id and scene_revision must be non-empty")
        if self.source not in {
            SceneEvidenceKind.RECORDED_SIM_OBSERVATION,
            SceneEvidenceKind.SIM_GROUND_TRUTH,
            SceneEvidenceKind.FIXTURE,
        }:
            raise ValueError(
                "observation source must be recorded_sim_observation, sim_ground_truth or fixture"
            )
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("observation timestamp must be finite and non-negative")
        _validate_sha256(self.rgb_sha256, "rgb_sha256")
        _validate_sha256(self.artifact_sha256, "artifact_sha256")
        if self.depth_sha256 is not None:
            _validate_sha256(self.depth_sha256, "depth_sha256")
        if abs(self.calibration.timestamp_s - self.timestamp_s) > 1e-6:
            raise ValueError("camera calibration timestamp must match observation timestamp")


@dataclass(frozen=True, slots=True)
class ObjectEvidence:
    """Per-object evidence; RGB estimates require measured depth and image coordinates."""

    object_id: str
    semantic_labels: tuple[str, ...]
    geometry: ObjectGeometry
    geometry_source: GeometrySource
    evidence_kind: SceneEvidenceKind
    visibility: ObjectVisibility
    identity_confidence: float
    pose_world: Pose6D | None = None
    pixel_center_uv: Pixel | None = None
    depth_m: float | None = None
    orientation_camera_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not self.object_id.strip() or not self.semantic_labels:
            raise ValueError("object identity and semantic labels must be non-empty")
        if not isfinite(self.identity_confidence) or not 0 <= self.identity_confidence <= 1:
            raise ValueError("identity_confidence must be finite and in [0, 1]")
        if len(self.orientation_camera_xyzw) != 4 or not all(
            isfinite(value) for value in self.orientation_camera_xyzw
        ):
            raise ValueError("orientation_camera_xyzw must contain four finite values")
        norm = sqrt(sum(value * value for value in self.orientation_camera_xyzw))
        if norm < 1e-8 or abs(norm - 1.0) > 1e-3:
            raise ValueError("orientation_camera_xyzw must be normalized")


def _quaternion_from_rotation(rotation: tuple[tuple[float, float, float], ...]) -> Quaternion:
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = sqrt(trace + 1.0) * 2
        q = ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    elif m00 > m11 and m00 > m22:
        s = sqrt(1.0 + m00 - m11 - m22) * 2
        q = (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    elif m11 > m22:
        s = sqrt(1.0 + m11 - m00 - m22) * 2
        q = ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    else:
        s = sqrt(1.0 + m22 - m00 - m11) * 2
        q = ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    norm = sqrt(sum(value * value for value in q))
    return tuple(value / norm for value in q)  # type: ignore[return-value]


def _multiply_quaternions(a: Quaternion, b: Quaternion) -> Quaternion:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _image_pose_world(
    evidence: ObjectEvidence, calibration: CameraCalibration, depth_sha256: str | None
) -> Pose6D:
    if depth_sha256 is None:
        raise ValueError("RGB image estimate requires a depth frame")
    if evidence.visibility is not ObjectVisibility.VISIBLE:
        raise ValueError("an occluded or absent object cannot produce a current RGB estimate")
    if evidence.pixel_center_uv is None or evidence.depth_m is None:
        raise ValueError("RGB image estimate requires pixel coordinates and depth")
    u, v = evidence.pixel_center_uv
    z = evidence.depth_m
    if (
        not isfinite(u)
        or not isfinite(v)
        or not 0 <= u < calibration.width_px
        or not 0 <= v < calibration.height_px
    ):
        raise ValueError("object pixel center must lie inside calibrated image bounds")
    if not isfinite(z) or z <= 0:
        raise ValueError("object depth must be finite and positive")
    k = calibration.intrinsic_matrix
    x = (u - k[0][2]) * z / k[0][0]
    y = (v - k[1][2]) * z / k[1][1]
    transform = calibration.camera_to_world
    point = tuple(
        sum(transform[row][column] * (x, y, z)[column] for column in range(3)) + transform[row][3]
        for row in range(3)
    )
    rotation = tuple(tuple(transform[i][j] for j in range(3)) for i in range(3))
    camera_q = _quaternion_from_rotation(rotation)
    world_q = _multiply_quaternions(camera_q, evidence.orientation_camera_xyzw)
    norm = sqrt(sum(value * value for value in world_q))
    normalized_q = tuple(value / norm for value in world_q)
    return Pose6D(point, normalized_q, calibration.world_frame_id)  # type: ignore[arg-type]


def build_scene_snapshot(
    record: SceneObservationRecord,
    objects: tuple[ObjectEvidence, ...],
    *,
    max_age_s: float,
    now_s: float,
    table_height_m: float = 0.0,
    destinations: tuple[DestinationRegion, ...] = (),
) -> SceneSnapshot:
    """Validate frame provenance and convert evidence into GeometryFirstGraspGenerator input."""

    if not isfinite(max_age_s) or max_age_s < 0 or not isfinite(now_s) or now_s < 0:
        raise ValueError("max_age_s and now_s must be finite and non-negative")
    age = now_s - record.timestamp_s
    if age < 0 or age > max_age_s:
        raise ValueError("scene observation is stale or from the future")
    if not isfinite(table_height_m):
        raise ValueError("table_height_m must be finite")
    calibration = record.calibration
    scene_objects: list[SceneObject] = []
    object_ids: set[str] = set()
    compatible_sources = {
        SceneEvidenceKind.FIXTURE: {
            SceneEvidenceKind.FIXTURE, SceneEvidenceKind.IMAGE_ESTIMATE
        },
        SceneEvidenceKind.RECORDED_SIM_OBSERVATION: {
            SceneEvidenceKind.RECORDED_SIM_OBSERVATION, SceneEvidenceKind.IMAGE_ESTIMATE
        },
        SceneEvidenceKind.SIM_GROUND_TRUTH: {SceneEvidenceKind.SIM_GROUND_TRUTH},
    }
    for evidence in objects:
        if evidence.evidence_kind not in compatible_sources[record.source]:
            raise ValueError("object evidence source conflicts with observation source")
        if evidence.object_id in object_ids:
            raise ValueError("object ids must be unique")
        object_ids.add(evidence.object_id)
        if evidence.evidence_kind is SceneEvidenceKind.IMAGE_ESTIMATE:
            if evidence.pose_world is not None:
                raise ValueError("image estimate must be projected, not supplied as world pose")
            if evidence.geometry_source is GeometrySource.SIM_GROUND_TRUTH:
                raise ValueError("image estimate cannot use privileged simulator geometry")
            pose = _image_pose_world(evidence, calibration, record.depth_sha256)
        else:
            if evidence.pose_world is None:
                raise ValueError("non-image object evidence requires an explicit world pose")
            if evidence.pose_world.frame_id != calibration.world_frame_id:
                raise ValueError("object world pose frame does not match calibrated world frame")
            if evidence.evidence_kind is SceneEvidenceKind.SIM_GROUND_TRUTH:
                if evidence.geometry_source is not GeometrySource.SIM_GROUND_TRUTH:
                    raise ValueError(
                        "privileged object evidence must retain sim_ground_truth geometry source"
                    )
            elif evidence.geometry_source is GeometrySource.SIM_GROUND_TRUTH:
                raise ValueError("non-privileged evidence cannot use sim_ground_truth geometry")
            pose = evidence.pose_world
        metadata: dict[str, object] = {
            "evidence_kind": evidence.evidence_kind.value,
            "observation_id": record.observation_id,
            "observation_timestamp_s": record.timestamp_s,
            "identity_confidence": evidence.identity_confidence,
            "control_eligible": False,
            "calibration_sha256": calibration.calibration_sha256,
        }
        if evidence.pixel_center_uv is not None:
            metadata["pixel_center_uv"] = evidence.pixel_center_uv
        if evidence.depth_m is not None:
            metadata["depth_m"] = evidence.depth_m
        scene_objects.append(
            SceneObject(
                object_id=evidence.object_id,
                pose=pose,
                geometry=evidence.geometry,
                geometry_source=evidence.geometry_source,
                semantic_labels=evidence.semantic_labels,
                identity_confidence=evidence.identity_confidence,
                visibility=evidence.visibility,
                metadata=metadata,
            )
        )
    if any(item.pose.frame_id != calibration.world_frame_id for item in destinations):
        raise ValueError("destination pose frame does not match calibrated world frame")
    source_name = {
        SceneEvidenceKind.FIXTURE: "fixture",
        SceneEvidenceKind.RECORDED_SIM_OBSERVATION: "recorded_native",
        SceneEvidenceKind.SIM_GROUND_TRUTH: "sim_ground_truth",
    }[record.source]
    return SceneSnapshot(
        scene_revision=record.scene_revision,
        transform_revision=calibration.transform_revision,
        frame_id=calibration.world_frame_id,
        objects=tuple(scene_objects),
        destinations=destinations,
        table_height_m=table_height_m,
        metadata={
            **record.metadata,
            "observation_source": source_name,
            "observation_revision": record.observation_id,
            "evidence_kind": record.source.value,
            "timestamp_s": record.timestamp_s,
            "scene_revision": record.scene_revision,
            "transform_revision": calibration.transform_revision,
            "camera_id": calibration.camera_id,
            "camera_frame_id": calibration.camera_frame_id,
            "rgb_sha256": record.rgb_sha256,
            "depth_sha256": record.depth_sha256,
            "artifact_sha256": record.artifact_sha256,
            "calibration_sha256": calibration.calibration_sha256,
            "camera_intrinsics": calibration.intrinsic_matrix,
            "camera_to_world": calibration.camera_to_world,
            "control_eligible": False,
        },
    )
