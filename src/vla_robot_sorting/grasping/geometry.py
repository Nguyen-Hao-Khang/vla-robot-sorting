"""Deterministic geometry-first full-6-DoF parallel-jaw grasp generation."""

from __future__ import annotations

from math import cos, isfinite, pi, sin, sqrt

from vla_robot_sorting.contracts.grasping import (
    GraspCandidate,
    GraspFailure,
    GraspFailureCode,
    GraspGenerationResult,
    GraspRequest,
)
from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.perception import (
    BoxGeometry,
    CylinderGeometry,
    GeometrySource,
    SceneObject,
    SceneSnapshot,
)

Vector3 = tuple[float, float, float]


def _dot(a: Vector3, b: Vector3) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(vector: Vector3) -> Vector3:
    norm = sqrt(_dot(vector, vector))
    if norm < 1e-12 or not isfinite(norm):
        raise ValueError("cannot normalize a zero/non-finite vector")
    return tuple(value / norm for value in vector)  # type: ignore[return-value]


def _rotate_vector_xyzw(quaternion: tuple[float, float, float, float], vector: Vector3) -> Vector3:
    x, y, z, w = quaternion
    vx, vy, vz = vector
    # R(q) @ v for a normalized xyzw quaternion.
    return (
        (1 - 2 * (y * y + z * z)) * vx
        + 2 * (x * y - z * w) * vy
        + 2 * (x * z + y * w) * vz,
        2 * (x * y + z * w) * vx
        + (1 - 2 * (x * x + z * z)) * vy
        + 2 * (y * z - x * w) * vz,
        2 * (x * z - y * w) * vx
        + 2 * (y * z + x * w) * vy
        + (1 - 2 * (x * x + y * y)) * vz,
    )


def _quaternion_xyzw_from_basis(
    x_axis: Vector3,
    y_axis: Vector3,
) -> tuple[float, float, float, float]:
    """Build xyzw quaternion from world-space local x/y basis axes."""

    x_axis = _normalize(x_axis)
    z_axis = _normalize(_cross(x_axis, y_axis))
    y_axis = _normalize(_cross(z_axis, x_axis))

    m00, m10, m20 = x_axis
    m01, m11, m21 = y_axis
    m02, m12, m22 = z_axis
    trace = m00 + m11 + m22

    if trace > 0:
        s = sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    norm = sqrt(x * x + y * y + z * z + w * w)
    return (x / norm, y / norm, z / norm, w / norm)


def _candidate_pose(
    center: Vector3,
    approach_world: Vector3,
    closing_world: Vector3,
    frame_id: str,
) -> Pose6D:
    return Pose6D(
        position_xyz=center,
        orientation_xyzw=_quaternion_xyzw_from_basis(approach_world, closing_world),
        frame_id=frame_id,
    )


def _pregrasp_pose(grasp: Pose6D, approach_world: Vector3, standoff_m: float) -> Pose6D:
    return Pose6D(
        position_xyz=tuple(
            point - standoff_m * axis
            for point, axis in zip(grasp.position_xyz, approach_world, strict=True)
        ),
        orientation_xyzw=grasp.orientation_xyzw,
        frame_id=grasp.frame_id,
    )


def _table_clearance(
    contact_center: Vector3,
    pregrasp_contact_center: Vector3,
    finger_height_m: float,
    table_height_m: float,
) -> float:
    lowest_center = min(contact_center[2], pregrasp_contact_center[2])
    return lowest_center - 0.5 * finger_height_m - table_height_m


def _width_quality(width_m: float, minimum_m: float, maximum_m: float) -> float:
    if maximum_m <= minimum_m:
        return 0.0
    normalized = (maximum_m - width_m) / (maximum_m - minimum_m)
    return max(0.0, min(1.0, normalized))


class GeometryFirstGraspGenerator:
    """Generate deterministic box/cylinder candidates in the current scene frame."""

    def __init__(
        self,
        *,
        cylinder_radial_samples: int = 8,
        cylinder_side_grasp_height_fractions: tuple[float, ...] = (0.0,),
        box_side_grasp_height_fractions: tuple[float, ...] = (0.0,),
        box_parallel_jaw_roll_variants: bool = False,
    ) -> None:
        if cylinder_radial_samples < 4:
            raise ValueError("cylinder_radial_samples must be >= 4")
        if not cylinder_side_grasp_height_fractions:
            raise ValueError("cylinder_side_grasp_height_fractions must not be empty")
        cylinder_fractions = tuple(
            float(value) for value in cylinder_side_grasp_height_fractions
        )
        if any(
            not isfinite(value) or not 0.0 <= value <= 1.0
            for value in cylinder_fractions
        ):
            raise ValueError("cylinder side-grasp height fractions must be finite in [0, 1]")
        if not box_side_grasp_height_fractions:
            raise ValueError("box_side_grasp_height_fractions must not be empty")
        box_fractions = tuple(float(value) for value in box_side_grasp_height_fractions)
        if any(
            not isfinite(value) or not 0.0 <= value <= 1.0
            for value in box_fractions
        ):
            raise ValueError("box side-grasp height fractions must be finite in [0, 1]")
        self._cylinder_radial_samples = cylinder_radial_samples
        self._cylinder_side_grasp_height_fractions = tuple(
            dict.fromkeys(cylinder_fractions)
        )
        self._box_side_grasp_height_fractions = tuple(dict.fromkeys(box_fractions))
        self._box_parallel_jaw_roll_variants = bool(box_parallel_jaw_roll_variants)

    def generate(self, request: GraspRequest, scene: SceneSnapshot) -> GraspGenerationResult:
        if request.scene_revision != scene.scene_revision:
            return GraspGenerationResult(
                (),
                GraspFailure(
                    GraspFailureCode.SCENE_STALE,
                    "scene revision does not match the grasp request",
                    {"request": request.scene_revision, "scene": scene.scene_revision},
                ),
            )
        if request.transform_revision != scene.transform_revision:
            return GraspGenerationResult(
                (),
                GraspFailure(
                    GraspFailureCode.TRANSFORM_STALE,
                    "transform revision does not match the grasp request",
                    {"request": request.transform_revision, "scene": scene.transform_revision},
                ),
            )

        try:
            target = scene.object(request.object_id)
        except KeyError:
            return GraspGenerationResult(
                (),
                GraspFailure(
                    GraspFailureCode.TARGET_NOT_FOUND,
                    f"object {request.object_id!r} is absent from scene {scene.scene_revision}",
                ),
            )

        if isinstance(target.geometry, BoxGeometry):
            candidates = self._box_candidates(target, request, scene)
        elif isinstance(target.geometry, CylinderGeometry):
            candidates = self._cylinder_candidates(target, request, scene)
        else:
            candidates = []

        if not candidates:
            return GraspGenerationResult(
                (),
                GraspFailure(
                    GraspFailureCode.NO_GEOMETRIC_CANDIDATE,
                    "no candidate passed jaw-width and table-clearance geometry gates",
                    {"object_id": target.object_id},
                ),
            )
        return GraspGenerationResult(tuple(candidates))

    def _make_candidate(
        self,
        *,
        target: SceneObject,
        request: GraspRequest,
        scene: SceneSnapshot,
        candidate_id: str,
        approach_local: Vector3,
        closing_local: Vector3,
        jaw_width_m: float,
        raw_quality: float,
        metadata: dict[str, object],
        contact_center_world: Vector3 | None = None,
    ) -> GraspCandidate | None:
        gripper = request.gripper
        if not gripper.min_width_m <= jaw_width_m <= gripper.max_width_m:
            return None

        approach_world = _normalize(
            _rotate_vector_xyzw(target.pose.orientation_xyzw, approach_local)
        )
        closing_world = _normalize(
            _rotate_vector_xyzw(target.pose.orientation_xyzw, closing_local)
        )
        if abs(_dot(approach_world, closing_world)) > 1e-6:
            return None

        contact_center = contact_center_world or target.pose.position_xyz
        gripper_frame_center = tuple(
            point - gripper.contact_center_offset_m * axis
            for point, axis in zip(contact_center, approach_world, strict=True)
        )
        pregrasp_contact_center = tuple(
            point - gripper.pregrasp_standoff_m * axis
            for point, axis in zip(contact_center, approach_world, strict=True)
        )
        grasp = _candidate_pose(
            gripper_frame_center,
            approach_world,
            closing_world,
            scene.frame_id,
        )
        pregrasp = _pregrasp_pose(grasp, approach_world, gripper.pregrasp_standoff_m)
        table_clearance = _table_clearance(
            contact_center,
            pregrasp_contact_center,
            gripper.finger_height_m,
            scene.table_height_m,
        )
        if table_clearance < request.minimum_table_clearance_m:
            return None

        compliance = "fragile-soft" if target.fragile else "standard"
        candidate_metadata = {
            **metadata,
            "table_clearance_m": table_clearance,
            "contact_center_position_xyz_m": contact_center,
            "pregrasp_contact_center_position_xyz_m": pregrasp_contact_center,
            "gripper_frame_contact_offset_m": gripper.contact_center_offset_m,
            "identity_confidence": target.identity_confidence,
            "privileged_geometry": target.geometry_source is GeometrySource.SIM_GROUND_TRUTH,
        }
        if target.fragile:
            candidate_metadata["recommended_force_scale"] = 0.45
        else:
            candidate_metadata["recommended_force_scale"] = 1.0

        return GraspCandidate(
            candidate_id=candidate_id,
            object_id=target.object_id,
            grasp_pose=grasp,
            pregrasp_pose=pregrasp,
            approach_direction=approach_world,
            closing_direction=closing_world,
            jaw_width_m=jaw_width_m,
            source="geometry_first",
            geometry_source=target.geometry_source.value,
            scene_revision=scene.scene_revision,
            transform_revision=scene.transform_revision,
            compliance_profile=compliance,
            raw_quality=max(0.0, min(1.0, raw_quality)),
            metadata=candidate_metadata,
        )


    def _box_candidates(
        self,
        target: SceneObject,
        request: GraspRequest,
        scene: SceneSnapshot,
    ) -> list[GraspCandidate]:
        assert isinstance(target.geometry, BoxGeometry)
        size = target.geometry.size_xyz_m
        axes: tuple[Vector3, ...] = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        world_axes = tuple(
            _normalize(_rotate_vector_xyzw(target.pose.orientation_xyzw, axis))
            for axis in axes
        )
        up_index = max(range(3), key=lambda index: abs(world_axes[index][2]))
        up_axis_world = world_axes[up_index]
        if up_axis_world[2] < 0.0:
            up_axis_world = tuple(-value for value in up_axis_world)  # type: ignore[assignment]
        max_up_offset_m = max(
            0.0,
            0.5 * size[up_index] - 0.5 * request.gripper.finger_height_m,
        )
        closing_signs = (
            (-1.0, 1.0) if self._box_parallel_jaw_roll_variants else (1.0,)
        )
        candidates: list[GraspCandidate] = []

        for closing_index, closing_axis_base in enumerate(axes):
            jaw_width = size[closing_index] + request.gripper.jaw_clearance_m
            quality = _width_quality(
                jaw_width,
                request.gripper.min_width_m,
                request.gripper.max_width_m,
            )
            for approach_index, approach_axis in enumerate(axes):
                if approach_index == closing_index:
                    continue
                side_grasp = up_index not in (closing_index, approach_index)
                height_fractions = (
                    self._box_side_grasp_height_fractions
                    if side_grasp and max_up_offset_m > 1e-9
                    else (0.0,)
                )
                for approach_sign in (-1.0, 1.0):
                    approach = tuple(
                        approach_sign * value for value in approach_axis
                    )
                    for closing_sign in closing_signs:
                        closing_axis = tuple(
                            closing_sign * value for value in closing_axis_base
                        )
                        for height_fraction in height_fractions:
                            center_offset_m = height_fraction * max_up_offset_m
                            grasp_center = tuple(
                                target.pose.position_xyz[index]
                                + center_offset_m * up_axis_world[index]
                                for index in range(3)
                            )
                            candidate_id = (
                                f"{target.object_id}:box:c{closing_index}:"
                                f"a{approach_index}:s{int(approach_sign):+d}"
                            )
                            if closing_sign < 0.0:
                                candidate_id += ":roll180"
                            if height_fraction > 0.0:
                                candidate_id += (
                                    f":up{round(height_fraction * 100.0):02d}"
                                )
                            candidate = self._make_candidate(
                                target=target,
                                request=request,
                                scene=scene,
                                candidate_id=candidate_id,
                                approach_local=approach,  # type: ignore[arg-type]
                                closing_local=closing_axis,  # type: ignore[arg-type]
                                jaw_width_m=jaw_width,
                                raw_quality=quality,
                                metadata={
                                    "shape": "box",
                                    "closing_axis_local_index": closing_index,
                                    "closing_sign": closing_sign,
                                    "approach_axis_local_index": approach_index,
                                    "approach_sign": approach_sign,
                                    "up_axis_local_index": up_index,
                                    "grasp_height_fraction": height_fraction,
                                    "grasp_center_up_offset_m": center_offset_m,
                                    "max_grasp_center_up_offset_m": max_up_offset_m,
                                },
                                contact_center_world=grasp_center,  # type: ignore[arg-type]
                            )
                            if candidate is not None:
                                candidates.append(candidate)
        return candidates

    def _cylinder_candidates(
        self,
        target: SceneObject,
        request: GraspRequest,
        scene: SceneSnapshot,
    ) -> list[GraspCandidate]:
        assert isinstance(target.geometry, CylinderGeometry)
        jaw_width = 2.0 * target.geometry.radius_m + request.gripper.jaw_clearance_m
        width_quality = _width_quality(
            jaw_width,
            request.gripper.min_width_m,
            request.gripper.max_width_m,
        )
        axis = _normalize(target.geometry.axis_local)
        reference: Vector3 = (1.0, 0.0, 0.0)
        if abs(_dot(axis, reference)) > 0.9:
            reference = (0.0, 1.0, 0.0)
        projection = _dot(reference, axis)
        radial_u = _normalize(
            tuple(
                reference[index] - projection * axis[index]
                for index in range(3)
            )  # type: ignore[arg-type]
        )
        radial_v = _normalize(_cross(axis, radial_u))

        axis_world = _normalize(
            _rotate_vector_xyzw(target.pose.orientation_xyzw, axis)
        )
        if axis_world[2] >= 0.0:
            up_axis_local = axis
            up_axis_world = axis_world
        else:
            up_axis_local = tuple(-value for value in axis)  # type: ignore[assignment]
            up_axis_world = tuple(-value for value in axis_world)  # type: ignore[assignment]
        max_up_offset_m = max(
            0.0,
            0.5 * target.geometry.height_m
            - 0.5 * request.gripper.finger_height_m,
        )
        candidates: list[GraspCandidate] = []

        for index in range(self._cylinder_radial_samples):
            angle = 2.0 * pi * index / self._cylinder_radial_samples
            closing = tuple(
                cos(angle) * radial_u[axis_index]
                + sin(angle) * radial_v[axis_index]
                for axis_index in range(3)
            )
            tangent = tuple(
                -sin(angle) * radial_u[axis_index]
                + cos(angle) * radial_v[axis_index]
                for axis_index in range(3)
            )

            for sign in (-1.0, 1.0):
                approach = tuple(sign * value for value in tangent)
                for height_fraction in self._cylinder_side_grasp_height_fractions:
                    center_offset_m = height_fraction * max_up_offset_m
                    contact_center = tuple(
                        target.pose.position_xyz[axis_index]
                        + center_offset_m * up_axis_world[axis_index]
                        for axis_index in range(3)
                    )
                    candidate_id = (
                        f"{target.object_id}:cyl:side:{index}:s{int(sign):+d}"
                    )
                    if height_fraction > 0.0:
                        candidate_id += (
                            f":up{round(height_fraction * 100.0):02d}"
                        )
                    candidate = self._make_candidate(
                        target=target,
                        request=request,
                        scene=scene,
                        candidate_id=candidate_id,
                        approach_local=approach,  # type: ignore[arg-type]
                        closing_local=closing,  # type: ignore[arg-type]
                        jaw_width_m=jaw_width,
                        raw_quality=min(1.0, 0.1 + 0.9 * width_quality),
                        metadata={
                            "shape": "cylinder",
                            "mode": "side",
                            "sample_index": index,
                            "axis_local": axis,
                            "grasp_height_fraction": height_fraction,
                            "grasp_center_up_offset_m": center_offset_m,
                            "max_grasp_center_up_offset_m": max_up_offset_m,
                        },
                        contact_center_world=contact_center,  # type: ignore[arg-type]
                    )
                    if candidate is not None:
                        candidates.append(candidate)

            top_down = self._make_candidate(
                target=target,
                request=request,
                scene=scene,
                candidate_id=f"{target.object_id}:cyl:top:{index}",
                approach_local=tuple(-value for value in up_axis_local),  # type: ignore[arg-type]
                closing_local=closing,  # type: ignore[arg-type]
                jaw_width_m=jaw_width,
                raw_quality=min(1.0, 0.05 + 0.85 * width_quality),
                metadata={
                    "shape": "cylinder",
                    "mode": "top",
                    "sample_index": index,
                    "axis_local": axis,
                },
            )
            if top_down is not None:
                candidates.append(top_down)

        return candidates
