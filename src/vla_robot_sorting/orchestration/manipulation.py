"""Fail-closed manipulation phase state machine for project-owned pick/place."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from vla_robot_sorting.contracts.manipulation import (
    ManipulationPhase,
    ManipulationSignal,
    ManipulationTransition,
)
from vla_robot_sorting.contracts.models import Pose6D


@dataclass(slots=True)
class ManipulationStateMachine:
    """Track pick/place phase progression without issuing robot commands.

    Native adapters execute motion/gripper actions separately. This state machine
    accepts only evidence-backed phase completion signals and routes stale/failure
    signals to recovery rather than silently advancing.
    """

    scene_revision: str
    transform_revision: str
    phase: ManipulationPhase = ManipulationPhase.PREGRASP

    def __post_init__(self) -> None:
        if not self.scene_revision.strip() or not self.transform_revision.strip():
            raise ValueError("manipulation revisions must be non-empty")

    def update_revisions(self, scene_revision: str, transform_revision: str) -> None:
        if not scene_revision.strip() or not transform_revision.strip():
            raise ValueError("manipulation revisions must be non-empty")
        self.scene_revision = scene_revision
        self.transform_revision = transform_revision

    def apply(self, signal: ManipulationSignal) -> ManipulationTransition:
        previous = self.phase
        if previous in {
            ManipulationPhase.COMPLETE,
            ManipulationPhase.RECOVERY_REQUIRED,
        }:
            return ManipulationTransition(
                previous,
                previous,
                signal,
                False,
                f"terminal phase {previous.value} rejects further progression",
            )

        if signal in {
            ManipulationSignal.MOTION_FAILED,
            ManipulationSignal.GRASP_FAILED,
            ManipulationSignal.SCENE_STALE,
        }:
            self.phase = ManipulationPhase.RECOVERY_REQUIRED
            return ManipulationTransition(
                previous,
                self.phase,
                signal,
                True,
                f"{signal.value} requires explicit recovery before more actuation",
            )

        expected = {
            ManipulationPhase.PREGRASP: (
                ManipulationSignal.MOTION_COMPLETED,
                ManipulationPhase.APPROACH,
            ),
            ManipulationPhase.APPROACH: (
                ManipulationSignal.MOTION_COMPLETED,
                ManipulationPhase.CLOSE,
            ),
            ManipulationPhase.CLOSE: (
                ManipulationSignal.GRIPPER_CLOSED,
                ManipulationPhase.VERIFY_GRASP,
            ),
            ManipulationPhase.VERIFY_GRASP: (
                ManipulationSignal.GRASP_CONFIRMED,
                ManipulationPhase.LIFT,
            ),
            ManipulationPhase.LIFT: (
                ManipulationSignal.MOTION_COMPLETED,
                ManipulationPhase.TRANSPORT,
            ),
            ManipulationPhase.TRANSPORT: (
                ManipulationSignal.MOTION_COMPLETED,
                ManipulationPhase.PLACE,
            ),
            ManipulationPhase.PLACE: (
                ManipulationSignal.MOTION_COMPLETED,
                ManipulationPhase.RELEASE,
            ),
            ManipulationPhase.RELEASE: (
                ManipulationSignal.GRIPPER_OPENED,
                ManipulationPhase.RETREAT,
            ),
            ManipulationPhase.RETREAT: (
                ManipulationSignal.MOTION_COMPLETED,
                ManipulationPhase.COMPLETE,
            ),
        }
        expected_signal, next_phase = expected[previous]
        if signal is not expected_signal:
            return ManipulationTransition(
                previous,
                previous,
                signal,
                False,
                (
                    f"phase {previous.value} requires {expected_signal.value}; "
                    f"received {signal.value}"
                ),
            )

        self.phase = next_phase
        return ManipulationTransition(
            previous,
            next_phase,
            signal,
            True,
            f"{previous.value} completed; advancing to {next_phase.value}",
        )


def _normalize_xyzw(q: tuple[float, float, float, float]) -> tuple[float, ...]:
    norm = sqrt(sum(value * value for value in q))
    if norm <= 0.0:
        raise ValueError("quaternion norm must be positive")
    return tuple(value / norm for value in q)


def _quat_conjugate_xyzw(q: tuple[float, float, float, float]) -> tuple[float, ...]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def _quat_multiply_xyzw(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, ...]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _rotate_xyzw(
    q: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    qn = _normalize_xyzw(q)
    vx, vy, vz = vector
    pure = (vx, vy, vz, 0.0)
    rotated = _quat_multiply_xyzw(
        _quat_multiply_xyzw(qn, pure),
        _quat_conjugate_xyzw(qn),
    )
    return (rotated[0], rotated[1], rotated[2])


def gripper_pose_for_desired_object_pose(
    current_object_pose: Pose6D,
    current_gripper_pose: Pose6D,
    desired_object_pose: Pose6D,
) -> Pose6D:
    """Preserve the rigid object->gripper transform for placement planning."""

    if not (
        current_object_pose.frame_id
        == current_gripper_pose.frame_id
        == desired_object_pose.frame_id
    ):
        raise ValueError("object/gripper poses must share one frame")

    current_object_q = _normalize_xyzw(current_object_pose.orientation_xyzw)
    current_gripper_q = _normalize_xyzw(current_gripper_pose.orientation_xyzw)
    desired_object_q = _normalize_xyzw(desired_object_pose.orientation_xyzw)

    world_offset = tuple(
        current_gripper_pose.position_xyz[index]
        - current_object_pose.position_xyz[index]
        for index in range(3)
    )
    object_offset = _rotate_xyzw(
        _quat_conjugate_xyzw(current_object_q),
        world_offset,
    )
    desired_world_offset = _rotate_xyzw(desired_object_q, object_offset)
    desired_position = tuple(
        desired_object_pose.position_xyz[index] + desired_world_offset[index]
        for index in range(3)
    )

    object_to_gripper_q = _quat_multiply_xyzw(
        _quat_conjugate_xyzw(current_object_q),
        current_gripper_q,
    )
    desired_gripper_q = _normalize_xyzw(
        _quat_multiply_xyzw(desired_object_q, object_to_gripper_q)
    )
    return Pose6D(
        position_xyz=desired_position,
        orientation_xyzw=desired_gripper_q,
        frame_id=current_object_pose.frame_id,
    )
