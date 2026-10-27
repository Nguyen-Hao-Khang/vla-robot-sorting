"""Translate Panthera/RoboTwin observations into project-owned contracts.

RoboTwin single-arm joint_action contains drive targets. It must not be
treated as measured proprioception. The mapper therefore requires the actual
state returned by Robot.get_left_arm_real_jointState() separately.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import monotonic
from typing import Any

from vla_robot_sorting.contracts.motion import JointState
from vla_robot_sorting.contracts.simulation import (
    CameraFrame,
    CameraRole,
    GripperState,
    SimulatorObservation,
)

PANTHERA_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)
_REQUIRED_CAMERAS = ("head_camera", "wrist_camera")


def _float_tuple(values: Sequence[Any], *, expected: int, label: str) -> tuple[float, ...]:
    if len(values) != expected:
        raise ValueError(f"{label} must contain exactly {expected} values")
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values") from exc


def _rgb_shape(image: Any, camera_id: str) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if shape is None:
        raise ValueError(f"{camera_id} RGB payload has no shape")
    try:
        dimensions = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{camera_id} RGB payload shape is invalid") from exc
    if len(dimensions) != 3 or dimensions[2] != 3:
        raise ValueError(f"{camera_id} RGB payload must have shape (H, W, 3)")
    height, width, _ = dimensions
    if height <= 0 or width <= 0:
        raise ValueError(f"{camera_id} RGB payload dimensions must be positive")
    return width, height


def _commanded_vector(observation: Mapping[str, Any]) -> tuple[float, ...] | None:
    joint_action = observation.get("joint_action")
    if not isinstance(joint_action, Mapping):
        return None
    vector = joint_action.get("vector")
    if vector is None:
        return None
    try:
        return tuple(float(value) for value in vector)
    except (TypeError, ValueError):
        return None


class PantheraObservationMapper:
    """Map the approved single-arm RGB observation into stable project types."""

    def from_robotwin(
        self,
        observation: Mapping[str, Any],
        real_joint_state: Sequence[Any],
        *,
        sequence_id: int,
        timestamp_s: float,
        scene_revision: str,
        transform_revision: str,
    ) -> SimulatorObservation:
        actual = _float_tuple(real_joint_state, expected=7, label="real_joint_state")
        camera_data = observation.get("observation")
        if not isinstance(camera_data, Mapping):
            raise ValueError("RoboTwin observation is missing camera data")

        frames: list[CameraFrame] = []
        for camera_id in _REQUIRED_CAMERAS:
            camera = camera_data.get(camera_id)
            if not isinstance(camera, Mapping) or "rgb" not in camera:
                raise ValueError(f"required RGB camera missing: {camera_id}")
            image = camera["rgb"]
            width, height = _rgb_shape(image, camera_id)
            role = CameraRole.HEAD if camera_id == "head_camera" else CameraRole.WRIST
            frames.append(
                CameraFrame(
                    camera_id=camera_id,
                    role=role,
                    frame_id=camera_id,
                    width_px=width,
                    height_px=height,
                    timestamp_s=timestamp_s,
                    payload=image,
                )
            )

        metadata: dict[str, Any] = {
            "backend": "robotwin",
            "embodiment": "panthera-6dof",
            "proprioception_source": "Robot.get_left_arm_real_jointState",
        }
        commanded = _commanded_vector(observation)
        if commanded is not None:
            metadata["commanded_joint_vector"] = commanded
            metadata["commanded_joint_vector_semantics"] = "drive_target_plus_normalized_gripper"

        return SimulatorObservation(
            sequence_id=sequence_id,
            timestamp_s=timestamp_s,
            joint_state=JointState(
                joint_names=PANTHERA_JOINT_NAMES,
                positions=actual[:6],
                timestamp_s=timestamp_s,
            ),
            gripper_state=GripperState(normalized_position=actual[6]),
            cameras=tuple(frames),
            scene_revision=scene_revision,
            transform_revision=transform_revision,
            metadata=metadata,
        )


class PantheraRoboTwinObservationAdapter:
    """Read a live RoboTwin task environment through the stable observation contract."""

    def __init__(
        self,
        task_env: Any,
        *,
        scene_revision: Callable[[], str],
        transform_revision: Callable[[], str],
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._task_env = task_env
        self._scene_revision = scene_revision
        self._transform_revision = transform_revision
        self._clock = clock
        self._sequence_id = 0
        self._mapper = PantheraObservationMapper()

    def observe(self) -> SimulatorObservation:
        """Capture cameras and actual articulation qpos from one simulator instant."""

        timestamp_s = float(self._clock())
        vendor_observation = self._task_env.get_obs()
        real_joint_state = self._task_env.robot.get_left_arm_real_jointState()

        result = self._mapper.from_robotwin(
            vendor_observation,
            real_joint_state,
            sequence_id=self._sequence_id,
            timestamp_s=timestamp_s,
            scene_revision=self._scene_revision(),
            transform_revision=self._transform_revision(),
        )
        self._sequence_id += 1
        return result
