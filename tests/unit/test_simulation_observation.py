from dataclasses import dataclass

import pytest

from vla_robot_sorting.contracts.simulation import CameraRole
from vla_robot_sorting.simulation.panthera_observation import (
    PANTHERA_JOINT_NAMES,
    PantheraObservationMapper,
    PantheraRoboTwinObservationAdapter,
)


@dataclass
class FakeImage:
    shape: tuple[int, ...]


def robotwin_observation() -> dict[str, object]:
    return {
        "observation": {
            "head_camera": {"rgb": FakeImage((480, 640, 3))},
            "wrist_camera": {"rgb": FakeImage((240, 320, 3))},
        },
        "joint_action": {"vector": [9, 9, 9, 9, 9, 9, 0.1]},
    }


def map_observation(observation: dict[str, object] | None = None):
    return PantheraObservationMapper().from_robotwin(
        observation or robotwin_observation(),
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75],
        sequence_id=4,
        timestamp_s=12.5,
        scene_revision="scene-7",
        transform_revision="tf-2",
    )


def test_maps_required_rgb_and_actual_joint_state() -> None:
    result = map_observation()

    assert result.joint_state.joint_names == PANTHERA_JOINT_NAMES
    assert result.joint_state.positions == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    assert result.gripper_state.normalized_position == pytest.approx(0.75)
    assert result.camera("head_camera").role is CameraRole.HEAD
    assert result.camera("head_camera").width_px == 640
    assert result.camera("wrist_camera").role is CameraRole.WRIST
    assert result.camera("wrist_camera").height_px == 240


def test_commanded_drive_target_is_metadata_not_proprioception() -> None:
    result = map_observation()

    assert result.joint_state.positions != (9, 9, 9, 9, 9, 9)
    assert result.metadata["commanded_joint_vector"] == (9, 9, 9, 9, 9, 9, 0.1)
    assert result.metadata["proprioception_source"] == "Robot.get_left_arm_real_jointState"


def test_rejects_missing_wrist_rgb() -> None:
    observation = robotwin_observation()
    observation["observation"].pop("wrist_camera")  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="wrist_camera"):
        map_observation(observation)


def test_rejects_wrong_real_state_length() -> None:
    with pytest.raises(ValueError, match="exactly 7"):
        PantheraObservationMapper().from_robotwin(
            robotwin_observation(),
            [0.0] * 6,
            sequence_id=0,
            timestamp_s=0.0,
            scene_revision="scene",
            transform_revision="tf",
        )


def test_rejects_non_rgb_camera_payload() -> None:
    observation = robotwin_observation()
    cameras = observation["observation"]
    assert isinstance(cameras, dict)
    cameras["head_camera"] = {"rgb": FakeImage((480, 640))}

    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        map_observation(observation)


def test_rejects_gripper_outside_normalized_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PantheraObservationMapper().from_robotwin(
            robotwin_observation(),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5],
            sequence_id=0,
            timestamp_s=0.0,
            scene_revision="scene",
            transform_revision="tf",
        )


class FakeRobot:
    def get_left_arm_real_jointState(self) -> list[float]:
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


class FakeTaskEnv:
    def __init__(self) -> None:
        self.robot = FakeRobot()
        self.observation_count = 0

    def get_obs(self) -> dict[str, object]:
        self.observation_count += 1
        return robotwin_observation()


def test_live_adapter_reads_actual_state_and_sequences_observations() -> None:
    env = FakeTaskEnv()
    adapter = PantheraRoboTwinObservationAdapter(
        env,
        scene_revision=lambda: "scene-live",
        transform_revision=lambda: "tf-live",
        clock=lambda: 42.0,
    )

    first = adapter.observe()
    second = adapter.observe()

    assert first.sequence_id == 0
    assert second.sequence_id == 1
    assert first.timestamp_s == 42.0
    assert first.joint_state.positions == pytest.approx((0.0, 0.1, 0.2, 0.3, 0.4, 0.5))
    assert env.observation_count == 2
