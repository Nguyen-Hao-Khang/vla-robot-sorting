from __future__ import annotations

import pytest

from vla_robot_sorting.contracts import GripperCommand
from vla_robot_sorting.control import PantheraRoboTwinGripperExecutor


class FakeRobot:
    def __init__(self) -> None:
        self.gripper = 1.0
        self.commands: list[tuple[float, str, float]] = []

    def set_gripper(self, value: float, arm_tag: str, per_step: float) -> None:
        self.commands.append((value, arm_tag, per_step))
        self.gripper = value

    def get_left_gripper_val(self) -> float:
        return self.gripper


class FakeTask:
    def __init__(self) -> None:
        self.robot = FakeRobot()
        self.step_count = 0

    def set_gripper(self, *, left_pos: float, set_tag: str):
        assert set_tag == "left"
        start = self.robot.gripper
        midpoint = (start + left_pos) * 0.5
        return {
            "result": [midpoint, left_pos],
            "num_step": 2,
            "per_step": abs(left_pos - start) * 0.5,
        }

    def _step_scene(self) -> None:
        self.step_count += 1


def _command(*, force_scale: float = 1.0) -> GripperCommand:
    return GripperCommand(
        normalized_position=0.0,
        compliance_profile="standard",
        force_scale=force_scale,
        scene_revision="scene-1",
        transform_revision="tf-1",
    )


def test_gripper_executor_drives_plan_stepwise() -> None:
    task = FakeTask()
    executor = PantheraRoboTwinGripperExecutor(
        task,
        scene_revision=lambda: "scene-1",
        transform_revision=lambda: "tf-1",
        position_tolerance=0.01,
        contact_observed=lambda: True,
        holding_object=lambda: True,
    )

    result = executor.execute_gripper(_command())

    assert result.completed
    assert result.actual_normalized_position == pytest.approx(0.0)
    assert result.contact_observed
    assert result.holding_object
    assert task.step_count == 2
    assert [item[0] for item in task.robot.commands] == pytest.approx([0.5, 0.0])


def test_gripper_executor_rejects_stale_scene_before_actuation() -> None:
    task = FakeTask()
    executor = PantheraRoboTwinGripperExecutor(
        task,
        scene_revision=lambda: "scene-new",
        transform_revision=lambda: "tf-1",
    )

    result = executor.execute_gripper(_command())

    assert not result.completed
    assert result.reason is not None
    assert result.reason.startswith("scene_revision_changed:")
    assert task.robot.commands == []
    assert task.step_count == 0


def test_gripper_executor_cancels_mid_command_on_revision_change() -> None:
    task = FakeTask()
    executor = PantheraRoboTwinGripperExecutor(
        task,
        scene_revision=lambda: "scene-1" if task.step_count < 1 else "scene-2",
        transform_revision=lambda: "tf-1",
    )

    result = executor.execute_gripper(_command())

    assert not result.completed
    assert result.reason is not None
    assert result.reason.startswith("scene_revision_changed:")
    assert task.step_count == 1
    assert len(task.robot.commands) == 1


def test_unvalidated_force_scale_fails_closed() -> None:
    task = FakeTask()
    executor = PantheraRoboTwinGripperExecutor(
        task,
        scene_revision=lambda: "scene-1",
        transform_revision=lambda: "tf-1",
    )

    result = executor.execute_gripper(_command(force_scale=0.45))

    assert not result.completed
    assert result.reason is not None
    assert result.reason.startswith("unsupported_force_scale:")
    assert task.robot.commands == []
