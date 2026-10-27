import pytest

from vla_robot_sorting.contracts import (
    ExecutionSegment,
    ExecutionStatus,
    Trajectory,
    TrajectoryWaypoint,
)
from vla_robot_sorting.control import PantheraRoboTwinTrajectoryExecutor
from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES


def _trajectory() -> Trajectory:
    return Trajectory(
        joint_names=PANTHERA_JOINT_NAMES,
        source="unit-test",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,) * 6, velocities=(0.0,) * 6),
            TrajectoryWaypoint(0.01, (0.1,) * 6, velocities=(1.0,) * 6),
            TrajectoryWaypoint(0.02, (0.2,) * 6, velocities=(1.0,) * 6),
        ),
        scene_revision="scene-1",
        transform_revision="tf-1",
    )


def _segment(horizon_s: float = 0.02) -> ExecutionSegment:
    return ExecutionSegment(
        segment_id="seg-1",
        trajectory=_trajectory(),
        scene_revision="scene-1",
        transform_revision="tf-1",
        horizon_end_s=horizon_s,
        controller_period_s=0.01,
    )


class FakeRobot:
    def __init__(self, *, follow_commands: bool = True) -> None:
        self.follow_commands = follow_commands
        self.actual = [0.0] * 6
        self.commands: list[tuple[tuple[float, ...], tuple[float, ...], str]] = []

    def set_arm_joints(
        self,
        position: tuple[float, ...],
        velocity: tuple[float, ...],
        arm_tag: str,
    ) -> None:
        self.commands.append((tuple(position), tuple(velocity), arm_tag))
        if self.follow_commands:
            self.actual = list(position)

    def get_left_arm_real_jointState(self) -> list[float]:
        return [*self.actual, 0.5]


class FakeTask:
    def __init__(self, *, follow_commands: bool = True) -> None:
        self.robot = FakeRobot(follow_commands=follow_commands)
        self.step_count = 0

    def _step_scene(self) -> None:
        self.step_count += 1


def test_executor_runs_bounded_segment_and_checks_tracking() -> None:
    task = FakeTask()
    executor = PantheraRoboTwinTrajectoryExecutor(
        task,
        scene_revision=lambda: "scene-1",
        transform_revision=lambda: "tf-1",
        tracking_tolerance_rad=0.01,
    )

    feedback = executor.execute(_segment())

    assert feedback.status is ExecutionStatus.COMPLETED
    assert feedback.executed_until_s == pytest.approx(0.02)
    assert task.step_count == 2
    assert [command[0][0] for command in task.robot.commands] == pytest.approx([0.1, 0.2])
    assert all(command[2] == "left" for command in task.robot.commands)
    assert executor.last_actual_state is not None
    assert executor.last_actual_state.positions == pytest.approx((0.2,) * 6)


def test_executor_cancels_before_motion_when_scene_is_stale() -> None:
    task = FakeTask()
    executor = PantheraRoboTwinTrajectoryExecutor(
        task,
        scene_revision=lambda: "scene-new",
        transform_revision=lambda: "tf-1",
    )

    feedback = executor.execute(_segment())

    assert feedback.status is ExecutionStatus.CANCELLED
    assert feedback.replan_required
    assert feedback.executed_until_s == 0.0
    assert task.robot.commands == []


def test_executor_cancels_mid_segment_on_revision_change() -> None:
    task = FakeTask()
    executor = PantheraRoboTwinTrajectoryExecutor(
        task,
        scene_revision=lambda: "scene-1" if task.step_count < 1 else "scene-2",
        transform_revision=lambda: "tf-1",
    )

    feedback = executor.execute(_segment())

    assert feedback.status is ExecutionStatus.CANCELLED
    assert feedback.replan_required
    assert feedback.executed_until_s == pytest.approx(0.01)
    assert task.step_count == 1


def test_executor_reports_tracking_error() -> None:
    task = FakeTask(follow_commands=False)
    executor = PantheraRoboTwinTrajectoryExecutor(
        task,
        scene_revision=lambda: "scene-1",
        transform_revision=lambda: "tf-1",
        tracking_tolerance_rad=0.05,
    )

    feedback = executor.execute(_segment())

    assert feedback.status is ExecutionStatus.FAILED
    assert feedback.replan_required
    assert feedback.reason is not None
    assert feedback.reason.startswith("tracking_error:")


def test_executor_rejects_horizon_between_control_samples() -> None:
    task = FakeTask()
    executor = PantheraRoboTwinTrajectoryExecutor(
        task,
        scene_revision=lambda: "scene-1",
        transform_revision=lambda: "tf-1",
    )

    with pytest.raises(ValueError, match="align"):
        executor.execute(_segment(0.015))


def test_executor_holds_dilated_command_for_multiple_physics_ticks() -> None:
    trajectory = Trajectory(
        joint_names=PANTHERA_JOINT_NAMES,
        source="dilated-test",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,) * 6, velocities=(0.0,) * 6),
            TrajectoryWaypoint(0.04, (0.1,) * 6, velocities=(0.25,) * 6),
            TrajectoryWaypoint(0.08, (0.2,) * 6, velocities=(0.25,) * 6),
        ),
        scene_revision="scene-1",
        transform_revision="tf-1",
    )
    segment = ExecutionSegment(
        segment_id="dilated",
        trajectory=trajectory,
        scene_revision="scene-1",
        transform_revision="tf-1",
        horizon_end_s=0.08,
        controller_period_s=0.04,
    )
    task = FakeTask()
    executor = PantheraRoboTwinTrajectoryExecutor(
        task,
        scene_revision=lambda: "scene-1",
        transform_revision=lambda: "tf-1",
        physics_period_s=0.01,
    )

    feedback = executor.execute(segment)

    assert feedback.status is ExecutionStatus.COMPLETED
    assert feedback.executed_until_s == pytest.approx(0.08)
    assert task.step_count == 8
    assert len(task.robot.commands) == 2


def test_executor_rejects_controller_period_not_aligned_to_physics() -> None:
    task = FakeTask()
    executor = PantheraRoboTwinTrajectoryExecutor(
        task,
        scene_revision=lambda: "scene-1",
        transform_revision=lambda: "tf-1",
        physics_period_s=0.004,
    )

    with pytest.raises(ValueError, match="integer multiple"):
        executor.execute(_segment())


def test_command_observer_runs_after_drive_apply_before_physics():
    task = FakeTask()
    events = []

    def observe(index, position, velocity, time_s):
        events.append((index, len(task.robot.commands), task.step_count, tuple(position), time_s))

    executor = PantheraRoboTwinTrajectoryExecutor(
        task,
        scene_revision=lambda: "scene-1",
        transform_revision=lambda: "tf-1",
        command_observer=observe,
    )
    assert executor.execute(_segment()).status is ExecutionStatus.COMPLETED
    assert [(v[1], v[2]) for v in events] == [(1, 0), (2, 1)]
    assert events[-1][3] == (0.2,) * 6


def test_command_observer_failure_stops_before_physics_and_next_command():
    task = FakeTask()

    def observe(*args):
        raise RuntimeError("synthetic telemetry API failure")

    executor = PantheraRoboTwinTrajectoryExecutor(
        task,
        scene_revision=lambda: "scene-1",
        transform_revision=lambda: "tf-1",
        command_observer=observe,
    )
    feedback = executor.execute(_segment())
    assert feedback.status is ExecutionStatus.FAILED
    assert task.step_count == 0 and len(task.robot.commands) == 1
