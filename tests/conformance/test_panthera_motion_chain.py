import pytest

from vla_robot_sorting.contracts import (
    ExecutionSegment,
    ExecutionStatus,
    JointLimits,
    JointState,
    MotionGoal,
    PlanningContext,
    Pose6D,
)
from vla_robot_sorting.control import PantheraRoboTwinTrajectoryExecutor
from vla_robot_sorting.planning import (
    MinimumJerkStopParameterizer,
    PantheraRoboTwinMotionPlanner,
)
from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits
from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES


class FakeRobot:
    def __init__(self) -> None:
        self.actual = [0.0] * 6
        self.commands: list[tuple[float, ...]] = []

    def get_left_arm_real_jointState(self) -> list[float]:
        return [*self.actual, 0.5]

    def left_plan_path(self, target: tuple[float, ...]) -> dict[str, object]:
        del target
        return {
            "status": "Success",
            "position": [[0.0] * 6, [0.08, -0.04, 0.03, 0.02, -0.01, 0.05]],
            "velocity": [[0.0] * 6, [0.1] * 6],
        }

    def set_arm_joints(
        self,
        position: tuple[float, ...],
        velocity: tuple[float, ...],
        arm_tag: str,
    ) -> None:
        del velocity
        assert arm_tag == "left"
        self.actual = list(position)
        self.commands.append(tuple(position))


class FakeTask:
    def __init__(self) -> None:
        self.robot = FakeRobot()
        self.steps = 0

    def _step_scene(self) -> None:
        self.steps += 1


def _limits() -> JointLimits:
    return JointLimits(
        joint_names=PANTHERA_JOINT_NAMES,
        lower_position_rad=(-3.0,) * 6,
        upper_position_rad=(3.0,) * 6,
        max_velocity_rad_s=(1.5,) * 6,
        max_acceleration_rad_s2=(3.0,) * 6,
        max_jerk_rad_s3=(30.0,) * 6,
    )


def test_project_owned_panthera_motion_chain() -> None:
    task = FakeTask()
    context = PlanningContext(scene_revision="scene-1", transform_revision="tf-1")
    state = JointState(joint_names=PANTHERA_JOINT_NAMES, positions=(0.0,) * 6)
    goal = MotionGoal(
        pose=Pose6D(
            position_xyz=(0.42, -0.1, 0.31),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        )
    )


    plan = PantheraRoboTwinMotionPlanner(task.robot).plan(
        state,
        goal,
        context,
        scene={},
    )
    assert plan.succeeded
    assert plan.path is not None

    trajectory = MinimumJerkStopParameterizer(
        controller_period_s=0.01
    ).parameterize(plan.path, _limits())
    safety = validate_joint_trajectory_limits(trajectory, _limits())
    assert safety.is_safe, safety.violations

    executor = PantheraRoboTwinTrajectoryExecutor(
        task,
        scene_revision=lambda: "scene-1",
        transform_revision=lambda: "tf-1",
        tracking_tolerance_rad=1e-6,
    )
    segment = ExecutionSegment(
        segment_id="conformance-segment",
        trajectory=trajectory,
        scene_revision="scene-1",
        transform_revision="tf-1",
        horizon_end_s=trajectory.duration_s,
        controller_period_s=0.01,
    )
    feedback = executor.execute(segment)

    assert feedback.status is ExecutionStatus.COMPLETED
    assert feedback.executed_until_s == pytest.approx(trajectory.duration_s)
    assert task.robot.commands
    assert task.robot.actual == pytest.approx(list(trajectory.waypoints[-1].positions))
    assert task.steps == len(trajectory.waypoints) - 1
