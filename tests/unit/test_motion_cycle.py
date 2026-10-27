from vla_robot_sorting.contracts import (
    JointLimits,
    JointState,
    MotionFailureCode,
    MotionGoal,
    PlanningContext,
    Pose6D,
)
from vla_robot_sorting.orchestration import BoundedMotionController, MotionCycleStatus
from vla_robot_sorting.safety import JointLimitSafetyValidator
from vla_robot_sorting.testing import (
    RecordingTrajectoryExecutor,
    StraightLineMotionPlanner,
    UniformTrajectoryParameterizer,
)

JOINT_NAMES = ("j1", "j2")


def _goal() -> MotionGoal:
    return MotionGoal(
        pose=Pose6D(
            position_xyz=(0.4, 0.0, 0.2),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="world",
        )
    )


def _limits() -> JointLimits:
    return JointLimits(
        joint_names=JOINT_NAMES,
        lower_position_rad=(-2.0, -2.0),
        upper_position_rad=(2.0, 2.0),
        max_velocity_rad_s=(2.0, 2.0),
        max_acceleration_rad_s2=(5.0, 5.0),
        max_jerk_rad_s3=(20.0, 20.0),
    )


def _context() -> PlanningContext:
    return PlanningContext(scene_revision="scene-1", transform_revision="fake-transform")


def _state() -> JointState:
    return JointState(joint_names=JOINT_NAMES, positions=(0.0, 0.0))


def _controller(planner: StraightLineMotionPlanner) -> BoundedMotionController:
    limits = _limits()
    return BoundedMotionController(
        planner=planner,
        parameterizer=UniformTrajectoryParameterizer(segment_duration_s=1.0),
        safety=JointLimitSafetyValidator(limits),
        executor=RecordingTrajectoryExecutor(),
        limits=limits,
        controller_period_s=1.0,
        horizon_s=1.0,
    )


def test_bounded_motion_cycle_executes_safe_plan() -> None:
    controller = _controller(
        StraightLineMotionPlanner(
            joint_names=JOINT_NAMES,
            target_positions=(0.5, -0.25),
        )
    )

    result = controller.run(
        _state(),
        _goal(),
        _context(),
        scene={},
        segment_id="seg-1",
    )

    assert result.status is MotionCycleStatus.EXECUTED
    assert result.succeeded
    assert result.execution is not None
    assert result.execution.executed_until_s == 1.0


def test_bounded_motion_cycle_stops_on_planner_failure() -> None:
    controller = _controller(
        StraightLineMotionPlanner(
            joint_names=JOINT_NAMES,
            target_positions=(0.0, 0.0),
            force_failure=MotionFailureCode.COLLISION,
        )
    )

    result = controller.run(
        _state(),
        _goal(),
        _context(),
        scene={},
        segment_id="seg-fail",
    )

    assert result.status is MotionCycleStatus.PLANNING_FAILED
    assert not result.succeeded
    assert result.trajectory is None
    assert result.execution is None


def test_bounded_motion_cycle_stops_before_timing_when_path_is_unsafe() -> None:
    controller = _controller(
        StraightLineMotionPlanner(
            joint_names=JOINT_NAMES,
            target_positions=(3.0, 0.0),
        )
    )

    result = controller.run(
        _state(),
        _goal(),
        _context(),
        scene={},
        segment_id="seg-unsafe",
    )

    assert result.status is MotionCycleStatus.PATH_UNSAFE
    assert result.path_safety is not None
    assert not result.path_safety.is_safe
    assert result.trajectory is None
