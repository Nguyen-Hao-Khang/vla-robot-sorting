from vla_robot_sorting.contracts import (
    ExecutionSegment,
    ExecutionStatus,
    JointLimits,
    JointState,
    MotionFailureCode,
    MotionGoal,
    PlanningContext,
    Pose6D,
)
from vla_robot_sorting.testing import (
    DeterministicIKSolver,
    PassThroughSafetyValidator,
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


def _state() -> JointState:
    return JointState(joint_names=JOINT_NAMES, positions=(0.0, 0.0))


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
    return PlanningContext(
        scene_revision="scene-1",
        transform_revision="tf-1",
        timestamp_s=1.0,
    )


def test_reference_adapter_chain_preserves_project_contracts() -> None:
    state = _state()
    context = _context()
    ik = DeterministicIKSolver(
        joint_names=JOINT_NAMES,
        solution_positions=(0.25, -0.1),
    )
    ik_result = ik.solve(_goal(), state, context, scene={})
    assert ik_result.succeeded
    assert ik_result.solution is not None
    assert ik_result.solution.joint_names == JOINT_NAMES

    planner = StraightLineMotionPlanner(
        joint_names=JOINT_NAMES,
        target_positions=ik_result.solution.positions,
    )
    plan_result = planner.plan(state, _goal(), context, scene={})
    assert plan_result.succeeded
    assert plan_result.path is not None
    assert plan_result.path.metadata["scene_revision"] == "scene-1"

    safety = PassThroughSafetyValidator()
    assert safety.validate_path(plan_result.path, context, scene={}).is_safe

    trajectory = UniformTrajectoryParameterizer().parameterize(
        plan_result.path,
        _limits(),
    )
    assert trajectory.duration_s == 1.0
    assert trajectory.scene_revision == "scene-1"
    assert safety.validate_trajectory(
        trajectory,
        _limits(),
        context,
        scene={},
    ).is_safe

    executor = RecordingTrajectoryExecutor()
    segment = ExecutionSegment(
        segment_id="seg-1",
        trajectory=trajectory,
        scene_revision=context.scene_revision,
        transform_revision=context.transform_revision,
        horizon_end_s=0.5,
        controller_period_s=0.01,
        predicted_stop_state=JointState(
            joint_names=JOINT_NAMES,
            positions=(0.125, -0.05),
        ),
    )
    feedback = executor.execute(segment)
    assert feedback.status is ExecutionStatus.COMPLETED
    assert feedback.executed_until_s == 0.5
    assert executor.executed_segments == [segment]


def test_reference_adapters_surface_typed_failures() -> None:
    state = _state()
    context = _context()
    ik = DeterministicIKSolver(
        joint_names=JOINT_NAMES,
        solution_positions=(0.0, 0.0),
        force_failure=MotionFailureCode.IK_UNREACHABLE,
    )
    ik_result = ik.solve(_goal(), state, context, scene={})
    assert not ik_result.succeeded
    assert ik_result.failure is not None
    assert ik_result.failure.code is MotionFailureCode.IK_UNREACHABLE

    planner = StraightLineMotionPlanner(
        joint_names=JOINT_NAMES,
        target_positions=(0.0, 0.0),
        force_failure=MotionFailureCode.COLLISION,
    )
    plan_result = planner.plan(state, _goal(), context, scene={})
    assert not plan_result.succeeded
    assert plan_result.failure is not None
    assert plan_result.failure.code is MotionFailureCode.COLLISION
