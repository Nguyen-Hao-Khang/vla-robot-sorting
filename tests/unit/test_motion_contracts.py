import pytest

from vla_robot_sorting.contracts import (
    ExecutionFeedback,
    ExecutionSegment,
    ExecutionStatus,
    IKResult,
    IKSolution,
    JointLimits,
    JointPath,
    JointState,
    MotionFailure,
    MotionFailureCode,
    PlannerResult,
    PlanningContext,
    Pose6D,
    Trajectory,
    TrajectoryWaypoint,
    WorkspaceBounds,
)
from vla_robot_sorting.contracts.motion import MotionGoal


def _trajectory() -> Trajectory:
    return Trajectory(
        joint_names=("j1", "j2"),
        source="unit-test",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0, 0.0)),
            TrajectoryWaypoint(0.5, (0.1, -0.2)),
            TrajectoryWaypoint(1.0, (0.2, -0.4)),
        ),
    )


def test_joint_state_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        JointState(joint_names=("j1", "j1"), positions=(0.0, 0.0))


def test_joint_limits_require_positive_derivative_limits() -> None:
    with pytest.raises(ValueError, match="positive"):
        JointLimits(
            joint_names=("j1",),
            lower_position_rad=(-1.0,),
            upper_position_rad=(1.0,),
            max_velocity_rad_s=(0.0,),
            max_acceleration_rad_s2=(1.0,),
            max_jerk_rad_s3=(1.0,),
        )


def test_joint_path_requires_multiple_positions() -> None:
    with pytest.raises(ValueError, match="at least two"):
        JointPath(
            joint_names=("j1",),
            positions=((0.0,),),
            source="planner",
        )


def test_trajectory_requires_strictly_increasing_time() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        Trajectory(
            joint_names=("j1",),
            source="parameterizer",
            waypoints=(
                TrajectoryWaypoint(0.0, (0.0,)),
                TrajectoryWaypoint(0.0, (0.1,)),
            ),
        )


def test_planner_result_requires_exactly_one_payload() -> None:
    failure = MotionFailure(
        code=MotionFailureCode.IK_UNREACHABLE,
        message="goal unreachable",
        recoverable=True,
    )
    with pytest.raises(ValueError, match="exactly one"):
        PlannerResult(
            path=None,
            failure=None,
            planning_time_s=0.01,
            planner_id="planner",
        )
    assert not PlannerResult(
        path=None,
        failure=failure,
        planning_time_s=0.01,
        planner_id="planner",
    ).succeeded


def test_ik_result_success_contract() -> None:
    solution = IKSolution(
        joint_names=("j1", "j2"),
        positions=(0.1, -0.1),
        position_error_m=0.0005,
        orientation_error_rad=0.001,
    )
    result = IKResult(
        solution=solution,
        failure=None,
        solve_time_s=0.002,
        solver_id="ik-test",
    )
    assert result.succeeded


def test_execution_segment_enforces_bounded_horizon() -> None:
    trajectory = _trajectory()
    segment = ExecutionSegment(
        segment_id="seg-1",
        trajectory=trajectory,
        scene_revision="scene-42",
        transform_revision="tf-7",
        horizon_end_s=0.4,
        controller_period_s=0.005,
    )
    assert segment.horizon_end_s < trajectory.duration_s

    with pytest.raises(ValueError, match="cannot exceed"):
        ExecutionSegment(
            segment_id="seg-2",
            trajectory=trajectory,
            scene_revision="scene-42",
            transform_revision="tf-7",
            horizon_end_s=2.0,
            controller_period_s=0.005,
        )


def test_replan_feedback_requires_reason() -> None:
    with pytest.raises(ValueError, match="include a reason"):
        ExecutionFeedback(
            segment_id="seg-1",
            status=ExecutionStatus.CANCELLED,
            executed_until_s=0.2,
            replan_required=True,
        )


def test_motion_goal_requires_matching_workspace_frame() -> None:
    pose = Pose6D(
        position_xyz=(0.4, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    workspace = WorkspaceBounds(
        frame_id="robot_base",
        minimum_xyz_m=(0.0, -0.5, 0.0),
        maximum_xyz_m=(0.8, 0.5, 0.8),
    )
    with pytest.raises(ValueError, match="frame_id"):
        MotionGoal(pose=pose, workspace=workspace)


def test_workspace_bounds_reject_inverted_axis() -> None:
    with pytest.raises(ValueError, match="minimum"):
        WorkspaceBounds(
            frame_id="world",
            minimum_xyz_m=(0.5, -0.5, 0.0),
            maximum_xyz_m=(0.4, 0.5, 0.8),
        )


def test_trajectory_must_start_at_zero() -> None:
    with pytest.raises(ValueError, match="start at time_from_start_s=0"):
        Trajectory(
            joint_names=("j1",),
            source="parameterizer",
            waypoints=(
                TrajectoryWaypoint(0.1, (0.0,)),
                TrajectoryWaypoint(0.2, (0.1,)),
            ),
        )


def test_workspace_bounds_require_three_axes() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        WorkspaceBounds(
            frame_id="world",
            minimum_xyz_m=(0.0, 0.0),
            maximum_xyz_m=(1.0, 1.0),
        )


def test_planning_context_requires_version_provenance() -> None:
    context = PlanningContext(
        scene_revision="scene-1",
        transform_revision="tf-1",
        timestamp_s=1.25,
    )
    assert context.scene_revision == "scene-1"

    with pytest.raises(ValueError, match="must be non-empty"):
        PlanningContext(scene_revision="", transform_revision="tf-1")


def test_trajectory_rejects_limit_joint_order_mismatch() -> None:
    limits = JointLimits(
        joint_names=("j2", "j1"),
        lower_position_rad=(-1.0, -1.0),
        upper_position_rad=(1.0, 1.0),
        max_velocity_rad_s=(1.0, 1.0),
        max_acceleration_rad_s2=(1.0, 1.0),
        max_jerk_rad_s3=(1.0, 1.0),
    )
    with pytest.raises(ValueError, match="limits_used joint order"):
        Trajectory(
            joint_names=("j1", "j2"),
            source="parameterizer",
            limits_used=limits,
            waypoints=(
                TrajectoryWaypoint(0.0, (0.0, 0.0)),
                TrajectoryWaypoint(1.0, (0.1, 0.1)),
            ),
        )


def test_execution_segment_validates_controller_period_and_stop_state() -> None:
    trajectory = _trajectory()
    stop_state = JointState(
        joint_names=("j1", "j2"),
        positions=(0.1, -0.2),
    )
    segment = ExecutionSegment(
        segment_id="seg-3",
        trajectory=trajectory,
        scene_revision="scene-42",
        transform_revision="tf-7",
        horizon_end_s=0.5,
        controller_period_s=0.005,
        predicted_stop_state=stop_state,
    )
    assert segment.predicted_stop_state == stop_state

    with pytest.raises(ValueError, match="controller_period_s"):
        ExecutionSegment(
            segment_id="seg-4",
            trajectory=trajectory,
            scene_revision="scene-42",
            transform_revision="tf-7",
            horizon_end_s=0.5,
            controller_period_s=0.0,
        )
