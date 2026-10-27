import pytest

from vla_robot_sorting.contracts import PlanningContext, Trajectory, TrajectoryWaypoint
from vla_robot_sorting.control import build_execution_segment


def _trajectory() -> Trajectory:
    return Trajectory(
        joint_names=("j1",),
        source="test",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,), velocities=(0.0,)),
            TrajectoryWaypoint(0.01, (0.1,), velocities=(1.0,)),
            TrajectoryWaypoint(0.02, (0.2,), velocities=(1.0,)),
            TrajectoryWaypoint(0.03, (0.3,), velocities=(0.0,)),
        ),
        scene_revision="scene-1",
        transform_revision="tf-1",
    )


def _context() -> PlanningContext:
    return PlanningContext(scene_revision="scene-1", transform_revision="tf-1")


def test_segment_snaps_horizon_to_previous_waypoint() -> None:
    segment = build_execution_segment(
        _trajectory(),
        _context(),
        segment_id="seg-1",
        requested_horizon_s=0.025,
        controller_period_s=0.01,
    )

    assert segment.horizon_end_s == pytest.approx(0.02)
    assert segment.predicted_stop_state is not None
    assert segment.predicted_stop_state.positions == pytest.approx((0.2,))
    assert segment.predicted_stop_state.velocities == pytest.approx((1.0,))


def test_segment_caps_requested_horizon_at_trajectory_end() -> None:
    segment = build_execution_segment(
        _trajectory(),
        _context(),
        segment_id="seg-2",
        requested_horizon_s=10.0,
        controller_period_s=0.01,
    )

    assert segment.horizon_end_s == pytest.approx(0.03)


def test_segment_rejects_stale_scene_revision() -> None:
    with pytest.raises(ValueError, match="scene_revision"):
        build_execution_segment(
            _trajectory(),
            PlanningContext(scene_revision="scene-2", transform_revision="tf-1"),
            segment_id="seg-stale",
            requested_horizon_s=0.02,
            controller_period_s=0.01,
        )


def test_segment_rejects_horizon_before_first_command_sample() -> None:
    with pytest.raises(ValueError, match="command waypoint"):
        build_execution_segment(
            _trajectory(),
            _context(),
            segment_id="seg-too-short",
            requested_horizon_s=0.005,
            controller_period_s=0.01,
        )
