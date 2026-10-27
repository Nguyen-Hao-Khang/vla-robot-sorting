import math

import pytest

from vla_robot_sorting.contracts import Pose6D, Trajectory, TrajectoryWaypoint
from vla_robot_sorting.telemetry.motion_metrics import (
    joint_path_length_rad,
    pose_error,
    trajectory_kinematics,
    trajectory_splice_discontinuity,
)


def _linear_trajectory() -> Trajectory:
    return Trajectory(
        joint_names=("j1",),
        source="unit-test",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,)),
            TrajectoryWaypoint(1.0, (1.0,)),
            TrajectoryWaypoint(2.0, (2.0,)),
            TrajectoryWaypoint(3.0, (3.0,)),
        ),
    )


def test_linear_trajectory_metrics_are_deterministic() -> None:
    trajectory = _linear_trajectory()
    metrics = trajectory_kinematics(trajectory)

    assert joint_path_length_rad(trajectory) == pytest.approx(3.0)
    assert metrics.duration_s == pytest.approx(3.0)
    assert metrics.peak_velocity_rad_s == pytest.approx(1.0)
    assert metrics.rms_velocity_rad_s == pytest.approx(1.0)
    assert metrics.peak_acceleration_rad_s2 == pytest.approx(0.0)
    assert metrics.peak_jerk_rad_s3 == pytest.approx(0.0)


def test_pose_error_handles_quaternion_sign_equivalence() -> None:
    expected = Pose6D(
        position_xyz=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    actual = Pose6D(
        position_xyz=(0.1, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, -1.0),
        frame_id="world",
    )
    error = pose_error(expected, actual)
    assert error.translation_m == pytest.approx(0.1)
    assert error.rotation_rad == pytest.approx(0.0)


def test_pose_error_reports_180_degree_rotation() -> None:
    expected = Pose6D(
        position_xyz=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    actual = Pose6D(
        position_xyz=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 1.0, 0.0),
        frame_id="world",
    )
    error = pose_error(expected, actual)
    assert error.translation_m == pytest.approx(0.0)
    assert error.rotation_rad == pytest.approx(math.pi)


def test_pose_error_requires_matching_frames() -> None:
    expected = Pose6D(
        position_xyz=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    actual = Pose6D(
        position_xyz=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="robot_base",
    )
    with pytest.raises(ValueError, match="matching frame"):
        pose_error(expected, actual)


def test_splice_metric_reports_continuous_boundary() -> None:
    previous = Trajectory(
        joint_names=("j1",),
        source="previous",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,)),
            TrajectoryWaypoint(1.0, (1.0,)),
        ),
    )
    following = Trajectory(
        joint_names=("j1",),
        source="following",
        waypoints=(
            TrajectoryWaypoint(0.0, (1.0,)),
            TrajectoryWaypoint(1.0, (2.0,)),
        ),
    )
    discontinuity = trajectory_splice_discontinuity(previous, following)
    assert discontinuity.position_rad_max == pytest.approx(0.0)
    assert discontinuity.velocity_rad_s_max == pytest.approx(0.0)
    assert discontinuity.acceleration_rad_s2_max is None


def test_splice_metric_detects_position_and_velocity_jump() -> None:
    previous = Trajectory(
        joint_names=("j1",),
        source="previous",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,)),
            TrajectoryWaypoint(1.0, (1.0,)),
        ),
    )
    following = Trajectory(
        joint_names=("j1",),
        source="following",
        waypoints=(
            TrajectoryWaypoint(0.0, (1.2,)),
            TrajectoryWaypoint(0.5, (2.2,)),
        ),
    )
    discontinuity = trajectory_splice_discontinuity(previous, following)
    assert discontinuity.position_rad_max == pytest.approx(0.2)
    assert discontinuity.velocity_rad_s_max == pytest.approx(1.0)
