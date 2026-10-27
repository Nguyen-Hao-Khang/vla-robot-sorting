from itertools import pairwise

import pytest

from vla_robot_sorting.contracts import JointLimits, JointPath
from vla_robot_sorting.planning import MinimumJerkStopParameterizer
from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits

JOINT_NAMES = ("j1", "j2")


def _limits() -> JointLimits:
    return JointLimits(
        joint_names=JOINT_NAMES,
        lower_position_rad=(-2.0, -2.0),
        upper_position_rad=(2.0, 2.0),
        max_velocity_rad_s=(1.0, 1.2),
        max_acceleration_rad_s2=(2.0, 2.5),
        max_jerk_rad_s3=(10.0, 12.0),
    )


def _path() -> JointPath:
    return JointPath(
        joint_names=JOINT_NAMES,
        positions=((0.0, 0.0), (0.5, -0.3), (0.8, 0.1)),
        source="unit-path",
        metadata={"scene_revision": "scene-a", "transform_revision": "tf-a"},
    )


def test_minimum_jerk_trajectory_is_uniform_c2_and_limit_safe() -> None:
    parameterizer = MinimumJerkStopParameterizer(controller_period_s=0.02)

    trajectory = parameterizer.parameterize(_path(), _limits())

    assert trajectory.waypoints[0].time_from_start_s == 0.0
    assert trajectory.waypoints[0].positions == pytest.approx((0.0, 0.0))
    assert trajectory.waypoints[-1].positions == pytest.approx((0.8, 0.1))
    assert trajectory.scene_revision == "scene-a"
    assert trajectory.transform_revision == "tf-a"
    assert trajectory.interpolation == "quintic-minimum-jerk-stop-at-knots"

    times = [point.time_from_start_s for point in trajectory.waypoints]
    assert all(
        right - left == pytest.approx(0.02)
        for left, right in pairwise(times)
    )
    assert trajectory.waypoints[0].velocities == pytest.approx((0.0, 0.0))
    assert trajectory.waypoints[0].accelerations == pytest.approx((0.0, 0.0))
    assert trajectory.waypoints[-1].velocities == pytest.approx((0.0, 0.0))
    assert trajectory.waypoints[-1].accelerations == pytest.approx((0.0, 0.0))

    report = validate_joint_trajectory_limits(trajectory, _limits())
    assert report.is_safe, report.violations


def test_minimum_jerk_stops_at_internal_geometric_knot() -> None:
    trajectory = MinimumJerkStopParameterizer(
        controller_period_s=0.02
    ).parameterize(_path(), _limits())

    middle = [
        point
        for point in trajectory.waypoints
        if point.positions == pytest.approx((0.5, -0.3))
    ]

    assert len(middle) == 1
    assert middle[0].velocities == pytest.approx((0.0, 0.0))
    assert middle[0].accelerations == pytest.approx((0.0, 0.0))


def test_parameterizer_rejects_joint_order_mismatch() -> None:
    bad_limits = JointLimits(
        joint_names=("other1", "other2"),
        lower_position_rad=(-2.0, -2.0),
        upper_position_rad=(2.0, 2.0),
        max_velocity_rad_s=(1.0, 1.0),
        max_acceleration_rad_s2=(2.0, 2.0),
        max_jerk_rad_s3=(10.0, 10.0),
    )

    with pytest.raises(ValueError, match="joint order"):
        MinimumJerkStopParameterizer().parameterize(_path(), bad_limits)
