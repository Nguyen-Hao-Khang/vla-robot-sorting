import pytest

from vla_robot_sorting.contracts import JointLimits, JointPath
from vla_robot_sorting.planning import RoboTwinVendorTrajectoryParameterizer


def _limits() -> JointLimits:
    return JointLimits(
        joint_names=("j1", "j2"),
        lower_position_rad=(-2.0, -2.0),
        upper_position_rad=(2.0, 2.0),
        max_velocity_rad_s=(5.0, 5.0),
        max_acceleration_rad_s2=(20.0, 20.0),
        max_jerk_rad_s3=(1000.0, 1000.0),
    )


def _path() -> JointPath:
    return JointPath(
        joint_names=("j1", "j2"),
        positions=((0.0, 0.0), (0.01, -0.02), (0.02, -0.03)),
        source="robotwin-curobo",
        metadata={
            "scene_revision": "scene-1",
            "transform_revision": "tf-1",
            "vendor_interpolation_dt_s": 0.004,
            "vendor_velocity_samples_rad_s": (
                (0.0, 0.0),
                (2.5, -5.0),
                (2.5, -2.5),
            ),
        },
    )


def test_vendor_timing_preserves_source_interpolation() -> None:
    trajectory = RoboTwinVendorTrajectoryParameterizer().parameterize(_path(), _limits())

    assert [point.time_from_start_s for point in trajectory.waypoints] == pytest.approx(
        [0.0, 0.004, 0.008]
    )
    assert trajectory.waypoints[1].velocities == pytest.approx((2.5, -5.0))
    assert trajectory.scene_revision == "scene-1"
    assert trajectory.transform_revision == "tf-1"
    assert trajectory.metadata["compatibility_bridge"] is True
    assert trajectory.metadata["controller_period_s"] == pytest.approx(0.004)


def test_vendor_timing_requires_aligned_velocity_samples() -> None:
    path = _path()
    path.metadata["vendor_velocity_samples_rad_s"] = ((0.0, 0.0),)

    with pytest.raises(ValueError, match="align"):
        RoboTwinVendorTrajectoryParameterizer().parameterize(path, _limits())


def test_vendor_timing_requires_positive_sample_period() -> None:
    path = _path()
    path.metadata["vendor_interpolation_dt_s"] = 0.0

    with pytest.raises(ValueError, match="positive"):
        RoboTwinVendorTrajectoryParameterizer().parameterize(path, _limits())
