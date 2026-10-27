from pathlib import Path

import pytest

from vla_robot_sorting.planning.panthera_limits import load_joint_limits_from_urdf


def _write_urdf(path: Path) -> None:
    path.write_text(
        """<robot name="test">
  <joint name="joint1" type="revolute">
    <limit lower="-1.0" upper="1.2" velocity="2.0" effort="10"/>
  </joint>
  <joint name="joint2" type="revolute">
    <limit lower="-2.0" upper="2.1" velocity="3.0" effort="10"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )


def test_loads_position_velocity_and_explicit_derivative_limits(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    _write_urdf(urdf)

    limits = load_joint_limits_from_urdf(
        urdf,
        ("joint1", "joint2"),
        max_acceleration_rad_s2=2.5,
        max_jerk_rad_s3=(100.0, 120.0),
    )

    assert limits.lower_position_rad == pytest.approx((-1.0, -2.0))
    assert limits.upper_position_rad == pytest.approx((1.2, 2.1))
    assert limits.max_velocity_rad_s == pytest.approx((2.0, 3.0))
    assert limits.max_acceleration_rad_s2 == pytest.approx((2.5, 2.5))
    assert limits.max_jerk_rad_s3 == pytest.approx((100.0, 120.0))


def test_missing_joint_fails_closed(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    _write_urdf(urdf)

    with pytest.raises(ValueError, match="joint3"):
        load_joint_limits_from_urdf(
            urdf,
            ("joint1", "joint3"),
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=100.0,
        )


def test_invalid_derivative_limit_length_is_rejected(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    _write_urdf(urdf)

    with pytest.raises(ValueError, match="max_acceleration"):
        load_joint_limits_from_urdf(
            urdf,
            ("joint1", "joint2"),
            max_acceleration_rad_s2=(1.0,),
            max_jerk_rad_s3=100.0,
        )
