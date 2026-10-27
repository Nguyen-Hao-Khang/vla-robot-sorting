import pytest

from vla_robot_sorting.contracts import JointLimits, JointPath
from vla_robot_sorting.planning import (
    PhysicsAlignedTimeDilationParameterizer,
    RoboTwinVendorTrajectoryParameterizer,
    required_uniform_time_scale,
)
from vla_robot_sorting.safety.joint_limits import validate_joint_trajectory_limits


def _limits() -> JointLimits:
    return JointLimits(
        joint_names=("j1",),
        lower_position_rad=(-2.0,),
        upper_position_rad=(2.0,),
        max_velocity_rad_s=(2.0,),
        max_acceleration_rad_s2=(2.0,),
        max_jerk_rad_s3=(50.0,),
    )


def _unsafe_vendor_path() -> JointPath:
    return JointPath(
        joint_names=("j1",),
        positions=((0.0,), (0.02,), (0.08,), (0.10,), (0.10,)),
        source="robotwin-curobo",
        metadata={
            "vendor_interpolation_dt_s": 0.004,
            "vendor_velocity_samples_rad_s": (
                (0.0,),
                (5.0,),
                (15.0,),
                (5.0,),
                (0.0,),
            ),
        },
    )


def test_physics_aligned_dilation_makes_unsafe_vendor_timing_safe() -> None:
    limits = _limits()
    base = RoboTwinVendorTrajectoryParameterizer().parameterize(
        _unsafe_vendor_path(),
        limits,
    )
    assert not validate_joint_trajectory_limits(base, limits).is_safe

    parameterizer = PhysicsAlignedTimeDilationParameterizer(
        RoboTwinVendorTrajectoryParameterizer(),
        physics_period_s=0.004,
    )
    safe = parameterizer.parameterize(_unsafe_vendor_path(), limits)

    assert validate_joint_trajectory_limits(safe, limits).is_safe
    factor = safe.metadata["applied_integer_time_dilation_factor"]
    assert isinstance(factor, int)
    assert factor >= required_uniform_time_scale(base, limits)
    assert safe.metadata["controller_period_s"] == pytest.approx(0.004 * factor)
    assert safe.duration_s == pytest.approx(base.duration_s * factor)


def test_dilation_scales_declared_velocity_fields() -> None:
    limits = _limits()
    parameterizer = PhysicsAlignedTimeDilationParameterizer(
        RoboTwinVendorTrajectoryParameterizer(),
        physics_period_s=0.004,
    )
    safe = parameterizer.parameterize(_unsafe_vendor_path(), limits)
    factor = safe.metadata["applied_integer_time_dilation_factor"]

    assert safe.waypoints[2].velocities is not None
    assert safe.waypoints[2].velocities[0] == pytest.approx(15.0 / factor)


def test_dilation_rejects_non_physics_aligned_delegate_output() -> None:
    path = _unsafe_vendor_path()
    path.metadata["vendor_interpolation_dt_s"] = 0.005
    parameterizer = PhysicsAlignedTimeDilationParameterizer(
        RoboTwinVendorTrajectoryParameterizer(),
        physics_period_s=0.004,
    )

    with pytest.raises(ValueError, match="physics_period_s"):
        parameterizer.parameterize(path, _limits())
