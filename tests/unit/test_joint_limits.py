from vla_robot_sorting.contracts import (
    JointLimits,
    JointPath,
    PlanningContext,
    Trajectory,
    TrajectoryWaypoint,
)
from vla_robot_sorting.safety.joint_limits import (
    JointLimitSafetyValidator,
    validate_joint_path_limits,
    validate_joint_trajectory_limits,
)


def _limits() -> JointLimits:
    return JointLimits(
        joint_names=("j1",),
        lower_position_rad=(-1.0,),
        upper_position_rad=(1.0,),
        max_velocity_rad_s=(2.0,),
        max_acceleration_rad_s2=(5.0,),
        max_jerk_rad_s3=(20.0,),
    )


def test_safe_trajectory_passes_joint_limit_validation() -> None:
    trajectory = Trajectory(
        joint_names=("j1",),
        source="unit-test",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,)),
            TrajectoryWaypoint(1.0, (0.5,)),
            TrajectoryWaypoint(2.0, (1.0,)),
            TrajectoryWaypoint(3.0, (0.5,)),
        ),
    )
    report = validate_joint_trajectory_limits(trajectory, _limits())
    assert report.is_safe
    assert report.violations == ()


def test_joint_order_mismatch_is_reported() -> None:
    trajectory = Trajectory(
        joint_names=("other",),
        source="unit-test",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,)),
            TrajectoryWaypoint(1.0, (0.1,)),
        ),
    )
    report = validate_joint_trajectory_limits(trajectory, _limits())
    assert not report.is_safe
    assert report.violations[0].code == "joint_order_mismatch"


def test_position_limit_violation_is_reported() -> None:
    trajectory = Trajectory(
        joint_names=("j1",),
        source="unit-test",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,)),
            TrajectoryWaypoint(1.0, (1.2,)),
        ),
    )
    report = validate_joint_trajectory_limits(trajectory, _limits())
    assert any(violation.code == "position_limit" for violation in report.violations)


def test_velocity_limit_violation_is_reported() -> None:
    trajectory = Trajectory(
        joint_names=("j1",),
        source="unit-test",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,)),
            TrajectoryWaypoint(0.1, (0.5,)),
        ),
    )
    report = validate_joint_trajectory_limits(trajectory, _limits())
    assert any(violation.code == "velocity_limit" for violation in report.violations)


def test_acceleration_limit_violation_is_reported() -> None:
    limits = JointLimits(
        joint_names=("j1",),
        lower_position_rad=(-2.0,),
        upper_position_rad=(2.0,),
        max_velocity_rad_s=(100.0,),
        max_acceleration_rad_s2=(0.5,),
        max_jerk_rad_s3=(100.0,),
    )
    trajectory = Trajectory(
        joint_names=("j1",),
        source="unit-test",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,)),
            TrajectoryWaypoint(1.0, (0.0,)),
            TrajectoryWaypoint(2.0, (1.0,)),
        ),
    )
    report = validate_joint_trajectory_limits(trajectory, limits)
    assert any(violation.code == "acceleration_limit" for violation in report.violations)


def test_jerk_limit_violation_is_reported() -> None:
    limits = JointLimits(
        joint_names=("j1",),
        lower_position_rad=(-2.0,),
        upper_position_rad=(2.0,),
        max_velocity_rad_s=(100.0,),
        max_acceleration_rad_s2=(100.0,),
        max_jerk_rad_s3=(0.25,),
    )
    trajectory = Trajectory(
        joint_names=("j1",),
        source="unit-test",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,)),
            TrajectoryWaypoint(1.0, (0.0,)),
            TrajectoryWaypoint(2.0, (1.0,)),
            TrajectoryWaypoint(3.0, (1.0,)),
        ),
    )
    report = validate_joint_trajectory_limits(trajectory, limits)
    assert any(violation.code == "jerk_limit" for violation in report.violations)


def test_joint_path_position_limits_are_checked() -> None:
    path = JointPath(
        joint_names=("j1",),
        positions=((0.0,), (1.2,)),
        source="unit-test",
    )

    report = validate_joint_path_limits(path, _limits())

    assert any(violation.code == "position_limit" for violation in report.violations)


def test_concrete_validator_rejects_stale_path_revision() -> None:
    path = JointPath(
        joint_names=("j1",),
        positions=((0.0,), (0.5,)),
        source="unit-test",
        metadata={"scene_revision": "scene-old", "transform_revision": "tf-1"},
    )
    validator = JointLimitSafetyValidator(_limits())
    context = PlanningContext(scene_revision="scene-new", transform_revision="tf-1")

    report = validator.validate_path(path, context, scene={})

    assert any(
        violation.code == "scene_revision_mismatch"
        for violation in report.violations
    )


def test_concrete_validator_accepts_safe_matching_trajectory() -> None:
    trajectory = Trajectory(
        joint_names=("j1",),
        source="unit-test",
        waypoints=(
            TrajectoryWaypoint(0.0, (0.0,)),
            TrajectoryWaypoint(1.0, (0.5,)),
            TrajectoryWaypoint(2.0, (1.0,)),
            TrajectoryWaypoint(3.0, (0.5,)),
        ),
        scene_revision="scene-1",
        transform_revision="tf-1",
    )
    validator = JointLimitSafetyValidator(_limits())
    context = PlanningContext(scene_revision="scene-1", transform_revision="tf-1")

    report = validator.validate_trajectory(
        trajectory,
        _limits(),
        context,
        scene={},
    )

    assert report.is_safe
