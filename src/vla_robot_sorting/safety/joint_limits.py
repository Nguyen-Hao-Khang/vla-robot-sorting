"""Simulator-independent joint-limit validation for timed trajectories."""

from __future__ import annotations

from vla_robot_sorting.contracts.motion import (
    JointLimits,
    JointPath,
    PlanningContext,
    SafetyReport,
    SafetyViolation,
    Trajectory,
)

_TOLERANCE = 1e-9


def _differentiate(
    samples: tuple[tuple[float, ...], ...],
    times: tuple[float, ...],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    if len(samples) < 2:
        return (), ()

    derivatives: list[tuple[float, ...]] = []
    derivative_times: list[float] = []
    for left, right, t0, t1 in zip(samples[:-1], samples[1:], times[:-1], times[1:], strict=True):
        dt = t1 - t0
        derivatives.append(
            tuple((r_value - l_value) / dt for l_value, r_value in zip(left, right, strict=True))
        )
        derivative_times.append((t0 + t1) / 2.0)
    return tuple(derivatives), tuple(derivative_times)


def _append_derivative_violations(
    violations: list[SafetyViolation],
    samples: tuple[tuple[float, ...], ...],
    limits: tuple[float, ...],
    code: str,
    unit_label: str,
    waypoint_offset: int,
) -> None:
    for sample_index, sample in enumerate(samples):
        for joint_index, (value, limit) in enumerate(zip(sample, limits, strict=True)):
            if abs(value) > limit + _TOLERANCE:
                violations.append(
                    SafetyViolation(
                        code=code,
                        message=(
                            f"joint[{joint_index}] {unit_label}={value:.6g} exceeds "
                            f"limit={limit:.6g}"
                        ),
                        waypoint_index=sample_index + waypoint_offset,
                    )
                )


def validate_joint_trajectory_limits(
    trajectory: Trajectory,
    limits: JointLimits,
) -> SafetyReport:
    """Check position/velocity/acceleration/jerk limits using finite differences."""

    if trajectory.joint_names != limits.joint_names:
        return SafetyReport(
            (
                SafetyViolation(
                    code="joint_order_mismatch",
                    message="trajectory and JointLimits joint_names must match exactly",
                ),
            )
        )

    violations: list[SafetyViolation] = []
    for waypoint_index, waypoint in enumerate(trajectory.waypoints):
        for joint_index, (value, lower, upper) in enumerate(
            zip(
                waypoint.positions,
                limits.lower_position_rad,
                limits.upper_position_rad,
                strict=True,
            )
        ):
            if value < lower - _TOLERANCE or value > upper + _TOLERANCE:
                violations.append(
                    SafetyViolation(
                        code="position_limit",
                        message=(
                            f"joint[{joint_index}] position={value:.6g} outside "
                            f"[{lower:.6g}, {upper:.6g}]"
                        ),
                        waypoint_index=waypoint_index,
                    )
                )

    positions = tuple(waypoint.positions for waypoint in trajectory.waypoints)
    times = tuple(waypoint.time_from_start_s for waypoint in trajectory.waypoints)

    velocity, velocity_times = _differentiate(positions, times)
    acceleration, acceleration_times = _differentiate(velocity, velocity_times)
    jerk, _ = _differentiate(acceleration, acceleration_times)

    _append_derivative_violations(
        violations,
        velocity,
        limits.max_velocity_rad_s,
        "velocity_limit",
        "velocity",
        1,
    )
    _append_derivative_violations(
        violations,
        acceleration,
        limits.max_acceleration_rad_s2,
        "acceleration_limit",
        "acceleration",
        2,
    )
    _append_derivative_violations(
        violations,
        jerk,
        limits.max_jerk_rad_s3,
        "jerk_limit",
        "jerk",
        3,
    )
    return SafetyReport(tuple(violations))


def validate_joint_path_limits(
    path: JointPath,
    limits: JointLimits,
) -> SafetyReport:
    """Check geometric path positions against declared joint bounds."""

    if path.joint_names != limits.joint_names:
        return SafetyReport(
            (
                SafetyViolation(
                    code="joint_order_mismatch",
                    message="path and JointLimits joint_names must match exactly",
                ),
            )
        )

    violations: list[SafetyViolation] = []
    for waypoint_index, position in enumerate(path.positions):
        for joint_index, (value, lower, upper) in enumerate(
            zip(
                position,
                limits.lower_position_rad,
                limits.upper_position_rad,
                strict=True,
            )
        ):
            if value < lower - _TOLERANCE or value > upper + _TOLERANCE:
                violations.append(
                    SafetyViolation(
                        code="position_limit",
                        message=(
                            f"joint[{joint_index}] position={value:.6g} outside "
                            f"[{lower:.6g}, {upper:.6g}]"
                        ),
                        waypoint_index=waypoint_index,
                    )
                )
    return SafetyReport(tuple(violations))


def _revision_violations(
    *,
    scene_revision: str | None,
    transform_revision: str | None,
    context: PlanningContext,
) -> tuple[SafetyViolation, ...]:
    violations: list[SafetyViolation] = []
    if scene_revision is not None and scene_revision != context.scene_revision:
        violations.append(
            SafetyViolation(
                code="scene_revision_mismatch",
                message=(
                    f"artifact scene_revision={scene_revision!r} does not match "
                    f"context={context.scene_revision!r}"
                ),
            )
        )
    if transform_revision is not None and transform_revision != context.transform_revision:
        violations.append(
            SafetyViolation(
                code="transform_revision_mismatch",
                message=(
                    f"artifact transform_revision={transform_revision!r} does not match "
                    f"context={context.transform_revision!r}"
                ),
            )
        )
    return tuple(violations)


class JointLimitSafetyValidator:
    """Concrete safety validator for revisions and joint/dynamic limits."""

    def __init__(self, limits: JointLimits) -> None:
        self._limits = limits

    def validate_path(
        self,
        path: JointPath,
        context: PlanningContext,
        scene: object,
    ) -> SafetyReport:
        del scene
        revision_violations = _revision_violations(
            scene_revision=_metadata_text(path.metadata.get("scene_revision")),
            transform_revision=_metadata_text(path.metadata.get("transform_revision")),
            context=context,
        )
        position_report = validate_joint_path_limits(path, self._limits)
        return SafetyReport((*revision_violations, *position_report.violations))

    def validate_trajectory(
        self,
        trajectory: Trajectory,
        limits: JointLimits,
        context: PlanningContext,
        scene: object,
    ) -> SafetyReport:
        del scene
        if limits != self._limits:
            return SafetyReport(
                (
                    SafetyViolation(
                        code="limit_set_mismatch",
                        message="trajectory validation used a different JointLimits set",
                    ),
                )
            )

        revision_violations = _revision_violations(
            scene_revision=trajectory.scene_revision,
            transform_revision=trajectory.transform_revision,
            context=context,
        )
        dynamic_report = validate_joint_trajectory_limits(trajectory, limits)
        return SafetyReport((*revision_violations, *dynamic_report.violations))


def _metadata_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None
