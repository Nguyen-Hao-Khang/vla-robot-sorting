"""Physics-aligned uniform time dilation for dynamically safe trajectories."""

from __future__ import annotations

from itertools import pairwise
from math import ceil, isfinite, sqrt

from vla_robot_sorting.contracts.interfaces import TrajectoryParameterizer
from vla_robot_sorting.contracts.motion import (
    JointLimits,
    JointPath,
    Trajectory,
    TrajectoryWaypoint,
)


def _differentiate(
    samples: tuple[tuple[float, ...], ...],
    times: tuple[float, ...],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    if len(samples) < 2:
        return (), ()
    rows: list[tuple[float, ...]] = []
    sample_times: list[float] = []
    for left, right, t0, t1 in zip(
        samples[:-1], samples[1:], times[:-1], times[1:], strict=True
    ):
        dt = t1 - t0
        if dt <= 0:
            raise ValueError("trajectory times must be strictly increasing")
        rows.append(
            tuple(
                (right_value - left_value) / dt
                for left_value, right_value in zip(left, right, strict=True)
            )
        )
        sample_times.append((t0 + t1) / 2.0)
    return tuple(rows), tuple(sample_times)


def _peak_ratio(
    samples: tuple[tuple[float, ...], ...],
    limits: tuple[float, ...],
) -> float:
    if not samples:
        return 0.0
    return max(
        abs(value) / limit
        for row in samples
        for value, limit in zip(row, limits, strict=True)
    )


def required_uniform_time_scale(
    trajectory: Trajectory,
    limits: JointLimits,
) -> float:
    """Return the continuous scale required by v/a/jerk finite-difference limits."""

    if trajectory.joint_names != limits.joint_names:
        raise ValueError("trajectory/limits joint order mismatch")

    positions = tuple(point.positions for point in trajectory.waypoints)
    times = tuple(point.time_from_start_s for point in trajectory.waypoints)
    velocity, velocity_times = _differentiate(positions, times)
    acceleration, acceleration_times = _differentiate(velocity, velocity_times)
    jerk, _ = _differentiate(acceleration, acceleration_times)

    velocity_ratio = _peak_ratio(velocity, limits.max_velocity_rad_s)
    acceleration_ratio = _peak_ratio(acceleration, limits.max_acceleration_rad_s2)
    jerk_ratio = _peak_ratio(jerk, limits.max_jerk_rad_s3)

    return max(
        1.0,
        velocity_ratio,
        sqrt(acceleration_ratio),
        jerk_ratio ** (1.0 / 3.0),
    )


class PhysicsAlignedTimeDilationParameterizer:
    """Wrap another parameterizer and slow it by an integer physics-tick factor.

    The geometric path is unchanged. Every original control interval becomes an
    integer multiple of the physics period, so an executor can honor the retimed
    trajectory without changing SAPIEN's fixed integration timestep.
    """

    def __init__(
        self,
        delegate: TrajectoryParameterizer,
        *,
        physics_period_s: float = 1.0 / 250.0,
        max_dilation_factor: int = 64,
        period_tolerance_s: float = 1e-9,
    ) -> None:
        if not isfinite(physics_period_s) or physics_period_s <= 0:
            raise ValueError("physics_period_s must be finite and positive")
        if max_dilation_factor < 1:
            raise ValueError("max_dilation_factor must be at least 1")
        if not isfinite(period_tolerance_s) or period_tolerance_s < 0:
            raise ValueError("period_tolerance_s must be finite and non-negative")
        self._delegate = delegate
        self.physics_period_s = physics_period_s
        self.max_dilation_factor = max_dilation_factor
        self.period_tolerance_s = period_tolerance_s

    def parameterize(self, path: JointPath, limits: JointLimits) -> Trajectory:
        baseline = self._delegate.parameterize(path, limits)
        self._require_physics_aligned_baseline(baseline)

        required_scale = required_uniform_time_scale(baseline, limits)
        applied_factor = max(1, ceil(required_scale - 1e-12))
        if applied_factor > self.max_dilation_factor:
            raise ValueError(
                "required time dilation exceeds configured maximum: "
                f"required={required_scale:.6g}, factor={applied_factor}, "
                f"max={self.max_dilation_factor}"
            )

        factor = float(applied_factor)
        waypoints = tuple(
            TrajectoryWaypoint(
                time_from_start_s=point.time_from_start_s * factor,
                positions=point.positions,
                velocities=(
                    tuple(value / factor for value in point.velocities)
                    if point.velocities is not None
                    else None
                ),
                accelerations=(
                    tuple(value / factor**2 for value in point.accelerations)
                    if point.accelerations is not None
                    else None
                ),
            )
            for point in baseline.waypoints
        )
        metadata = dict(baseline.metadata)
        metadata.update(
            {
                "timing_safety_layer": "physics-aligned-uniform-time-dilation",
                "baseline_source": baseline.source,
                "baseline_duration_s": baseline.duration_s,
                "required_continuous_time_scale": required_scale,
                "applied_integer_time_dilation_factor": applied_factor,
                "physics_period_s": self.physics_period_s,
                "controller_period_s": self.physics_period_s * applied_factor,
            }
        )
        return Trajectory(
            joint_names=baseline.joint_names,
            waypoints=waypoints,
            source=f"{baseline.source}+safe-time-dilation",
            frame_id=baseline.frame_id,
            limits_used=limits,
            interpolation=baseline.interpolation,
            scene_revision=baseline.scene_revision,
            transform_revision=baseline.transform_revision,
            metadata=metadata,
        )

    def _require_physics_aligned_baseline(self, trajectory: Trajectory) -> None:
        times = tuple(point.time_from_start_s for point in trajectory.waypoints)
        for left, right in pairwise(times):
            dt = right - left
            if abs(dt - self.physics_period_s) > self.period_tolerance_s:
                raise ValueError(
                    "baseline trajectory sample period must match physics_period_s; "
                    f"got {dt:.9g}, expected {self.physics_period_s:.9g}"
                )
