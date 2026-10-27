"""Conservative minimum-jerk timing for simulator-independent joint paths.

This is a deterministic fallback/validation parameterizer. It stops at every
geometric path knot, so it favors bounded derivatives over throughput.
"""

from __future__ import annotations

from math import ceil, isfinite, sqrt

from vla_robot_sorting.contracts.motion import (
    JointLimits,
    JointPath,
    Trajectory,
    TrajectoryWaypoint,
)

_MAX_S_D1 = 15.0 / 8.0
_MAX_S_D2 = 10.0 / sqrt(3.0)
_MAX_S_D3 = 60.0


def _segment_duration(
    delta: tuple[float, ...],
    limits: JointLimits,
    *,
    minimum_duration_s: float,
    duration_margin: float,
) -> float:
    required = minimum_duration_s
    for distance, vmax, amax, jmax in zip(
        (abs(value) for value in delta),
        limits.max_velocity_rad_s,
        limits.max_acceleration_rad_s2,
        limits.max_jerk_rad_s3,
        strict=True,
    ):
        required = max(
            required,
            _MAX_S_D1 * distance / vmax,
            sqrt(_MAX_S_D2 * distance / amax) if distance else 0.0,
            (_MAX_S_D3 * distance / jmax) ** (1.0 / 3.0) if distance else 0.0,
        )
    return required * duration_margin


def _quintic_state(u: float) -> tuple[float, float, float]:
    """Return normalized position, first derivative and second derivative."""

    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    ds = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    d2s = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
    return s, ds, d2s


class MinimumJerkStopParameterizer:
    """Create a C2 trajectory with zero velocity/acceleration at each path knot."""

    def __init__(
        self,
        *,
        controller_period_s: float = 1.0 / 250.0,
        minimum_segment_duration_s: float = 0.05,
        duration_margin: float = 1.15,
    ) -> None:
        if not isfinite(controller_period_s) or controller_period_s <= 0:
            raise ValueError("controller_period_s must be finite and positive")
        if not isfinite(minimum_segment_duration_s) or minimum_segment_duration_s <= 0:
            raise ValueError("minimum_segment_duration_s must be finite and positive")
        if not isfinite(duration_margin) or duration_margin < 1.0:
            raise ValueError("duration_margin must be finite and >= 1")
        self.controller_period_s = controller_period_s
        self.minimum_segment_duration_s = minimum_segment_duration_s
        self.duration_margin = duration_margin


    def parameterize(self, path: JointPath, limits: JointLimits) -> Trajectory:
        if path.joint_names != limits.joint_names:
            raise ValueError("path/limits joint order mismatch")

        waypoints: list[TrajectoryWaypoint] = []
        elapsed_s = 0.0

        for segment_index, (start, end) in enumerate(
            zip(path.positions[:-1], path.positions[1:], strict=True)
        ):
            delta = tuple(right - left for left, right in zip(start, end, strict=True))
            raw_duration = _segment_duration(
                delta,
                limits,
                minimum_duration_s=self.minimum_segment_duration_s,
                duration_margin=self.duration_margin,
            )
            sample_count = max(1, ceil(raw_duration / self.controller_period_s))
            duration_s = sample_count * self.controller_period_s

            first_sample = 0 if segment_index == 0 else 1
            for sample_index in range(first_sample, sample_count + 1):
                u = sample_index / sample_count
                s, ds, d2s = _quintic_state(u)
                positions = tuple(
                    left + s * distance for left, distance in zip(start, delta, strict=True)
                )
                velocities = tuple(distance * ds / duration_s for distance in delta)
                accelerations = tuple(distance * d2s / duration_s**2 for distance in delta)
                waypoints.append(
                    TrajectoryWaypoint(
                        time_from_start_s=elapsed_s + sample_index * self.controller_period_s,
                        positions=positions,
                        velocities=velocities,
                        accelerations=accelerations,
                    )
                )

            elapsed_s += duration_s


        return Trajectory(
            joint_names=path.joint_names,
            waypoints=tuple(waypoints),
            source="minimum-jerk-stop-parameterizer",
            limits_used=limits,
            interpolation="quintic-minimum-jerk-stop-at-knots",
            scene_revision=_optional_metadata_string(path, "scene_revision"),
            transform_revision=_optional_metadata_string(path, "transform_revision"),
            metadata={
                "controller_period_s": self.controller_period_s,
                "duration_margin": self.duration_margin,
                "geometric_path_source": path.source,
            },
        )


def _optional_metadata_string(path: JointPath, key: str) -> str | None:
    value = path.metadata.get(key)
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None
