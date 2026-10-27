"""Deterministic motion metrics that do not depend on a simulator backend."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import acos, sqrt

from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.contracts.motion import Trajectory


@dataclass(frozen=True, slots=True)
class PoseError:
    translation_m: float
    rotation_rad: float


@dataclass(frozen=True, slots=True)
class MotionKinematicsMetrics:
    duration_s: float
    joint_path_length_rad: float
    peak_velocity_rad_s: float
    rms_velocity_rad_s: float
    peak_acceleration_rad_s2: float
    rms_acceleration_rad_s2: float
    peak_jerk_rad_s3: float
    rms_jerk_rad_s3: float


def _differentiate(
    samples: tuple[tuple[float, ...], ...],
    times: tuple[float, ...],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    if len(samples) != len(times):
        raise ValueError("samples/times length mismatch")
    if len(samples) < 2:
        return (), ()

    derivatives: list[tuple[float, ...]] = []
    derivative_times: list[float] = []
    for left, right, t0, t1 in zip(
        samples[:-1], samples[1:], times[:-1], times[1:], strict=True
    ):
        dt = t1 - t0
        if dt <= 0:
            raise ValueError("times must be strictly increasing")
        derivatives.append(
            tuple((r_value - l_value) / dt for l_value, r_value in zip(left, right, strict=True))
        )
        derivative_times.append((t0 + t1) / 2.0)
    return tuple(derivatives), tuple(derivative_times)


def _peak_and_rms(samples: tuple[tuple[float, ...], ...]) -> tuple[float, float]:
    values = [abs(value) for sample in samples for value in sample]
    if not values:
        return 0.0, 0.0
    peak = max(values)
    rms = sqrt(sum(value * value for value in values) / len(values))
    return peak, rms


def joint_path_length_rad(trajectory: Trajectory) -> float:
    total = 0.0
    points = tuple(waypoint.positions for waypoint in trajectory.waypoints)
    for left, right in pairwise(points):
        total += sqrt(
            sum((r_value - l_value) ** 2 for l_value, r_value in zip(left, right, strict=True))
        )
    return total


def trajectory_kinematics(trajectory: Trajectory) -> MotionKinematicsMetrics:
    samples = tuple(waypoint.positions for waypoint in trajectory.waypoints)
    times = tuple(waypoint.time_from_start_s for waypoint in trajectory.waypoints)

    velocity, velocity_times = _differentiate(samples, times)
    acceleration, acceleration_times = _differentiate(velocity, velocity_times)
    jerk, _ = _differentiate(acceleration, acceleration_times)

    peak_velocity, rms_velocity = _peak_and_rms(velocity)
    peak_acceleration, rms_acceleration = _peak_and_rms(acceleration)
    peak_jerk, rms_jerk = _peak_and_rms(jerk)

    return MotionKinematicsMetrics(
        duration_s=trajectory.duration_s,
        joint_path_length_rad=joint_path_length_rad(trajectory),
        peak_velocity_rad_s=peak_velocity,
        rms_velocity_rad_s=rms_velocity,
        peak_acceleration_rad_s2=peak_acceleration,
        rms_acceleration_rad_s2=rms_acceleration,
        peak_jerk_rad_s3=peak_jerk,
        rms_jerk_rad_s3=rms_jerk,
    )


def pose_error(expected: Pose6D, actual: Pose6D) -> PoseError:
    if expected.frame_id != actual.frame_id:
        raise ValueError("Pose error requires matching frame_id values")

    translation = sqrt(
        sum(
            (actual_value - expected_value) ** 2
            for expected_value, actual_value in zip(
                expected.position_xyz, actual.position_xyz, strict=True
            )
        )
    )

    dot = abs(
        sum(
            expected_value * actual_value
            for expected_value, actual_value in zip(
                expected.orientation_xyzw, actual.orientation_xyzw, strict=True
            )
        )
    )
    dot = min(1.0, max(-1.0, dot))
    return PoseError(translation_m=translation, rotation_rad=2.0 * acos(dot))


@dataclass(frozen=True, slots=True)
class SpliceDiscontinuity:
    """Boundary mismatch when replacing one trajectory with another."""

    position_rad_max: float
    velocity_rad_s_max: float | None
    acceleration_rad_s2_max: float | None


def _max_component_delta(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    return max(
        abs(right_value - left_value)
        for left_value, right_value in zip(left, right, strict=True)
    )


def trajectory_splice_discontinuity(
    previous: Trajectory,
    following: Trajectory,
) -> SpliceDiscontinuity:
    if previous.joint_names != following.joint_names:
        raise ValueError("Splice comparison requires matching joint_names")
    previous_positions = tuple(point.positions for point in previous.waypoints)
    following_positions = tuple(point.positions for point in following.waypoints)
    previous_times = tuple(point.time_from_start_s for point in previous.waypoints)
    following_times = tuple(point.time_from_start_s for point in following.waypoints)

    position_delta = _max_component_delta(
        previous_positions[-1],
        following_positions[0],
    )

    previous_velocity, previous_velocity_times = _differentiate(
        previous_positions, previous_times
    )
    following_velocity, following_velocity_times = _differentiate(
        following_positions, following_times
    )
    velocity_delta = (
        _max_component_delta(previous_velocity[-1], following_velocity[0])
        if previous_velocity and following_velocity
        else None
    )

    previous_acceleration, _ = _differentiate(
        previous_velocity, previous_velocity_times
    )
    following_acceleration, _ = _differentiate(
        following_velocity, following_velocity_times
    )
    acceleration_delta = (
        _max_component_delta(previous_acceleration[-1], following_acceleration[0])
        if previous_acceleration and following_acceleration
        else None
    )

    return SpliceDiscontinuity(
        position_rad_max=position_delta,
        velocity_rad_s_max=velocity_delta,
        acceleration_rad_s2_max=acceleration_delta,
    )
