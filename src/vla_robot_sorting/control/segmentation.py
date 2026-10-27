"""Build revision-locked bounded execution segments from timed trajectories."""

from __future__ import annotations

from math import isfinite

from vla_robot_sorting.contracts.motion import (
    ExecutionSegment,
    JointState,
    PlanningContext,
    Trajectory,
)


def build_execution_segment(
    trajectory: Trajectory,
    context: PlanningContext,
    *,
    segment_id: str,
    requested_horizon_s: float,
    controller_period_s: float,
    alignment_tolerance_s: float = 1e-6,
) -> ExecutionSegment:
    """Choose the greatest trajectory waypoint not beyond the requested horizon."""

    if not segment_id.strip():
        raise ValueError("segment_id must be non-empty")
    if not isfinite(requested_horizon_s) or requested_horizon_s <= 0:
        raise ValueError("requested_horizon_s must be finite and positive")
    if not isfinite(controller_period_s) or controller_period_s <= 0:
        raise ValueError("controller_period_s must be finite and positive")
    if alignment_tolerance_s < 0 or not isfinite(alignment_tolerance_s):
        raise ValueError("alignment_tolerance_s must be finite and non-negative")


    if (
        trajectory.scene_revision is not None
        and trajectory.scene_revision != context.scene_revision
    ):
        raise ValueError("trajectory scene_revision does not match planning context")
    if (
        trajectory.transform_revision is not None
        and trajectory.transform_revision != context.transform_revision
    ):
        raise ValueError("trajectory transform_revision does not match planning context")

    upper = min(requested_horizon_s, trajectory.duration_s)
    candidates = [
        waypoint
        for waypoint in trajectory.waypoints[1:]
        if waypoint.time_from_start_s <= upper + alignment_tolerance_s
    ]
    if not candidates:
        raise ValueError("requested horizon does not include a command waypoint")

    stop = candidates[-1]
    predicted = JointState(
        joint_names=trajectory.joint_names,
        positions=stop.positions,
        velocities=stop.velocities,
        timestamp_s=stop.time_from_start_s,
    )


    return ExecutionSegment(
        segment_id=segment_id,
        trajectory=trajectory,
        scene_revision=context.scene_revision,
        transform_revision=context.transform_revision,
        horizon_end_s=stop.time_from_start_s,
        controller_period_s=controller_period_s,
        predicted_stop_state=predicted,
    )
