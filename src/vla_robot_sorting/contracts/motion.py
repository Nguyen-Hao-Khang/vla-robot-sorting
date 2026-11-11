"""Vendor-neutral motion and control data types: joint states, paths, trajectories, safety reports.

All units are SI: seconds, metres, radians, and their derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any

from .models import Pose6D


def _finite_tuple(values: tuple[float, ...], label: str) -> None:
    if not values or not all(isfinite(value) for value in values):
        raise ValueError(f"{label} must contain finite values")


def _same_dof(values: tuple[float, ...], joint_names: tuple[str, ...], label: str) -> None:
    if len(values) != len(joint_names):
        raise ValueError(f"{label} must match joint_names length")


class MotionFailureCode(str, Enum):
    """Stable failure classes independent of any planner implementation."""

    INVALID_GOAL = "invalid_goal"
    TRANSFORM_STALE = "transform_stale"
    IK_UNREACHABLE = "ik_unreachable"
    COLLISION = "collision"
    OPTIMIZATION_FAILED = "optimization_failed"
    TIMEOUT = "timeout"
    DYNAMIC_LIMIT_FAILED = "dynamic_limit_failed"
    TRACKING_ERROR = "tracking_error"
    EXECUTION_STALE = "execution_stale"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    INTERNAL_ERROR = "internal_error"


class ExecutionStatus(str, Enum):
    """Lifecycle state for a bounded receding-horizon execution segment."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PlanningContext:
    """Versioned scene/transform context used to reject stale plans."""

    scene_revision: str
    transform_revision: str
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        if not self.scene_revision.strip() or not self.transform_revision.strip():
            raise ValueError("scene_revision and transform_revision must be non-empty")
        if self.timestamp_s is not None and (
            not isfinite(self.timestamp_s) or self.timestamp_s < 0
        ):
            raise ValueError("timestamp_s must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class JointState:
    """Robot joint state with an explicit joint ordering."""

    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...] | None = None
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        if not self.joint_names or any(not name.strip() for name in self.joint_names):
            raise ValueError("joint_names must be non-empty")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        _finite_tuple(self.positions, "positions")
        _same_dof(self.positions, self.joint_names, "positions")
        if self.velocities is not None:
            _finite_tuple(self.velocities, "velocities")
            _same_dof(self.velocities, self.joint_names, "velocities")
        if self.timestamp_s is not None and (
            not isfinite(self.timestamp_s) or self.timestamp_s < 0
        ):
            raise ValueError("timestamp_s must be a finite non-negative value")


@dataclass(frozen=True, slots=True)
class JointLimits:
    """Per-joint position and derivative limits in the declared joint order."""

    joint_names: tuple[str, ...]
    lower_position_rad: tuple[float, ...]
    upper_position_rad: tuple[float, ...]
    max_velocity_rad_s: tuple[float, ...]
    max_acceleration_rad_s2: tuple[float, ...]
    max_jerk_rad_s3: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.joint_names
            or any(not name.strip() for name in self.joint_names)
            or len(set(self.joint_names)) != len(self.joint_names)
        ):
            raise ValueError("joint_names must be non-empty and unique")
        series = (
            ("lower_position_rad", self.lower_position_rad),
            ("upper_position_rad", self.upper_position_rad),
            ("max_velocity_rad_s", self.max_velocity_rad_s),
            ("max_acceleration_rad_s2", self.max_acceleration_rad_s2),
            ("max_jerk_rad_s3", self.max_jerk_rad_s3),
        )
        for label, values in series:
            _finite_tuple(values, label)
            _same_dof(values, self.joint_names, label)
        for lower, upper in zip(
            self.lower_position_rad, self.upper_position_rad, strict=True
        ):
            if lower >= upper:
                raise ValueError("lower_position_rad must be < upper_position_rad")
        for label, values in series[2:]:
            if any(value <= 0 for value in values):
                raise ValueError(f"{label} values must be positive")


@dataclass(frozen=True, slots=True)
class JointPath:
    """Untimed collision-aware joint-space path returned by a planner."""

    joint_names: tuple[str, ...]
    positions: tuple[tuple[float, ...], ...]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.joint_names
            or any(not name.strip() for name in self.joint_names)
            or len(set(self.joint_names)) != len(self.joint_names)
        ):
            raise ValueError("JointPath joint_names must be non-empty and unique")
        if len(self.positions) < 2:
            raise ValueError("JointPath requires at least two positions")
        if not self.source.strip():
            raise ValueError("JointPath.source must be non-empty")
        for index, position in enumerate(self.positions):
            _finite_tuple(position, f"positions[{index}]")
            _same_dof(position, self.joint_names, f"positions[{index}]")


@dataclass(frozen=True, slots=True)
class TrajectoryWaypoint:
    """Timed joint waypoint; derivative fields are optional planner outputs."""

    time_from_start_s: float
    positions: tuple[float, ...]
    velocities: tuple[float, ...] | None = None
    accelerations: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.time_from_start_s) or self.time_from_start_s < 0:
            raise ValueError("time_from_start_s must be a finite non-negative value")
        _finite_tuple(self.positions, "positions")
        if self.velocities is not None:
            _finite_tuple(self.velocities, "velocities")
            if len(self.velocities) != len(self.positions):
                raise ValueError("velocities must match positions length")
        if self.accelerations is not None:
            _finite_tuple(self.accelerations, "accelerations")
            if len(self.accelerations) != len(self.positions):
                raise ValueError("accelerations must match positions length")


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Time-parameterized trajectory ready for safety validation/execution."""

    joint_names: tuple[str, ...]
    waypoints: tuple[TrajectoryWaypoint, ...]
    source: str
    frame_id: str = "joint"
    limits_used: JointLimits | None = None
    interpolation: str | None = None
    scene_revision: str | None = None
    transform_revision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError("Trajectory requires at least two waypoints")
        if not self.source.strip() or not self.frame_id.strip():
            raise ValueError("Trajectory source/frame_id must be non-empty")
        if (
            not self.joint_names
            or any(not name.strip() for name in self.joint_names)
            or len(set(self.joint_names)) != len(self.joint_names)
        ):
            raise ValueError("joint_names must be non-empty and unique")

        if abs(self.waypoints[0].time_from_start_s) > 1e-12:
            raise ValueError("Trajectory must start at time_from_start_s=0")
        if self.limits_used is not None and self.limits_used.joint_names != self.joint_names:
            raise ValueError("Trajectory limits_used joint order must match joint_names")
        for label, value in (
            ("interpolation", self.interpolation),
            ("scene_revision", self.scene_revision),
            ("transform_revision", self.transform_revision),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"Trajectory {label} must be non-empty when provided")

        previous_time = -1.0
        for index, waypoint in enumerate(self.waypoints):
            if waypoint.time_from_start_s <= previous_time:
                raise ValueError("Trajectory waypoint times must be strictly increasing")
            previous_time = waypoint.time_from_start_s
            _same_dof(waypoint.positions, self.joint_names, f"waypoints[{index}].positions")
            if waypoint.velocities is not None:
                _same_dof(
                    waypoint.velocities, self.joint_names, f"waypoints[{index}].velocities"
                )
            if waypoint.accelerations is not None:
                _same_dof(
                    waypoint.accelerations,
                    self.joint_names,
                    f"waypoints[{index}].accelerations",
                )

    @property
    def duration_s(self) -> float:
        return self.waypoints[-1].time_from_start_s - self.waypoints[0].time_from_start_s


@dataclass(frozen=True, slots=True)
class MotionFailure:
    """Planner/IK failure that remains meaningful across third-party backends."""

    code: MotionFailureCode
    message: str
    recoverable: bool
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("MotionFailure.message must be non-empty")


@dataclass(frozen=True, slots=True)
class PlannerResult:
    """Exactly one of path or failure must be present."""

    path: JointPath | None
    failure: MotionFailure | None
    planning_time_s: float
    planner_id: str

    def __post_init__(self) -> None:
        if (self.path is None) == (self.failure is None):
            raise ValueError("PlannerResult requires exactly one of path or failure")
        if not isfinite(self.planning_time_s) or self.planning_time_s < 0:
            raise ValueError("planning_time_s must be a finite non-negative value")
        if not self.planner_id.strip():
            raise ValueError("planner_id must be non-empty")

    @property
    def succeeded(self) -> bool:
        return self.path is not None


@dataclass(frozen=True, slots=True)
class IKSolution:
    """Validated IK solution and residual error metrics."""

    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    position_error_m: float
    orientation_error_rad: float

    def __post_init__(self) -> None:
        if (
            not self.joint_names
            or any(not name.strip() for name in self.joint_names)
            or len(set(self.joint_names)) != len(self.joint_names)
        ):
            raise ValueError("IKSolution joint_names must be non-empty and unique")
        _finite_tuple(self.positions, "positions")
        _same_dof(self.positions, self.joint_names, "positions")
        if self.position_error_m < 0 or not isfinite(self.position_error_m):
            raise ValueError("position_error_m must be finite and non-negative")
        if self.orientation_error_rad < 0 or not isfinite(self.orientation_error_rad):
            raise ValueError("orientation_error_rad must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class IKResult:
    """Exactly one of solution or failure must be present."""

    solution: IKSolution | None
    failure: MotionFailure | None
    solve_time_s: float
    solver_id: str

    def __post_init__(self) -> None:
        if (self.solution is None) == (self.failure is None):
            raise ValueError("IKResult requires exactly one of solution or failure")
        if not isfinite(self.solve_time_s) or self.solve_time_s < 0:
            raise ValueError("solve_time_s must be a finite non-negative value")
        if not self.solver_id.strip():
            raise ValueError("solver_id must be non-empty")

    @property
    def succeeded(self) -> bool:
        return self.solution is not None


@dataclass(frozen=True, slots=True)
class SafetyViolation:
    """A machine-readable violation found before or during execution."""

    code: str
    message: str
    waypoint_index: int | None = None

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise ValueError("SafetyViolation code/message must be non-empty")
        if self.waypoint_index is not None and self.waypoint_index < 0:
            raise ValueError("waypoint_index must be non-negative")


@dataclass(frozen=True, slots=True)
class SafetyReport:
    """Result of validating a path/trajectory against locked constraints."""

    violations: tuple[SafetyViolation, ...] = ()

    @property
    def is_safe(self) -> bool:
        return not self.violations


@dataclass(frozen=True, slots=True)
class ExecutionSegment:
    """Bounded trajectory slice executed under one scene revision."""

    segment_id: str
    trajectory: Trajectory
    scene_revision: str
    transform_revision: str
    horizon_end_s: float
    controller_period_s: float
    predicted_stop_state: JointState | None = None

    def __post_init__(self) -> None:
        if (
            not self.segment_id.strip()
            or not self.scene_revision.strip()
            or not self.transform_revision.strip()
        ):
            raise ValueError(
                "segment_id, scene_revision and transform_revision must be non-empty"
            )
        if not isfinite(self.horizon_end_s) or self.horizon_end_s <= 0:
            raise ValueError("horizon_end_s must be finite and positive")
        if self.horizon_end_s > self.trajectory.duration_s + 1e-9:
            raise ValueError("horizon_end_s cannot exceed trajectory duration")
        if not isfinite(self.controller_period_s) or self.controller_period_s <= 0:
            raise ValueError("controller_period_s must be finite and positive")
        if (
            self.predicted_stop_state is not None
            and self.predicted_stop_state.joint_names != self.trajectory.joint_names
        ):
            raise ValueError("predicted_stop_state joint order must match trajectory")


@dataclass(frozen=True, slots=True)
class ExecutionFeedback:
    """Local execution result used to decide continue/replan/recover/abort."""

    segment_id: str
    status: ExecutionStatus
    executed_until_s: float
    replan_required: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.segment_id.strip():
            raise ValueError("segment_id must be non-empty")
        if not isfinite(self.executed_until_s) or self.executed_until_s < 0:
            raise ValueError("executed_until_s must be finite and non-negative")
        if self.replan_required and not (self.reason and self.reason.strip()):
            raise ValueError("replan_required feedback must include a reason")


@dataclass(frozen=True, slots=True)
class WorkspaceBounds:
    """Axis-aligned Cartesian safety boundary expressed in one frame."""

    frame_id: str
    minimum_xyz_m: tuple[float, float, float]
    maximum_xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("WorkspaceBounds.frame_id must be non-empty")
        if len(self.minimum_xyz_m) != 3 or len(self.maximum_xyz_m) != 3:
            raise ValueError("WorkspaceBounds min/max must each contain exactly 3 values")
        values = (*self.minimum_xyz_m, *self.maximum_xyz_m)
        if not all(isfinite(value) for value in values):
            raise ValueError("WorkspaceBounds values must be finite")
        if any(
            lower >= upper
            for lower, upper in zip(self.minimum_xyz_m, self.maximum_xyz_m, strict=True)
        ):
            raise ValueError("WorkspaceBounds minimum must be < maximum on every axis")


@dataclass(frozen=True, slots=True)
class MotionGoal:
    """Full SE(3) goal plus optional task-level execution constraints."""

    pose: Pose6D
    gripper_width_m: float | None = None
    workspace: WorkspaceBounds | None = None

    def __post_init__(self) -> None:
        if self.gripper_width_m is not None and (
            not isfinite(self.gripper_width_m) or self.gripper_width_m < 0
        ):
            raise ValueError("gripper_width_m must be finite and non-negative")
        if self.workspace is not None and self.workspace.frame_id != self.pose.frame_id:
            raise ValueError("MotionGoal pose/workspace frame_id must match")
