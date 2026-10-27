"""Deterministic motion-stack test doubles.

These are for Windows unit/conformance tests only. They are not simulator backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vla_robot_sorting.contracts.motion import (
    ExecutionFeedback,
    ExecutionSegment,
    ExecutionStatus,
    IKResult,
    IKSolution,
    JointLimits,
    JointPath,
    JointState,
    MotionFailure,
    MotionFailureCode,
    MotionGoal,
    PlannerResult,
    PlanningContext,
    SafetyReport,
    Trajectory,
    TrajectoryWaypoint,
)


@dataclass(slots=True)
class DeterministicIKSolver:
    """Map a 6-DoF goal to a fixed valid joint vector, or a forced failure."""

    joint_names: tuple[str, ...]
    solution_positions: tuple[float, ...]
    force_failure: MotionFailureCode | None = None

    def solve(
        self,
        goal: MotionGoal,
        seed_state: JointState,
        context: PlanningContext,
        scene: object,
    ) -> IKResult:
        del goal, context, scene
        if seed_state.joint_names != self.joint_names:
            raise ValueError("seed_state joint order mismatch")

        if self.force_failure is not None:
            return IKResult(
                solution=None,
                failure=MotionFailure(
                    code=self.force_failure,
                    message="forced deterministic IK failure",
                    recoverable=True,
                ),
                solve_time_s=0.001,
                solver_id="fake-ik",
            )
        return IKResult(
            solution=IKSolution(
                joint_names=self.joint_names,
                positions=self.solution_positions,
                position_error_m=0.0,
                orientation_error_rad=0.0,
            ),
            failure=None,
            solve_time_s=0.001,
            solver_id="fake-ik",
        )


@dataclass(slots=True)
class StraightLineMotionPlanner:
    """Create a two-point joint path to a fixed target for conformance tests."""

    joint_names: tuple[str, ...]
    target_positions: tuple[float, ...]
    force_failure: MotionFailureCode | None = None

    def plan(
        self,
        start_state: JointState,
        goal: MotionGoal,
        context: PlanningContext,
        scene: object,
    ) -> PlannerResult:
        del goal, scene
        if start_state.joint_names != self.joint_names:
            raise ValueError("start_state joint order mismatch")
        if self.force_failure is not None:
            return PlannerResult(
                path=None,
                failure=MotionFailure(
                    code=self.force_failure,
                    message="forced deterministic planner failure",
                    recoverable=True,
                ),
                planning_time_s=0.002,
                planner_id="fake-planner",
            )

        return PlannerResult(
            path=JointPath(
                joint_names=self.joint_names,
                positions=(start_state.positions, self.target_positions),
                source="fake-planner",
                metadata={"scene_revision": context.scene_revision},
            ),
            failure=None,
            planning_time_s=0.002,
            planner_id="fake-planner",
        )


@dataclass(slots=True)
class UniformTrajectoryParameterizer:
    """Parameterize every geometric path segment at a fixed duration."""

    segment_duration_s: float = 1.0
    def parameterize(self, path: JointPath, limits: JointLimits) -> Trajectory:
        if path.joint_names != limits.joint_names:
            raise ValueError("path/limits joint order mismatch")
        if self.segment_duration_s <= 0:
            raise ValueError("segment_duration_s must be positive")

        waypoints = tuple(
            TrajectoryWaypoint(
                time_from_start_s=index * self.segment_duration_s,
                positions=position,
            )
            for index, position in enumerate(path.positions)
        )
        return Trajectory(
            joint_names=path.joint_names,
            waypoints=waypoints,
            source="fake-parameterizer",
            limits_used=limits,
            interpolation="linear-test-double",
            scene_revision=str(path.metadata.get("scene_revision", "unknown")),
            transform_revision="fake-transform",
        )


@dataclass(slots=True)
class RecordingTrajectoryExecutor:
    """Record bounded segments and return deterministic completion feedback."""

    executed_segments: list[ExecutionSegment] = field(default_factory=list)
    def execute(self, segment: ExecutionSegment) -> ExecutionFeedback:
        self.executed_segments.append(segment)
        return ExecutionFeedback(
            segment_id=segment.segment_id,
            status=ExecutionStatus.COMPLETED,
            executed_until_s=segment.horizon_end_s,
            replan_required=False,
        )


@dataclass(slots=True)
class PassThroughSafetyValidator:
    """Return a clean report; scene-specific safety remains adapter-owned."""

    def validate_path(
        self,
        path: JointPath,
        context: PlanningContext,
        scene: object,
    ) -> SafetyReport:
        del path, context, scene
        return SafetyReport()

    def validate_trajectory(
        self,
        trajectory: Trajectory,
        limits: JointLimits,
        context: PlanningContext,
        scene: object,
    ) -> SafetyReport:
        del trajectory, limits, context, scene
        return SafetyReport()
