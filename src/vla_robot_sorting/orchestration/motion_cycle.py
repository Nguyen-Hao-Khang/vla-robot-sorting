"""One bounded plan-validate-parameterize-execute cycle.

Closed-loop repetition/recovery belongs to S1.5; this module establishes the
single-cycle contract that S1.5 will repeatedly invoke.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from vla_robot_sorting.contracts.interfaces import (
    MotionPlanner,
    SafetyValidator,
    TrajectoryExecutor,
    TrajectoryParameterizer,
)
from vla_robot_sorting.contracts.motion import (
    ExecutionFeedback,
    ExecutionStatus,
    JointLimits,
    JointState,
    MotionGoal,
    PlannerResult,
    PlanningContext,
    SafetyReport,
    Trajectory,
)
from vla_robot_sorting.control import build_execution_segment


class MotionCycleStatus(str, Enum):
    PLANNING_FAILED = "planning_failed"
    PATH_UNSAFE = "path_unsafe"
    TRAJECTORY_UNSAFE = "trajectory_unsafe"
    EXECUTED = "executed"
    EXECUTION_FAILED = "execution_failed"


@dataclass(frozen=True, slots=True)
class MotionCycleResult:
    status: MotionCycleStatus
    plan: PlannerResult
    path_safety: SafetyReport | None = None
    trajectory: Trajectory | None = None
    trajectory_safety: SafetyReport | None = None
    execution: ExecutionFeedback | None = None

    @property
    def succeeded(self) -> bool:
        return (
            self.status is MotionCycleStatus.EXECUTED
            and self.execution is not None
            and self.execution.status is ExecutionStatus.COMPLETED
        )


class BoundedMotionController:
    """Execute one revision-locked bounded control cycle."""

    def __init__(
        self,
        *,
        planner: MotionPlanner,
        parameterizer: TrajectoryParameterizer,
        safety: SafetyValidator,
        executor: TrajectoryExecutor,
        limits: JointLimits,
        controller_period_s: float,
        horizon_s: float,
    ) -> None:
        if controller_period_s <= 0:
            raise ValueError("controller_period_s must be positive")
        if horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        self._planner = planner
        self._parameterizer = parameterizer
        self._safety = safety
        self._executor = executor
        self._limits = limits
        self._controller_period_s = controller_period_s
        self._horizon_s = horizon_s


    def run(
        self,
        start_state: JointState,
        goal: MotionGoal,
        context: PlanningContext,
        scene: Any,
        *,
        segment_id: str,
    ) -> MotionCycleResult:
        plan = self._planner.plan(start_state, goal, context, scene)
        if not plan.succeeded:
            return MotionCycleResult(
                status=MotionCycleStatus.PLANNING_FAILED,
                plan=plan,
            )

        assert plan.path is not None
        path_safety = self._safety.validate_path(plan.path, context, scene)
        if not path_safety.is_safe:
            return MotionCycleResult(
                status=MotionCycleStatus.PATH_UNSAFE,
                plan=plan,
                path_safety=path_safety,
            )

        trajectory = self._parameterizer.parameterize(plan.path, self._limits)
        trajectory_safety = self._safety.validate_trajectory(
            trajectory,
            self._limits,
            context,
            scene,
        )


        if not trajectory_safety.is_safe:
            return MotionCycleResult(
                status=MotionCycleStatus.TRAJECTORY_UNSAFE,
                plan=plan,
                path_safety=path_safety,
                trajectory=trajectory,
                trajectory_safety=trajectory_safety,
            )

        segment = build_execution_segment(
            trajectory,
            context,
            segment_id=segment_id,
            requested_horizon_s=self._horizon_s,
            controller_period_s=self._controller_period_s,
        )
        execution = self._executor.execute(segment)
        status = (
            MotionCycleStatus.EXECUTED
            if execution.status is ExecutionStatus.COMPLETED
            else MotionCycleStatus.EXECUTION_FAILED
        )
        return MotionCycleResult(
            status=status,
            plan=plan,
            path_safety=path_safety,
            trajectory=trajectory,
            trajectory_safety=trajectory_safety,
            execution=execution,
        )
