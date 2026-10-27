"""Protocols that keep third-party implementations behind project-owned boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .manipulation import GripperCommand, GripperExecutionResult
from .models import GraspTarget, SkillCommand
from .motion import (
    ExecutionFeedback,
    ExecutionSegment,
    IKResult,
    JointLimits,
    JointPath,
    JointState,
    MotionGoal,
    PlannerResult,
    PlanningContext,
    SafetyReport,
    Trajectory,
)
from .simulation import SimulatorObservation


class SemanticReasoner(Protocol):
    async def resolve(self, command_text: str, observations: Any) -> SkillCommand:
        """Convert language and observations into validated semantic intent."""


class GroundingBackend(Protocol):
    def ground(self, command: SkillCommand, observations: Any) -> Any:
        """Resolve semantic targets and destinations into scene entities."""


class GraspPlanner(Protocol):
    def generate(self, grounded_target: Any, observations: Any) -> Sequence[GraspTarget]:
        """Return ranked or rankable full 6-DoF grasp candidates."""


class IKSolver(Protocol):
    def solve(
        self,
        goal: MotionGoal,
        seed_state: JointState,
        context: PlanningContext,
        scene: Any,
    ) -> IKResult:
        """Solve a full SE(3) goal without leaking backend-specific result objects."""


class MotionPlanner(Protocol):
    def plan(
        self,
        start_state: JointState,
        goal: MotionGoal,
        context: PlanningContext,
        scene: Any,
    ) -> PlannerResult:
        """Return a collision-aware untimed joint-space path or classified failure."""


class TrajectoryParameterizer(Protocol):
    def parameterize(self, path: JointPath, limits: JointLimits) -> Trajectory:
        """Convert a geometric joint path into a continuous timed trajectory."""


class SafetyValidator(Protocol):
    def validate_path(
        self,
        path: JointPath,
        context: PlanningContext,
        scene: Any,
    ) -> SafetyReport:
        """Validate an untimed path against geometric/workspace constraints."""

    def validate_trajectory(
        self,
        trajectory: Trajectory,
        limits: JointLimits,
        context: PlanningContext,
        scene: Any,
    ) -> SafetyReport:
        """Validate timed trajectory limits and scene safety."""


class TrajectoryExecutor(Protocol):
    def execute(self, segment: ExecutionSegment) -> ExecutionFeedback:
        """Execute one bounded segment and return local feedback for replanning."""


class GripperExecutor(Protocol):
    def execute_gripper(self, command: GripperCommand) -> GripperExecutionResult:
        """Execute one revision-locked gripper command."""


class SimulatorObservationSource(Protocol):
    def observe(self) -> SimulatorObservation:
        """Read visual and measured proprioceptive simulator observations."""


class SimulatorAdapter(SimulatorObservationSource, Protocol):
    def execute(self, segment: ExecutionSegment) -> ExecutionFeedback:
        """Execute a validated bounded segment through the simulator backend."""
