"""Bounded Panthera/RoboTwin trajectory execution behind project contracts."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

from vla_robot_sorting.contracts.manipulation import (
    GripperCommand,
    GripperExecutionResult,
)
from vla_robot_sorting.contracts.motion import (
    ExecutionFeedback,
    ExecutionSegment,
    ExecutionStatus,
    JointState,
    TrajectoryWaypoint,
)
from vla_robot_sorting.simulation.panthera_observation import PANTHERA_JOINT_NAMES


class PantheraRoboTwinTrajectoryExecutor:
    """Execute one bounded single-arm segment on the live RoboTwin task environment."""

    def __init__(
        self,
        task_env: Any,
        *,
        scene_revision: Callable[[], str],
        transform_revision: Callable[[], str],
        tracking_tolerance_rad: float = 0.08,
        period_tolerance_s: float = 1e-6,
        physics_period_s: float | None = None,
        command_observer: Callable[[int, Any, Any, float], None] | None = None,
    ) -> None:
        if tracking_tolerance_rad < 0 or not isfinite(tracking_tolerance_rad):
            raise ValueError("tracking_tolerance_rad must be finite and non-negative")
        if period_tolerance_s < 0 or not isfinite(period_tolerance_s):
            raise ValueError("period_tolerance_s must be finite and non-negative")
        if physics_period_s is not None and (
            physics_period_s <= 0 or not isfinite(physics_period_s)
        ):
            raise ValueError("physics_period_s must be finite and positive when provided")
        self._task_env = task_env
        self._scene_revision = scene_revision
        self._transform_revision = transform_revision
        self._tracking_tolerance_rad = tracking_tolerance_rad
        self._period_tolerance_s = period_tolerance_s
        self._physics_period_s = physics_period_s
        if command_observer is not None and not callable(command_observer):
            raise ValueError("command_observer must be callable")
        self._command_observer = command_observer
        self.last_actual_state: JointState | None = None

    def execute(self, segment: ExecutionSegment) -> ExecutionFeedback:
        trajectory = segment.trajectory
        if trajectory.joint_names != PANTHERA_JOINT_NAMES:
            raise ValueError("trajectory joint order does not match Panthera joint1..joint6")
        if segment.controller_period_s <= 0:
            raise ValueError("controller_period_s must be positive")

        stale_reason = self._stale_reason(segment)
        if stale_reason is not None:
            return ExecutionFeedback(
                segment_id=segment.segment_id,
                status=ExecutionStatus.CANCELLED,
                executed_until_s=0.0,
                replan_required=True,
                reason=stale_reason,
            )

        command_indices = self._command_indices(segment)
        physics_steps = self._physics_steps_per_command(segment.controller_period_s)
        executed_until_s = 0.0
        previous = trajectory.waypoints[0]

        for index in command_indices:
            waypoint = trajectory.waypoints[index]
            stale_reason = self._stale_reason(segment)
            if stale_reason is not None:
                return ExecutionFeedback(
                    segment_id=segment.segment_id,
                    status=ExecutionStatus.CANCELLED,
                    executed_until_s=executed_until_s,
                    replan_required=True,
                    reason=stale_reason,
                )

            velocity = waypoint.velocities
            if velocity is None:
                velocity = self._derive_velocity(previous, waypoint)

            try:
                self._task_env.robot.set_arm_joints(
                    waypoint.positions,
                    velocity,
                    "left",
                )
                if self._command_observer is not None:
                    self._command_observer(
                        index, waypoint.positions, velocity, waypoint.time_from_start_s
                    )
                interval_start_s = previous.time_from_start_s
                for step_index in range(physics_steps):
                    self._task_env._step_scene()
                    if self._physics_period_s is None:
                        executed_until_s = waypoint.time_from_start_s
                    else:
                        executed_until_s = min(
                            waypoint.time_from_start_s,
                            interval_start_s + (step_index + 1) * self._physics_period_s,
                        )
                    stale_reason = self._stale_reason(segment)
                    if stale_reason is not None:
                        return ExecutionFeedback(
                            segment_id=segment.segment_id,
                            status=ExecutionStatus.CANCELLED,
                            executed_until_s=executed_until_s,
                            replan_required=True,
                            reason=stale_reason,
                        )
            except Exception as exc:  # vendor backend boundary
                return ExecutionFeedback(
                    segment_id=segment.segment_id,
                    status=ExecutionStatus.FAILED,
                    executed_until_s=executed_until_s,
                    replan_required=True,
                    reason=f"backend_error:{type(exc).__name__}:{exc}",
                )

            previous = waypoint
            executed_until_s = waypoint.time_from_start_s

        try:
            actual = self._actual_state(executed_until_s)
        except (TypeError, ValueError) as exc:
            return ExecutionFeedback(
                segment_id=segment.segment_id,
                status=ExecutionStatus.FAILED,
                executed_until_s=executed_until_s,
                replan_required=True,
                reason=f"invalid_proprioception:{exc}",
            )

        self.last_actual_state = actual
        expected = self._expected_stop_state(segment, command_indices)
        tracking_error = max(
            abs(commanded - measured)
            for commanded, measured in zip(expected.positions, actual.positions, strict=True)
        )

        if tracking_error > self._tracking_tolerance_rad:
            return ExecutionFeedback(
                segment_id=segment.segment_id,
                status=ExecutionStatus.FAILED,
                executed_until_s=executed_until_s,
                replan_required=True,
                reason=f"tracking_error:max_rad={tracking_error:.6g}",
            )

        return ExecutionFeedback(
            segment_id=segment.segment_id,
            status=ExecutionStatus.COMPLETED,
            executed_until_s=executed_until_s,
            replan_required=False,
        )

    def _stale_reason(self, segment: ExecutionSegment) -> str | None:
        scene = self._scene_revision()
        if scene != segment.scene_revision:
            return f"scene_revision_changed:{segment.scene_revision}->{scene}"
        transform = self._transform_revision()
        if transform != segment.transform_revision:
            return f"transform_revision_changed:{segment.transform_revision}->{transform}"
        return None

    def _command_indices(self, segment: ExecutionSegment) -> tuple[int, ...]:
        times = tuple(point.time_from_start_s for point in segment.trajectory.waypoints)
        indices = tuple(
            index
            for index, time_s in enumerate(times)
            if index > 0 and time_s <= segment.horizon_end_s + self._period_tolerance_s
        )
        if not indices:
            raise ValueError("execution horizon contains no command waypoint")
        last_time = times[indices[-1]]
        if abs(last_time - segment.horizon_end_s) > self._period_tolerance_s:
            raise ValueError("horizon_end_s must align with a trajectory waypoint")

        previous_time = times[0]
        for index in indices:
            dt = times[index] - previous_time
            if abs(dt - segment.controller_period_s) > self._period_tolerance_s:
                raise ValueError("trajectory sample period must match segment.controller_period_s")
            previous_time = times[index]
        return indices

    def _physics_steps_per_command(self, controller_period_s: float) -> int:
        if self._physics_period_s is None:
            return 1
        ratio = controller_period_s / self._physics_period_s
        steps = round(ratio)
        misalignment = abs(controller_period_s - steps * self._physics_period_s)
        if steps < 1 or misalignment > self._period_tolerance_s:
            raise ValueError("controller_period_s must be an integer multiple of physics_period_s")
        return steps

    @staticmethod
    def _derive_velocity(
        previous: TrajectoryWaypoint,
        waypoint: TrajectoryWaypoint,
    ) -> tuple[float, ...]:
        dt = waypoint.time_from_start_s - previous.time_from_start_s
        return tuple(
            (right - left) / dt
            for left, right in zip(previous.positions, waypoint.positions, strict=True)
        )

    def _actual_state(self, timestamp_s: float) -> JointState:
        raw_state = self._task_env.robot.get_left_arm_real_jointState()
        values = tuple(float(value) for value in raw_state)
        if len(values) != 7:
            raise ValueError("real joint state must contain six joints plus gripper")
        return JointState(
            joint_names=PANTHERA_JOINT_NAMES,
            positions=values[:6],
            timestamp_s=timestamp_s,
        )

    @staticmethod
    def _expected_stop_state(
        segment: ExecutionSegment,
        command_indices: tuple[int, ...],
    ) -> JointState:
        if segment.predicted_stop_state is not None:
            return segment.predicted_stop_state
        waypoint = segment.trajectory.waypoints[command_indices[-1]]
        return JointState(
            joint_names=segment.trajectory.joint_names,
            positions=waypoint.positions,
            timestamp_s=waypoint.time_from_start_s,
        )


class PantheraRoboTwinGripperExecutor:
    """Revision-locked single-arm gripper execution on RoboTwin.

    The executor intentionally drives the vendor gripper plan one physics step at
    a time so scene/transform revisions can be checked between commands.
    """

    def __init__(
        self,
        task_env: Any,
        *,
        scene_revision: Callable[[], str],
        transform_revision: Callable[[], str],
        position_tolerance: float = 0.08,
        contact_observed: Callable[[], bool] | None = None,
        holding_object: Callable[[], bool] | None = None,
    ) -> None:
        if position_tolerance < 0 or not isfinite(position_tolerance):
            raise ValueError("position_tolerance must be finite and non-negative")
        self._task_env = task_env
        self._scene_revision = scene_revision
        self._transform_revision = transform_revision
        self._position_tolerance = position_tolerance
        self._contact_observed = contact_observed or (lambda: False)
        self._holding_object = holding_object or (lambda: False)

    def execute_gripper(self, command: GripperCommand) -> GripperExecutionResult:
        stale = self._stale_reason(command)
        if stale is not None:
            return self._failure(stale)

        if abs(command.force_scale - 1.0) > 1e-12:
            return self._failure(
                "unsupported_force_scale:"
                f"{command.force_scale:.6g}; native force-control mapping not validated"
            )

        try:
            plan = self._task_env.set_gripper(
                left_pos=command.normalized_position,
                set_tag="left",
            )
            result = plan["result"]
            num_step = int(plan["num_step"])
            per_step = float(plan["per_step"])
        except Exception as exc:
            return self._failure(f"gripper_plan_error:{type(exc).__name__}:{exc}")

        if num_step < 1:
            return self._failure("gripper_plan_error:num_step must be positive")
        if len(result) < num_step:
            return self._failure("gripper_plan_error:result shorter than num_step")

        try:
            for index in range(num_step):
                stale = self._stale_reason(command)
                if stale is not None:
                    return self._failure(stale)
                self._task_env.robot.set_gripper(
                    float(result[index]),
                    "left",
                    per_step,
                )
                self._task_env._step_scene()
        except Exception as exc:
            return self._failure(f"gripper_backend_error:{type(exc).__name__}:{exc}")

        stale = self._stale_reason(command)
        if stale is not None:
            return self._failure(stale)

        actual = self._actual_position()
        error = abs(actual - command.normalized_position)
        if error > self._position_tolerance:
            return GripperExecutionResult(
                completed=False,
                actual_normalized_position=actual,
                contact_observed=bool(self._contact_observed()),
                holding_object=bool(self._holding_object()),
                reason=f"gripper_tracking_error:normalized={error:.6g}",
            )

        return GripperExecutionResult(
            completed=True,
            actual_normalized_position=actual,
            contact_observed=bool(self._contact_observed()),
            holding_object=bool(self._holding_object()),
        )

    def _stale_reason(self, command: GripperCommand) -> str | None:
        scene = self._scene_revision()
        if scene != command.scene_revision:
            return f"scene_revision_changed:{command.scene_revision}->{scene}"
        transform = self._transform_revision()
        if transform != command.transform_revision:
            return f"transform_revision_changed:{command.transform_revision}->{transform}"
        return None

    def _actual_position(self) -> float:
        value = float(self._task_env.robot.get_left_gripper_val())
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("RoboTwin gripper state must be finite and in [0, 1]")
        return value

    def _failure(self, reason: str) -> GripperExecutionResult:
        try:
            actual = self._actual_position()
        except Exception:
            actual = 0.0
        return GripperExecutionResult(
            completed=False,
            actual_normalized_position=actual,
            contact_observed=bool(self._contact_observed()),
            holding_object=bool(self._holding_object()),
            reason=reason,
        )
