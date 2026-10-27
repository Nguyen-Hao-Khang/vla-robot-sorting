"""Bounded recovery policy for S1.5 closed-loop adaptation."""

from __future__ import annotations

from dataclasses import replace

from vla_robot_sorting.contracts.recovery import (
    RecoveryAction,
    RecoveryBudget,
    RecoveryDecision,
    RecoveryEvent,
    RecoveryState,
    RecoveryTrigger,
    SceneChangeKind,
)


class ClosedLoopRecoveryPolicy:
    """Choose one high-level recovery action without issuing actuator commands."""

    def __init__(self, budget: RecoveryBudget | None = None) -> None:
        self.budget = budget or RecoveryBudget()

    def decide(self, event: RecoveryEvent, state: RecoveryState) -> RecoveryDecision:
        if not event.recoverable:
            return RecoveryDecision(
                RecoveryAction.ABORT,
                event.reason or "event marked non-recoverable",
                state,
            )

        if event.trigger is RecoveryTrigger.SCENE_CHANGE:
            assert event.scene_change is not None
            kind = event.scene_change.kind
            if kind is SceneChangeKind.NONE:
                return RecoveryDecision(
                    RecoveryAction.CONTINUE,
                    "scene remains within change thresholds",
                    state,
                )
            if kind in {
                SceneChangeKind.TRANSFORM_CHANGED,
                SceneChangeKind.TARGET_MOVED,
                SceneChangeKind.DESTINATION_MOVED,
            }:
                return self._bounded_replan(state, f"scene change requires replan: {kind.value}")
            if kind is SceneChangeKind.TARGET_OCCLUDED:
                return self._bounded_reobserve(state, "target is occluded")
            if kind in {SceneChangeKind.TARGET_MISSING, SceneChangeKind.DESTINATION_MISSING}:
                return self._bounded_reason(state, f"scene entity unavailable: {kind.value}")
            raise AssertionError(f"unhandled scene change: {kind}")

        if event.trigger is RecoveryTrigger.GRASP_FAILURE:
            return self._bounded_regrasp(state, event.reason or "observable grasp failure")

        if event.trigger in {
            RecoveryTrigger.MOTION_FAILURE,
            RecoveryTrigger.EXECUTION_REPLAN,
        }:
            return self._bounded_replan(state, event.reason or event.trigger.value)

        if event.trigger is RecoveryTrigger.SEMANTIC_UNCERTAINTY:
            return self._bounded_reason(state, event.reason or "semantic uncertainty")

        raise AssertionError(f"unhandled recovery trigger: {event.trigger}")

    def _bounded_replan(self, state: RecoveryState, reason: str) -> RecoveryDecision:
        if state.replans >= self.budget.max_replans:
            return RecoveryDecision(
                RecoveryAction.ABORT,
                f"replan budget exhausted after {state.replans} attempts: {reason}",
                state,
            )
        return RecoveryDecision(
            RecoveryAction.REPLAN,
            reason,
            replace(state, replans=state.replans + 1),
        )

    def _bounded_reobserve(self, state: RecoveryState, reason: str) -> RecoveryDecision:
        if state.reobserves >= self.budget.max_reobserves:
            return self._bounded_reason(
                state,
                f"reobserve budget exhausted after {state.reobserves} attempts: {reason}",
            )
        return RecoveryDecision(
            RecoveryAction.REOBSERVE,
            reason,
            replace(state, reobserves=state.reobserves + 1),
        )

    def _bounded_regrasp(self, state: RecoveryState, reason: str) -> RecoveryDecision:
        if state.regrasps >= self.budget.max_regrasps:
            return self._bounded_reason(
                state,
                f"regrasp budget exhausted after {state.regrasps} attempts: {reason}",
            )
        return RecoveryDecision(
            RecoveryAction.REGRASP,
            reason,
            replace(state, regrasps=state.regrasps + 1),
        )

    def _bounded_reason(self, state: RecoveryState, reason: str) -> RecoveryDecision:
        if state.reasoning_calls >= self.budget.max_reasoning_calls:
            return RecoveryDecision(
                RecoveryAction.ABORT,
                f"reasoning budget exhausted after {state.reasoning_calls} attempts: {reason}",
                state,
            )
        return RecoveryDecision(
            RecoveryAction.REASON,
            reason,
            replace(state, reasoning_calls=state.reasoning_calls + 1),
        )
