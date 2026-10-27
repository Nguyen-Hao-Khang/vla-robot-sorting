from __future__ import annotations

import pytest

from vla_robot_sorting.contracts.recovery import (
    RecoveryAction,
    RecoveryBudget,
    RecoveryEvent,
    RecoveryState,
    RecoveryTiming,
    RecoveryTrigger,
    SceneChange,
    SceneChangeKind,
)
from vla_robot_sorting.orchestration.recovery import ClosedLoopRecoveryPolicy


def scene_event(kind: SceneChangeKind) -> RecoveryEvent:
    return RecoveryEvent(
        trigger=RecoveryTrigger.SCENE_CHANGE,
        observed_at_s=1.0,
        scene_change=SceneChange(
            kind,
            "scene-1",
            "scene-2",
            target_object_id="target",
            destination_id="bin",
        ),
    )


def test_moved_target_triggers_bounded_replan() -> None:
    policy = ClosedLoopRecoveryPolicy(RecoveryBudget(max_replans=1))

    first = policy.decide(scene_event(SceneChangeKind.TARGET_MOVED), RecoveryState())
    second = policy.decide(scene_event(SceneChangeKind.TARGET_MOVED), first.next_state)

    assert first.action is RecoveryAction.REPLAN
    assert first.next_state.replans == 1
    assert second.action is RecoveryAction.ABORT


def test_occlusion_escalates_from_reobserve_to_reason() -> None:
    policy = ClosedLoopRecoveryPolicy(
        RecoveryBudget(max_reobserves=1, max_reasoning_calls=1)
    )

    first = policy.decide(scene_event(SceneChangeKind.TARGET_OCCLUDED), RecoveryState())
    second = policy.decide(scene_event(SceneChangeKind.TARGET_OCCLUDED), first.next_state)
    third = policy.decide(scene_event(SceneChangeKind.TARGET_OCCLUDED), second.next_state)

    assert first.action is RecoveryAction.REOBSERVE
    assert second.action is RecoveryAction.REASON
    assert third.action is RecoveryAction.ABORT


def test_grasp_failure_regrasps_then_escalates() -> None:
    policy = ClosedLoopRecoveryPolicy(
        RecoveryBudget(max_regrasps=1, max_reasoning_calls=1)
    )
    event = RecoveryEvent(
        trigger=RecoveryTrigger.GRASP_FAILURE,
        observed_at_s=2.0,
        reason="gripper closed without object retention",
    )

    first = policy.decide(event, RecoveryState())
    second = policy.decide(event, first.next_state)
    third = policy.decide(event, second.next_state)

    assert [first.action, second.action, third.action] == [
        RecoveryAction.REGRASP,
        RecoveryAction.REASON,
        RecoveryAction.ABORT,
    ]


def test_nonrecoverable_event_aborts_immediately() -> None:
    policy = ClosedLoopRecoveryPolicy()
    event = RecoveryEvent(
        trigger=RecoveryTrigger.MOTION_FAILURE,
        observed_at_s=3.0,
        recoverable=False,
        reason="hard safety violation",
    )

    decision = policy.decide(event, RecoveryState())

    assert decision.action is RecoveryAction.ABORT
    assert decision.next_state == RecoveryState()


def test_recovery_timing_reports_latency_components() -> None:
    timing = RecoveryTiming(
        observed_at_s=10.0,
        detected_at_s=10.2,
        decision_at_s=10.25,
        resolved_at_s=11.0,
    )

    assert timing.detection_latency_s == pytest.approx(0.2)
    assert timing.decision_latency_s == pytest.approx(0.05)
    assert timing.recovery_latency_s == pytest.approx(0.8)
