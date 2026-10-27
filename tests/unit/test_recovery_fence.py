"""S1-R fail-closed offline contracts; no simulator or actuator execution."""

from __future__ import annotations

from dataclasses import replace

import pytest

from vla_robot_sorting.contracts.recovery import (
    RecoveryAction,
    RecoveryBudget,
    RecoveryEvent,
    RecoveryTrigger,
    SceneChange,
    SceneChangeKind,
)
from vla_robot_sorting.orchestration.recovery_fence import (
    FenceCode,
    ObservationVersion,
    RecoveryCompletion,
    RecoveryFence,
)


def observation(source: str = "native_observed") -> ObservationVersion:
    return ObservationVersion(
        scene_revision="scene-0",
        transform_revision="tf-0",
        observation_revision="obs-0",
        observed_at_s=10.0,
        source_kind=source,
    )


def failure(
    trigger: RecoveryTrigger = RecoveryTrigger.MOTION_FAILURE, *,
    recoverable: bool = True,
) -> RecoveryEvent:
    return RecoveryEvent(
        trigger=trigger, observed_at_s=10.1,
        recoverable=recoverable, reason="independent motion failure",
    )


def test_clean_live_revision_only_allows_later_native_checks_not_motor_commands():
    fence = RecoveryFence(observation(), initial_plan_id="initial-plan")
    verdict = fence.preflight(observation(), now_s=10.5, plan_id="initial-plan")
    assert verdict.code is FenceCode.DOWNSTREAM_CHECKS_REQUIRED
    assert verdict.downstream_checks_permitted is True
    assert verdict.motor_command_authorized is False


@pytest.mark.parametrize("source", [
    "fixture", "manual_visual_annotation", "recorded_native",
    "recorded_sim_observation", "sim_ground_truth", "rgb_manual",
])
def test_replayed_fixture_manual_and_privileged_sources_never_pass_preflight(source):
    fence = RecoveryFence(observation(source), initial_plan_id="initial-plan")
    verdict = fence.preflight(observation(source), now_s=10.01)
    assert verdict.code is FenceCode.INVALID_PROVENANCE
    assert verdict.motor_command_authorized is False


@pytest.mark.parametrize("delta", [
    {"scene_revision": "scene-1"},
    {"transform_revision": "tf-1"},
    {"observation_revision": "obs-1"},
])
def test_any_scene_transform_or_observation_change_vetoes_stale_plan(delta):
    fence = RecoveryFence(observation(), initial_plan_id="initial-plan")
    verdict = fence.preflight(replace(observation(), **delta), now_s=10.2)
    assert verdict.code is FenceCode.REVISION_MISMATCH


@pytest.mark.parametrize("now_s", [9.99, 11.01, 99.0])
def test_future_and_expired_observations_fail_closed(now_s):
    fence = RecoveryFence(observation(), initial_plan_id="initial-plan")
    assert fence.preflight(observation(), now_s=now_s).code is FenceCode.STALE_OBSERVATION


def test_even_same_revision_changed_timestamp_is_not_accepted():
    fence = RecoveryFence(observation(), initial_plan_id="initial-plan")
    newer_same_revision = replace(observation(), observed_at_s=10.4)
    assert fence.preflight(newer_same_revision, now_s=10.5).code is FenceCode.STALE_OBSERVATION


def test_replan_blocks_all_preflight_until_new_native_revision_and_plan_id():
    fence = RecoveryFence(
        observation(), initial_plan_id="initial-plan",
        budget=RecoveryBudget(max_replans=2),
    )
    decision = fence.observe_failure(failure())
    assert decision.action is RecoveryAction.REPLAN
    assert fence.pending_action is RecoveryAction.REPLAN
    assert fence.preflight(observation(), now_s=10.2).code is FenceCode.RECOVERY_PENDING

    missing_plan = RecoveryCompletion(
        RecoveryAction.REPLAN,
        replace(observation(), observation_revision="obs-1", observed_at_s=10.3),
    )
    assert fence.complete_recovery(missing_plan).code is FenceCode.BAD_RECOVERY_PROOF
    assert fence.pending_action is RecoveryAction.REPLAN

    stale = RecoveryCompletion(RecoveryAction.REPLAN, observation(), plan_id="plan-B")
    assert fence.complete_recovery(stale).code is FenceCode.BAD_RECOVERY_PROOF
    assert fence.pending_action is RecoveryAction.REPLAN

    new_obs = replace(
        observation(), scene_revision="scene-1",
        observation_revision="obs-1", observed_at_s=10.4,
    )
    accepted = fence.complete_recovery(
        RecoveryCompletion(RecoveryAction.REPLAN, new_obs, plan_id="plan-B"),
    )
    assert accepted.code is FenceCode.DOWNSTREAM_CHECKS_REQUIRED
    assert accepted.motor_command_authorized is False
    assert fence.preflight(new_obs, now_s=10.5, plan_id="plan-A").code is (
        FenceCode.REVISION_MISMATCH
    )
    assert fence.preflight(new_obs, now_s=10.5, plan_id="plan-B").code is (
        FenceCode.DOWNSTREAM_CHECKS_REQUIRED
    )


def test_non_native_recovery_proof_cannot_reopen_gate():
    fence = RecoveryFence(observation(), initial_plan_id="initial-plan")
    fence.observe_failure(failure())
    bad = replace(
        observation(), source_kind="manual_visual_annotation",
        observation_revision="obs-new", observed_at_s=10.2,
    )
    result = fence.complete_recovery(
        RecoveryCompletion(RecoveryAction.REPLAN, bad, plan_id="plan-B")
    )
    assert result.code is FenceCode.INVALID_PROVENANCE
    assert fence.preflight(observation(), now_s=10.3).code is FenceCode.RECOVERY_PENDING


def test_new_failure_during_pending_recovery_causes_sticky_abort():
    fence = RecoveryFence(observation(), initial_plan_id="initial-plan")
    fence.observe_failure(failure())
    decision = fence.observe_failure(failure(RecoveryTrigger.GRASP_FAILURE))
    assert decision.action is RecoveryAction.ABORT
    assert fence.aborted is True
    assert fence.preflight(observation(), now_s=10.2).code is FenceCode.ABORTED
    assert fence.complete_recovery(
        RecoveryCompletion(RecoveryAction.REPLAN, observation(), plan_id="p1")
    ).code is FenceCode.ABORTED


def test_hard_safety_violation_aborts_without_consuming_recovery_budget():
    fence = RecoveryFence(observation(), initial_plan_id="initial-plan")
    decision = fence.observe_failure(failure(recoverable=False))
    assert decision.action is RecoveryAction.ABORT
    assert fence.recovery_state.replans == 0
    assert fence.preflight(observation(), now_s=10.2).code is FenceCode.ABORTED


def test_regrasp_needs_new_observation_and_new_plan_then_retains_veto_boundary():
    fence = RecoveryFence(observation(), initial_plan_id="initial-plan")
    decision = fence.observe_failure(failure(RecoveryTrigger.GRASP_FAILURE))
    assert decision.action is RecoveryAction.REGRASP
    changed = replace(observation(), observation_revision="obs-1", observed_at_s=10.2)
    assert fence.complete_recovery(
        RecoveryCompletion(RecoveryAction.REGRASP, changed)
    ).code is FenceCode.BAD_RECOVERY_PROOF
    assert fence.complete_recovery(
        RecoveryCompletion(RecoveryAction.REGRASP, changed, plan_id="regrasp-1")
    ).downstream_checks_permitted
    assert not fence.preflight(
        changed, now_s=10.3, plan_id="regrasp-1"
    ).motor_command_authorized


def test_occlusion_reobserve_can_clear_only_with_new_native_observation():
    fence = RecoveryFence(
        observation(), initial_plan_id="initial-plan",
        budget=RecoveryBudget(max_reobserves=1),
    )
    event = RecoveryEvent(
        trigger=RecoveryTrigger.SCENE_CHANGE,
        observed_at_s=10.1,
        scene_change=SceneChange(
            kind=SceneChangeKind.TARGET_OCCLUDED,
            previous_scene_revision="scene-0",
            current_scene_revision="scene-1",
        ),
    )
    assert fence.observe_failure(event).action is RecoveryAction.REOBSERVE
    changed = replace(observation(), observation_revision="obs-1", observed_at_s=10.5)
    assert fence.complete_recovery(
        RecoveryCompletion(RecoveryAction.REOBSERVE, changed, plan_id="new-reobserved-plan")
    ).downstream_checks_permitted


def test_scene_change_none_does_not_bypass_independent_revision_gate():
    fence = RecoveryFence(observation(), initial_plan_id="initial-plan")
    event = RecoveryEvent(
        trigger=RecoveryTrigger.SCENE_CHANGE,
        observed_at_s=10.1,
        scene_change=SceneChange(
            kind=SceneChangeKind.NONE,
            previous_scene_revision="scene-0",
            current_scene_revision="scene-1",
        ),
    )
    assert fence.observe_failure(event).action is RecoveryAction.CONTINUE
    updated = replace(observation(), scene_revision="scene-1")
    assert fence.preflight(updated, now_s=10.2).code is FenceCode.REVISION_MISMATCH


def test_repeated_replans_cannot_bypass_budget():
    fence = RecoveryFence(
        observation(), initial_plan_id="initial-plan",
        budget=RecoveryBudget(max_replans=1),
    )
    assert fence.observe_failure(failure()).action is RecoveryAction.REPLAN
    changed = replace(observation(), observation_revision="obs-1", observed_at_s=10.2)
    assert fence.complete_recovery(
        RecoveryCompletion(RecoveryAction.REPLAN, changed, plan_id="p1")
    ).downstream_checks_permitted
    assert fence.observe_failure(failure()).action is RecoveryAction.ABORT
    assert fence.aborted


def test_malformed_age_and_observation_revision_rejected():
    with pytest.raises(ValueError):
        RecoveryFence(observation(), initial_plan_id="initial-plan", max_observation_age_s=0)
    with pytest.raises(ValueError):
        RecoveryFence(
            observation(), initial_plan_id="initial-plan",
            max_observation_age_s=float("nan"),
        )
    with pytest.raises(ValueError):
        observation().__class__("", "tf-0", "obs-0", 10.0, "native_observed")
    with pytest.raises(ValueError):
        replace(observation(), observed_at_s=float("inf"))


def test_missing_plan_id_fails_closed_even_for_native_observation():
    fence = RecoveryFence(observation(), initial_plan_id="initial-plan")
    verdict = fence.preflight(observation(), now_s=10.5)
    assert verdict.code is FenceCode.MISSING_PLAN
    assert not verdict.motor_command_authorized


def test_reobservation_does_not_clear_fence_without_new_plan():
    fence = RecoveryFence(observation(), initial_plan_id="initial-plan")
    event = RecoveryEvent(
        trigger=RecoveryTrigger.SCENE_CHANGE,
        observed_at_s=10.1,
        scene_change=SceneChange(
            kind=SceneChangeKind.TARGET_OCCLUDED,
            previous_scene_revision="scene-0",
            current_scene_revision="scene-1",
        ),
    )
    assert fence.observe_failure(event).action is RecoveryAction.REOBSERVE
    newer = replace(observation(), observation_revision="obs-new", observed_at_s=10.2)
    assert fence.complete_recovery(
        RecoveryCompletion(RecoveryAction.REOBSERVE, newer)
    ).code is FenceCode.BAD_RECOVERY_PROOF
    assert fence.pending_action is RecoveryAction.REOBSERVE
    assert fence.complete_recovery(
        RecoveryCompletion(RecoveryAction.REOBSERVE, newer, plan_id="fresh-plan")
    ).downstream_checks_permitted
    assert fence.preflight(newer, now_s=10.3, plan_id="old-plan").code is (
        FenceCode.REVISION_MISMATCH
    )
    assert fence.preflight(newer, now_s=10.3, plan_id="fresh-plan").downstream_checks_permitted
