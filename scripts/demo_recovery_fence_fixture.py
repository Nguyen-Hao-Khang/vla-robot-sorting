"""Reproducible synthetic S1-R recovery *veto* matrix (no motors or physics)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from vla_robot_sorting.contracts.recovery import (
    RecoveryAction,
    RecoveryBudget,
    RecoveryEvent,
    RecoveryTrigger,
    SceneChange,
    SceneChangeKind,
)
from vla_robot_sorting.orchestration.recovery_fence import (
    ObservationVersion,
    RecoveryCompletion,
    RecoveryFence,
)


def _initial() -> ObservationVersion:
    # "native_observed" here is simulated input to the checker, NOT an
    # actual sensor/camera measurement or native acceptance evidence.
    return ObservationVersion(
        scene_revision="scene-0",
        transform_revision="tf-0",
        observation_revision="obs-0",
        observed_at_s=10.0,
        source_kind="native_observed",
    )


def evaluate_fixture_matrix() -> dict[str, object]:
    """Exercise independent gate scenarios and retain strict provenance."""
    old = _initial()
    results: list[dict[str, object]] = []

    def record(name: str, fence: RecoveryFence, obs: ObservationVersion, *,
               now_s: float, plan_id: str | None = "plan-0") -> None:
        verdict = fence.preflight(obs, now_s=now_s, plan_id=plan_id)
        results.append({
            "scenario": name,
            "gate_code": verdict.code.value,
            "downstream_checks_permitted": verdict.downstream_checks_permitted,
            "motor_command_authorized": verdict.motor_command_authorized,
        })

    def new_fence() -> RecoveryFence:
        return RecoveryFence(old, initial_plan_id="plan-0")

    record("nominal_revision_preflight", new_fence(), old, now_s=10.2)
    record("no_plan", new_fence(), old, now_s=10.2, plan_id=None)
    record(
        "manual_observation_veto", new_fence(),
        replace(old, source_kind="manual_visual_annotation"), now_s=10.2,
    )
    record("stale_frame_veto", new_fence(), old, now_s=12.0)
    record(
        "changed_transform_veto", new_fence(),
        replace(old, transform_revision="tf-1"), now_s=10.2,
    )

    moved = new_fence()
    scene_event = RecoveryEvent(
        trigger=RecoveryTrigger.SCENE_CHANGE,
        observed_at_s=10.2,
        scene_change=SceneChange(
            kind=SceneChangeKind.TARGET_MOVED,
            previous_scene_revision="scene-0",
            current_scene_revision="scene-1",
        ),
    )
    decision = moved.observe_failure(scene_event)
    record("moved_target_awaiting_replan", moved, old, now_s=10.3)
    changed = replace(
        old, scene_revision="scene-1",
        observation_revision="obs-1", observed_at_s=10.4,
    )
    proof = moved.complete_recovery(
        RecoveryCompletion(RecoveryAction.REPLAN, changed, plan_id="plan-1"),
    )
    record("moved_target_old_plan_veto", moved, changed, now_s=10.5)
    record("moved_target_new_plan_preflight", moved, changed, now_s=10.5, plan_id="plan-1")

    occluded = new_fence()
    occluded.observe_failure(RecoveryEvent(
        trigger=RecoveryTrigger.SCENE_CHANGE,
        observed_at_s=10.2,
        scene_change=SceneChange(
            kind=SceneChangeKind.TARGET_OCCLUDED,
            previous_scene_revision="scene-0",
            current_scene_revision="scene-1",
        ),
    ))
    occlusion_obs = replace(old, observation_revision="obs-2", observed_at_s=10.5)
    bad_proof = occluded.complete_recovery(
        RecoveryCompletion(RecoveryAction.REOBSERVE, occlusion_obs),
    )
    record("occluded_unreplanned_veto", occluded, occlusion_obs, now_s=10.6)
    occluded.complete_recovery(
        RecoveryCompletion(RecoveryAction.REOBSERVE, occlusion_obs, plan_id="plan-2"),
    )
    record(
        "occluded_after_reobserve_and_plan", occluded, occlusion_obs,
        now_s=10.6, plan_id="plan-2",
    )

    hard_failure = new_fence()
    hard_failure.observe_failure(RecoveryEvent(
        trigger=RecoveryTrigger.MOTION_FAILURE,
        observed_at_s=10.2,
        recoverable=False,
        reason="forbidden contact",
    ))
    record("hard_contact_sticky_abort", hard_failure, old, now_s=10.3)

    exhausted = RecoveryFence(
        old, initial_plan_id="plan-0", budget=RecoveryBudget(max_replans=0),
    )
    exhausted.observe_failure(RecoveryEvent(
        trigger=RecoveryTrigger.MOTION_FAILURE,
        observed_at_s=10.2,
        reason="IK unreachable",
    ))
    record("zero_replan_budget_abort", exhausted, old, now_s=10.3)
    return {
        "schema": "damh.s1r-recovery-fence-fixture.v1",
        "source_classification": "synthetic_fixture",
        "native_simulation_calls": 0,
        "physics_steps": 0,
        "network_calls": 0,
        "motor_commands_issued": 0,
        "native_evidence_accepted": False,
        "human_accepted_task_oracle": False,
        "scene_recovery_action": decision.action.value,
        "scene_recovery_proof_result": proof.code.value,
        "invalid_occlusion_proof_result": bad_proof.code.value,
        "scenarios": results,
        "case_count": len(results),
        "vetoed_count": sum(not item["downstream_checks_permitted"] for item in results),
        "allowed_later_checks_count": sum(
            item["downstream_checks_permitted"] for item in results
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_fixture_matrix()
    # The replay intentionally never executes external commands/physics.
    assert all(not item["motor_command_authorized"] for item in report["scenarios"])
    assert report["case_count"] == 12
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"S1R_SYNTHETIC_RECOVERY_VETO_PASS cases={report['case_count']} "
        f"vetoed={report['vetoed_count']} "
        f"downstream_preflight_only={report['allowed_later_checks_count']} "
        "motor_calls=0 native_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
