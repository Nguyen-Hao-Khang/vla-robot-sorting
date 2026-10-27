from __future__ import annotations

from vla_robot_sorting.contracts.grasping import (
    FeasibilityEvaluation,
    GraspCandidate,
)
from vla_robot_sorting.contracts.models import Pose6D
from vla_robot_sorting.grasping.ranking import GraspRanker


def candidate(candidate_id: str, quality: float = 0.8) -> GraspCandidate:
    pose = Pose6D((0.2, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0), "world")
    pregrasp = Pose6D((0.1, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0), "world")
    return GraspCandidate(
        candidate_id=candidate_id,
        object_id="obj",
        grasp_pose=pose,
        pregrasp_pose=pregrasp,
        approach_direction=(1.0, 0.0, 0.0),
        closing_direction=(0.0, 1.0, 0.0),
        jaw_width_m=0.04,
        source="geometry_first",
        geometry_source="sim_ground_truth",
        scene_revision="scene-1",
        transform_revision="tf-1",
        compliance_profile="standard",
        raw_quality=quality,
    )


def evaluation(
    candidate_id: str,
    *,
    reachable: bool = True,
    collision_free: bool = True,
    clearance: float = 0.04,
    approach: float = 0.05,
    quality: float = 0.9,
    cost: float = 0.2,
    identity: float = 0.9,
) -> FeasibilityEvaluation:
    return FeasibilityEvaluation(
        candidate_id=candidate_id,
        reachable=reachable,
        collision_free=collision_free,
        clearance_margin_m=clearance,
        approach_clearance_m=approach,
        grasp_quality=quality,
        motion_cost=cost,
        object_identity_confidence=identity,
    )


def test_ranker_prefers_eligible_high_quality_candidate() -> None:
    first = candidate("a", quality=0.9)
    second = candidate("b", quality=0.5)

    ranked = GraspRanker().rank(
        (first, second),
        (
            evaluation("a", clearance=0.045, quality=0.95, cost=0.1),
            evaluation("b", clearance=0.02, quality=0.6, cost=0.5),
        ),
    )

    assert ranked.scores[0].candidate_id == "a"
    assert ranked.scores[0].eligible
    assert ranked.scores[0].total > ranked.scores[1].total


def test_collision_candidate_is_ineligible_even_with_good_raw_score() -> None:
    item = candidate("blocked", quality=1.0)

    score = GraspRanker().score(
        item,
        evaluation(
            "blocked",
            collision_free=False,
            clearance=0.05,
            approach=0.08,
            quality=1.0,
            cost=0.0,
            identity=1.0,
        ),
    )

    assert not score.eligible
    assert score.total == 0.0
    assert "collision" in score.rejection_reasons


def test_low_identity_and_approach_clearance_are_separate_reasons() -> None:
    item = candidate("uncertain")

    score = GraspRanker().score(
        item,
        evaluation("uncertain", identity=0.2, approach=0.001),
    )

    assert not score.eligible
    assert set(score.rejection_reasons) == {"identity_confidence", "approach_clearance"}


def test_ranker_rejects_mixed_scene_revisions() -> None:
    first = candidate("a")
    second = candidate("b")
    second = GraspCandidate(
        candidate_id=second.candidate_id,
        object_id=second.object_id,
        grasp_pose=second.grasp_pose,
        pregrasp_pose=second.pregrasp_pose,
        approach_direction=second.approach_direction,
        closing_direction=second.closing_direction,
        jaw_width_m=second.jaw_width_m,
        source=second.source,
        geometry_source=second.geometry_source,
        scene_revision="scene-2",
        transform_revision=second.transform_revision,
        compliance_profile=second.compliance_profile,
        raw_quality=second.raw_quality,
        metadata=second.metadata,
    )

    import pytest

    with pytest.raises(ValueError, match="different scene/transform revisions"):
        GraspRanker().rank(
            (first, second),
            (evaluation("a"), evaluation("b")),
        )
