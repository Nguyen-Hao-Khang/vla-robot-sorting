"""Measurable grasp candidate ranking for the geometry-first S1.3 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from vla_robot_sorting.contracts.grasping import (
    CandidateScore,
    FeasibilityEvaluation,
    GraspCandidate,
    RankedGraspSet,
)


@dataclass(frozen=True, slots=True)
class GraspRankingConfig:
    identity_threshold: float = 0.50
    minimum_approach_clearance_m: float = 0.01
    clearance_reference_m: float = 0.05
    approach_clearance_reference_m: float = 0.08
    motion_cost_reference: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.identity_threshold <= 1.0:
            raise ValueError("identity_threshold must be in [0, 1]")
        if self.minimum_approach_clearance_m < 0:
            raise ValueError("minimum_approach_clearance_m must be non-negative")
        if self.clearance_reference_m <= 0:
            raise ValueError("clearance_reference_m must be positive")
        if self.approach_clearance_reference_m <= 0:
            raise ValueError("approach_clearance_reference_m must be positive")
        if self.motion_cost_reference <= 0:
            raise ValueError("motion_cost_reference must be positive")


class GraspRanker:
    """Rank already-evaluated candidates without hiding feasibility failures."""

    _WEIGHTS: ClassVar[dict[str, float]] = {
        "reachability": 0.28,
        "clearance_margin": 0.22,
        "grasp_quality": 0.18,
        "approach_clearance": 0.15,
        "motion_cost": 0.10,
        "object_identity_confidence": 0.07,
    }

    def __init__(self, config: GraspRankingConfig | None = None) -> None:
        self.config = config or GraspRankingConfig()

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def score(
        self,
        candidate: GraspCandidate,
        evaluation: FeasibilityEvaluation,
    ) -> CandidateScore:
        if evaluation.candidate_id != candidate.candidate_id:
            raise ValueError("evaluation candidate_id does not match candidate")

        cfg = self.config
        factors = {
            "reachability": 1.0 if evaluation.reachable else 0.0,
            "clearance_margin": self._clamp01(
                evaluation.clearance_margin_m / cfg.clearance_reference_m
            ),
            "grasp_quality": self._clamp01(
                0.5 * candidate.raw_quality + 0.5 * evaluation.grasp_quality
            ),
            "approach_clearance": self._clamp01(
                evaluation.approach_clearance_m / cfg.approach_clearance_reference_m
            ),
            "motion_cost": 1.0
            - self._clamp01(evaluation.motion_cost / cfg.motion_cost_reference),
            "object_identity_confidence": self._clamp01(
                evaluation.object_identity_confidence
            ),
        }

        reasons: list[str] = []
        if not evaluation.reachable:
            reasons.append("unreachable")
        if not evaluation.collision_free:
            reasons.append("collision")
        if evaluation.approach_clearance_m < cfg.minimum_approach_clearance_m:
            reasons.append("approach_clearance")
        if evaluation.object_identity_confidence < cfg.identity_threshold:
            reasons.append("identity_confidence")

        eligible = not reasons
        total = sum(self._WEIGHTS[name] * factors[name] for name in self._WEIGHTS)
        if not eligible:
            total = 0.0

        return CandidateScore(
            candidate_id=candidate.candidate_id,
            total=self._clamp01(total),
            factors=factors,
            eligible=eligible,
            rejection_reasons=tuple(reasons),
        )

    def rank(
        self,
        candidates: tuple[GraspCandidate, ...],
        evaluations: tuple[FeasibilityEvaluation, ...],
    ) -> RankedGraspSet:
        if not candidates:
            raise ValueError("rank requires at least one candidate")

        evaluation_by_id = {item.candidate_id: item for item in evaluations}
        if len(evaluation_by_id) != len(evaluations):
            raise ValueError("duplicate feasibility evaluations are not allowed")

        first = candidates[0]
        if any(
            candidate.scene_revision != first.scene_revision
            or candidate.transform_revision != first.transform_revision
            for candidate in candidates[1:]
        ):
            raise ValueError("rank cannot mix candidates from different scene/transform revisions")

        scores: list[CandidateScore] = []
        for candidate in candidates:
            try:
                evaluation = evaluation_by_id[candidate.candidate_id]
            except KeyError as exc:
                raise ValueError(
                    f"missing feasibility evaluation for {candidate.candidate_id}"
                ) from exc
            scores.append(self.score(candidate, evaluation))

        scores.sort(
            key=lambda score: (
                not score.eligible,
                -score.total,
                score.candidate_id,
            )
        )
        return RankedGraspSet(
            scene_revision=first.scene_revision,
            transform_revision=first.transform_revision,
            scores=tuple(scores),
        )
