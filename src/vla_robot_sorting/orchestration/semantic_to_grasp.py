"""Fail-closed semantic-to-geometric grasp bridge for recorded/fixture scenes.

A grasp *proposal* is not an executable motion plan: no IK, collision,
native physics or Gemini call occurs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vla_robot_sorting.contracts.grasping import (
    GraspCandidate,
    GraspRequest,
    ParallelJawGripper,
)
from vla_robot_sorting.contracts.perception import ObjectVisibility, SceneSnapshot
from vla_robot_sorting.contracts.semantic import (
    GroundingConstraint,
    Operation,
    SemanticDecision,
    SemanticResult,
)
from vla_robot_sorting.grasping.geometry import GeometryFirstGraspGenerator
from vla_robot_sorting.semantics.grounding import (
    CatalogDestinationGrounder,
    CatalogGrounder,
    GroundingRequest,
)


class BridgeFailureCode(str, Enum):
    SEMANTIC_REJECTED = "semantic_rejected"
    OBSERVATION_STALE = "observation_stale"
    PROVENANCE_MISSING = "provenance_missing"
    CONSTRAINT_UNSUPPORTED = "constraint_unsupported"
    GROUNDING_FAILED = "grounding_failed"
    TARGET_UNAVAILABLE = "target_unavailable"
    CONFIDENCE_LOW = "confidence_low"
    NO_GRASP = "no_grasp"


@dataclass(frozen=True, slots=True)
class GraspProposal:
    """Geometry-only candidates. Never send these directly to a motor executor."""

    target_id: str
    destination_id: str
    scene_revision: str
    transform_revision: str
    observation_revision: str
    observation_source: str
    geometry_source: str
    candidate_ids: tuple[str, ...]
    candidates: tuple[GraspCandidate, ...]
    source_evidence_kind: str
    control_eligible: bool = False
    feasibility_checked: bool = False

    def __post_init__(self) -> None:
        if not self.candidates or not self.candidate_ids:
            raise ValueError("grasp proposal requires geometric candidates")
        if self.feasibility_checked:
            raise ValueError("this bridge cannot certify motion feasibility")
        if self.control_eligible:
            raise ValueError("geometry-only grasp proposals are never controller eligible")
        if not self.source_evidence_kind.strip():
            raise ValueError("grasp proposal must retain object evidence provenance")
        if self.candidate_ids != tuple(c.candidate_id for c in self.candidates):
            raise ValueError("candidate_ids must match candidates")
        if any(
            c.object_id != self.target_id
            or c.scene_revision != self.scene_revision
            or c.transform_revision != self.transform_revision
            for c in self.candidates
        ):
            raise ValueError("grasp proposal candidates do not match locked scene")


@dataclass(frozen=True, slots=True)
class BridgeResult:
    proposal: GraspProposal | None
    failure_code: BridgeFailureCode | None
    reason: str | None

    def __post_init__(self) -> None:
        if (self.proposal is None) == (self.failure_code is None):
            raise ValueError("exactly one of proposal and failure_code is required")

    @property
    def succeeded(self) -> bool:
        return self.proposal is not None


def _rejected(code: BridgeFailureCode, reason: str) -> BridgeResult:
    return BridgeResult(None, code, reason)


def _words(text: str) -> tuple[str, ...]:
    return tuple(text.casefold().strip().split())


def _matches_metadata(constraint: GroundingConstraint, labels: tuple[str, ...]) -> bool:
    """Lexical validation only; never pretend to measure image attributes."""

    if constraint.relation is not None or constraint.relation_reference is not None:
        return False
    normalized = tuple(_words(label) for label in labels)
    pieces = (
        constraint.category,
        constraint.attributes.color,
        constraint.attributes.shape,
        constraint.attributes.size,
        constraint.attributes.location,
    )
    for piece in pieces:
        if piece is None:
            continue
        tokens = _words(piece)
        if not tokens or not any(
            any(
                row[index : index + len(tokens)] == tokens
                for index in range(len(row) - len(tokens) + 1)
            )
            for row in normalized
        ):
            return False
    return True


class SemanticToGraspBridge:
    """Integrate validated semantic output with explicit catalog scene metadata."""

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.5,
        grasp_generator: GeometryFirstGraspGenerator | None = None,
        minimum_table_clearance_m: float = 0.0,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if minimum_table_clearance_m < 0.0:
            raise ValueError("minimum_table_clearance_m must not be negative")
        self.minimum_confidence = minimum_confidence
        self._grasp_generator = grasp_generator or GeometryFirstGraspGenerator()
        self._minimum_table_clearance_m = minimum_table_clearance_m

    def propose(
        self,
        semantic: SemanticResult,
        scene: SceneSnapshot,
        gripper: ParallelJawGripper,
    ) -> BridgeResult:
        if (
            semantic.decision is not SemanticDecision.STRUCTURED_SUCCESS
            or semantic.operation is not Operation.PICK_AND_PLACE
            or semantic.target is None
            or semantic.destination is None
        ):
            return _rejected(BridgeFailureCode.SEMANTIC_REJECTED, "non-executable intent")
        if semantic.observation_revision != scene.metadata.get("observation_revision"):
            return _rejected(BridgeFailureCode.OBSERVATION_STALE, "observation revision differs")
        source = scene.metadata.get("observation_source")
        if source not in ("fixture", "recorded_native", "sim_ground_truth"):
            return _rejected(
                BridgeFailureCode.PROVENANCE_MISSING,
                "scene observation source must be explicit",
            )
        if not semantic.target.reference or not semantic.destination.reference:
            return _rejected(
                BridgeFailureCode.CONSTRAINT_UNSUPPORTED,
                "explicit target and destination references are required",
            )
        if (
            semantic.target.relation is not None
            or semantic.target.relation_reference is not None
            or semantic.destination.relation is not None
            or semantic.destination.relation_reference is not None
        ):
            return _rejected(
                BridgeFailureCode.CONSTRAINT_UNSUPPORTED,
                "spatial relations require a separately validated resolver",
            )
        if semantic.semantic_confidence < self.minimum_confidence:
            return _rejected(BridgeFailureCode.CONFIDENCE_LOW, "semantic confidence too low")

        language = semantic.source_language.value
        target = CatalogGrounder().ground(
            GroundingRequest(semantic.target.reference, scene.scene_revision, language),
            scene,
        )
        destination = CatalogDestinationGrounder().ground(
            GroundingRequest(semantic.destination.reference, scene.scene_revision, language),
            scene,
        )
        if not target.succeeded or not destination.succeeded:
            details = (
                target.failure.code.value if target.failure is not None else "target_ok",
                destination.failure.code.value if destination.failure is not None else "dest_ok",
            )
            return _rejected(BridgeFailureCode.GROUNDING_FAILED, "/".join(details))

        assert target.object_id is not None and destination.destination_id is not None
        scene_object = scene.object(target.object_id)
        scene_destination = scene.destination(destination.destination_id)
        if scene_object.visibility is not ObjectVisibility.VISIBLE:
            return _rejected(BridgeFailureCode.TARGET_UNAVAILABLE, "target not visible")
        if not (
            _matches_metadata(semantic.target, scene_object.semantic_labels)
            and _matches_metadata(semantic.destination, scene_destination.semantic_labels)
        ):
            return _rejected(
                BridgeFailureCode.CONSTRAINT_UNSUPPORTED,
                "attributes not independently supported by scene labels",
            )
        if scene_object.identity_confidence < self.minimum_confidence:
            return _rejected(BridgeFailureCode.CONFIDENCE_LOW, "object confidence too low")

        generated = self._grasp_generator.generate(
            GraspRequest(
                object_id=target.object_id,
                scene_revision=scene.scene_revision,
                transform_revision=scene.transform_revision,
                gripper=gripper,
                minimum_table_clearance_m=self._minimum_table_clearance_m,
            ),
            scene,
        )
        if not generated.succeeded:
            assert generated.failure is not None
            return _rejected(BridgeFailureCode.NO_GRASP, generated.failure.code.value)
        candidates = tuple(
            sorted(generated.candidates, key=lambda c: (-c.raw_quality, c.candidate_id))
        )
        return BridgeResult(
            GraspProposal(
                target_id=target.object_id,
                destination_id=destination.destination_id,
                scene_revision=scene.scene_revision,
                transform_revision=scene.transform_revision,
                observation_revision=semantic.observation_revision,
                observation_source=source,
                geometry_source=scene_object.geometry_source.value,
                candidate_ids=tuple(c.candidate_id for c in candidates),
                candidates=candidates,
                source_evidence_kind=str(
                    scene_object.metadata.get("evidence_kind", "fixture_catalog")
                    if source == "fixture"
                    else scene_object.metadata.get("evidence_kind", "untyped")
                ),
                control_eligible=False,
            ),
            None,
            None,
        )
