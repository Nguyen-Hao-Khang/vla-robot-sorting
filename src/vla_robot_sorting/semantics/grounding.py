"""Grounding contracts and deterministic catalog baseline for S1.3/S1.4."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from vla_robot_sorting.contracts.perception import SceneSnapshot


class GroundingFailureCode(str, Enum):
    INVALID_QUERY = "invalid_query"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    STALE_SCENE = "stale_scene"


@dataclass(frozen=True, slots=True)
class GroundingFailure:
    code: GroundingFailureCode
    message: str
    candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("GroundingFailure.message must be non-empty")


@dataclass(frozen=True, slots=True)
class GroundingRequest:
    query: str
    scene_revision: str
    language: str | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("GroundingRequest.query must be non-empty")
        if not self.scene_revision.strip():
            raise ValueError("GroundingRequest.scene_revision must be non-empty")


@dataclass(frozen=True, slots=True)
class GroundingResult:
    object_id: str | None
    confidence: float
    failure: GroundingFailure | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("GroundingResult.confidence must be in [0, 1]")
        if (self.object_id is None) == (self.failure is None):
            raise ValueError("GroundingResult requires exactly one of object_id/failure")

    @property
    def succeeded(self) -> bool:
        return self.object_id is not None


@dataclass(frozen=True, slots=True)
class DestinationGroundingResult:
    destination_id: str | None
    confidence: float
    failure: GroundingFailure | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("DestinationGroundingResult.confidence must be in [0, 1]")
        if (self.destination_id is None) == (self.failure is None):
            raise ValueError(
                "DestinationGroundingResult requires exactly one of destination_id/failure"
            )

    @property
    def succeeded(self) -> bool:
        return self.destination_id is not None


class SemanticGrounder(Protocol):
    def ground(self, request: GroundingRequest, scene: SceneSnapshot) -> GroundingResult: ...


def _normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())


class CatalogGrounder:
    """Deterministic lexical baseline; open-vocabulary providers plug into the same protocol."""

    def ground(self, request: GroundingRequest, scene: SceneSnapshot) -> GroundingResult:
        if request.scene_revision != scene.scene_revision:
            return GroundingResult(
                None,
                0.0,
                GroundingFailure(
                    GroundingFailureCode.STALE_SCENE,
                    "grounding request scene revision is stale",
                ),
            )

        query = _normalize(request.query)
        exact: list[str] = []
        partial: list[str] = []

        for item in scene.objects:
            labels = {
                _normalize(item.object_id),
                *(_normalize(label) for label in item.semantic_labels),
            }
            if query in labels:
                exact.append(item.object_id)
            elif any(query in label or label in query for label in labels):
                partial.append(item.object_id)

        matches = exact if exact else partial
        if not matches:
            return GroundingResult(
                None,
                0.0,
                GroundingFailure(
                    GroundingFailureCode.NOT_FOUND,
                    f"no scene object matches {request.query!r}",
                ),
            )
        if len(matches) > 1:
            return GroundingResult(
                None,
                0.0,
                GroundingFailure(
                    GroundingFailureCode.AMBIGUOUS,
                    f"multiple scene objects match {request.query!r}",
                    tuple(sorted(matches)),
                ),
            )

        object_id = matches[0]
        confidence = scene.object(object_id).identity_confidence
        return GroundingResult(object_id, confidence)


class CatalogDestinationGrounder:
    """Deterministic destination-label baseline sharing the object grounding semantics."""

    def ground(
        self,
        request: GroundingRequest,
        scene: SceneSnapshot,
    ) -> DestinationGroundingResult:
        if request.scene_revision != scene.scene_revision:
            return DestinationGroundingResult(
                None,
                0.0,
                GroundingFailure(
                    GroundingFailureCode.STALE_SCENE,
                    "grounding request scene revision is stale",
                ),
            )

        query = _normalize(request.query)
        exact: list[str] = []
        partial: list[str] = []
        for item in scene.destinations:
            labels = {
                _normalize(item.destination_id),
                *(_normalize(label) for label in item.semantic_labels),
            }
            if query in labels:
                exact.append(item.destination_id)
            elif any(query in label or label in query for label in labels):
                partial.append(item.destination_id)

        matches = exact if exact else partial
        if not matches:
            return DestinationGroundingResult(
                None,
                0.0,
                GroundingFailure(
                    GroundingFailureCode.NOT_FOUND,
                    f"no scene destination matches {request.query!r}",
                ),
            )
        if len(matches) > 1:
            return DestinationGroundingResult(
                None,
                0.0,
                GroundingFailure(
                    GroundingFailureCode.AMBIGUOUS,
                    f"multiple scene destinations match {request.query!r}",
                    tuple(sorted(matches)),
                ),
            )
        return DestinationGroundingResult(matches[0], 1.0)
