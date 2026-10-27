"""Validated semantic reasoning contracts for S1.4."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any


class SemanticDecision(str, Enum):
    STRUCTURED_SUCCESS = "structured_success"
    CLARIFY = "clarify"
    REOBSERVE = "reobserve"
    ABORT = "abort"


class SourceLanguage(str, Enum):
    EN = "en"
    VI = "vi"
    CODE_SWITCH = "code_switch"


class Operation(str, Enum):
    PICK_AND_PLACE = "pick_and_place"


@dataclass(frozen=True, slots=True)
class ObjectAttributes:
    color: str | None = None
    shape: str | None = None
    size: str | None = None
    location: str | None = None


@dataclass(frozen=True, slots=True)
class GroundingConstraint:
    reference: str | None = None
    category: str | None = None
    attributes: ObjectAttributes = field(default_factory=ObjectAttributes)
    relation: str | None = None
    relation_reference: str | None = None

    def __post_init__(self) -> None:
        for value in (self.reference, self.category, self.relation, self.relation_reference):
            if value is not None and not value.strip():
                raise ValueError("grounding constraint strings must be non-empty when present")


@dataclass(frozen=True, slots=True)
class SemanticAmbiguity:
    reason: str
    clarification_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("ambiguity reason must be non-empty")
        if any(not slot.strip() for slot in self.clarification_slots):
            raise ValueError("clarification slots must be non-empty strings")


@dataclass(frozen=True, slots=True)
class SemanticProvenance:
    provider: str
    model_id: str
    prompt_version: str
    prompt_sha256: str
    schema_sha256: str
    fallback_used: bool

    def __post_init__(self) -> None:
        for value in (
            self.provider,
            self.model_id,
            self.prompt_version,
            self.prompt_sha256,
            self.schema_sha256,
        ):
            if not value.strip():
                raise ValueError("semantic provenance fields must be non-empty")
        if len(self.prompt_sha256) != 64 or len(self.schema_sha256) != 64:
            raise ValueError("semantic provenance hashes must be 64-character SHA256 digests")


@dataclass(frozen=True, slots=True)
class SemanticRequest:
    command: str
    source_language: SourceLanguage
    observation_revision: str
    scene_summary: dict[str, Any] = field(default_factory=dict)
    case_id: str | None = None

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("SemanticRequest.command must be non-empty")
        if not self.observation_revision.strip():
            raise ValueError("SemanticRequest.observation_revision must be non-empty")
        if self.case_id is not None and not self.case_id.strip():
            raise ValueError("case_id must be non-empty when present")


@dataclass(frozen=True, slots=True)
class SemanticResult:
    decision: SemanticDecision
    normalized_intent_id: str | None
    operation: Operation | None
    target: GroundingConstraint | None
    destination: GroundingConstraint | None
    semantic_confidence: float
    ambiguity: SemanticAmbiguity | None
    reason: str | None
    observation_revision: str
    source_language: SourceLanguage
    provenance: SemanticProvenance

    def __post_init__(self) -> None:
        if not isfinite(self.semantic_confidence) or not 0.0 <= self.semantic_confidence <= 1.0:
            raise ValueError("semantic_confidence must be finite and in [0, 1]")
        if not self.observation_revision.strip():
            raise ValueError("observation_revision must be non-empty")
        if self.normalized_intent_id is not None and not self.normalized_intent_id.strip():
            raise ValueError("normalized_intent_id must be non-empty when present")
        if self.reason is not None and not self.reason.strip():
            raise ValueError("reason must be non-empty when present")

        if self.decision is SemanticDecision.STRUCTURED_SUCCESS:
            if self.operation is None or self.target is None or self.destination is None:
                raise ValueError("structured_success requires operation, target and destination")
            if self.normalized_intent_id is None:
                raise ValueError("structured_success requires normalized_intent_id")
            if self.ambiguity is not None:
                raise ValueError("structured_success cannot contain ambiguity")
        elif self.operation is not None:
            raise ValueError("non-success semantic results must not emit an executable operation")

        if self.decision is SemanticDecision.CLARIFY and self.ambiguity is None:
            raise ValueError("clarify requires ambiguity details")
        if self.decision is not SemanticDecision.CLARIFY and self.ambiguity is not None:
            raise ValueError("ambiguity details are only valid for clarify")
        if (
            self.decision in {SemanticDecision.REOBSERVE, SemanticDecision.ABORT}
            and self.reason is None
        ):
            raise ValueError(f"{self.decision.value} requires a reason")
