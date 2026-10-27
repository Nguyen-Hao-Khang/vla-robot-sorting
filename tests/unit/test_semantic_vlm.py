from __future__ import annotations

import pytest

from vla_robot_sorting.contracts.semantic import (
    SemanticDecision,
    SemanticProvenance,
    SemanticRequest,
    SourceLanguage,
)
from vla_robot_sorting.semantics.vlm import parse_semantic_payload


def provenance() -> SemanticProvenance:
    return SemanticProvenance(
        provider="test",
        model_id="model",
        prompt_version="v1",
        prompt_sha256="a" * 64,
        schema_sha256="b" * 64,
        fallback_used=False,
    )


def request(language: SourceLanguage = SourceLanguage.EN) -> SemanticRequest:
    return SemanticRequest(
        command="Put the red bottle in the left tray.",
        source_language=language,
        observation_revision="obs-7",
        scene_summary={"objects": 3},
    )


def success_payload() -> dict[str, object]:
    return {
        "decision": "structured_success",
        "normalized_intent_id": "sort:red-bottle:left-tray",
        "operation": "pick_and_place",
        "target": {
            "reference": "red bottle",
            "category": "bottle",
            "attributes": {
                "color": "red",
                "shape": None,
                "size": None,
                "location": None,
            },
            "relation": None,
            "relation_reference": None,
        },
        "destination": {
            "reference": "left tray",
            "category": "tray",
            "attributes": {
                "color": None,
                "shape": None,
                "size": None,
                "location": "left",
            },
            "relation": None,
            "relation_reference": None,
        },
        "semantic_confidence": 0.93,
        "ambiguity": None,
        "reason": None,
    }


def test_success_payload_preserves_host_observation_revision() -> None:
    result = parse_semantic_payload(
        success_payload(),
        request=request(SourceLanguage.VI),
        provenance=provenance(),
    )

    assert result.decision is SemanticDecision.STRUCTURED_SUCCESS
    assert result.observation_revision == "obs-7"
    assert result.source_language is SourceLanguage.VI
    assert result.target is not None
    assert result.target.attributes.color == "red"


def test_parser_rejects_unknown_motor_field() -> None:
    payload = success_payload()
    payload["joint_velocity"] = [1, 2, 3]

    with pytest.raises(ValueError, match="keys mismatch"):
        parse_semantic_payload(payload, request=request(), provenance=provenance())


def test_clarify_requires_bounded_ambiguity_details() -> None:
    payload = success_payload()
    payload.update(
        {
            "decision": "clarify",
            "normalized_intent_id": None,
            "operation": None,
            "ambiguity": {
                "reason": "two red bottles match",
                "clarification_slots": ["target.location"],
            },
            "reason": None,
        }
    )

    result = parse_semantic_payload(payload, request=request(), provenance=provenance())

    assert result.decision is SemanticDecision.CLARIFY
    assert result.ambiguity is not None
    assert result.ambiguity.clarification_slots == ("target.location",)


def test_abort_cannot_emit_operation() -> None:
    payload = success_payload()
    payload.update(
        {
            "decision": "abort",
            "normalized_intent_id": None,
            "operation": "pick_and_place",
            "target": None,
            "destination": None,
            "reason": "contradictory request",
        }
    )

    with pytest.raises(ValueError, match="must not emit an executable operation"):
        parse_semantic_payload(payload, request=request(), provenance=provenance())
