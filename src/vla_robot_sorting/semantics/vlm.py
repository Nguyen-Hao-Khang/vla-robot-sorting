"""Provider-independent semantic reasoning boundary for S1.4."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Protocol

from vla_robot_sorting.contracts.semantic import (
    GroundingConstraint,
    ObjectAttributes,
    Operation,
    SemanticAmbiguity,
    SemanticDecision,
    SemanticProvenance,
    SemanticRequest,
    SemanticResult,
)

PROMPT_VERSION = "s1.4-semantic-v1"
SYSTEM_PROMPT = """You are the semantic reasoner for a robot sorting system.
Convert the user's English, Vietnamese, or code-switched command into the required JSON schema.
You never command motors, joints, poses, velocities, torques, or trajectories.
Choose exactly one decision:
- structured_success: the requested pick-and-place intent is semantically resolvable.
- clarify: multiple incompatible readings remain; identify bounded clarification slots.
- reobserve: the wording refers to stale, occluded, or insufficient scene evidence.
- abort: the request is contradictory, impossible, or unsafe to interpret.
For structured_success, normalize the operation as pick_and_place and emit target/destination
grounding constraints rather than scene instance IDs. Do not guess through ambiguity.
Use null for fields that are not applicable. Preserve meaning across English and Vietnamese.
"""
PROMPT_SHA256 = sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


class SemanticReasoner(Protocol):
    def reason(self, request: SemanticRequest) -> SemanticResult: ...


def semantic_response_schema() -> dict[str, Any]:
    attributes = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "color": {"type": ["string", "null"]},
            "shape": {"type": ["string", "null"]},
            "size": {"type": ["string", "null"]},
            "location": {"type": ["string", "null"]},
        },
        "required": ["color", "shape", "size", "location"],
    }
    constraint = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reference": {"type": ["string", "null"]},
            "category": {"type": ["string", "null"]},
            "attributes": attributes,
            "relation": {"type": ["string", "null"]},
            "relation_reference": {"type": ["string", "null"]},
        },
        "required": [
            "reference",
            "category",
            "attributes",
            "relation",
            "relation_reference",
        ],
    }
    ambiguity = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reason": {"type": "string"},
            "clarification_slots": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
        },
        "required": ["reason", "clarification_slots"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "string",
                "enum": [item.value for item in SemanticDecision],
            },
            "normalized_intent_id": {"type": ["string", "null"]},
            "operation": {
                "type": ["string", "null"],
                "enum": [Operation.PICK_AND_PLACE.value, None],
            },
            "target": {"anyOf": [constraint, {"type": "null"}]},
            "destination": {"anyOf": [constraint, {"type": "null"}]},
            "semantic_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "ambiguity": {"anyOf": [ambiguity, {"type": "null"}]},
            "reason": {"type": ["string", "null"]},
        },
        "required": [
            "decision",
            "normalized_intent_id",
            "operation",
            "target",
            "destination",
            "semantic_confidence",
            "ambiguity",
            "reason",
        ],
    }


def semantic_schema_sha256() -> str:
    encoded = json.dumps(
        semantic_response_schema(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_user_prompt(request: SemanticRequest) -> str:
    payload = {
        "command": request.command,
        "source_language": request.source_language.value,
        "observation_revision": request.observation_revision,
        "scene_summary": request.scene_summary,
        "case_id": request.case_id,
    }
    return (
        "Interpret this request. observation_revision is context supplied by the host and "
        "must not be modified or invented by the model.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _require_exact_keys(data: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be null or a non-empty string")
    return value


def _parse_constraint(value: Any, label: str) -> GroundingConstraint | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object or null")
    _require_exact_keys(
        value,
        {"reference", "category", "attributes", "relation", "relation_reference"},
        label,
    )
    attributes = value["attributes"]
    if not isinstance(attributes, dict):
        raise ValueError(f"{label}.attributes must be an object")
    _require_exact_keys(attributes, {"color", "shape", "size", "location"}, f"{label}.attributes")
    return GroundingConstraint(
        reference=_optional_string(value["reference"], f"{label}.reference"),
        category=_optional_string(value["category"], f"{label}.category"),
        attributes=ObjectAttributes(
            color=_optional_string(attributes["color"], f"{label}.attributes.color"),
            shape=_optional_string(attributes["shape"], f"{label}.attributes.shape"),
            size=_optional_string(attributes["size"], f"{label}.attributes.size"),
            location=_optional_string(attributes["location"], f"{label}.attributes.location"),
        ),
        relation=_optional_string(value["relation"], f"{label}.relation"),
        relation_reference=_optional_string(
            value["relation_reference"],
            f"{label}.relation_reference",
        ),
    )


def parse_semantic_payload(
    payload: dict[str, Any],
    *,
    request: SemanticRequest,
    provenance: SemanticProvenance,
) -> SemanticResult:
    if not isinstance(payload, dict):
        raise ValueError("semantic payload must be a JSON object")
    _require_exact_keys(
        payload,
        {
            "decision",
            "normalized_intent_id",
            "operation",
            "target",
            "destination",
            "semantic_confidence",
            "ambiguity",
            "reason",
        },
        "semantic payload",
    )

    try:
        decision = SemanticDecision(payload["decision"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid semantic decision") from exc

    operation_value = payload["operation"]
    try:
        operation = None if operation_value is None else Operation(operation_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid semantic operation") from exc


    confidence = payload["semantic_confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("semantic_confidence must be numeric")

    ambiguity_value = payload["ambiguity"]
    ambiguity = None
    if ambiguity_value is not None:
        if not isinstance(ambiguity_value, dict):
            raise ValueError("ambiguity must be an object or null")
        _require_exact_keys(
            ambiguity_value,
            {"reason", "clarification_slots"},
            "ambiguity",
        )
        slots = ambiguity_value["clarification_slots"]
        if not isinstance(slots, list) or any(not isinstance(slot, str) for slot in slots):
            raise ValueError("clarification_slots must be an array of strings")
        ambiguity = SemanticAmbiguity(
            reason=_optional_string(ambiguity_value["reason"], "ambiguity.reason") or "",
            clarification_slots=tuple(slots),
        )

    return SemanticResult(
        decision=decision,
        normalized_intent_id=_optional_string(
            payload["normalized_intent_id"],
            "normalized_intent_id",
        ),
        operation=operation,
        target=_parse_constraint(payload["target"], "target"),
        destination=_parse_constraint(payload["destination"], "destination"),
        semantic_confidence=float(confidence),
        ambiguity=ambiguity,
        reason=_optional_string(payload["reason"], "reason"),
        observation_revision=request.observation_revision,
        source_language=request.source_language,
        provenance=provenance,
    )
