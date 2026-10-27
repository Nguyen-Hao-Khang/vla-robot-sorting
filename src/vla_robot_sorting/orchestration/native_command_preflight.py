"""Explicit fixed-task bilingual command compiler for simulator-ground-truth baseline.

This is not Gemini and not a detector: user command picks an existing pinned
RoboTwin task; real actors, scene and collision geometry are verified separately.
"""
from __future__ import annotations

from hashlib import sha256

from vla_robot_sorting.contracts.semantic import (
    GroundingConstraint,
    Operation,
    SemanticDecision,
    SemanticProvenance,
    SemanticResult,
    SourceLanguage,
)

SUPPORTED = {
    ("en", "Pick up the pill bottle and place it on the pad"): (
        SourceLanguage.EN, "pill bottle",
    ),
    ("vi", "Gắp chai thuốc và đặt lên tấm đệm"): (
        SourceLanguage.VI, "pill bottle",
    ),
}
MODEL_ID = "fixed_task_compiler_not_gemini"
PROMPT_SHA = sha256(b"fixed-pinned-robotwin-task-v1").hexdigest()
SCHEMA_SHA = sha256(b"SemanticResult-v1").hexdigest()


def compile_native_command(
    command: str,
    language: str,
    observation_revision: str,
) -> tuple[SemanticResult, str]:
    if (language, command) not in SUPPORTED:
        raise ValueError("unsupported command; no guessing or universal language parsing")
    if not observation_revision.strip():
        raise ValueError("observation revision required")
    source_language, reference = SUPPORTED[(language, command)]
    semantic = SemanticResult(
        decision=SemanticDecision.STRUCTURED_SUCCESS,
        normalized_intent_id="move_pillbottle_pad",
        operation=Operation.PICK_AND_PLACE,
        target=GroundingConstraint(reference=reference),
        destination=GroundingConstraint(reference="pad"),
        semantic_confidence=1.0,
        ambiguity=None,
        reason=None,
        observation_revision=observation_revision,
        source_language=source_language,
        provenance=SemanticProvenance(
            provider="explicit_fixed_task_compiler",
            model_id=MODEL_ID,
            prompt_version="native-fixed-task-v1",
            prompt_sha256=PROMPT_SHA,
            schema_sha256=SCHEMA_SHA,
            fallback_used=False,
        ),
    )
    return semantic, sha256(command.encode("utf-8")).hexdigest()
