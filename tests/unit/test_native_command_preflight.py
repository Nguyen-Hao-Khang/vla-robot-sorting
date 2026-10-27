"""Native operator command is explicit fixed-task mapping, not Gemini/open-vocabulary."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from vla_robot_sorting.contracts.semantic import SemanticDecision, SourceLanguage
from vla_robot_sorting.orchestration.native_command_preflight import (
    MODEL_ID,
    compile_native_command,
)


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("vi", "Gắp chai thuốc và đặt lên tấm đệm"),
        ("en", "Pick up the pill bottle and place it on the pad"),
    ],
)
def test_two_fixed_commands_compile(language: str, text: str) -> None:
    result, digest = compile_native_command(text, language, "native-revision-1")
    assert result.decision is SemanticDecision.STRUCTURED_SUCCESS
    assert result.normalized_intent_id == "move_pillbottle_pad"
    assert result.source_language is SourceLanguage(language)
    assert result.target is not None and result.target.reference == "pill bottle"
    assert result.destination is not None and result.destination.reference == "pad"
    assert result.provenance.model_id == MODEL_ID
    assert result.provenance.provider == "explicit_fixed_task_compiler"
    assert digest == sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("vi", "Gắp chai đỏ"),
        ("en", "Sort all objects"),
        ("en", "Gắp chai thuốc và đặt lên tấm đệm"),
        ("vi", "Pick up the pill bottle and place it on the pad"),
        ("en", "Pick up the pill bottle and place it on the pad  "),
        ("en", ""),
    ],
)
def test_unrecognized_or_mixed_command_rejected(language: str, text: str) -> None:
    with pytest.raises(ValueError, match="unsupported command"):
        compile_native_command(text, language, "native-revision-1")


def test_missing_observation_revision_rejected() -> None:
    with pytest.raises(ValueError, match="observation revision"):
        compile_native_command(
            "Pick up the pill bottle and place it on the pad", "en", ""
        )


def test_native_semantic_command_digest_changes_on_rewording() -> None:
    a, digest = compile_native_command(
        "Pick up the pill bottle and place it on the pad", "en", "scene-1"
    )
    assert a.semantic_confidence == 1.0  # exact fixed command, NOT visual confidence
    assert digest != sha256(b"Pick another object").hexdigest()
    assert replace(a, observation_revision="old").observation_revision == "old"
