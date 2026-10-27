from __future__ import annotations

import io
from collections import deque
from typing import Any
from urllib.error import HTTPError

import pytest

from vla_robot_sorting.contracts.semantic import SemanticRequest, SourceLanguage
from vla_robot_sorting.semantics.gemini import (
    GeminiModelConfig,
    GeminiRESTTransport,
    GeminiSemanticReasoner,
    SemanticProviderError,
)


def payload() -> dict[str, Any]:
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
        "semantic_confidence": 0.9,
        "ambiguity": None,
        "reason": None,
    }


class FakeTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, dict)
        return response


def request() -> SemanticRequest:
    return SemanticRequest(
        command="Lấy chai đỏ bỏ vào khay trái.",
        source_language=SourceLanguage.VI,
        observation_revision="obs-12",
    )


def test_primary_model_success_records_reproducible_provenance() -> None:
    transport = FakeTransport([payload()])
    reasoner = GeminiSemanticReasoner(
        transport,
        config=GeminiModelConfig("primary-model", "fallback-model"),
    )

    result = reasoner.reason(request())

    assert result.provenance.model_id == "primary-model"
    assert not result.provenance.fallback_used
    assert len(result.provenance.prompt_sha256) == 64
    assert transport.calls[0]["response_schema"]["additionalProperties"] is False


def test_invalid_primary_payload_falls_back() -> None:
    bad = payload()
    bad["motor_command"] = "move_joint_1"
    transport = FakeTransport([bad, payload()])
    reasoner = GeminiSemanticReasoner(
        transport,
        config=GeminiModelConfig("primary-model", "fallback-model"),
    )

    result = reasoner.reason(request())

    assert result.provenance.model_id == "fallback-model"
    assert result.provenance.fallback_used
    assert [call["model_id"] for call in transport.calls] == [
        "primary-model",
        "fallback-model",
    ]


def test_both_models_failing_is_typed_provider_failure() -> None:
    transport = FakeTransport(
        [
            SemanticProviderError("primary unavailable"),
            SemanticProviderError("fallback unavailable"),
        ]
    )
    reasoner = GeminiSemanticReasoner(
        transport,
        config=GeminiModelConfig("primary-model", "fallback-model"),
    )

    with pytest.raises(SemanticProviderError, match="both Gemini semantic models failed"):
        reasoner.reason(request())


def test_benchmark_can_disable_automatic_fallback() -> None:
    transport = FakeTransport([SemanticProviderError("primary unavailable"), payload()])
    reasoner = GeminiSemanticReasoner(
        transport,
        config=GeminiModelConfig("primary-model", "fallback-model"),
        fallback_enabled=False,
    )

    with pytest.raises(SemanticProviderError, match="Gemini semantic request failed"):
        reasoner.reason(request())

    assert [call["model_id"] for call in transport.calls] == ["primary-model"]


def test_rest_http_error_is_sanitized_and_exposes_retry_after(monkeypatch) -> None:
    def fail_request(*args, **kwargs):
        raise HTTPError(
            "https://example.invalid",
            429,
            "rate limited",
            {"Retry-After": "3"},
            io.BytesIO(b"private prompt fragment"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fail_request)
    transport = GeminiRESTTransport(api_key="never-log-this")

    with pytest.raises(SemanticProviderError) as exc_info:
        transport.generate_json(
            model_id="test-model",
            system_instruction="private system",
            user_prompt="private user",
            response_schema={},
        )

    assert "private" not in str(exc_info.value)
    assert "never-log-this" not in str(exc_info.value)
    assert exc_info.value.http_status == 429
    assert exc_info.value.retry_after_s == 3.0


def test_production_default_fallback_is_low_cost_structured_flash_lite(monkeypatch) -> None:
    """No key is exposed and default fallback remains a REST JSON model."""
    from vla_robot_sorting.semantics.gemini import GeminiRESTTransport

    assert GeminiModelConfig().primary_model == "gemini-robotics-er-2-preview"
    assert GeminiModelConfig().fallback_model == "gemini-3.5-flash-lite"

    monkeypatch.setenv("GEMINI_API_KEY", "fake-value-for-local-test-only")
    monkeypatch.delenv("VLM_PRIMARY_MODEL", raising=False)
    monkeypatch.delenv("VLM_FALLBACK_MODEL", raising=False)
    reasoner = GeminiSemanticReasoner.from_env()
    assert isinstance(reasoner._transport, GeminiRESTTransport)
    assert reasoner._config.fallback_model == "gemini-3.5-flash-lite"
    assert reasoner._config.primary_model != reasoner._config.fallback_model
