"""Gemini provider adapter with strict semantic schema and model fallback."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from vla_robot_sorting.contracts.semantic import SemanticProvenance, SemanticRequest, SemanticResult
from vla_robot_sorting.semantics.vlm import (
    PROMPT_SHA256,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_semantic_payload,
    semantic_response_schema,
    semantic_schema_sha256,
)


class SemanticProviderError(RuntimeError):
    def __init__(
        self, message: str, *, http_status: int | None = None, retry_after_s: float | None = None
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.retry_after_s = retry_after_s


class GeminiTransport(Protocol):
    def generate_json(
        self,
        *,
        model_id: str,
        system_instruction: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class GeminiModelConfig:
    primary_model: str = "gemini-robotics-er-2-preview"
    fallback_model: str = "gemini-3.5-flash-lite"

    def __post_init__(self) -> None:
        if not self.primary_model.strip() or not self.fallback_model.strip():
            raise ValueError("Gemini model IDs must be non-empty")
        if self.primary_model == self.fallback_model:
            raise ValueError("primary and fallback model IDs must differ")


class GeminiRESTTransport:
    """Minimal stdlib REST transport; API keys are never stored in provenance."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_s: float = 60.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Gemini API key must be non-empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._timeout_s = timeout_s

    def generate_json(
        self,
        *,
        model_id: str,
        system_instruction: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        model = urllib.parse.quote(model_id, safe="-._")
        url = f"{self._endpoint}/models/{model}:generateContent"
        body = {
            "systemInstruction": {
                "parts": [{"text": system_instruction}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseFormat": {
                    "text": {
                        "mimeType": "APPLICATION_JSON",
                        "schema": response_schema,
                    }
                },
            },
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                retry_after_s = float(retry_after) if retry_after is not None else None
            except ValueError:
                retry_after_s = None
            # Never retain provider response bodies: they may contain request data.
            raise SemanticProviderError(
                f"Gemini HTTP {exc.code} for model {model_id}",
                http_status=exc.code,
                retry_after_s=retry_after_s,
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SemanticProviderError(
                f"Gemini request failed for model {model_id}: {type(exc).__name__}"
            ) from exc

        try:
            candidates = raw["candidates"]
            text = candidates[0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SemanticProviderError(
                f"Gemini response for model {model_id} has no candidate JSON text"
            ) from exc

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SemanticProviderError(
                f"Gemini model {model_id} returned non-JSON structured output"
            ) from exc
        if not isinstance(payload, dict):
            raise SemanticProviderError(
                f"Gemini model {model_id} returned a non-object semantic payload"
            )
        return payload


class GeminiSemanticReasoner:
    def __init__(
        self,
        transport: GeminiTransport,
        *,
        config: GeminiModelConfig | None = None,
        fallback_enabled: bool = True,
    ) -> None:
        self._transport = transport
        self._config = config or GeminiModelConfig()
        self._fallback_enabled = fallback_enabled

    @classmethod
    def from_env(cls) -> GeminiSemanticReasoner:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise SemanticProviderError("GEMINI_API_KEY is not configured")
        config = GeminiModelConfig(
            primary_model=os.environ.get(
                "VLM_PRIMARY_MODEL",
                "gemini-robotics-er-2-preview",
            ),
            fallback_model=os.environ.get(
                "VLM_FALLBACK_MODEL",
                "gemini-3.5-flash-lite",
            ),
        )
        return cls(GeminiRESTTransport(api_key=api_key), config=config)

    def _reason_with_model(
        self,
        request: SemanticRequest,
        *,
        model_id: str,
        fallback_used: bool,
    ) -> SemanticResult:
        payload = self._transport.generate_json(
            model_id=model_id,
            system_instruction=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(request),
            response_schema=semantic_response_schema(),
        )
        provenance = SemanticProvenance(
            provider="google-gemini",
            model_id=model_id,
            prompt_version=PROMPT_VERSION,
            prompt_sha256=PROMPT_SHA256,
            schema_sha256=semantic_schema_sha256(),
            fallback_used=fallback_used,
        )
        return parse_semantic_payload(payload, request=request, provenance=provenance)

    def reason(self, request: SemanticRequest) -> SemanticResult:
        primary_error: Exception | None = None
        try:
            return self._reason_with_model(
                request,
                model_id=self._config.primary_model,
                fallback_used=False,
            )
        except (SemanticProviderError, ValueError, KeyError, TypeError) as exc:
            primary_error = exc

        if not self._fallback_enabled:
            raise SemanticProviderError(
                f"Gemini semantic request failed: {type(primary_error).__name__}"
            ) from primary_error

        try:
            return self._reason_with_model(
                request,
                model_id=self._config.fallback_model,
                fallback_used=True,
            )
        except (SemanticProviderError, ValueError, KeyError, TypeError) as fallback_error:
            raise SemanticProviderError(
                "both Gemini semantic models failed validation/request; "
                f"primary={type(primary_error).__name__}, "
                f"fallback={type(fallback_error).__name__}"
            ) from fallback_error
