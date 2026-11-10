#!/usr/bin/env python3
# ruff: noqa: E501
"""Model router for the digital twin: Flash Lite first, Gemini Robotics-ER only when needed.

Primary chain (fast, 500 free requests/day each): gemini-3.5-flash-lite, then
gemini-3.1-flash-lite once the first has used its daily budget (450 by default, counted per
Pacific day in ~/.cache/damh_twin/model_usage.json). Escalation: gemini-robotics-er-2-preview
(20/day) is asked only when the primary answer is a refusal/clarification (second opinion)
or when no primary model produced a valid answer. Models that hit HTTP 429 are skipped for a
cool-down (or for the session when the server asks for a long wait), unavailable models
(HTTP 404) are skipped, and a model that rejects structured output (HTTP 400) is asked once
more with the schema written into the prompt. The API key is read from GEMINI_API_KEY, sent
only in the request header, never printed or logged; provider response bodies are not stored.

Overrides: DAMH_MODEL_CHAIN="a,b", DAMH_MODEL_ESCALATION="c" (or "none"),
DAMH_MODEL_BUDGETS="gemini-3.5-flash-lite=450,gemini-robotics-er-2-preview=18".
List the models your key can call (no generation request is made):
  python tools/twin/twin_models.py --list
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_CHAIN = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
DEFAULT_ESCALATION = ("gemini-robotics-er-2-preview",)
# Requests per Pacific day before a model is skipped (free tier: 500 / 500 / 20).
DEFAULT_BUDGETS = {
    "gemini-3.5-flash-lite": 450,
    "gemini-3.1-flash-lite": 450,
    "gemini-robotics-er-2-preview": 18,
}
USAGE_FILE = Path.home() / ".cache" / "damh_twin" / "model_usage.json"
SHORT_COOLDOWN_S = 65.0  # per-minute limits clear after about a minute
LONG_WAIT_S = 300.0  # longer Retry-After means a daily quota: skip for the session


class ProviderError(RuntimeError):
    def __init__(
        self, message: str, *, status: int | None = None, retry_after_s: float | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after_s = retry_after_s


def _models_from(env: dict[str, str] | None, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    env = os.environ if env is None else env
    raw = env.get(key, "").strip()
    if not raw:
        return default
    if raw.lower() == "none":
        return ()
    chain = tuple(dict.fromkeys(m.strip() for m in raw.split(",") if m.strip()))
    return chain or default


def model_chain(env: dict[str, str] | None = None) -> tuple[str, ...]:
    return _models_from(env, "DAMH_MODEL_CHAIN", DEFAULT_CHAIN) or DEFAULT_CHAIN


def escalation_chain(env: dict[str, str] | None = None) -> tuple[str, ...]:
    return _models_from(env, "DAMH_MODEL_ESCALATION", DEFAULT_ESCALATION)


def extract_json(text: str) -> dict:
    """Parse a JSON object from model text (plain JSON, ```json fences, or prose around it)."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in model output") from None
        value = json.loads(cleaned[start : end + 1])
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("model output is not a JSON object")
    return value


def schema_instruction(schema: dict) -> str:
    return (
        "\n\nReturn ONLY one JSON object (no prose, no code fences) that matches this JSON schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def candidate_text(raw: dict) -> str:
    """Join the non-thought text parts of the first candidate."""
    try:
        parts = raw["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("response has no candidate text") from exc
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought")]
    text = "".join(texts).strip()
    if not text:
        raise ProviderError("response candidate text is empty")
    return text


@dataclass(frozen=True)
class Image:
    data: bytes
    mime_type: str = "image/jpeg"


class GeminiAPI:
    """Minimal stdlib client for generateContent / models.list."""

    def __init__(self, api_key: str, endpoint: str = ENDPOINT, timeout_s: float = 90.0) -> None:
        if not api_key.strip():
            raise ValueError("GEMINI_API_KEY is empty")
        self._key, self._endpoint, self._timeout = api_key, endpoint.rstrip("/"), timeout_s

    def _request(self, url: str, body: dict | None) -> dict:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "x-goog-api-key": self._key},
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retry = exc.headers.get("Retry-After") if exc.headers else None
            try:
                retry_s = float(retry) if retry is not None else None
            except ValueError:
                retry_s = None
            raise ProviderError(f"HTTP {exc.code}", status=exc.code, retry_after_s=retry_s) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"request failed: {type(exc).__name__}") from exc

    def generate(
        self,
        model_id: str,
        system: str,
        prompt: str,
        schema: dict | None,
        images: tuple[Image, ...] = (),
    ) -> dict:
        parts: list[dict[str, Any]] = [
            {
                "inline_data": {
                    "mime_type": img.mime_type,
                    "data": base64.b64encode(img.data).decode(),
                }
            }
            for img in images
        ]
        parts.append({"text": prompt})
        config: dict[str, Any] = {"temperature": 0.0}
        if schema is not None:
            config["responseFormat"] = {"text": {"mimeType": "APPLICATION_JSON", "schema": schema}}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": config,
        }
        url = f"{self._endpoint}/models/{urllib.parse.quote(model_id, safe='-._')}:generateContent"
        try:
            return extract_json(candidate_text(self._request(url, body)))
        except ValueError as exc:
            raise ProviderError(f"unparseable output: {exc}") from exc

    def list_models(self) -> list[dict]:
        rows, token = [], ""
        while True:
            query = "pageSize=1000" + (f"&pageToken={urllib.parse.quote(token)}" if token else "")
            page = self._request(f"{self._endpoint}/models?{query}", None)
            rows += page.get("models", [])
            token = page.get("nextPageToken", "")
            if not token:
                return rows


@dataclass
class ModelState:
    requests: int = 0
    ok: int = 0
    invalid: int = 0
    errors: int = 0
    quota_hits: int = 0
    disabled_until: float = 0.0
    disabled_reason: str = ""
    schema_in_prompt: bool = False
    seconds: float = 0.0


@dataclass
class Attempt:
    model: str
    outcome: str  # ok | invalid | quota | unavailable | error | skipped | budget | escalated
    detail: str = ""
    seconds: float = 0.0


@dataclass
class RouterResult:
    model: str | None
    payload: dict | None
    attempts: list[Attempt] = field(default_factory=list)
    escalated: bool = False

    @property
    def error(self) -> str:
        return "; ".join(
            f"{a.model}: {a.outcome} {a.detail}".strip() for a in self.attempts if a.outcome != "ok"
        )

    @property
    def seconds(self) -> float:
        return sum(a.seconds for a in self.attempts)


def pacific_day(now: float | None = None) -> str:
    """Gemini API daily quotas reset at midnight Pacific time."""
    stamp = time.time() if now is None else now
    try:
        from zoneinfo import ZoneInfo

        return datetime.fromtimestamp(stamp, ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:  # no tz database: fixed UTC-8
        return datetime.fromtimestamp(stamp - 8 * 3600, timezone.utc).strftime("%Y-%m-%d")


class UsageStore:
    """Requests per model per (Pacific) day, persisted so budgets survive restarts."""

    def __init__(
        self, path: Path | None = USAGE_FILE, day: Callable[[], str] = pacific_day
    ) -> None:
        self.path, self.day = path, day
        self.data: dict[str, dict[str, int]] = {}
        if path is not None and path.is_file():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def used(self, model: str) -> int:
        return int(self.data.get(self.day(), {}).get(model, 0))

    def add(self, model: str) -> None:
        today = self.day()
        self.data = {today: self.data.get(today, {})}  # keep only today
        self.data[today][model] = self.data[today].get(model, 0) + 1
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
            except OSError:
                pass


class ModelRouter:
    """Primary chain (cheap, fast) plus escalation models used only when really needed.

    For each request the primary models are tried in order; a model is skipped when it is
    cooling down after HTTP 429, unavailable (404), or has used its daily budget (default:
    450 of the 500 free requests). The first answer that passes ``validate`` is used. If
    ``escalate_when(answer)`` is true (e.g. a refusal or clarification) the escalation models
    (Gemini Robotics-ER) are asked for a second opinion; they are also the last resort when no
    primary model produced a valid answer.
    """

    def __init__(
        self,
        api: GeminiAPI,
        chain: tuple[str, ...] = DEFAULT_CHAIN,
        escalation: tuple[str, ...] = DEFAULT_ESCALATION,
        budgets: dict[str, int] | None = None,
        usage: UsageStore | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api, self.chain, self.escalation, self.clock = api, chain, escalation, clock
        self.budgets = dict(DEFAULT_BUDGETS if budgets is None else budgets)
        self.usage = usage if usage is not None else UsageStore(None)
        self.state = {m: ModelState() for m in (*chain, *escalation)}

    def _disable(self, model: str, seconds: float, reason: str) -> None:
        state = self.state[model]
        state.disabled_until = self.clock() + seconds
        state.disabled_reason = reason

    def _available(self, model: str, result: RouterResult) -> bool:
        state = self.state[model]
        if state.disabled_until > self.clock():
            result.attempts.append(Attempt(model, "skipped", state.disabled_reason))
            return False
        budget = self.budgets.get(model)
        if budget is not None and self.usage.used(model) >= budget:
            result.attempts.append(Attempt(model, "budget", f"daily budget {budget} used"))
            return False
        return True

    def _first_valid(
        self, models, system, prompt, schema, images, validate, result
    ) -> tuple[str, dict] | None:
        for model in models:
            if not self._available(model, result):
                continue
            payload = self._ask(model, self.state[model], system, prompt, schema, images, result)
            if payload is None:
                continue
            problem = validate(payload)
            if problem:
                self.state[model].invalid += 1
                result.attempts.append(Attempt(model, "invalid", problem, self._last_seconds))
                continue
            self.state[model].ok += 1
            return model, payload
        return None

    def call(
        self,
        system: str,
        prompt: str,
        schema: dict,
        validate: Callable[[dict], str | None] = lambda _p: None,
        images: tuple[Image, ...] = (),
        escalate_when: Callable[[dict], bool] | None = None,
    ) -> RouterResult:
        result = RouterResult(None, None)
        found = self._first_valid(self.chain, system, prompt, schema, images, validate, result)
        if found is not None:
            model, payload = found
            result.attempts.append(Attempt(model, "ok", "", self._last_seconds))
            result.model, result.payload = model, payload
            if not (escalate_when and escalate_when(payload)) or not self.escalation:
                return result
            result.attempts[-1].detail = "asked for a second opinion"
        second = self._first_valid(
            self.escalation, system, prompt, schema, images, validate, result
        )
        if second is not None:
            model, payload = second
            result.attempts.append(Attempt(model, "ok", "escalated", self._last_seconds))
            result.model, result.payload, result.escalated = model, payload, True
        return result

    _last_seconds = 0.0

    def _ask(self, model, state, system, prompt, schema, images, result) -> dict | None:
        for _ in range(2):  # second pass only when the model rejected the schema settings
            state.requests += 1
            self.usage.add(model)
            use_schema = None if state.schema_in_prompt else schema
            text = prompt + (schema_instruction(schema) if state.schema_in_prompt else "")
            started = time.perf_counter()
            try:
                payload = self.api.generate(model, system, text, use_schema, images)
                self._last_seconds = time.perf_counter() - started
                state.seconds += self._last_seconds
                return payload
            except ProviderError as exc:
                spent = time.perf_counter() - started
                state.seconds += spent
                if exc.status == 429:
                    state.quota_hits += 1
                    wait = exc.retry_after_s or SHORT_COOLDOWN_S
                    long_wait = wait >= LONG_WAIT_S
                    self._disable(
                        model,
                        1e9 if long_wait else wait,
                        "quota exhausted" if long_wait else "rate limited",
                    )
                    result.attempts.append(
                        Attempt(model, "quota", f"retry after {wait:.0f}s", spent)
                    )
                    return None
                if exc.status == 404:
                    self._disable(model, 1e9, "not available for this key")
                    result.attempts.append(Attempt(model, "unavailable", "HTTP 404", spent))
                    return None
                if (exc.status == 400 or "unparseable" in str(exc)) and not state.schema_in_prompt:
                    state.schema_in_prompt = True  # retry once with the schema in the prompt
                    continue
                state.errors += 1
                result.attempts.append(Attempt(model, "error", str(exc), spent))
                return None
        return None

    def summary(self) -> list[str]:
        rows = []
        now = self.clock()
        for model in (*self.chain, *self.escalation):
            s = self.state[model]
            status = s.disabled_reason if s.disabled_until > now else "ready"
            role = "escalation" if model in self.escalation else "primary"
            budget = self.budgets.get(model)
            used = (
                f"{self.usage.used(model)}/{budget}"
                if budget is not None
                else str(self.usage.used(model))
            )
            avg = s.seconds / s.requests if s.requests else 0.0
            rows.append(
                f"{model} ({role}): {status}; today {used}; session requests={s.requests} ok={s.ok} "
                f"invalid={s.invalid} errors={s.errors} quota={s.quota_hits}; avg {avg:.1f} s"
            )
        return rows


def parse_budgets(raw: str) -> dict[str, int]:
    out = {}
    for item in raw.split(","):
        if "=" in item:
            name, value = item.split("=", 1)
            if name.strip() and value.strip().isdigit():
                out[name.strip()] = int(value)
    return out


def router_from_env() -> ModelRouter:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in this terminal.")
    budgets = {**DEFAULT_BUDGETS, **parse_budgets(os.environ.get("DAMH_MODEL_BUDGETS", ""))}
    return ModelRouter(GeminiAPI(key), model_chain(), escalation_chain(), budgets, UsageStore())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list", action="store_true", help="list generateContent models for this key"
    )
    args = parser.parse_args(argv)
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("GEMINI_API_KEY_MISSING: export GEMINI_API_KEY in this terminal.")
        return 3
    print(
        "Model chain: "
        + " -> ".join(model_chain())
        + " | escalation: "
        + (", ".join(escalation_chain()) or "none")
    )
    if args.list:
        api = GeminiAPI(key)
        try:
            models = api.list_models()
        except ProviderError as exc:
            print(f"LIST_FAILED: {exc}")
            return 5
        names = sorted(
            m.get("name", "").removeprefix("models/")
            for m in models
            if "generateContent" in m.get("supportedGenerationMethods", [])
        )
        chain = set(model_chain()) | set(escalation_chain())
        for name in names:
            if name.startswith(("gemma", "gemini")):
                print(("* " if name in chain else "  ") + name)
        missing = [m for m in (*model_chain(), *escalation_chain()) if m not in names]
        print("CHAIN_MISSING=" + (",".join(missing) if missing else "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
