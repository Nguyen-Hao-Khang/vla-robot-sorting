"""Model router: chain order, quota/404/400 handling, validation escalation, JSON extraction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("twin_models", ROOT / "tools/twin/twin_models.py")
tm = importlib.util.module_from_spec(spec)
sys.modules["twin_models"] = tm
spec.loader.exec_module(tm)

SCHEMA = {"type": "object"}


class FakeAPI:
    def __init__(self, script):
        self.script = script  # model -> list of results (dict or ProviderError)
        self.calls = []

    def generate(self, model, system, prompt, schema, images=()):
        self.calls.append(
            (model, schema is not None, "Return ONLY one JSON object" in prompt, len(images))
        )
        result = self.script[model].pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def R(api, chain, escalation=(), budgets=None, usage=None, clock=None):
    kwargs = {"clock": clock} if clock else {}
    return tm.ModelRouter(
        api, chain, escalation, budgets or {}, usage or tm.UsageStore(None), **kwargs
    )


def test_default_chain_flash_lite_first_robotics_only_for_escalation() -> None:
    assert tm.DEFAULT_CHAIN == ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
    assert tm.DEFAULT_ESCALATION == ("gemini-robotics-er-2-preview",)
    assert tm.DEFAULT_BUDGETS["gemini-3.5-flash-lite"] == 450
    assert tm.model_chain({"DAMH_MODEL_CHAIN": "a, b,a"}) == ("a", "b")
    assert tm.model_chain({}) == tm.DEFAULT_CHAIN
    assert tm.escalation_chain({"DAMH_MODEL_ESCALATION": "none"}) == ()
    assert tm.parse_budgets("a=10, b=x, c=3") == {"a": 10, "c": 3}


def test_first_valid_answer_wins() -> None:
    api = FakeAPI({"m1": [{"x": 1}], "m2": [{"x": 2}]})
    result = R(api, ("m1", "m2")).call("s", "p", SCHEMA)
    assert result.model == "m1" and result.payload == {"x": 1} and len(api.calls) == 1


def test_invalid_answer_falls_to_next_model() -> None:
    api = FakeAPI({"m1": [{"bad": 1}], "m2": [{"good": 1}]})
    router = R(api, ("m1", "m2"))
    result = router.call(
        "s", "p", SCHEMA, validate=lambda p: None if "good" in p else "wrong names"
    )
    assert result.model == "m2" and [a.outcome for a in result.attempts] == ["invalid", "ok"]
    assert router.state["m1"].invalid == 1


def test_refusal_gets_second_opinion_from_escalation_model() -> None:
    api = FakeAPI({"lite": [{"decision": "refuse"}], "er": [{"decision": "execute"}]})
    result = R(api, ("lite",), ("er",)).call(
        "s", "p", SCHEMA, escalate_when=lambda p: p["decision"] == "refuse"
    )
    assert result.model == "er" and result.escalated and result.payload == {"decision": "execute"}
    assert result.attempts[0].detail == "asked for a second opinion"


def test_accepted_answer_does_not_touch_escalation_model() -> None:
    api = FakeAPI({"lite": [{"decision": "execute"}], "er": []})
    result = R(api, ("lite",), ("er",)).call(
        "s", "p", SCHEMA, escalate_when=lambda p: p["decision"] == "refuse"
    )
    assert result.model == "lite" and not result.escalated and [c[0] for c in api.calls] == ["lite"]


def test_refusal_kept_when_escalation_unavailable() -> None:
    api = FakeAPI(
        {
            "lite": [{"decision": "refuse"}],
            "er": [tm.ProviderError("HTTP 429", status=429, retry_after_s=3600)],
        }
    )
    result = R(api, ("lite",), ("er",)).call(
        "s", "p", SCHEMA, escalate_when=lambda p: p["decision"] == "refuse"
    )
    assert (
        result.model == "lite" and result.payload == {"decision": "refuse"} and not result.escalated
    )


def test_escalation_is_last_resort_when_primaries_fail() -> None:
    api = FakeAPI({"lite": [tm.ProviderError("HTTP 500", status=500)], "er": [{"ok": 1}]})
    result = R(api, ("lite",), ("er",)).call("s", "p", SCHEMA)
    assert result.model == "er" and result.escalated


def test_daily_budget_swaps_to_second_flash_lite(tmp_path) -> None:
    day = {"d": "2026-10-18"}
    usage = tm.UsageStore(tmp_path / "usage.json", day=lambda: day["d"])
    api = FakeAPI({"a": [{"n": i} for i in range(3)], "b": [{"m": 1}, {"m": 2}]})
    router = R(api, ("a", "b"), budgets={"a": 2, "b": 5}, usage=usage)
    assert [router.call("s", "p", SCHEMA).model for _ in range(3)] == ["a", "a", "b"]
    assert router.call("s", "p", SCHEMA).attempts[0].outcome == "budget"
    reloaded = tm.UsageStore(tmp_path / "usage.json", day=lambda: day["d"])
    assert reloaded.used("a") == 2 and reloaded.used("b") == 2  # survives a restart
    assert "today 2/2" in "\n".join(router.summary())
    day["d"] = "2026-10-19"
    assert reloaded.used("a") == 0  # new Pacific day, fresh budget


def test_pacific_day_format() -> None:
    assert len(tm.pacific_day(0)) == 10 and tm.pacific_day(0).startswith("1969-12-31")


def test_quota_short_cooldown_then_recovers() -> None:
    clock = Clock()
    api = FakeAPI(
        {
            "m1": [tm.ProviderError("HTTP 429", status=429, retry_after_s=30), {"x": 1}],
            "m2": [{"y": 1}],
        }
    )
    router = R(api, ("m1", "m2"), clock=clock)
    assert router.call("s", "p", SCHEMA).model == "m2"
    api.script["m2"].append({"y": 2})
    assert router.call("s", "p", SCHEMA).attempts[0].outcome == "skipped"  # still cooling down
    clock.t = 31
    assert router.call("s", "p", SCHEMA).model == "m1"


def test_daily_quota_and_missing_model_disable_for_session() -> None:
    clock = Clock()
    api = FakeAPI(
        {
            "m1": [tm.ProviderError("HTTP 429", status=429, retry_after_s=3600)],
            "m2": [tm.ProviderError("HTTP 404", status=404)],
            "m3": [{"ok": 1}, {"ok": 2}],
        }
    )
    router = R(api, ("m1", "m2", "m3"), clock=clock)
    assert router.call("s", "p", SCHEMA).model == "m3"
    clock.t = 10_000
    second = router.call("s", "p", SCHEMA)
    assert second.model == "m3" and [a.outcome for a in second.attempts[:2]] == [
        "skipped",
        "skipped",
    ]
    summary = "\n".join(router.summary())
    assert "quota exhausted" in summary and "not available" in summary


def test_schema_rejection_retries_once_with_schema_in_prompt() -> None:
    api = FakeAPI({"gemma": [tm.ProviderError("HTTP 400", status=400), {"x": 1}, {"x": 2}]})
    router = R(api, ("gemma",))
    assert router.call("s", "p", SCHEMA).payload == {"x": 1}
    assert api.calls[0][1:3] == (True, False) and api.calls[1][1:3] == (False, True)
    router.call("s", "p", SCHEMA)
    assert api.calls[2][1:3] == (False, True)  # remembered for the session


def test_other_errors_fall_through_and_report() -> None:
    api = FakeAPI(
        {
            "m1": [tm.ProviderError("request failed: URLError")],
            "m2": [tm.ProviderError("HTTP 500", status=500)],
        }
    )
    result = R(api, ("m1", "m2")).call("s", "p", SCHEMA)
    assert result.payload is None and "URLError" in result.error and "HTTP 500" in result.error


def test_images_are_forwarded() -> None:
    api = FakeAPI({"m1": [{"objects": []}]})
    R(api, ("m1",)).call("s", "p", SCHEMA, images=(tm.Image(b"jpg"),))
    assert api.calls[0][3] == 1


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Sure! Here it is: {"a": 1} hope it helps',
        '[{"a": 1}]',
    ],
)
def test_extract_json_variants(text) -> None:
    assert tm.extract_json(text) == {"a": 1}


def test_extract_json_rejects_non_objects() -> None:
    with pytest.raises(ValueError):
        tm.extract_json("no json here")
    with pytest.raises(ValueError):
        tm.extract_json("[1, 2]")


def test_candidate_text_skips_thoughts() -> None:
    raw = {
        "candidates": [
            {"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": '{"a": 1}'}]}}
        ]
    }
    assert json.loads(tm.candidate_text(raw)) == {"a": 1}
    with pytest.raises(tm.ProviderError):
        tm.candidate_text({"candidates": []})


def test_request_body_carries_images_schema_and_key_only_in_header(monkeypatch) -> None:
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
            ).encode()

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(tm.urllib.request, "urlopen", fake_urlopen)
    api = tm.GeminiAPI("SECRET-KEY")
    assert api.generate("gemma-4-31b-it", "sys", "prompt", SCHEMA, (tm.Image(b"\xff\xd8"),)) == {
        "ok": True
    }
    assert "SECRET-KEY" not in seen["url"] and "SECRET-KEY" not in json.dumps(seen["body"])
    parts = seen["body"]["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "image/jpeg" and parts[-1]["text"] == "prompt"
    assert seen["body"]["generationConfig"]["responseFormat"]["text"]["schema"] == SCHEMA
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"


def test_main_without_key_makes_no_request(monkeypatch, capsys) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert tm.main(["--list"]) == 3 and "MISSING" in capsys.readouterr().out
