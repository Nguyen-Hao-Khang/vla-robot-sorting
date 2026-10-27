"""Digital-twin step 1/2 smoke tools: argument guards, prompt and scene validation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


view = _load("twin_view", "scripts/linux/twin_view.py")
gem = _load("gemini_command_smoke", "scripts/linux/gemini_command_smoke.py")


def test_twin_view_defaults_and_guards() -> None:
    args = view.parse_args([])
    assert args.task == "move_pillbottle_pad" and args.render_freq == 1
    with pytest.raises(SystemExit):
        view.parse_args(["--render-freq", "0"])


def test_twin_view_requires_display(monkeypatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    assert view.main([]) == 3


def test_gemini_prompt_lists_scene_and_command() -> None:
    prompt = gem.build_prompt("gắp chai thuốc", ["pill bottle", "pad"])
    assert "- pill bottle" in prompt and prompt.endswith("gắp chai thuốc")


def test_gemini_scene_validation_rejects_invented_objects() -> None:
    scene = ["pill bottle (white)", "pad (blue square)"]
    ok = {"decision": "pick_and_place", "target_object": "pill bottle", "destination": "pad"}
    bad = {"decision": "pick_and_place", "target_object": "red cup", "destination": "pad"}
    assert gem.validate(ok, scene) == []
    assert gem.validate(bad, scene) == ["target_object_not_in_scene"]
    assert gem.validate({"decision": "refuse"}, scene) == []


def test_gemini_missing_key_makes_no_request(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert gem.main(["gắp chai thuốc"]) == 3


def test_gemini_falls_back_once_then_reports(monkeypatch, capsys) -> None:
    calls = []

    class Fake:
        def __init__(self, **_):
            pass

        def generate_json(self, *, model_id, **_):
            calls.append(model_id)
            if len(calls) == 1:
                raise gem.SemanticProviderError("boom")
            return {
                "decision": "refuse",
                "language": "vi",
                "target_object": None,
                "destination": None,
                "reason": "x",
            }

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(gem, "GeminiRESTTransport", Fake)
    assert gem.main(["đừng gắp"]) == 0
    assert len(calls) == 2 and calls[0] != calls[1]
    assert "SCENE_CHECK=OK" in capsys.readouterr().out
