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


view = _load("twin_view", "tools/twin/twin_view.py")


def test_twin_view_defaults_and_guards() -> None:
    args = view.parse_args([])
    assert args.task == "move_pillbottle_pad" and args.render_freq == 1
    with pytest.raises(SystemExit):
        view.parse_args(["--render-freq", "0"])


def test_twin_view_requires_display(monkeypatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    assert view.main([]) == 3
