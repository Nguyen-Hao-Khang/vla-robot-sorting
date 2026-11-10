"""Probe display hook: optional SAPIEN viewer redraw that never changes physics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "probe_display", ROOT / "scripts/linux/probe_panthera_project_manipulation.py"
)
probe_mod = importlib.util.module_from_spec(spec)
sys.modules["probe_display"] = probe_mod
spec.loader.exec_module(probe_mod)


def test_probe_render_freq_defaults_to_headless(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["p", "--robotwin-root", "/r", "--project-root", "/p", "--output", "/o"]
    )
    assert probe_mod.parse_args().render_freq == 0


def test_display_hook_draws_every_n_steps_without_touching_physics() -> None:
    class Viewer:
        closed = False

        def __init__(self) -> None:
            self.frames = 0

        def render(self) -> None:
            self.frames += 1

    class Task:
        render_freq = 4

        def __init__(self) -> None:
            self.viewer = Viewer()
            self.steps = 0
            self.renders = 0

        def _step_scene(self) -> str:
            self.steps += 1
            return "stepped"

        def _update_render(self) -> None:
            self.renders += 1

    task = Task()
    info = probe_mod.install_display_viewer(task, 3)
    assert task.render_freq == 0
    results = [task._step_scene() for _ in range(10)]
    assert results == ["stepped"] * 10 and task.steps == 10
    assert task.viewer.frames == 3 and task.renders == 3
    assert info["counter"] == {"steps": 10, "frames": 3} and info["physics_affecting"] is False


def test_display_hook_requires_viewer() -> None:
    class Task:
        viewer = None

    with pytest.raises(RuntimeError, match="no viewer"):
        probe_mod.install_display_viewer(Task(), 2)
