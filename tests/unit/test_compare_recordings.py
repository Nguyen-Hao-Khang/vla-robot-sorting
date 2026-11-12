"""Reproducibility check for recordings (tools/twin/compare_recordings.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "twin"))

import compare_recordings as cr  # noqa: E402


def _episode(root: Path, index: int, states: np.ndarray, **meta) -> None:
    folder = root / f"episode_{index:06d}"
    folder.mkdir(parents=True)
    np.save(folder / "states.npy", states.astype(np.float32))
    base = {
        "task": "put the red cube in the left tray",
        "targets": ["small red cube"],
        "spawns": {"a": [0, 0, 0]},
    }
    (folder / "meta.json").write_text(json.dumps({**base, **meta}), encoding="utf-8")


def test_identical_and_close_runs_pass(tmp_path) -> None:
    states = np.linspace(0, 1, 30).reshape(10, 3)
    for run, noise in (("a", 0.0), ("b", 2e-4)):
        _episode(tmp_path / run, 0, states + noise, summary=f"took {run}")
        _episode(tmp_path / run, 1, states * 2)
    assert cr.main([str(tmp_path / "a"), str(tmp_path / "b")]) == 0


def test_differences_are_reported(tmp_path, capsys) -> None:
    states = np.zeros((5, 3))
    _episode(tmp_path / "a", 0, states)
    _episode(tmp_path / "b", 0, states + 0.1)
    _episode(tmp_path / "a", 1, states)
    _episode(tmp_path / "b", 1, states, task="move the red cube to the left tray")
    _episode(tmp_path / "a", 2, states)
    out = tmp_path / "r.json"
    assert cr.main([str(tmp_path / "a"), str(tmp_path / "b"), "--json", str(out)]) == 1
    text = capsys.readouterr().out
    assert "identical task definitions: 1/2" in text and "episode count differs" in text
    results = json.loads(out.read_text())
    assert results[1]["task_fields_differ"] == ["task"]
    assert results[0]["max_state_diff"] > 0.09
