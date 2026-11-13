"""Reproducibility check for recordings (tools/twin/compare_recordings.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "twin"))

import compare_recordings as cr  # noqa: E402


def _episode(root: Path, index: int, states: np.ndarray, key: str | None = None, **meta) -> None:
    folder = root / f"episode_{index:06d}"
    folder.mkdir(parents=True)
    np.save(folder / "states.npy", states.astype(np.float32))
    base = {"task": "put the red cube in the left tray", "targets": ["small red cube"]}
    if key is not None:
        base["episode_key"] = key
    (folder / "meta.json").write_text(json.dumps({**base, **meta}), encoding="utf-8")


def test_close_runs_with_different_lengths_pass(tmp_path, capsys) -> None:
    t = np.linspace(0, 1, 40)[:, None] * np.ones((1, 3))
    _episode(tmp_path / "a", 0, t, key="0-0", summary="a")
    _episode(tmp_path / "b", 0, np.linspace(0, 1, 42)[:, None] * np.ones((1, 3)) + 1e-3, key="0-0")
    assert cr.main([str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    out = capsys.readouterr().out
    assert (
        "identical task definitions: 1/1" in out and "length difference (frames): median 2" in out
    )


def test_pairs_by_key_after_a_failed_episode(tmp_path, capsys) -> None:
    s = np.zeros((10, 3))
    # run b lost layout 1 (failed episode), so its later episodes have smaller indices
    for i, key in enumerate(["0-0", "0-1", "1-0", "2-0", "2-1"]):
        _episode(tmp_path / "a", i, s, key=key, task=f"t{key}")
    for i, key in enumerate(["0-0", "0-1", "2-0", "2-1"]):
        _episode(tmp_path / "b", i, s, key=key, task=f"t{key}")
    results, extra = cr.compare(tmp_path / "a", tmp_path / "b")
    assert all(r["task_identical"] for r in results) and extra["only_in_a"] == ["1-0"]
    assert cr.main([str(tmp_path / "a"), str(tmp_path / "b")]) == 1  # 4/5 paired < 90%


def test_task_differences_are_reported(tmp_path, capsys) -> None:
    s = np.zeros((5, 3))
    _episode(tmp_path / "a", 0, s)
    _episode(tmp_path / "b", 0, s, task="move the red cube to the left tray")
    out = tmp_path / "r.json"
    assert cr.main([str(tmp_path / "a"), str(tmp_path / "b"), "--json", str(out)]) == 1
    assert json.loads(out.read_text())["episodes"][0]["task_fields_differ"] == ["task"]
