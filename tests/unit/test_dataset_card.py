"""Dataset card statistics (tools/data/dataset_card.py) on synthetic sorting metadata."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "twin"))

import sorting_task as st  # noqa: E402

spec = importlib.util.spec_from_file_location("dataset_card", ROOT / "tools/data/dataset_card.py")
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)


def _write(root: Path, index: int, seed: int, success: bool = True) -> dict:
    rng = random.Random(seed)
    scene = st.sample_scene(rng)
    cmd = st.make_command(scene, rng, ("en", "vi")[index % 2], 0.5)
    meta = {
        "task": cmd.text,
        "language": cmd.language,
        "command_kind": cmd.kind,
        "object_reference": dict(cmd.object_reference),
        "container_reference": list(cmd.container_reference),
        "object": cmd.targets[0],
        "destination": cmd.destination,
        "scene": {
            "objects": [{"name": o.name} for o in scene.objects],
            "containers": [
                {"colour": c.colour, "outline": c.outline, "depth": c.depth, "position": c.position}
                for c in scene.containers
            ],
        },
        "success": success,
        "fps": 15,
        "frames": 100 + index,
        "seed": 0,
    }
    folder = root / f"episode_{index:06d}"
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def test_statistics_and_card(tmp_path, capsys) -> None:
    for i in range(10):
        _write(tmp_path, i, seed=i)
    _write(tmp_path, 10, seed=10, success=False)
    stats = dc.statistics_of(dc.load_metas(tmp_path))
    assert stats["episodes"] == 10 and stats["failed_episodes_kept"] == 1
    assert stats["language"] == {"en": 5, "vi": 5}
    assert sum(stats["command_kind"].values()) == 10
    assert sum(stats["destination_depth"].values()) == 10 and "?" not in stats["destination_depth"]
    assert stats["frames_per_episode"]["min"] == 100 and stats["seeds"] == [0]
    assert dc.main([str(tmp_path), "--name", "demo"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# demo") and "| en | 5 | 50% |" in out
