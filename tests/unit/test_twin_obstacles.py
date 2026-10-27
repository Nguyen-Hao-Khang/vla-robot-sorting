"""Scene obstacles for the CuRobo planner (pure numpy; CuRobo itself is not required)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/linux"))

import twin_obstacles as tob  # noqa: E402
import twin_scene as ts  # noqa: E402

TABLE = 0.74


def test_object_and_wall_boxes() -> None:
    level = ts.LEVELS[6]
    xyz = {n: [x, y, TABLE + level.object(n).half_height] for n, (x, y) in level.spawn_xy().items()}
    q = {n: (1.0, 0.0, 0.0, 0.0) for n in xyz}
    boxes = tob.scene_boxes(level, xyz, q, TABLE, exclude={"box 1"})
    names = [b.name for b in boxes]
    assert "box 1" not in names and "fresh milk carton" in names
    assert sum("wall" in n for n in names) == 8  # tray + bin walls, no floor, no table spot
    carton = next(b for b in boxes if b.name == "fresh milk carton")
    assert np.allclose(carton.dims, [0.045 + 0.012, 0.038 + 0.012, 0.10 + 0.012])
    bin_walls = [b for b in boxes if b.name.startswith("deep bin")]
    assert all(
        math.isclose(b.dims[2], ts.DEEP_BIN.wall_height + ts.FLOOR_T, abs_tol=0.005)
        for b in bin_walls
    )
    tetra = tob.object_box(ts.LEVELS[3].object("tetrahedron"), [0, 0, 0.77], (1, 0, 0, 0), 0.0)
    assert math.isclose(tetra.dims[2], 2 * ts.LEVELS[3].object("tetrahedron").half_height)


def test_planner_frame_matches_the_target_transform() -> None:
    yaw90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    base = [0.0, -0.45, 0.74, *yaw90]
    box = tob.Box("b", (0.2, -0.17, 0.79), (1.0, 0.0, 0.0, 0.0), (0.05, 0.05, 0.05))
    pose = tob.to_planner_frame(box, base, [0.01, 0.0, 0.0])
    assert np.allclose(pose[:3], [0.28 + 0.01, -0.2, 0.05], atol=1e-9)
    assert np.allclose(np.abs(pose[3:]), np.abs([yaw90[0], 0, 0, -yaw90[3]]), atol=1e-9)


def test_cache_patch_and_world_loading_degrade_gracefully() -> None:
    assert (
        "unavailable" in tob.enlarge_collision_cache() or "cache" in tob.enlarge_collision_cache()
    )

    class NoPlanner:
        robot = object()

    assert tob.load_world(NoPlanner(), []) == "no in-process CuRobo planner"
