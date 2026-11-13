"""Scene obstacles for RoboTwin's CuRobo planner, so arm trajectories avoid the objects.

Upstream RoboTwin gives CuRobo a world with the table only, so every planned arm motion may
sweep through objects and container walls (live logs: a 10 cm carton in the deep bin tipped
while the arm reached for a box in the front row). This module turns the twin scene into
oriented boxes (objects with a small margin, container walls) in the planner's base frame and
loads them into the planner's MotionGen instances before each pick.

CuRobo builds its collision cache from the initial world (table only) and, with CUDA graphs,
cannot grow it later; ``enlarge_collision_cache`` therefore patches MotionGenConfig before
RoboTwin creates the planner so the cache has room for the scene. Everything is optional:
without CuRobo (or with an unexpected version) the twin falls back to its own safe-height
motions and reports the planner as "table only".
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import twin_scene as ts

OBSTACLE_PAD = 0.006  # margin added around every object box (each side)
CACHE_OBB = 48  # oriented boxes reserved in CuRobo's collision cache


@dataclass(frozen=True)
class Box:
    name: str
    centre: tuple[float, float, float]  # world
    quat: tuple[float, float, float, float]  # world (w, x, y, z)
    dims: tuple[float, float, float]  # full lengths


def object_box(obj: ts.ObjectSpec, xyz, q, pad: float = OBSTACLE_PAD) -> Box:
    """Oriented bounding box of an object in its current pose."""
    if obj.shape == "box" and obj.half_extents:
        half = np.asarray(obj.half_extents, float)
        centre_local = np.zeros(3)
    elif obj.shape in ("prism", "tetrahedron", "halfcyl"):
        verts = np.asarray(ts.mesh_vertices(obj))
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        half, centre_local = (hi - lo) / 2, (hi + lo) / 2
    else:  # cube, cylinder, bottle, sphere
        r = obj.radius
        half = np.array([r, r, obj.half_height if obj.shape != "sphere" else r])
        centre_local = np.zeros(3)
    rot = np.asarray(ts.quat_matrix(q), float)
    centre = np.asarray(xyz, float) + rot @ centre_local
    dims = tuple(float(v) for v in 2 * half + 2 * pad)
    return Box(obj.name, tuple(float(v) for v in centre), tuple(float(v) for v in q), dims)


def container_boxes(container: ts.ContainerSpec, table_top: float) -> list[Box]:
    """The four walls of a tray/bin (pads and the table spot have none)."""
    if container.kind in ("pad", "table") or container.wall_height <= 0:
        return []
    out = []
    for i, (centre, half) in enumerate(ts.container_boxes(container)):
        if centre[2] - half[2] < 0.002 and half[2] < 0.004:  # skip the floor slab
            continue
        world = (container.xy[0] + centre[0], container.xy[1] + centre[1], table_top + centre[2])
        dims = tuple(2 * h for h in half)
        out.append(Box(f"{container.name} wall {i}", world, (1.0, 0.0, 0.0, 0.0), dims))
    return out


def scene_boxes(
    level: ts.Level,
    objects_xyz: dict[str, list[float]],
    orientations: dict[str, list[float]],
    table_top: float,
    exclude: set[str] = frozenset(),
) -> list[Box]:
    boxes = []
    for container in level.containers:
        boxes += container_boxes(container, table_top)
    for name, xyz in objects_xyz.items():
        if name in exclude:
            continue
        boxes.append(object_box(level.object(name), xyz, orientations.get(name, (1, 0, 0, 0))))
    return boxes


def to_planner_frame(box: Box, base_pose, frame_bias) -> list[float]:
    """[x, y, z, qw, qx, qy, qz] in the planner base frame (same transform as its targets)."""
    base_p = np.asarray(base_pose[:3], float)
    base_r = np.asarray(ts.quat_matrix(base_pose[3:]), float)
    p = base_r.T @ (np.asarray(box.centre, float) - base_p) + np.asarray(frame_bias, float)
    r = base_r.T @ np.asarray(ts.quat_matrix(box.quat), float)
    from twin_geometry import mat_to_quat

    return [*map(float, p), *map(float, mat_to_quat(r))]


def enlarge_collision_cache(obb: int = CACHE_OBB) -> str:
    """Give CuRobo room for scene obstacles (call before RoboTwin builds its planner)."""
    try:
        from curobo.wrap.reacher.motion_gen import MotionGenConfig
    except Exception as exc:  # no CuRobo in this environment
        return f"curobo unavailable ({type(exc).__name__})"
    original = MotionGenConfig.load_from_robot_config
    if getattr(original, "_twin_patched", False):
        return "already patched"
    try:
        params = inspect.signature(original).parameters
    except (TypeError, ValueError):
        return "cannot inspect MotionGenConfig.load_from_robot_config"
    if "collision_cache" not in params:
        return "this CuRobo has no collision_cache option"

    def patched(*args, **kwargs):
        if kwargs.get("collision_cache") is None:
            kwargs["collision_cache"] = {"obb": obb}
        return original(*args, **kwargs)

    patched._twin_patched = True  # type: ignore[attr-defined]
    MotionGenConfig.load_from_robot_config = patched
    return f"collision cache {obb} boxes"


def planner_of(task):
    """RoboTwin's in-process CuRobo planner for the left arm, or None."""
    robot = getattr(task, "robot", None)
    planner = getattr(robot, "left_planner", None)
    if planner is None or not hasattr(planner, "motion_gen"):
        return None
    return planner


def load_world(task, boxes: list[Box]) -> str:
    """Replace the planner world with table + ``boxes``; returns a status string."""
    planner = planner_of(task)
    if planner is None:
        return "no in-process CuRobo planner"
    try:
        from curobo.geom.types import Cuboid, WorldConfig
    except Exception as exc:
        return f"curobo unavailable ({type(exc).__name__})"
    base = planner.robot_origion_pose
    base_pose = [*base.p, *base.q]
    bias = list(getattr(planner, "frame_bias", [0.0, 0.0, 0.0]))
    cuboids = [
        Cuboid(
            name="table",
            pose=[float(base.p[1]), 0.0, 0.74 - float(base.p[2]), 1.0, 0.0, 0.0, 0.0],
            dims=[0.7, 2.0, 0.04],
        )
    ]
    for box in boxes[: CACHE_OBB - 1]:
        cuboids.append(
            Cuboid(
                name=box.name.replace(" ", "_"),
                pose=to_planner_frame(box, base_pose, bias),
                dims=list(box.dims),
            )
        )
    world = WorldConfig(cuboid=cuboids)
    try:
        planner.motion_gen.update_world(world)
        batch = getattr(planner, "motion_gen_batch", None)
        if batch is not None:
            batch.update_world(world)
    except Exception as exc:
        return f"update_world failed: {type(exc).__name__}: {exc}"
    return f"{len(cuboids) - 1} obstacle box(es)"


def describe_world(task, boxes: list[Box]) -> dict:
    """Base pose, frame bias and a few boxes in world and planner frames (for the log)."""
    planner = planner_of(task)
    if planner is None:
        return {}
    base = planner.robot_origion_pose
    base_pose = [float(v) for v in (*base.p, *base.q)]
    bias = [float(v) for v in getattr(planner, "frame_bias", [0.0, 0.0, 0.0])]
    sample = [
        {
            "name": b.name,
            "world": [round(v, 4) for v in b.centre],
            "planner": [round(v, 4) for v in to_planner_frame(b, base_pose, bias)],
            "dims": [round(v, 4) for v in b.dims],
        }
        for b in boxes[:6]
    ]
    return {"base_pose": base_pose, "frame_bias": bias, "boxes": sample}
