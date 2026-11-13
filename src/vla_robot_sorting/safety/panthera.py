"""Panthera-HT forward kinematics in the robot base frame, pure Python.

The joint chain is copied from ``panthera_6dof.urdf`` (Panthera-HT_RoboTwin commit 2aee31ad,
all joint origins with rpy = 0), the same chain as ``tools/twin/twin_kinematics.py``. It is
kept dependency-free so the safety layer can run on the robot computer without NumPy.
The base frame has z up with z = 0 at the mounting surface (the table top).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vec = tuple[float, float, float]
Mat = tuple[Vec, Vec, Vec]

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
# (origin xyz in the parent link, joint axis in the child link) for joint1..joint6
JOINTS: tuple[tuple[Vec, Vec], ...] = (
    ((0.0, 0.0, 0.0584), (0.0, 0.0, 1.0)),
    ((0.018199, 0.0, 0.053), (0.0, 1.0, 0.0)),
    ((-0.26, 0.0, 0.0), (0.0, -1.0, 0.0)),
    ((0.23, 0.0, 0.06), (0.0, -1.0, 0.0)),
    ((0.07, 0.0, 0.036319), (0.0, 0.0, -1.0)),
    ((0.02345, 0.0, -0.039), (1.0, 0.0, 0.0)),
)
LIMITS = ((-2.4, 2.4), (0.0, 3.2), (0.0, 4.0), (-1.6, 1.6), (-1.7, 1.7), (-2.5, 2.5))
TOOL: Vec = (0.165, 0.0, 0.0)  # tool_link (between the fingertips) in link6

_IDENTITY: Mat = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _matmul(a: Mat, b: Mat) -> Mat:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def _apply(r: Mat, v: Vec) -> Vec:
    return tuple(sum(r[i][k] * v[k] for k in range(3)) for i in range(3))  # type: ignore[return-value]


def _rotation(axis: Vec, angle: float) -> Mat:
    """Rodrigues rotation about a unit axis."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return (
        (c + x * x * t, x * y * t - z * s, x * z * t + y * s),
        (y * x * t + z * s, c + y * y * t, y * z * t - x * s),
        (z * x * t - y * s, z * y * t + x * s, c + z * z * t),
    )


def link_points(q: Sequence[float]) -> list[Vec]:
    """Joint origins 1-6 followed by the tool point, in the base frame."""
    if len(q) < len(JOINTS):
        raise ValueError(f"expected {len(JOINTS)} joint values, got {len(q)}")
    r, p = _IDENTITY, (0.0, 0.0, 0.0)
    points: list[Vec] = []
    for (offset, axis), angle in zip(JOINTS, q, strict=False):
        d = _apply(r, offset)
        p = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
        points.append(p)
        r = _matmul(r, _rotation(axis, float(angle)))
    d = _apply(r, TOOL)
    points.append((p[0] + d[0], p[1] + d[1], p[2] + d[2]))
    return points


def tool_position(q: Sequence[float]) -> Vec:
    return link_points(q)[-1]
