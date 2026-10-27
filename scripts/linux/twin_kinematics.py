"""Panthera-HT 6-DoF kinematics for the digital twin (pure numpy, unit-tested).

The joint chain below is copied from ``panthera_6dof.urdf`` (Panthera-HT_RoboTwin commit
2aee31ad, embodiment ``panthera-6dof``: all joint origins have rpy = 0). RoboTwin places the
base at world (0, -0.45, 0.75) with yaw +90 deg and plans the ``tool_link`` pose (gripper x
= approach, fingers along gripper y), the same pose convention as ``Base_Task.get_grasp_pose``.

Used only to rank grasp candidates before any motion: a candidate whose pre-grasp, grasp,
lift, transport and release poses all have an inverse-kinematics solution within the joint
limits is tried first. CuRobo still plans and checks collisions for every real motion, so a
small model error can only change the order in which candidates are tried, never skip one.

Finding that drives the grasp strategy (offline scan with this model): a gripper pointing
straight down is reachable only up to about 0.86-0.92 m (table top 0.74 m) and only near
the robot, so it can grasp a box but cannot lift it 15 cm; tilting the approach by 35-45 deg
(still in a vertical plane, fingers horizontal) reaches 1.07 m everywhere on the work area.
"""

from __future__ import annotations

import math

import numpy as np

# (origin xyz in the parent link, joint axis in the child link) for joint1..joint6
JOINTS = (
    ((0.0, 0.0, 0.0584), (0.0, 0.0, 1.0)),
    ((0.018199, 0.0, 0.053), (0.0, 1.0, 0.0)),
    ((-0.26, 0.0, 0.0), (0.0, -1.0, 0.0)),
    ((0.23, 0.0, 0.06), (0.0, -1.0, 0.0)),
    ((0.07, 0.0, 0.036319), (0.0, 0.0, -1.0)),
    ((0.02345, 0.0, -0.039), (1.0, 0.0, 0.0)),
)
LIMITS = ((-2.4, 2.4), (0.0, 3.2), (0.0, 4.0), (-1.6, 1.6), (-1.7, 1.7), (-2.5, 2.5))
TOOL = (0.165, 0.0, 0.0)  # tool_link in link6
BASE_XYZ = (0.0, -0.45, 0.75)
BASE_YAW = math.pi / 2
LIMIT_MARGIN = 0.05  # rad kept free at every joint limit (tracking error, planner margin)
POS_TOL = 0.002  # m
ROT_TOL = 0.02  # rad
_LO = np.array([lo for lo, _ in LIMITS]) + LIMIT_MARGIN
_HI = np.array([hi for _, hi in LIMITS]) - LIMIT_MARGIN
# Seeds that cover the elbow-up / wrist-down postures this arm uses above the table.
SEEDS = (
    (0.0, 1.2, 1.6, 0.4, 0.0, 0.0),
    (0.0, 1.6, 2.2, 0.8, 0.0, 0.0),
    (0.0, 0.8, 1.0, -0.2, 0.0, 0.0),
    (0.0, 2.0, 2.8, 1.2, 0.0, 0.0),
    (0.0, 1.4, 2.0, -0.6, 0.0, 0.0),
)


def _rot(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


_BASE_R = _rot((0.0, 0.0, 1.0), BASE_YAW)
_BASE_P = np.array(BASE_XYZ)


def fk(q) -> tuple[np.ndarray, np.ndarray, list, list]:
    """tool_link (position, rotation) in the world, plus joint positions and world axes."""
    r, p = _BASE_R.copy(), _BASE_P.copy()
    origins, axes = [], []
    for (offset, axis), angle in zip(JOINTS, q, strict=True):
        p = p + r @ np.asarray(offset)
        origins.append(p.copy())
        axes.append(r @ np.asarray(axis))
        r = r @ _rot(axis, float(angle))
    return p + r @ np.asarray(TOOL), r, origins, axes


def _error(q, target_p, target_r) -> tuple[np.ndarray, np.ndarray, list, list]:
    p, r, origins, axes = fk(q)
    rot_err = 0.5 * sum(np.cross(r[:, i], target_r[:, i]) for i in range(3))
    return np.concatenate([target_p - p, rot_err]), p, origins, axes


def ik(target_p, target_r, seed=None, iters: int = 60) -> np.ndarray | None:
    """Joint angles reaching (target_p, target_r) within the limits, or None."""
    target_p = np.asarray(target_p, float)
    target_r = np.asarray(target_r, float)
    seeds = ([np.asarray(seed, float)] if seed is not None else []) + [
        np.asarray(s, float) for s in SEEDS
    ]
    for q in seeds:
        q = np.clip(q.copy(), _LO, _HI)
        checkpoint = math.inf
        for it in range(iters):
            err, p, origins, axes = _error(q, target_p, target_r)
            if np.linalg.norm(err[:3]) < POS_TOL and np.linalg.norm(err[3:]) < ROT_TOL:
                r = fk(q)[1]
                if np.trace(r.T @ target_r) > 1.0 + 2.0 * math.cos(2 * ROT_TOL):
                    return q
                break  # converged to the half-turn flip: not this orientation
            size = float(np.linalg.norm(err[:3]) + 0.3 * np.linalg.norm(err[3:]))
            if it % 10 == 0:
                if size > 0.7 * checkpoint:
                    break  # stalled (usually pressed against a joint limit): next seed
                checkpoint = size
            jac = np.zeros((6, 6))
            for i, (o, a) in enumerate(zip(origins, axes, strict=True)):
                jac[:3, i] = np.cross(a, p - o)
                jac[3:, i] = a
            w = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])  # metres vs radians
            jw = jac * w[:, None]
            step = jw.T @ np.linalg.solve(jw @ jw.T + 1e-4 * np.eye(6), err * w)
            q = np.clip(q + np.clip(step, -0.25, 0.25), _LO, _HI)
    return None


def quat_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def reachable_chain(poses) -> bool:
    """True if every pose [x, y, z, qw, qx, qy, qz] has an IK solution (each one seeded
    with the previous solution, as the arm moves through them in order)."""
    seed = None
    for pose in poses:
        q = ik(pose[:3], quat_matrix(pose[3:7]), seed)
        if q is None:
            return False
        seed = q
    return True
