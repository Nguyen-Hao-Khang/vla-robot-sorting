"""Deterministic, bounded outward post-release goals; never physics certification."""

from __future__ import annotations

from math import cos, hypot, isclose, isfinite, radians, sin, sqrt
from typing import Any

from vla_robot_sorting.contracts import Pose6D

# Original bounded candidates are preserved in exactly their historical order.
DIRECTIONS: tuple[tuple[str, float, float, bool], ...] = (
    ("radial_up_60", 0.0, 60.0, True),
    ("radial_up_45", 0.0, 45.0, True),
    ("radial_up_75", 0.0, 75.0, True),
    ("radial_up_30", 0.0, 30.0, True),
    ("radial_yaw_pos_30", 30.0, 0.0, True),
    ("radial_yaw_neg_30", -30.0, 0.0, True),
    # 40 mm vertical caused forbidden physical target contact in older evidence.
    ("vertical_up_separation_only", 0.0, 90.0, False),
)

# Release has completed, so task does not constrain gripper wrist orientation.
# Introduce bounded local-Y rotations only for the failing 40mm retreat.
# All paths still require full joint, FCL, dynamic, PhysX contact safety.
RETREAT_WRIST_CHOICES: tuple[tuple[str, float, float], ...] = (
    ("radial_pitch_neg_20", 0.0, -20.0),
    ("radial_pitch_pos_20", 0.0, 20.0),
    ("radial_up45_pitch_neg_20", 45.0, -20.0),
    ("radial_up45_pitch_pos_20", 45.0, 20.0),
    ("radial_pitch_neg_10", 0.0, -10.0),
    ("radial_pitch_pos_10", 0.0, 10.0),
    ("radial_up45_pitch_neg_10", 45.0, -10.0),
    ("radial_up45_pitch_pos_10", 45.0, 10.0),
    ("radial_pitch_neg_30", 0.0, -30.0),
    ("radial_pitch_pos_30", 0.0, 30.0),
    ("radial_up60_pitch_neg_20", 60.0, -20.0),
    ("radial_up60_pitch_pos_20", 60.0, 20.0),
)


def local_y_rotated_quaternion(
    original_xyzw: tuple[float, float, float, float], pitch_deg: float
) -> tuple[float, float, float, float]:
    """Right-compose a bounded local gripper pitch, without changing frames."""
    q = tuple(float(v) for v in original_xyzw)
    if len(q) != 4 or not all(isfinite(v) for v in q):
        raise ValueError("nonfinite gripper orientation")
    if not isclose(sum(v * v for v in q), 1.0, abs_tol=1e-4):
        raise ValueError("nonunit gripper orientation")
    if not isfinite(pitch_deg) or abs(pitch_deg) > 30.0:
        raise ValueError("postrelease wrist pitch exceeds 30 degrees")
    x, y, z, w = q
    h = radians(pitch_deg) / 2.0
    c, s = cos(h), sin(h)
    return (
        c * x - s * z,
        c * y + s * w,
        c * z + s * x,
        c * w - s * y,
    )


def alternative_goals(
    gripper: Pose6D,
    target: Pose6D,
    *,
    distance_m: float,
    phase: str,
) -> tuple[tuple[str, Pose6D, dict[str, Any]], ...]:
    if phase not in ("separation", "retreat"):
        raise ValueError("unsupported release motion phase")
    expected = 0.01 if phase == "separation" else 0.04
    if not isfinite(distance_m) or not isclose(distance_m, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("unexpected bounded postrelease motion length")
    if gripper.frame_id != target.frame_id:
        raise ValueError("postrelease reference frame mismatch")
    start = tuple(float(x) for x in gripper.position_xyz)
    obj = tuple(float(x) for x in target.position_xyz)
    if not all(isfinite(x) for x in (*start, *obj)):
        raise ValueError("nonfinite postrelease geometry")
    dx, dy = start[0] - obj[0], start[1] - obj[1]
    radius = hypot(dx, dy)
    if radius < 0.005:
        return ()
    ux, uy = dx / radius, dy / radius

    specs: list[tuple[str, float, float, float]] = [
        (label, yaw_deg, elevation_deg, 0.0)
        for label, yaw_deg, elevation_deg, retreat_allowed in DIRECTIONS
        if phase == "separation" or retreat_allowed
    ]
    if phase == "retreat":
        specs.extend(
            (label, 0.0, elevation_deg, pitch_deg)
            for label, elevation_deg, pitch_deg in RETREAT_WRIST_CHOICES
        )
    options = []
    for label, yaw_deg, elevation_deg, pitch_deg in specs:
        yaw, elevation = radians(yaw_deg), radians(elevation_deg)
        projection = cos(elevation)
        direction = (
            projection * (ux * cos(yaw) - uy * sin(yaw)),
            projection * (ux * sin(yaw) + uy * cos(yaw)),
            sin(elevation),
        )
        outward = direction[0] * ux + direction[1] * uy
        if outward < -1e-12 or direction[2] < -1e-12:
            raise ValueError("alternative direction is inward/downward")
        if not isclose(sqrt(sum(x * x for x in direction)), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("nonunit alternative direction")
        goal_quat = local_y_rotated_quaternion(gripper.orientation_xyzw, pitch_deg)
        goal = Pose6D(
            position_xyz=tuple(start[i] + distance_m * direction[i] for i in range(3)),
            orientation_xyzw=goal_quat,
            frame_id=gripper.frame_id,
        )
        geometry = {
            "strategy": "bounded_outward_direction_search_v1",
            "direction_label": label,
            "direction_xyz_unit": list(direction),
            "initial_robot_object_xy_distance_m": radius,
            "reference_object_xyz_m": list(obj),
            "start_gripper_xyz_m": list(start),
            "start_gripper_orientation_xyzw": list(gripper.orientation_xyzw),
            "goal_gripper_orientation_xyzw": list(goal_quat),
            "wrist_local_y_pitch_deg": pitch_deg,
        }
        options.append((label, goal, geometry))
    return tuple(options)
