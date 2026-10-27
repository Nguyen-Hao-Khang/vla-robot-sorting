"""Fail-closed interior URDF margin gate for planned Panthera pregrasp.

The gate examines raw and retimed commands; it never changes a waypoint,
expands the URDF bounds, or excuses a measured runtime limit violation.
"""

from __future__ import annotations

import math
from typing import Any

PREGRASP_MINIMUM_MARGIN_RAD = 0.005


def audit_pregrasp_joint_reserve(
    path: Any,
    trajectory: Any,
    limits: Any,
    *,
    required_margin_rad: float = PREGRASP_MINIMUM_MARGIN_RAD,
) -> dict[str, Any]:
    """Check all arm joints at every raw and timed waypoint before physics."""
    if not math.isfinite(required_margin_rad) or required_margin_rad <= 0:
        raise ValueError("interior reserve must be positive and finite")
    names = tuple(limits.joint_names)
    if len(names) != 6 or names != tuple(f"joint{i}" for i in range(1, 7)):
        raise ValueError("unsupported pregrasp joint name/order")
    if tuple(path.joint_names) != names or tuple(trajectory.joint_names) != names:
        raise ValueError("planned joint order mismatch")
    raw = tuple(path.positions)
    timed = tuple(point.positions for point in trajectory.waypoints)
    if len(raw) < 2 or len(timed) < 2:
        raise ValueError("missing pregrasp trajectory waypoints")

    minimum = math.inf
    limiting_joint = None
    limiting_source = None
    rejected = 0
    for label, samples in (("raw", raw), ("retimed", timed)):
        for index, row in enumerate(samples):
            if len(row) != len(names):
                raise ValueError("planned waypoint joint count mismatch")
            for name, q, low, high in zip(
                names, row, limits.lower_position_rad, limits.upper_position_rad, strict=True
            ):
                if not math.isfinite(q):
                    raise ValueError("nonfinite planned joint command")
                margin = min(q - low, high - q)
                if margin < minimum:
                    minimum = margin
                    limiting_joint = name
                    limiting_source = f"{label}:{index}"
                rejected += int(margin < required_margin_rad)
    return {
        "schema": "s1m.pregrasp-interior-joint-reserve.v1",
        "passed": rejected == 0,
        "minimum_margin_rad": minimum,
        "required_margin_rad": required_margin_rad,
        "limiting_joint": limiting_joint,
        "limiting_waypoint": limiting_source,
        "violating_joint_samples": rejected,
        "raw_waypoint_count": len(raw),
        "retimed_waypoint_count": len(timed),
        "joint_bounds_unchanged": True,
        "command_waypoints_modified": False,
        "physical_execution_performed": False,
    }
