"""Evidence-bound placement lookahead for the near-limit carried-object wrist.

Only ranks *pre-screened* candidate plans. It cannot authorize physical motion.
"""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from typing import Any

NEAR_JOINT5_LIMIT_RAD = 0.10
TARGET_JOINT5_RESERVE_RAD = 0.125
REQUIRED_GAIN_RAD = 0.10
MINIMUM_INTERIOR_MARGIN_RAD = 0.005


def joint5_margin(positions: tuple[float, ...], limits: Any) -> float:
    if tuple(limits.joint_names) != tuple(f"joint{i}" for i in range(1, 7)):
        raise ValueError("invalid Panthera joint order")
    if len(positions) != 6 or not all(isfinite(float(q)) for q in positions):
        raise ValueError("invalid terminal joint vector")
    return min(
        float(positions[4]) - float(limits.lower_position_rad[4]),
        float(limits.upper_position_rad[4]) - float(positions[4]),
    )


def fingerprint_joint_path(path: Any) -> str:
    if tuple(path.joint_names) != tuple(f"joint{i}" for i in range(1, 7)):
        raise ValueError("invalid path joint order")
    rows = [[float(v) for v in sample] for sample in path.positions]
    if len(rows) < 2 or any(len(row) != 6 or not all(isfinite(v) for v in row) for row in rows):
        raise ValueError("invalid planned waypoints")
    canonical = json.dumps(
        {"joint_names": list(path.joint_names), "positions": rows},
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def choose_safe_yaw(
    records: list[dict[str, Any]],
    *,
    baseline_margin_rad: float,
) -> dict[str, Any] | None:
    """Prefer the best joint5 reserve only after all full preflight vetoes pass."""
    if not isfinite(baseline_margin_rad) or baseline_margin_rad < 0:
        raise ValueError("invalid baseline joint5 margin")
    if baseline_margin_rad >= NEAR_JOINT5_LIMIT_RAD:
        return None
    eligible = [
        row
        for row in records
        if row.get("yaw_offset_rad") not in (None, 0.0)
        and row.get("planner_success") is True
        and row.get("dynamic_safe") is True
        and row.get("raw_self_collision_free") is True
        and row.get("timed_self_collision_free") is True
        and row.get("joint_interior_safe") is True
        and row.get("eligible") is True
        and isinstance(row.get("path_sha256"), str)
        and len(row["path_sha256"]) == 64
        and isinstance(row.get("joint5_margin_rad"), (int, float))
        and isfinite(float(row["joint5_margin_rad"]))
        and row["joint5_margin_rad"] >= TARGET_JOINT5_RESERVE_RAD
        and row["joint5_margin_rad"] - baseline_margin_rad >= REQUIRED_GAIN_RAD
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda row: float(row["joint5_margin_rad"]))
