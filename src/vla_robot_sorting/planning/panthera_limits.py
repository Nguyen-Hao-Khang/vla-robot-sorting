"""Load Panthera joint limits from the exact runtime URDF.

Acceleration and jerk are not standard URDF joint-limit fields, so callers
must provide those values from an explicitly versioned planner/model config.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from math import isfinite
from pathlib import Path

from vla_robot_sorting.contracts.motion import JointLimits


def _expand_limit(
    value: float | Sequence[float],
    *,
    dof: int,
    label: str,
) -> tuple[float, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(float(item) for item in value)
    else:
        result = (float(value),) * dof
    if len(result) != dof or not all(isfinite(item) and item > 0 for item in result):
        raise ValueError(f"{label} must provide {dof} finite positive values")
    return result


def load_joint_limits_from_urdf(
    urdf_path: str | Path,
    joint_names: tuple[str, ...],
    *,
    max_acceleration_rad_s2: float | Sequence[float],
    max_jerk_rad_s3: float | Sequence[float],
) -> JointLimits:
    """Return exact position/velocity limits plus explicit derivative limits."""

    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("joint_names must be non-empty and unique")

    root = ET.parse(Path(urdf_path)).getroot()
    joints = {joint.attrib.get("name"): joint for joint in root.findall("joint")}

    lower: list[float] = []
    upper: list[float] = []
    velocity: list[float] = []

    for joint_name in joint_names:
        joint = joints.get(joint_name)
        if joint is None:
            raise ValueError(f"URDF joint missing: {joint_name}")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"URDF joint {joint_name} has no <limit> element")


        try:
            low = float(limit.attrib["lower"])
            high = float(limit.attrib["upper"])
            max_velocity = float(limit.attrib["velocity"])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"URDF joint {joint_name} must define numeric lower/upper/velocity limits"
            ) from exc

        if not all(isfinite(item) for item in (low, high, max_velocity)):
            raise ValueError(f"URDF joint {joint_name} has non-finite limits")
        if low >= high or max_velocity <= 0:
            raise ValueError(f"URDF joint {joint_name} has invalid limits")

        lower.append(low)
        upper.append(high)
        velocity.append(max_velocity)

    acceleration = _expand_limit(
        max_acceleration_rad_s2,
        dof=len(joint_names),
        label="max_acceleration_rad_s2",
    )
    jerk = _expand_limit(
        max_jerk_rad_s3,
        dof=len(joint_names),
        label="max_jerk_rad_s3",
    )


    return JointLimits(
        joint_names=joint_names,
        lower_position_rad=tuple(lower),
        upper_position_rad=tuple(upper),
        max_velocity_rad_s=tuple(velocity),
        max_acceleration_rad_s2=acceleration,
        max_jerk_rad_s3=jerk,
    )
