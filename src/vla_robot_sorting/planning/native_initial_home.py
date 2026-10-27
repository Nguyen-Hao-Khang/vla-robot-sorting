"""Pinned pre-settling Panthera home target and exact measured startup gate.

The setting changes RoboTwin's drive target before setup's physics steps.
It never clips a measured qpos, changes the URDF, or authorizes motion.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from vla_robot_sorting.planning.manipulation_recovery import (
    CANONICAL_LOWER_RAD,
    CANONICAL_UPPER_RAD,
    CANONICAL_URDF_SHA256,
    RecoveryContractError,
)

ARM_NAMES = tuple(f"joint{i}" for i in range(1, 7))
FULL_NAMES = (*ARM_NAMES, "gripper_L_joint", "gripper_R_joint")
JOINT2_INTERIOR_HOME_RAD = (CANONICAL_UPPER_RAD[1] - CANONICAL_LOWER_RAD[1]) / 64
JOINT3_INTERIOR_HOME_RAD = (CANONICAL_UPPER_RAD[2] - CANONICAL_LOWER_RAD[2]) / 64
MIN_MEASURED_JOINT2_MARGIN_RAD = JOINT2_INTERIOR_HOME_RAD / 10
MIN_MEASURED_JOINT3_MARGIN_RAD = JOINT3_INTERIOR_HOME_RAD / 10


def _checked_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise RecoveryContractError("missing native Panthera embodiment config")
    if config.get("urdf_path") != "./panthera_6dof.urdf":
        raise RecoveryContractError("unexpected native Panthera URDF")
    if config.get("arm_joints_name") != [list(ARM_NAMES), list(ARM_NAMES)]:
        raise RecoveryContractError("unexpected Panthera arm joint order")
    home = config.get("homestate")
    if (
        not isinstance(home, list)
        or len(home) != 2
        or any(not isinstance(row, list) or row != [0.0] * 6 for row in home)
    ):
        raise RecoveryContractError("unexpected canonical Panthera zero home")
    return deepcopy(config)


def prepare_native_initial_home(task_args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return copied setup args with joints 2/3 driven inside their exact limits."""
    if (
        not isinstance(task_args, dict)
        or task_args.get("task_name") != "move_pillbottle_pad"
        or task_args.get("arm_mode") != "single"
        or task_args.get("robot_type") != "panthera-6dof"
        or CANONICAL_LOWER_RAD[1] != 0.0
        or CANONICAL_UPPER_RAD[1] != 3.2
        or CANONICAL_LOWER_RAD[2] != 0.0
        or CANONICAL_UPPER_RAD[2] != 4.0
    ):
        raise RecoveryContractError("unsupported native initial-home task or limits")
    prepared = task_args.copy()
    for side in ("left", "right"):
        key = f"{side}_embodiment_config"
        config = _checked_config(task_args.get(key))
        for row in config["homestate"]:
            row[1] = JOINT2_INTERIOR_HOME_RAD
            row[2] = JOINT3_INTERIOR_HOME_RAD
        prepared[key] = config
    policy = {
        "schema": "s1m.native-initial-home.v2",
        "method": "pre_setup_drive_target",
        "joint2_original_target_rad": 0.0,
        "joint2_configured_target_rad": JOINT2_INTERIOR_HOME_RAD,
        "joint2_lower_rad": CANONICAL_LOWER_RAD[1],
        "joint2_upper_rad": CANONICAL_UPPER_RAD[1],
        "joint3_original_target_rad": 0.0,
        "joint3_configured_target_rad": JOINT3_INTERIOR_HOME_RAD,
        "joint3_lower_rad": CANONICAL_LOWER_RAD[2],
        "joint3_upper_rad": CANONICAL_UPPER_RAD[2],
        "minimum_measured_joint2_margin_rad": MIN_MEASURED_JOINT2_MARGIN_RAD,
        "minimum_measured_joint3_margin_rad": MIN_MEASURED_JOINT3_MARGIN_RAD,
        "urdf_sha256": CANONICAL_URDF_SHA256,
        "simulator_state_modified_after_setup": False,
        "controller_eligible": False,
        "native_acceptance": False,
    }
    return prepared, policy


def audit_native_initial_home(entity: Any) -> dict[str, Any]:
    """Read exact live articulation state after settling; never modify it."""
    try:
        names = tuple(joint.get_name() for joint in entity.get_active_joints())
        raw = list(entity.get_qpos())
    except (AttributeError, TypeError, ValueError) as exc:
        raise RecoveryContractError("missing native articulation state") from exc
    if names != FULL_NAMES or len(raw) != len(FULL_NAMES):
        raise RecoveryContractError("native articulation joint ordering changed")
    if any(isinstance(v, bool) for v in raw):
        raise RecoveryContractError("invalid native articulation qpos")
    try:
        q = [float(v) for v in raw]
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecoveryContractError("invalid native articulation qpos") from exc
    if not all(math.isfinite(v) for v in q):
        raise RecoveryContractError("nonfinite native articulation qpos")
    margins = [
        min(value - lo, hi - value)
        for value, lo, hi in zip(q[:6], CANONICAL_LOWER_RAD, CANONICAL_UPPER_RAD, strict=True)
    ]
    exact_bounds = all(margin >= 0 for margin in margins)
    joint2_reserve = q[1] - CANONICAL_LOWER_RAD[1]
    joint3_reserve = q[2] - CANONICAL_LOWER_RAD[2]
    joint2_passed = joint2_reserve >= MIN_MEASURED_JOINT2_MARGIN_RAD
    joint3_passed = joint3_reserve >= MIN_MEASURED_JOINT3_MARGIN_RAD
    return {
        "schema": "s1m.native-initial-state.v2",
        "full_joint_names": list(names),
        "measured_full_qpos": q,
        "arm_joint_margins_rad": margins,
        "exact_urdf_bounds_passed": exact_bounds,
        "joint2_margin_above_lower_rad": joint2_reserve,
        "joint2_reserve_passed": joint2_passed,
        "joint3_margin_above_lower_rad": joint3_reserve,
        "joint3_reserve_passed": joint3_passed,
        "passed": exact_bounds and joint2_passed and joint3_passed,
        "urdf_sha256": CANONICAL_URDF_SHA256,
        "state_modified": False,
        "controller_eligible": False,
        "native_acceptance": False,
    }


def initialize_native_task_with_home(
    task: Any, task_args: dict[str, Any], *, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Install the interior drive target before upstream setup settles physics."""
    if type(seed) is not int or seed not in (0, 1, 2):
        raise RecoveryContractError("unsupported canonical seed")
    prepared, policy = prepare_native_initial_home(task_args)
    task.setup_demo(now_ep_num=0, seed=seed, **prepared)
    measured = audit_native_initial_home(task.robot.left_entity)
    return prepared, {"policy": policy, "measured": measured}
