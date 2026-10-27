"""Configure pinned native CuRobo search before robot setup, without upstream edits.

Project owns this context; the vendor YAML/URDF and CuRobo installed source
remain immutable. CuRobo natively implements cspace.position_limit_clip.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

PINNED_CUROBO_YAML_LF_SHA256 = "5e73d20566ca49fe818f1cf759baa08c13eb118881a6ab807f63dc86039f0bc7"
CUROBO_POSITION_LIMIT_CLIP_RAD = 0.0075
INTERIOR_RETRACT_CONFIG_RAD = (0.0, 0.05, 0.0625, 0.0, 0.0, 0.0)
PRIMARY_IK_SEEDS = 64
PRIMARY_TRAJOPT_SEEDS = 2


def verify_pinned_curobo_yaml(path: Path) -> None:
    raw = path.read_bytes()
    if b"\r" in raw.replace(b"\r\n", b""):
        raise RuntimeError("unexpected pinned CuRobo YAML line ending")
    normalized = raw.replace(b"\r\n", b"\n")
    if hashlib.sha256(normalized).hexdigest() != PINNED_CUROBO_YAML_LF_SHA256:
        raise RuntimeError("pinned CuRobo YAML sha256 mismatch")


@contextmanager
def native_curobo_interior_setup(config_path: Path) -> Iterator[dict[str, Any]]:
    """Temporarily patch the *in-memory loader* while upstream builds MotionGen.

    The loader delegates to the original exact pinned YAML reader and modifies
    only its returned dictionary. The factory changes only search width for
    vendor's primary single-arm planner; batch planner budget is not enlarged.
    Both hooks are reverted even if upstream setup raises.
    """
    import curobo.wrap.reacher.motion_gen as motion_gen

    config_path = config_path.resolve(strict=True)
    verify_pinned_curobo_yaml(config_path)
    original_loader = motion_gen.load_yaml
    original_factory = motion_gen.MotionGenConfig.load_from_robot_config
    provenance: dict[str, Any] = {
        "schema": "s1m.native-curobo-interior-planner.v1",
        "method": "scoped_in_memory_native_config_loader",
        "pinned_yaml_sha256": PINNED_CUROBO_YAML_LF_SHA256,
        "position_limit_clip_rad": CUROBO_POSITION_LIMIT_CLIP_RAD,
        "interior_retract_config_rad": list(INTERIOR_RETRACT_CONFIG_RAD),
        "original_urdf_and_vendor_yaml_modified": False,
        "factory_calls": 0,
        "config_intercepts": 0,
        "single_arm_search_ik_seeds": PRIMARY_IK_SEEDS,
        "single_arm_search_trajopt_seeds": PRIMARY_TRAJOPT_SEEDS,
        "runtime_joint_limits_waived": False,
        "controller_eligible": False,
    }

    def guarded_yaml_loader(path: Any) -> Any:
        document = original_loader(path)
        if Path(path).resolve() != config_path:
            return document
        if not isinstance(document, dict):
            raise RuntimeError("unexpected CuRobo YAML object")
        clone = deepcopy(document)
        cspace = clone["robot_cfg"]["kinematics"]["cspace"]
        names = cspace.get("joint_names")
        if names != [f"joint{i}" for i in range(1, 7)]:
            raise RuntimeError("unexpected pinned CuRobo arm joint order")
        if cspace.get("position_limit_clip", 0.0) != 0.0:
            raise RuntimeError("pinned CuRobo YAML joint clip was modified")
        if cspace.get("retract_config") != [0.0] * 6:
            raise RuntimeError("pinned CuRobo retract configuration changed")
        cspace["position_limit_clip"] = CUROBO_POSITION_LIMIT_CLIP_RAD
        cspace["retract_config"] = list(INTERIOR_RETRACT_CONFIG_RAD)
        provenance["config_intercepts"] += 1
        return clone

    def bounded_factory(robot_cfg: Any, *args: Any, **kwargs: Any) -> Any:
        new_kwargs = dict(kwargs)
        provenance["factory_calls"] += 1
        if "num_graph_seeds" not in kwargs:
            # Vendor's primary single-plan MotionGen. Leave batch memory alone.
            if kwargs.get("num_trajopt_seeds") != 1:
                raise RuntimeError("vendor primary trajopt seeds changed")
            new_kwargs["num_trajopt_seeds"] = PRIMARY_TRAJOPT_SEEDS
            new_kwargs["num_ik_seeds"] = PRIMARY_IK_SEEDS
        return original_factory(robot_cfg, *args, **new_kwargs)

    motion_gen.load_yaml = guarded_yaml_loader
    motion_gen.MotionGenConfig.load_from_robot_config = staticmethod(bounded_factory)
    try:
        yield provenance
        if provenance["config_intercepts"] != 2 or provenance["factory_calls"] != 2:
            raise RuntimeError(
                "native CuRobo config injection did not reach both MotionGen instances"
            )
    finally:
        motion_gen.load_yaml = original_loader
        motion_gen.MotionGenConfig.load_from_robot_config = staticmethod(original_factory)


@contextmanager
def scoped_native_curobo_search(robot: Any) -> Iterator[dict[str, Any]]:
    """Bound primary vendor MotionGenPlanConfig attempts; preserve vendor code."""
    import sys

    attempts = 24
    planner = getattr(robot, "left_planner", None)
    if planner is None or type(planner).__module__ != "envs.robot.planner":
        raise RuntimeError("unexpected native CuRobo planner implementation")
    module = sys.modules.get(type(planner).__module__)
    if module is None or getattr(module, "MotionGenPlanConfig", None) is None:
        raise RuntimeError("pinned CuRobo plan config unavailable")
    original = module.MotionGenPlanConfig
    policy: dict[str, Any] = {
        "strategy": "scoped_native_curobo_search_v1",
        "max_attempts": attempts,
        "factory_invocations": 0,
        "vendor_original_restored": False,
    }

    def bounded_plan_config(*args: Any, **kwargs: Any) -> Any:
        if args or kwargs.get("max_attempts") != 10:
            raise RuntimeError("unexpected pinned CuRobo query configuration")
        policy["factory_invocations"] += 1
        options = dict(kwargs)
        options["max_attempts"] = attempts
        return original(**options)

    module.MotionGenPlanConfig = bounded_plan_config
    try:
        yield policy
    finally:
        module.MotionGenPlanConfig = original
        policy["vendor_original_restored"] = True
