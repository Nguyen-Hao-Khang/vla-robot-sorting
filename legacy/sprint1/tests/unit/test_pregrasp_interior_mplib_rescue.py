"""Bounded independent pregrasp rescue without PhysX, clipping, or safety waivers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from vla_robot_sorting.contracts import MotionGoal, Pose6D
from vla_robot_sorting.contracts.motion import (
    JointLimits,
    JointPath,
    JointState,
    MotionFailure,
    MotionFailureCode,
    PlannerResult,
    PlanningContext,
)

ROOT = Path(__file__).resolve().parents[2]
NAMES = tuple(f"joint{i}" for i in range(1, 7))


def _module(name: str):
    script = ROOT / "scripts/linux" / name
    spec = importlib.util.spec_from_file_location("test_native_" + name[:-3], script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture(joint5: float, *, planner_id="robotwin-curobo"):
    start = (0.0, 0.05, 0.062, 0.0, 0.0, 0.0)
    end = (0.14, 2.19, 0.88, 1.295, joint5, 0.01)
    path = JointPath(
        joint_names=NAMES,
        positions=(start, end),
        source=planner_id,
        metadata={
            "scene_revision": "rev",
            "transform_revision": "t",
            "target_pose_robotwin_wxyz": (0.12, 0.1, 0.9, 1.0, 0.0, 0.0, 0.0),
        },
    )
    return PlannerResult(path=path, failure=None, planning_time_s=0.02, planner_id=planner_id)


def _limits():
    return JointLimits(
        joint_names=NAMES,
        lower_position_rad=(-2.4, 0, 0, -1.6, -1.7, -2.5),
        upper_position_rad=(2.4, 3.2, 4, 1.6, 1.7, 2.5),
        max_velocity_rad_s=(2,) * 6,
        max_acceleration_rad_s2=(3,) * 6,
        max_jerk_rad_s3=(4,) * 6,
    )


def _call(primary, alternate):
    helpers = _module("probe_panthera_grasp_recovery.py")
    original = {
        "primary_planner_id": "robotwin-curobo",
        "primary_success": True,
        "fallback_attempted": False,
        "fallback_planner_id": None,
        "fallback_success": None,
        "selected_planner_id": "robotwin-curobo",
        "plan_success": True,
        "planning_time_s": 0.02,
    }
    calls = []

    def primary_fn(*args, **kwargs):
        calls.append("curobo")
        return primary, original

    class Fallback:
        def plan(self, *args, **kwargs):
            calls.append("mplib")
            return alternate

    class Timer:
        def parameterize(self, path, limits):
            del limits
            return SimpleNamespace(
                joint_names=path.joint_names,
                waypoints=tuple(SimpleNamespace(positions=q) for q in path.positions),
            )

    plan, policy = helpers.plan_pregrasp_with_interior_rescue(
        SimpleNamespace(robot="stub"),
        JointState(NAMES, primary.path.positions[0]),
        object(),
        PlanningContext("rev", "t"),
        scene="scene",
        limits=_limits(),
        timer=Timer(),
        primary_fallback=primary_fn,
        mplib_planner=Fallback(),
    )
    return plan, policy, calls


def test_mplib_rescue_accepts_only_interior_trajectory():
    primary = _fixture(1.6999998092651367)
    alt = _fixture(1.398, planner_id="robotwin-mplib-rrt")
    selected, policy, calls = _call(primary, alt)
    assert calls == ["curobo", "mplib"]
    assert selected is alt
    assert policy["strategy"] == "pregrasp_interior_mplib_rescue_v1"
    assert policy["primary_success"] is True  # vendor success faithfully preserved
    assert policy["primary_reserve"]["passed"] is False
    assert policy["interior_rescue"]["fallback_reserve"]["passed"] is True
    from vla_robot_sorting.evidence.manipulation import _planner_policy_ok

    assert _planner_policy_ok(policy)


def test_successful_curobo_interior_never_invokes_rescue():
    primary = _fixture(1.398)
    _, policy, calls = _call(primary, _fixture(1.39, planner_id="robotwin-mplib-rrt"))
    assert calls == ["curobo"]
    assert policy["fallback_attempted"] is False


def test_mplib_also_at_boundary_is_rejected_without_waiver():
    primary = _fixture(1.6999998092651367)
    alt = _fixture(1.6999998092651367, planner_id="robotwin-mplib-rrt")
    chosen, policy, calls = _call(primary, alt)
    assert calls == ["curobo", "mplib"]
    assert chosen is primary
    assert policy["interior_rescue"]["accepted"] is False
    assert policy["interior_rescue"]["fallback_reserve"]["passed"] is False


def test_missing_or_failed_mplib_rescue_preserves_original_unsafe_path():
    primary = _fixture(1.6999998092651367)
    failed = PlannerResult(
        path=None,
        failure=MotionFailure(
            MotionFailureCode.OPTIMIZATION_FAILED,
            message="IK unavailable",
            recoverable=True,
        ),
        planning_time_s=0.01,
        planner_id="robotwin-mplib-rrt",
    )
    chosen, policy, calls = _call(primary, failed)
    assert calls == ["curobo", "mplib"]
    assert chosen is primary
    assert policy["interior_rescue"]["accepted"] is False
    assert policy["interior_rescue"]["fallback_native_success"] is False


def test_prepared_pregrasp_checks_start_goal_and_revision():
    module = _module("probe_panthera_project_manipulation.py")
    valid = _fixture(1.398)
    policy = {"primary_planner_id": "robotwin-curobo"}
    goal = MotionGoal(
        Pose6D(
            position_xyz=(0.12, 0.1, 0.9),
            orientation_xyzw=(0, 0, 0, 1),
            frame_id="world",
        )
    )
    start = JointState(NAMES, valid.path.positions[0])
    planner = module.PreparedPregraspInteriorPlanner(valid, policy, "selected-B5")
    good = planner.plan(start, goal, PlanningContext("rev", "t"), "scene")
    assert good is valid
    wrong = JointState(NAMES, (0.1, *start.positions[1:]))
    assert not planner.plan(wrong, goal, PlanningContext("rev", "t"), "scene").succeeded
    assert not planner.plan(start, goal, PlanningContext("wrong", "t"), "scene").succeeded
    other = MotionGoal(
        Pose6D(
            position_xyz=(0.1, 0.1, 0.9),
            orientation_xyzw=(0, 0, 0, 1),
            frame_id="world",
        )
    )
    assert not planner.plan(start, other, PlanningContext("rev", "t"), "scene").succeeded
