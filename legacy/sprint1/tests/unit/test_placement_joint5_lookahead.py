"""Bounded joint5 placement lookahead is planning-only and evidence-attested."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from dataclasses import replace
from math import pi
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_robot_sorting.contracts import JointPath, JointState, MotionGoal, PlannerResult, Pose6D
from vla_robot_sorting.evidence.manipulation import _joint5_placement_lookahead_attested
from vla_robot_sorting.planning.placement_lookahead import (
    choose_safe_yaw,
    fingerprint_joint_path,
    joint5_margin,
)

ROOT = Path(__file__).resolve().parents[2]
NAMES = tuple(f"joint{i}" for i in range(1, 7))
START = (0.0, 2.0, 1.0, 0.0, 1.54, 0.0)
GOAL = Pose6D((0.3, -0.1, 0.83), (0.0, 0.0, 0.0, 1.0), "world")


def probe():
    path = ROOT / "scripts/linux/probe_panthera_project_manipulation.py"
    spec = importlib.util.spec_from_file_location("placement_test_probe", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_path(endpoint=START):
    return JointPath(
        joint_names=NAMES,
        positions=(START, endpoint),
        source="robotwin-curobo-carried-object",
        metadata={"scene_revision": "s", "transform_revision": "t"},
    )


def eligible_row(yaw, margin):
    path = make_path()
    return {
        "yaw_offset_rad": yaw,
        "planner_id": "robotwin-curobo-carried-object",
        "planner_success": True,
        "dynamic_safe": True,
        "raw_self_collision_free": True,
        "timed_self_collision_free": True,
        "joint_interior_safe": True,
        "minimum_all_joint_margin_rad": 0.0075,
        "joint5_margin_rad": margin,
        "path_sha256": fingerprint_joint_path(path),
        "eligible": True,
    }


def test_planned_joint5_not_measured_state_controls_lookahead():
    limits = SimpleNamespace(
        joint_names=NAMES,
        lower_position_rad=(-2.4, 0, 0, -1.6, -1.7, -2.5),
        upper_position_rad=(2.4, 3.2, 4, 1.6, 1.7, 2.5),
    )
    assert joint5_margin((0.0, 2.0, 1.0, 0.0, 1.69, 0.0), limits) == pytest.approx(0.01)
    assert joint5_margin((0.0, 2.0, 1.0, 0.0, 0.3, 0.0), limits) == pytest.approx(1.4)
    with pytest.raises(ValueError):
        joint5_margin((0.0,) * 5, limits)


def test_fingerprint_changes_with_waypoints_not_metadata():
    a = make_path()
    b = replace(a, metadata={**a.metadata, "some": "change"})
    c = make_path((0.0, 2.0, 1.0, 0.01, 1.54, 0.0))
    assert fingerprint_joint_path(a) == fingerprint_joint_path(b)
    assert fingerprint_joint_path(a) != fingerprint_joint_path(c)


def test_policy_refuses_threshold_only_and_unsafe_choices():
    rows = [
        eligible_row(0.0, 0.012),
        eligible_row(pi / 2, 0.20),
        eligible_row(-pi / 2, 0.30),
        eligible_row(pi, 0.15),
    ]
    assert choose_safe_yaw(rows, baseline_margin_rad=0.012)["yaw_offset_rad"] == -pi / 2
    assert choose_safe_yaw(rows, baseline_margin_rad=0.12) is None
    changed = deepcopy(rows)
    changed[2]["timed_self_collision_free"] = False
    assert choose_safe_yaw(changed, baseline_margin_rad=0.012)["yaw_offset_rad"] == pi / 2
    for field in ("dynamic_safe", "joint_interior_safe", "raw_self_collision_free", "eligible"):
        rejected = deepcopy(rows)
        for row in rejected[1:]:
            row[field] = False
        assert choose_safe_yaw(rejected, baseline_margin_rad=0.012) is None


def make_attested():
    rows = [
        eligible_row(0.0, 0.012),
        eligible_row(pi / 2, 0.20),
        eligible_row(-pi / 2, 0.30),
        eligible_row(pi, 0.15),
    ]
    selected = rows[2]
    goal = [0.3, -0.1, 0.83, 1.0, 0.0, 0.0, 0.0]
    report = {
        "schema": "s1m.joint5-placement-lookahead.v1",
        "planning_only": True,
        "waivers": False,
        "physics_executions": 1,
        "max_candidate_plans": 4,
        "baseline_yaw_rad": 0.0,
        "baseline_joint5_margin_rad": 0.012,
        "candidate_yaw_offsets_rad": [0.0, pi / 2, -pi / 2, pi],
        "selected_yaw_offset_rad": -pi / 2,
        "selected_plan_sha256": selected["path_sha256"],
        "selected_goal_robotwin_wxyz": goal,
        "evaluated": rows,
    }
    recovery = {
        "selected_yaw_offset_rad": -pi / 2,
        "lookahead": report,
        "attempts": [
            {
                "yaw_offset_rad": -pi / 2,
                "passed": True,
                "plan_success": True,
                "execution_status": "completed",
            }
        ],
    }
    place = {
        "passed": True,
        "execution_status": "completed",
        "placement_yaw_offset_rad": -pi / 2,
        "planner_policy": {
            "strategy": "joint5_placement_lookahead_v1",
            "selected_yaw_offset_rad": -pi / 2,
            "prepared_placement_plan_verified": True,
        },
        "path_metadata": {
            "lookahead_plan_sha256": selected["path_sha256"],
            "lookahead_selected_yaw_offset_rad": -pi / 2,
            "lookahead_goal_robotwin_wxyz": goal,
        },
    }
    return place, recovery


def test_auditor_fails_closed_on_tampering():
    stage, rec = make_attested()
    assert _joint5_placement_lookahead_attested(stage, rec)
    for field, value in [
        ("physics_executions", 2),
        ("waivers", True),
        ("selected_yaw_offset_rad", pi / 2),
        ("baseline_joint5_margin_rad", 0.25),
    ]:
        changed = deepcopy(rec)
        changed["lookahead"][field] = value
        assert not _joint5_placement_lookahead_attested(stage, changed)
    altered = deepcopy(stage)
    altered["path_metadata"]["lookahead_plan_sha256"] = "a" * 64
    assert not _joint5_placement_lookahead_attested(altered, rec)


def test_cached_placement_checks_goal_start_scene_and_digest():
    module = probe()
    from vla_robot_sorting.planning.panthera_robotwin import pose6d_to_robotwin_target

    start = JointState(NAMES, START)
    plan = PlannerResult(
        path=replace(
            make_path(),
            metadata={
                "scene_revision": "s",
                "transform_revision": "t",
                "lookahead_plan_sha256": fingerprint_joint_path(make_path()),
                "lookahead_goal_robotwin_wxyz": list(pose6d_to_robotwin_target(MotionGoal(GOAL))),
            },
        ),
        failure=None,
        planning_time_s=0.1,
        planner_id="robotwin-curobo-carried-object",
    )
    planner = module.PreparedCarriedPlacePlanner(plan, GOAL, pi / 2)
    ctx = SimpleNamespace(scene_revision="s", transform_revision="t")
    assert planner.plan(start, MotionGoal(GOAL), ctx, scene=None).succeeded
    mismatch = Pose6D((0.33, -0.1, 0.83), GOAL.orientation_xyzw, "world")
    assert not planner.plan(start, MotionGoal(mismatch), ctx, scene=None).succeeded
    assert not planner.plan(
        JointState(NAMES, (START[0] + 0.03, *START[1:])),
        MotionGoal(GOAL),
        ctx,
        scene=None,
    ).succeeded
    assert not planner.plan(
        start,
        MotionGoal(GOAL),
        SimpleNamespace(scene_revision="changed", transform_revision="t"),
        scene=None,
    ).succeeded


def test_preflight_plans_four_yaws_but_executes_no_physics(monkeypatch):
    import vla_robot_sorting.orchestration as orchestration
    import vla_robot_sorting.safety.joint_limits as safety

    module = probe()
    object_pose = GOAL
    task = SimpleNamespace(
        pillbottle=object_pose,
        robot=SimpleNamespace(get_left_ee_pose=lambda: object_pose),
    )
    monkeypatch.setattr(module, "actor_pose6d", lambda item: item)
    monkeypatch.setattr(module, "robotwin_pose6d", lambda item: item)
    monkeypatch.setattr(
        orchestration,
        "gripper_pose_for_desired_object_pose",
        lambda obj, ee, target: target,
    )

    near_start = (*START[:4], 1.65, START[5])

    class Planner:
        def __init__(self):
            self.count = 0

        def plan(self, state, goal, context, scene):
            margin_targets = (1.69, 1.20, 0.90, 1.50)
            endpoint = (*START[:4], margin_targets[self.count], START[5])
            self.count += 1
            path = JointPath(
                joint_names=NAMES,
                positions=(near_start, endpoint),
                source="robotwin-curobo-carried-object",
                metadata={"scene_revision": "s", "transform_revision": "t"},
            )
            return PlannerResult(
                path=path,
                failure=None,
                planning_time_s=0.01,
                planner_id="robotwin-curobo-carried-object",
            )

    planner = Planner()
    helper = SimpleNamespace(
        current_arm_state=lambda task: JointState(NAMES, near_start),
        self_collision_recheck=lambda task, path: {"self_collision_free": True},
    )
    monkeypatch.setattr(
        module,
        "screen_executable_waypoints",
        lambda *a: {"self_collision_free": True},
    )
    monkeypatch.setattr(
        safety,
        "validate_joint_trajectory_limits",
        lambda *a: SimpleNamespace(is_safe=True, violations=()),
    )
    limits = SimpleNamespace(
        joint_names=NAMES,
        lower_position_rad=(-2.4, 0, 0, -1.6, -1.7, -2.5),
        upper_position_rad=(2.4, 3.2, 4, 1.6, 1.7, 2.5),
    )
    timer = SimpleNamespace(
        parameterize=lambda path, limits: SimpleNamespace(
            joint_names=path.joint_names,
            waypoints=tuple(SimpleNamespace(positions=row) for row in path.positions),
        )
    )
    context = SimpleNamespace(scene_revision="s", transform_revision="t")
    selected = module.preflight_place_joint5_lookahead(
        task=task,
        helpers=helper,
        carried_planner=planner,
        context=context,
        scene=None,
        limits=limits,
        timer=timer,
        desired_object_release=GOAL,
    )
    assert selected is not None
    wrapper, goal, report = selected
    assert planner.count == 4
    assert report["selected_yaw_offset_rad"] == -pi / 2
    assert report["physics_executions"] == 0
    assert len(report["evaluated"]) == 4
    actual = wrapper.plan(JointState(NAMES, near_start), MotionGoal(goal), context, scene=None)
    assert actual.succeeded
    assert actual.path.metadata["lookahead_plan_sha256"] == report["selected_plan_sha256"]
    assert actual.path.positions[-1][4] == pytest.approx(0.90)


def test_healthy_transport_state_skips_all_extra_planner_calls():
    module = probe()
    limits = SimpleNamespace(
        joint_names=NAMES,
        lower_position_rad=(-2.4, 0, 0, -1.6, -1.7, -2.5),
        upper_position_rad=(2.4, 3.2, 4, 1.6, 1.7, 2.5),
    )

    def never(*args, **kwargs):
        raise AssertionError("healthy seed may not use extra planner calls")

    result = module.preflight_place_joint5_lookahead(
        task=SimpleNamespace(),
        helpers=SimpleNamespace(current_arm_state=lambda t: JointState(NAMES, START)),
        carried_planner=SimpleNamespace(plan=never),
        context=SimpleNamespace(),
        scene=SimpleNamespace(),
        limits=limits,
        timer=SimpleNamespace(),
        desired_object_release=GOAL,
    )
    assert result is None
