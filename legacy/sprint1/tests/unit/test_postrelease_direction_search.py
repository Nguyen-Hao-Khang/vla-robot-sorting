"""Evidence-grounded 10mm separation / 40mm retreat bounded goal search."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_robot_sorting.contracts import Pose6D
from vla_robot_sorting.evidence.manipulation import (
    _bounded_postrelease_search_attested,
    _postrelease_direction_geometry_ok,
)
from vla_robot_sorting.planning.postrelease_direction_search import alternative_goals

REPO = Path(__file__).resolve().parents[2]
EE1 = Pose6D((0.02625081141930058, 0.09686381576262183, 0.8721085637108631), (0, 0, 0, 1), "world")
OBJ1 = Pose6D((0.1359870582818985, 0.09035910665988922, 0.7428584694862366), (0, 0, 0, 1), "world")
EE2 = Pose6D(
    (-0.05860698786482955, -0.13001918424498865, 0.8316987472543492), (0, 0, 0, 1), "world"
)
OBJ2 = Pose6D(
    (-0.05244169011712074, -0.03375954553484917, 0.7429836392402649), (0, 0, 0, 1), "world"
)


def probe():
    path = REPO / "scripts/linux/probe_panthera_project_manipulation.py"
    spec = importlib.util.spec_from_file_location("test_direction_probe", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "ee,obj,distance,phase,n",
    [
        (EE1, OBJ1, 0.04, "retreat", 18),
        (EE2, OBJ2, 0.01, "separation", 7),
    ],
)
def test_alternative_goals_are_exact_bounded_and_independently_auditable(
    ee, obj, distance, phase, n
):
    options = alternative_goals(ee, obj, distance_m=distance, phase=phase)
    assert len(options) == n
    for name, goal, geometry in options:
        assert name == geometry["direction_label"]
        assert math.isclose(
            math.dist(goal.position_xyz, ee.position_xyz), distance, rel_tol=0, abs_tol=1e-12
        )
        assert goal.position_xyz[2] >= ee.position_xyz[2] - 1e-12
        assert goal.frame_id == ee.frame_id
        if geometry["wrist_local_y_pitch_deg"] == 0.0:
            assert goal.orientation_xyzw == ee.orientation_xyzw
        else:
            assert geometry["start_gripper_orientation_xyzw"] == list(ee.orientation_xyzw)
            assert geometry["goal_gripper_orientation_xyzw"] == list(goal.orientation_xyzw)
            assert abs(geometry["wrist_local_y_pitch_deg"]) <= 30.0
        stage = {
            "requested_separation_m": distance,
            "requested_goal_xyz_m": list(goal.position_xyz),
            "separation_geometry": geometry,
        }
        assert _postrelease_direction_geometry_ok(stage), name
    if phase == "retreat":
        assert "vertical_up_separation_only" not in [name for name, *_ in options]


@pytest.mark.parametrize("bad", [-1, 0.0, 0.03, 0.05, float("nan"), float("inf")])
def test_bad_unbounded_distance_refused(bad):
    with pytest.raises(ValueError, match="length"):
        alternative_goals(EE1, OBJ1, distance_m=bad, phase="retreat")


def test_search_geometry_tampering_fails_independent_audit():
    _, goal, geometry = alternative_goals(EE2, OBJ2, distance_m=0.01, phase="separation")[0]
    data = {
        "requested_separation_m": 0.01,
        "requested_goal_xyz_m": list(goal.position_xyz),
        "separation_geometry": dict(geometry),
    }
    assert _postrelease_direction_geometry_ok(data)
    data["separation_geometry"]["direction_label"] = "forged"
    assert not _postrelease_direction_geometry_ok(data)


def test_selected_alternative_has_independent_path_and_search_attestation():
    label, goal, geometry = alternative_goals(EE1, OBJ1, distance_m=0.04, phase="retreat")[0]
    path_pose = [*goal.position_xyz, 1.0, 0.0, 0.0, 0.0]
    report = {
        "schema": "s1m.postrelease-outward-candidate-search.v1",
        "phase": "retreat",
        "baseline_failed_before_physics": True,
        "max_candidates": 6,
        "contact_and_bounds_waivers": False,
        "physics_alternative_executions": 1,
        "selected_label": label,
        "attempted": [
            {
                "label": label,
                "goal_xyz_m": list(goal.position_xyz),
                "planner_success": True,
                "raw_collision_free": True,
                "timed_collision_free": True,
                "dynamic_safe": True,
                "joint_interior_safe": True,
                "eligible": True,
            }
        ],
    }
    stage = {
        "bounded_alternative_search": report,
        "planner_policy": {
            "prepared_postrelease_direction": label,
            "prepared_postrelease_goal_verified": True,
        },
        "path_metadata": {"target_pose_robotwin_wxyz": path_pose},
    }
    geo = {
        **geometry,
        "requested_goal_xyz_m": list(goal.position_xyz),
        "requested_retreat_m": 0.04,
        "no_contact_override": True,
    }
    raw = {"postrelease_direction_search_contract_version": "outward_discrete_prephysics_v1"}
    assert _bounded_postrelease_search_attested(raw, stage, geo, "retreat")
    assert not _bounded_postrelease_search_attested({}, stage, geo, "retreat")
    for field, value in [
        ("physics_alternative_executions", 0),
        ("contact_and_bounds_waivers", True),
        ("selected_label", "forged"),
    ]:
        altered = {**stage, "bounded_alternative_search": {**report, field: value}}
        assert not _bounded_postrelease_search_attested(raw, altered, geo, "retreat")


def test_search_refuses_to_attempt_any_physics_after_baseline_execution(monkeypatch):
    module = probe()

    def forbidden(*args, **kwargs):
        raise AssertionError("no further attempts after physical execution")

    monkeypatch.setattr(module, "ReleasedTargetCollisionScreenedPlanner", forbidden)
    baseline = {
        "passed": False,
        "plan_success": True,
        "execution_status": "completed",
        "contacts": {"robot_target_contact_frames": 1},
        "start_joint_bounds": {"within_bounds": True},
    }
    result, goal = module.execute_bounded_postrelease_alternatives(
        task=None,
        helpers=None,
        context=None,
        scene=None,
        limits=None,
        timer=None,
        timing_gate={},
        phase="retreat",
        baseline=baseline,
        gripper_pose=EE1,
        object_pose=OBJ1,
        distance_m=0.04,
    )
    assert result is baseline and goal is None


def test_search_selects_only_first_fully_screened_cached_plan(monkeypatch):
    module = probe()
    import vla_robot_sorting.planning.pregrasp_joint_reserve as reserve
    import vla_robot_sorting.safety.joint_limits as limits_mod

    names = tuple(f"joint{i}" for i in range(1, 7))
    start = SimpleNamespace(joint_names=names, positions=(0.0,) * 6)
    path = SimpleNamespace(
        joint_names=names,
        positions=((0.0,) * 6, (0.001,) * 6),
        metadata={"scene_revision": "rev", "transform_revision": "trans"},
    )
    plan = SimpleNamespace(
        succeeded=True,
        path=path,
        planner_id="robotwin-curobo",
        planning_time_s=0.1,
    )
    calls = []

    class Planner:
        def __init__(self, task, helpers, phase):
            self.last_provenance = {
                "strategy": "released_target_collision_screen_v1",
                "primary_planner_id": "robotwin-curobo",
                "primary_success": True,
                "primary_native_success": True,
                "primary_collision_safe": True,
                "primary_collision_check": {"collision_free": True},
                "fallback_attempted": False,
                "selected_planner_id": "robotwin-curobo",
                "plan_success": True,
                "target_world_refreshed_after_separation": phase == "retreat",
                "target_world_refreshed_before_motion": True,
            }

        def plan(self, start_state, goal, context, scene):
            calls.append(("planned", goal.pose.position_xyz))
            return plan

    monkeypatch.setattr(module, "ReleasedTargetCollisionScreenedPlanner", Planner)
    monkeypatch.setattr(module, "screen_executable_waypoints", lambda *a: {"collision_free": True})
    monkeypatch.setattr(
        limits_mod,
        "validate_joint_trajectory_limits",
        lambda *a: SimpleNamespace(is_safe=True, violations=()),
    )
    monkeypatch.setattr(
        reserve,
        "audit_pregrasp_joint_reserve",
        lambda *a: {"passed": True, "minimum_margin_rad": 0.012},
    )
    monkeypatch.setattr(
        module,
        "execute_planned_motion",
        lambda **kw: (
            calls.append(("executed", kw["goal"].pose.position_xyz))
            or {"passed": True, "execution_status": "completed"}
        ),
    )
    helper = SimpleNamespace(
        current_arm_state=lambda t: start,
        collision_recheck=lambda *a: {"collision_free": True},
    )
    timer = SimpleNamespace(parameterize=lambda *a: SimpleNamespace())
    baseline = {
        "passed": False,
        "plan_success": False,
        "start_joint_bounds": {"within_bounds": True},
    }
    result, geometry = module.execute_bounded_postrelease_alternatives(
        task=SimpleNamespace(),
        helpers=helper,
        context=SimpleNamespace(scene_revision="rev", transform_revision="trans"),
        scene=SimpleNamespace(),
        limits=SimpleNamespace(),
        timer=timer,
        timing_gate={},
        phase="separation",
        baseline=baseline,
        gripper_pose=EE2,
        object_pose=OBJ2,
        distance_m=0.01,
    )
    assert result["passed"] is True
    assert geometry["direction_label"] == "radial_up_60"
    assert [x[0] for x in calls] == ["planned", "executed"]
    assert result["bounded_alternative_search"]["physics_alternative_executions"] == 1


def test_unexpected_planning_error_is_not_a_safe_candidate_search_trigger(monkeypatch):
    module = probe()

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected native error must not invoke further planning")

    monkeypatch.setattr(module, "ReleasedTargetCollisionScreenedPlanner", forbidden)
    baseline = {
        "passed": False,
        "plan_success": False,
        "start_joint_bounds": {"within_bounds": True},
        "failure": {"code": "internal_error", "message": "CUDA backend exception"},
    }
    result, geometry = module.execute_bounded_postrelease_alternatives(
        task=None,
        helpers=None,
        context=None,
        scene=None,
        limits=None,
        timer=None,
        timing_gate={},
        phase="separation",
        baseline=baseline,
        gripper_pose=EE2,
        object_pose=OBJ2,
        distance_m=0.01,
    )
    assert result is baseline and geometry is None


def test_v2_wrist_pitch_and_screw_provenance_are_independently_recomputed():
    from vla_robot_sorting.evidence.manipulation import (
        _planner_policy_ok,
        _released_target_retreat_policy_ok,
    )

    options = alternative_goals(EE1, OBJ1, distance_m=0.04, phase="retreat")
    target = next(item for item in options if item[0] == "radial_pitch_neg_20")
    label, goal, geometry = target
    assert _postrelease_direction_geometry_ok(
        {
            "requested_separation_m": 0.04,
            "requested_goal_xyz_m": list(goal.position_xyz),
            "separation_geometry": geometry,
        }
    )
    for key, value in (
        ("wrist_local_y_pitch_deg", 20.0),
        ("goal_gripper_orientation_xyzw", [0.0, 0.0, 0.0, 1.0]),
        ("direction_label", "radial_pitch_pos_20"),
    ):
        changed = {**geometry, key: value}
        assert not _postrelease_direction_geometry_ok(
            {
                "requested_separation_m": 0.04,
                "requested_goal_xyz_m": list(goal.position_xyz),
                "separation_geometry": changed,
            }
        )
    screw_policy = {
        "strategy": "mplib_screw_acquisition",
        "wrapper_method": "left_mplib_planner.plan_screw",
        "target_aware_planning_world_required": True,
        "plan_success": True,
        "vendor_status": "Success",
        "selected_planner_id": "robotwin-mplib-screw",
        "straight_screw_motion": True,
        "independent_collision_recheck_required": True,
    }
    policy = {
        "strategy": "released_target_screw_screen_v1",
        "primary_planner_id": "robotwin-mplib-screw",
        "selected_planner_id": "robotwin-mplib-screw",
        "primary_success": True,
        "primary_collision_safe": True,
        "primary_collision_check": {"collision_free": True},
        "fallback_attempted": False,
        "plan_success": True,
        "target_world_refreshed_after_separation": True,
        "target_world_refreshed_before_motion": True,
        "screw_planner_policy": screw_policy,
        "prepared_postrelease_direction": label,
        "prepared_postrelease_goal_verified": True,
    }
    assert _planner_policy_ok(policy)
    assert _released_target_retreat_policy_ok(policy, "robotwin-mplib-screw")
    report = {
        "schema": "s1m.postrelease-outward-candidate-search.v1",
        "phase": "retreat",
        "baseline_failed_before_physics": True,
        "max_candidates": 18,
        "contact_and_bounds_waivers": False,
        "physics_alternative_executions": 1,
        "selected_label": label,
        "attempted": [
            *[{"label": lbl, "eligible": False} for lbl, *_ in options[:6]],
            {
                "label": label,
                "goal_xyz_m": list(goal.position_xyz),
                "planner_success": True,
                "raw_collision_free": True,
                "timed_collision_free": True,
                "dynamic_safe": True,
                "joint_interior_safe": True,
                "eligible": True,
                "screw_probe": {
                    "attempted": True,
                    "planner_id": "robotwin-mplib-screw",
                    "native_success": True,
                    "collision_free": True,
                },
            },
        ],
    }
    stage = {
        "planner_id": "robotwin-mplib-screw",
        "planner_policy": policy,
        "path_metadata": {
            "target_pose_robotwin_wxyz": [
                *goal.position_xyz,
                goal.orientation_xyzw[3],
                *goal.orientation_xyzw[:3],
            ],
        },
        "bounded_alternative_search": report,
    }
    attested_geometry = {
        **geometry,
        "requested_goal_xyz_m": list(goal.position_xyz),
        "requested_retreat_m": 0.04,
        "no_contact_override": True,
    }
    raw = {"postrelease_direction_search_contract_version": "outward_discrete_prephysics_v2"}
    assert _bounded_postrelease_search_attested(raw, stage, attested_geometry, "retreat")
    altered = {
        **stage,
        "path_metadata": {
            "target_pose_robotwin_wxyz": [*goal.position_xyz, 1.0, 0.0, 0.0, 0.0],
        },
    }
    assert not _bounded_postrelease_search_attested(raw, altered, attested_geometry, "retreat")
    tampered = {
        **stage,
        "bounded_alternative_search": {
            **report,
            "attempted": [
                *report["attempted"][:-1],
                {**report["attempted"][-1], "screw_probe": {"attempted": False}},
            ],
        },
    }
    assert not _bounded_postrelease_search_attested(raw, tampered, attested_geometry, "retreat")


def test_bounded_retreat_screw_executes_only_cached_screened_path(monkeypatch):
    import vla_robot_sorting.planning as planning
    import vla_robot_sorting.planning.pregrasp_joint_reserve as reserve
    import vla_robot_sorting.safety.joint_limits as safety

    module = probe()
    from vla_robot_sorting.contracts import JointState

    names = tuple(f"joint{i}" for i in range(1, 7))
    start = JointState(names, (0.0, 0.4, 0.3, 0.2, 0.4, 0.0))
    context = SimpleNamespace(scene_revision="s", transform_revision="t")
    scene = SimpleNamespace()
    calls = []

    class Screw:
        def __init__(self, robot):
            self.last_provenance = {
                "strategy": "mplib_screw_acquisition",
                "wrapper_method": "left_mplib_planner.plan_screw",
                "target_aware_planning_world_required": True,
                "plan_success": True,
                "vendor_status": "Success",
                "selected_planner_id": "robotwin-mplib-screw",
                "straight_screw_motion": True,
                "independent_collision_recheck_required": True,
            }

        def plan(self, start_state, goal, context, scene):
            calls.append("screw_plan")
            from vla_robot_sorting.planning.panthera_robotwin import pose6d_to_robotwin_target

            path = SimpleNamespace(
                joint_names=names,
                positions=(start.positions, start.positions),
                metadata={
                    "scene_revision": context.scene_revision,
                    "transform_revision": context.transform_revision,
                    "target_pose_robotwin_wxyz": pose6d_to_robotwin_target(goal),
                    "vendor_status": "Success",
                },
            )
            result = SimpleNamespace(
                planner_id="robotwin-mplib-screw",
                succeeded=True,
                path=path,
                planning_time_s=0.1,
            )
            return result

    monkeypatch.setattr(planning, "PantheraRoboTwinMPlibScrewPlanner", Screw)
    monkeypatch.setattr(
        module,
        "ReleasedTargetCollisionScreenedPlanner",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("no alternate planner after screw accepted")
        ),
    )
    monkeypatch.setattr(
        module,
        "screen_executable_waypoints",
        lambda *a: {"collision_free": True},
    )
    monkeypatch.setattr(
        safety,
        "validate_joint_trajectory_limits",
        lambda *a: SimpleNamespace(is_safe=True, violations=()),
    )
    monkeypatch.setattr(
        reserve,
        "audit_pregrasp_joint_reserve",
        lambda *a: {"passed": True, "minimum_margin_rad": 0.09},
    )
    helper = SimpleNamespace(
        current_arm_state=lambda t: start,
        collision_recheck=lambda *a: {"collision_free": True},
    )

    def execute(**kwargs):
        calls.append("executed_once")
        actual = kwargs["planner"].plan(start, kwargs["goal"], context, scene)
        assert actual.planner_id == "robotwin-mplib-screw"
        return {
            "passed": True,
            "execution_status": "completed",
            "planner_id": actual.planner_id,
            "path_metadata": actual.path.metadata,
            "planner_policy": kwargs["planner"].last_provenance,
        }

    monkeypatch.setattr(module, "execute_planned_motion", execute)
    timer = SimpleNamespace(parameterize=lambda *args: SimpleNamespace())
    baseline = {
        "passed": False,
        "plan_success": False,
        "start_joint_bounds": {"within_bounds": True},
        "failure": {"code": "optimization_failed"},
    }
    result, geometry = module.execute_bounded_postrelease_alternatives(
        task=SimpleNamespace(robot=SimpleNamespace()),
        helpers=helper,
        context=context,
        scene=scene,
        limits=SimpleNamespace(),
        timer=timer,
        timing_gate={},
        phase="retreat",
        baseline=baseline,
        gripper_pose=EE1,
        object_pose=OBJ1,
        distance_m=0.04,
    )
    assert result["passed"] is True
    assert calls == ["screw_plan", "executed_once"]
    assert geometry["direction_label"] == "radial_up_60"
    assert result["bounded_alternative_search"]["max_candidates"] == 18
    assert result["bounded_alternative_search"]["attempted"][0]["screw_probe"]["collision_free"]
