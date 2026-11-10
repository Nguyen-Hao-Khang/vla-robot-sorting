"""Offline safety contracts for bounded native-IK diagnostic mode.

These tests do not launch RoboTwin, solve native IK, or execute physics.
"""
from __future__ import annotations

import importlib.util
from math import isclose, sqrt
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_robot_sorting.contracts import Pose6D

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/linux/probe_panthera_project_manipulation.py"
)


def load_probe():
    spec = importlib.util.spec_from_file_location("retreat_ik_diagnostic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def poses():
    # Coordinates from archived native seed1 :up75 post-separation.
    ee = Pose6D(
        (0.026456602811429752, 0.09694583444955211, 0.8718259969607843),
        (0.0, 0.0, 0.0, 1.0),
        "world",
    )
    target = Pose6D(
        (0.13673299551010132, 0.0906381830573082, 0.7428379058837891),
        (0.0, 0.0, 0.0, 1.0),
        "world",
    )
    return ee, target


def test_retreat_goal_family_is_exact_40mm_noninward_and_nonlowering():
    module = load_probe()
    start, target = poses()
    goals = module.bounded_retreat_diagnostic_goals(start, target)
    assert [name for name, _ in goals] == [
        "xy_plus_45deg", "xy_minus_45deg",
        "radial_plus_z_45deg", "vertical_plus_z",
    ]
    sx, sy, sz = start.position_xyz
    tx, ty, _ = target.position_xyz
    for name, goal in goals:
        x, y, z = goal.position_xyz
        length = sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2)
        outward = (x - sx)*(sx - tx) + (y - sy)*(sy - ty)
        assert isclose(length, 0.04, rel_tol=0, abs_tol=1e-12), name
        assert outward >= -1e-12, name
        assert z >= sz - 1e-12, name
        assert goal.orientation_xyzw == start.orientation_xyzw
        assert goal.frame_id == start.frame_id
    assert goals[-1][1].position_xyz[:2] == start.position_xyz[:2]


def test_retreat_goal_family_refuses_unsupported_geometry():
    module = load_probe()
    ee, target = poses()
    for distance in (-0.04, 0.0, 0.039, 0.05, float("nan")):
        with pytest.raises(ValueError, match="exactly 40mm"):
            module.bounded_retreat_diagnostic_goals(ee, target, distance)
    bad_frame = Pose6D(target.position_xyz, target.orientation_xyzw, "other")
    with pytest.raises(ValueError, match="same reference frame"):
        module.bounded_retreat_diagnostic_goals(ee, bad_frame)
    same_center = Pose6D(
        (ee.position_xyz[0], ee.position_xyz[1], target.position_xyz[2]),
        ee.orientation_xyzw, "world",
    )
    assert module.bounded_retreat_diagnostic_goals(ee, same_center) == ()


def test_native_planning_diagnostic_never_executes_or_certifies(monkeypatch):
    module = load_probe()
    ee, target = poses()
    calls = []
    screening_count = []

    class DummyPlanner:
        def __init__(self, task, helpers):
            self.last_provenance = {
                "primary_collision_safe": False,
                "fallback_attempted": True,
            }

        def plan(self, state, goal, context, scene):
            calls.append(tuple(goal.pose.position_xyz))
            idx = len(calls)
            return SimpleNamespace(
                planner_id="fake-native-planner",
                succeeded=idx != 2,
                path=SimpleNamespace(positions=((0.0,) * 6,)) if idx != 2 else None,
                failure=(
                    SimpleNamespace(
                        code=SimpleNamespace(value="optimization_failed"),
                        details={"native_status": "IK Failed"},
                    ) if idx == 2 else None
                ),
            )

    class Helpers:
        def collision_recheck(self, task, path):
            screening_count.append(path)
            # A planner path may succeed yet fail collision screening.
            return {
                "collision_free": len(screening_count) != 1,
                "waypoints_checked": 12,
                "collisions": [{"link": "gripper"}] if len(screening_count) == 1 else [],
            }

    monkeypatch.setattr(module, "ReleasedTargetCollisionScreenedPlanner", DummyPlanner)
    report = module.diagnose_retreat_alternative_plans(
        task=object(), helpers=Helpers(), context=object(), scene=object(),
        start_state=object(), gripper_pose=ee, object_pose=target,
    )
    assert len(report["attempts"]) == 4
    assert len(calls) == 4
    assert len(screening_count) == 3
    assert report["planning_only"] is True
    assert report["executed"] is False
    assert report["acceptance_override"] is False
    assert all(row["eligible_for_execution"] is False for row in report["attempts"])
    assert report["attempts"][0]["plan_success"] is True
    assert report["attempts"][0]["collision_recheck_pass"] is False
    assert report["attempts"][1]["native_status"] == "IK Failed"
    assert report["attempts"][1]["collision_recheck_pass"] is False
    assert report["attempts"][2]["collision_recheck_pass"] is True


def test_planner_exceptions_are_recorded_as_unknown_not_success(monkeypatch):
    module = load_probe()
    ee, target = poses()

    class BrokenPlanner:
        def __init__(self, task, helpers):
            pass

        def plan(self, state, goal, context, scene):
            raise RuntimeError("planner unavailable")

    monkeypatch.setattr(module, "ReleasedTargetCollisionScreenedPlanner", BrokenPlanner)
    result = module.diagnose_retreat_alternative_plans(
        task=None, helpers=None, context=None, scene=None,
        start_state=None, gripper_pose=ee, object_pose=target,
    )
    assert len(result["attempts"]) == 4
    for row in result["attempts"]:
        assert row["plan_success"] is False
        assert row["collision_recheck_pass"] is False
        assert row["eligible_for_execution"] is False
        assert "planner unavailable" in row["diagnostic_error"]


def test_native_ik_diagnostic_opt_in_and_strict_baseline_unchanged(monkeypatch):
    import sys

    module = load_probe()
    required = [
        "probe.py", "--robotwin-root", "/tmp/robotwin",
        "--project-root", "/tmp/project", "--output", "/tmp/out.json",
    ]
    monkeypatch.setattr(sys, "argv", required)
    assert module.parse_args().diagnose_native_ik_blockers is False
    monkeypatch.setattr(sys, "argv", [*required, "--diagnose-native-ik-blockers"])
    assert module.parse_args().diagnose_native_ik_blockers is True
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'and retreat.get("plan_success") is False' in text
    assert "if not place[\"passed\"]:" in text
    assert "stage=\"retreat\"" in text
    assert "allow_robot_target_contact=False" in text
    assert "retreat[\"retreat_remaining_m\"] = 0.04" in text
