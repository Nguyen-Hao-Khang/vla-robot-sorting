"""Synthetic regressions; no native solver, contact or motion evidence."""

from types import SimpleNamespace

import pytest

from vla_robot_sorting.planning.manipulation_recovery import RecoveryContractError
from vla_robot_sorting.planning.recovery_query_diagnostics import (
    MotionDiagnosticCapture,
    collect_ik_diagnostics,
    deterministic_solver_starts,
)

LOWER = [-2.4, 0, 0, -1.6, -1.7, -2.5, 0]
UPPER = [2.4, 3.2, 4, 1.6, 1.7, 2.5, 0.04]
CURRENT = [0, 1, 1, 1.5, 0.2, 0, 0.02]


def test_solver_initializations_are_deterministic_keep_gripper_and_never_change_start():
    current = CURRENT.copy()
    starts = deterministic_solver_starts(current, LOWER, UPPER, list(range(6)))
    assert [s["label"] for s in starts] == ["measured", "midpoint", "halfway"]
    assert starts == deterministic_solver_starts(current, LOWER, UPPER, list(range(6)))
    assert all(s["full_qpos"][-1] == 0.02 for s in starts)
    assert current == CURRENT
    assert starts[1]["full_qpos"][3] == 0


@pytest.mark.parametrize(
    "bad",
    [
        [0, 1, 1, 1.6000008583068848, 0.2, 0, 0.02],
        [True, 1, 1, 1.5, 0.2, 0, 0.02],
        [0, float("nan"), 1, 1.5, 0.2, 0, 0.02],
    ],
)
def test_invalid_measured_state_is_rejected_without_clipping(bad):
    with pytest.raises(RecoveryContractError):
        deterministic_solver_starts(bad, LOWER, UPPER, list(range(6)))


class FixtureIKBackend:
    source_kind = "synthetic_fixture"
    joint_names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
    arm_indices = tuple(range(6))
    lower = LOWER
    upper = UPPER

    def __init__(self, mode="ok"):
        self.q = CURRENT.copy()
        self.calls = []
        self.mode = mode

    def snapshot(self):
        return self.q.copy()

    def restore(self, q):
        self.q = q.copy()

    def solve(self, goal, initial):
        self.calls.append(initial.copy())
        self.q = initial.copy()
        q = CURRENT.copy()
        if self.mode == "bounds":
            q[3] = 1.6000008583068848
        return {
            "full_qpos": q,
            "solver_converged": self.mode != "nonconverged",
            "solver_error_6d": [0] * 6,
            "fk_goal_error": 0,
            "fk_link_poses_world_wxyz": {"tool": [0, 0, 1, 1, 0, 0, 0]},
            "collision_pairs": [{"pair": "fixture"}] if self.mode == "collision" else [],
            "target_signed_distance_m": -0.001 if self.mode == "collision" else 0.1,
        }

    def plan_qpos(self, goal, start, **kwargs):
        self.calls.append(kwargs)
        self.q = goal.copy()
        return {"status": "fixture-plan", "executed": False}


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("bounds", "outside_joint_bounds"),
        ("collision", "collision_filtered"),
        ("nonconverged", "solver_not_converged"),
    ],
)
def test_raw_ik_failures_are_separate_and_state_is_restored(mode, reason):
    b = FixtureIKBackend(mode)
    result = collect_ik_diagnostics(b, [0, 0, 1, 1, 0, 0, 0], phase="retreat", allow_rrt=False)
    assert len(result["queries"]) == 3
    assert all(q["classification"] == reason for q in result["queries"])
    assert b.q == CURRENT
    assert result["controller_eligible"] is False
    assert result["source_kind"] == "synthetic_fixture"


def test_single_rrt_call_explicitly_disables_clipping_and_is_never_executed():
    b = FixtureIKBackend()
    result = collect_ik_diagnostics(b, [0, 0, 1, 1, 0, 0, 0], phase="retreat", allow_rrt=True)
    assert len(b.calls) == 4
    assert b.calls[-1]["fix_joint_limits"] is False
    assert b.calls[-1]["planning_time"] == 5
    assert result["rrt"]["executed"] is False
    assert b.q == CURRENT


def test_carried_phase_cannot_use_fixed_target_rrt():
    b = FixtureIKBackend()
    with pytest.raises(RecoveryContractError):
        collect_ik_diagnostics(b, [0, 0, 1, 1, 0, 0, 0], phase="place", allow_rrt=True)
    assert b.calls == []


def test_collector_restores_after_solver_exception():
    b = FixtureIKBackend()

    def fail(*a):
        b.q = [0] * 7
        raise RuntimeError("synthetic unsupported API")

    b.solve = fail
    with pytest.raises(RuntimeError):
        collect_ik_diagnostics(b, [0, 0, 1, 1, 0, 0, 0], phase="retreat", allow_rrt=False)
    assert b.q == CURRENT


def test_bounded_motion_capture_retains_first_failure_and_terminal():
    capture = MotionDiagnosticCapture(capacity=3, stride=1)
    capture.on_command(4, [0, 1, 1, 1.59, 0, 0], [0] * 6, 0.1)
    for step in range(1, 7):
        capture.observe(
            step,
            lambda: {"fixture": True, "full_qpos": CURRENT.copy()},
            violation=(step == 4),
            terminal=(step == 6),
        )
    result = capture.payload()
    assert result["physics_steps_seen"] == 6
    assert result["truncated"] is True
    assert result["first_violation"]["physics_step"] == 4
    assert result["terminal"]["physics_step"] == 6
    assert result["first_violation"]["command"]["waypoint_index"] == 4
    assert len(result["samples"]) == 3
    assert result["controller_eligible"] is False


def test_invalid_command_telemetry_is_refused():
    capture = MotionDiagnosticCapture()
    with pytest.raises(RecoveryContractError):
        capture.on_command(1, [0, 1, 1, float("nan"), 0, 0], [0] * 6, 0.1)


def _fixture_task_for_snapshot():
    from types import SimpleNamespace

    import numpy as np

    from vla_robot_sorting.planning.recovery_query_diagnostics import read_motion_state

    pose = SimpleNamespace(p=np.array([0, 0, 1.0]), q=np.array([1.0, 0, 0, 0]))
    links = [
        SimpleNamespace(get_name=lambda name=name: name, get_pose=lambda: pose)
        for name in ("tool_link", "gripper_L_link", "gripper_R_link")
    ]
    point = SimpleNamespace(
        position=np.array([0, 0, 1.0]),
        normal=np.array([0, 0, 1.0]),
        impulse=np.array([0, 0, 0.001]),
        separation=-0.0001,
    )
    contact = SimpleNamespace(bodies=[SimpleNamespace(id=1), SimpleNamespace(id=2)], points=[point])
    entity = SimpleNamespace(
        get_qpos=lambda: np.array(CURRENT),
        get_active_joints=lambda: [
            SimpleNamespace(get_name=lambda name=name: name)
            for name in [f"joint{i}" for i in range(1, 7)] + ["gripper"]
        ],
        get_links=lambda: links,
    )
    task = SimpleNamespace(
        robot=SimpleNamespace(left_entity=entity),
        scene=SimpleNamespace(get_contacts=lambda: [contact]),
    )
    return task, point, read_motion_state


def test_live_snapshot_records_full_gripper_state_fk_and_raw_contact_points():
    task, _point, read = _fixture_task_for_snapshot()
    row = read(task, lambda body: body.id, source_kind="synthetic_fixture")
    assert row["full_qpos"] == CURRENT and row["full_joint_names"][-1] == "gripper"
    assert set(row["link_poses_world_wxyz"]) == {"tool_link", "gripper_L_link", "gripper_R_link"}
    assert row["physics_contacts"][0]["body_ids"] == [1, 2]
    assert row["physics_contacts"][0]["points"][0]["native_separation"] == -0.0001
    assert row["source_kind"] == "synthetic_fixture" and row["controller_eligible"] is False


def test_live_snapshot_refuses_missing_fk_and_nonfinite_contact():
    task, point, read = _fixture_task_for_snapshot()
    task.robot.left_entity.get_links = lambda: []
    with pytest.raises(RecoveryContractError):
        read(task, lambda body: body.id)
    task, point, read = _fixture_task_for_snapshot()
    point.separation = float("nan")
    with pytest.raises(RecoveryContractError):
        read(task, lambda body: body.id)


def test_violation_is_recorded_even_between_sampled_physics_steps():
    c = MotionDiagnosticCapture(stride=20)

    def read():
        return {"full_qpos": CURRENT.copy(), "source_kind": "synthetic_fixture"}

    c.observe(0, read)
    c.observe(7, read, violation=True)
    assert c.payload()["first_violation"]["physics_step"] == 7


def test_query_start_is_live_state_not_stale_planner_model():
    b = FixtureIKBackend()
    b.q = [0, 1, 1, 0, 0, 0, 0.02]
    b.measured_start = lambda: CURRENT.copy()
    result = collect_ik_diagnostics(b, [0, 0, 1, 1, 0, 0, 0], phase="retreat")
    assert result["query_start_full_qpos"] == CURRENT
    assert result["queries"][0]["solver_seed"]["full_qpos"] == CURRENT
    assert b.q == [0, 1, 1, 0, 0, 0, 0.02]


@pytest.mark.parametrize(
    "field,value",
    [
        ("solver_converged", "true"),
        ("fk_goal_error", float("nan")),
        ("collision_pairs", None),
        ("fk_link_poses_world_wxyz", {}),
    ],
)
def test_raw_query_refuses_ambiguous_backend_fields(field, value):
    b = FixtureIKBackend()
    original = b.solve

    def bad(*args):
        result = original(*args)
        result[field] = value
        return result

    b.solve = bad
    with pytest.raises(RecoveryContractError):
        collect_ik_diagnostics(b, [0, 0, 1, 1, 0, 0, 0], phase="retreat")
    assert b.q == CURRENT


def test_single_cast_does_not_explain_recorded_seven_ulp_excess():
    import struct

    nearest = struct.unpack("<f", struct.pack("<f", 1.6))[0]
    measured = 1.6000008583068848
    assert nearest == 1.600000023841858
    assert measured - nearest == 7 * (2**-23)
    assert measured > nearest > 1.6


def test_raw_query_refuses_masked_gripper_change():
    b = FixtureIKBackend()
    original = b.solve

    def bad(*args):
        result = original(*args)
        result["full_qpos"][-1] = 0.03
        return result

    b.solve = bad
    with pytest.raises(RecoveryContractError):
        collect_ik_diagnostics(b, [0, 0, 1, 1, 0, 0, 0], phase="retreat")
    assert b.q == CURRENT


def test_rejected_measured_full_state_produces_zero_solver_queries():
    b = FixtureIKBackend()
    bad = CURRENT.copy()
    bad[3] = 1.6000008583068848
    b.measured_start = lambda: bad.copy()
    result = collect_ik_diagnostics(b, [0, 0, 1, 1, 0, 0, 0], phase="retreat", allow_rrt=True)
    assert result["rejected_start"] is True and result["queries"] == []
    assert b.calls == [] and b.q == CURRENT
    assert result["controller_eligible"] is False


def test_timed_trace_preserves_actual_schedule_instead_of_nominal_index(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    import vla_robot_sorting.planning.native_plan_diagnostics as native

    path = (
        Path(__file__).resolve().parents[2] / "scripts/linux/probe_panthera_project_manipulation.py"
    )
    spec = importlib.util.spec_from_file_location("additional_probe_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class Backend:
        def __init__(self, *args):
            pass

        def collect(self, rows, times, output, **metadata):
            calls.append((rows, times, metadata))

    monkeypatch.setattr(native, "NativeMPlibDiagnosticBackend", Backend)
    task = SimpleNamespace(_plan_diagnostic_dir=tmp_path)
    plan = SimpleNamespace(
        path=SimpleNamespace(positions=[CURRENT[:6], CURRENT[:6]]),
        planner_id="synthetic_fixture",
        succeeded=True,
    )
    module.collect_optional_plan_trace(
        task,
        None,
        plan,
        None,
        "place-timed",
        SimpleNamespace(scene_revision="fixture", transform_revision="fixture"),
        sample_times=[0, 0.032],
        time_semantics="timed schedule fixture",
    )
    assert calls[0][1] == [0, 0.032]
    assert calls[0][2]["time_semantics"] == "timed schedule fixture"


def test_capture_reports_measured_displacement_without_granting_acceptance():
    capture = MotionDiagnosticCapture()
    capture.observe(
        0, lambda: {"full_qpos": CURRENT.copy(), "gripper_pose_world_xyzw": [0, 0, 1, 0, 0, 0, 1]}
    )
    capture.observe(
        20,
        lambda: {"full_qpos": CURRENT.copy(), "gripper_pose_world_xyzw": [0.04, 0, 1, 0, 0, 0, 1]},
        terminal=True,
    )
    result = capture.payload()
    assert result["measured_gripper_displacement_m"] == pytest.approx(0.04)
    assert result["displacement_authorizes_acceptance"] is False


@pytest.mark.parametrize(
    "mode,automatic", [("strict-acceptance", False), ("physical-diagnostic", True)]
)
def test_experimental_diagnostics_cannot_open_strict_acceptance(mode, automatic):
    from vla_robot_sorting.planning.recovery_query_diagnostics import validate_diagnostic_mode

    with pytest.raises(RecoveryContractError):
        validate_diagnostic_mode(mode, automatic)


def test_live_snapshot_can_use_geometry_verified_link_names():
    from vla_robot_sorting.planning.recovery_query_diagnostics import read_motion_state

    task, _point, _read = _fixture_task_for_snapshot()
    links = {link.get_name(): link for link in task.robot.left_entity.get_links()}
    task.robot.left_entity.find_link_by_name = lambda name: links[name]
    task.robot.left_entity.get_links = lambda: pytest.fail("unverified link name API not needed")
    row = read_motion_state(
        task, lambda body: body.id, source_kind="synthetic_fixture", link_names=list(links)
    )
    assert set(row["link_poses_world_wxyz"]) == set(links)
