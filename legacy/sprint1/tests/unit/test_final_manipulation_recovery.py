"""Synthetic safety tests; no native execution evidence."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_robot_sorting.contracts import JointLimits, JointState
from vla_robot_sorting.planning.manipulation_recovery import (
    RecoveryContractError,
    acceptance_readiness,
    audit_terminal_bounds,
    create_source_lock,
    joint_bound_report,
    verify_source_lock,
    write_new_json,
)
from vla_robot_sorting.planning.native_plan_diagnostics import (
    collect_plan_samples,
    validate_distance_selftest,
    validate_fk_agreement,
)


def limits():
    return JointLimits(
        ("joint1", "joint2"), (-1.0, -1.6), (1.0, 1.6), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)
    )


def test_observed_boundary_excess_is_rejected_without_clipping():
    state = JointState(("joint1", "joint2"), (0.0, 1.6000008583068848))
    report = joint_bound_report(state, limits())
    assert report["within_bounds"] is False
    assert report["joints"][1]["excess_rad"] == pytest.approx(8.583068848e-7)
    assert state.positions[1] == 1.6000008583068848


@pytest.mark.parametrize(
    "state",
    [
        JointState(("other", "joint2"), (0.0, 0.0)),
        JointState(("joint1",), (0.0,)),
    ],
)
def test_bound_report_rejects_wrong_joint_order(state):
    with pytest.raises(RecoveryContractError):
        joint_bound_report(state, limits())


def test_exact_bound_is_valid_but_has_zero_margin():
    report = joint_bound_report(JointState(("joint1", "joint2"), (1.0, -1.6)), limits())
    assert report["within_bounds"] is True
    assert report["minimum_margin_rad"] == 0.0


def test_json_writer_refuses_overwrite_and_nonfinite(tmp_path):
    p = tmp_path / "result.json"
    write_new_json(p, {"fixture": True})
    with pytest.raises(FileExistsError):
        write_new_json(p, {})
    with pytest.raises(ValueError):
        write_new_json(tmp_path / "nan.json", {"x": float("nan")})
    assert not (tmp_path / "nan.json").exists()


def git_repo(tmp_path):
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    (root / "source.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    return root


def test_source_lock_detects_changed_source_and_wrong_head(tmp_path):
    root = git_repo(tmp_path)
    lock = create_source_lock(root)
    assert verify_source_lock(root, lock)["verified_file_count"] == 1
    wrong = dict(lock, project_git_sha="0" * 40)
    with pytest.raises(RecoveryContractError):
        verify_source_lock(root, wrong)
    (root / "source.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(RecoveryContractError):
        verify_source_lock(root, lock)


@pytest.mark.parametrize("lock", [{}, {"schema": "wrong"}, {"files": {"../x": "0" * 64}}])
def test_source_lock_fails_closed_on_malformed(tmp_path, lock):
    with pytest.raises(RecoveryContractError):
        verify_source_lock(tmp_path, lock)


def test_plan_only_and_missing_evidence_cannot_unlock_acceptance(tmp_path):
    result = acceptance_readiness(tmp_path, "a" * 40)
    assert result["ready"] is False
    assert result["accepted_seed_count"] == 0
    assert result["controller_eligible"] is False
    write_new_json(
        tmp_path / "plan-only.json",
        {
            "passed": True,
            "physics_steps": 0,
            "source_kind": "synthetic_fixture",
        },
    )
    assert acceptance_readiness(tmp_path, "a" * 40)["ready"] is False


class FixtureBackend:
    """Source-shaped synthetic adapter; no native API invoked."""

    source_kind = "synthetic_fixture"

    def __init__(self, fail=False):
        self.q = [0.0, 0.0]
        self.fail = fail

    def snapshot(self):
        return list(self.q)

    def restore(self, state):
        self.q = list(state)

    def sample(self, q):
        self.q = list(q)
        if self.fail:
            raise RuntimeError("synthetic API failure")
        return {
            "full_qpos": list(q),
            "fk_link_poses_world_wxyz": {"tool": [0, 0, 1, 1, 0, 0, 0]},
            "collision_pairs": [],
            "target_signed_distance_m": 0.1,
        }


def test_collector_restores_state_and_never_authorizes_fixture():
    b = FixtureBackend()
    result = collect_plan_samples(b, [[0.1, 0.0], [0.2, 0.0]], [0.0, 0.1])
    assert b.q == [0.0, 0.0]
    assert result["samples"][1]["qpos_rad"] == [0.2, 0.0]
    assert result["source_kind"] == "synthetic_fixture"
    assert result["controller_eligible"] is False
    assert result["physics_steps_during_collection"] == 0


def test_collector_restores_even_when_api_raises():
    b = FixtureBackend(fail=True)
    with pytest.raises(RuntimeError, match="synthetic API failure"):
        collect_plan_samples(b, [[0.1, 0.0]], [0.0])
    assert b.q == [0.0, 0.0]


@pytest.mark.parametrize(
    "rows,times",
    [
        ([], []),
        ([[0.0]], []),
        ([[float("nan")]], [0.0]),
        ([[0.0], [0.1]], [0.1, 0.0]),
        ([[0.0], [0.1, 0.2]], [0.0, 0.1]),
        ([[True]], [0.0]),
    ],
)
def test_collector_refuses_malformed_paths_before_backend_call(rows, times):
    b = FixtureBackend()
    with pytest.raises(RecoveryContractError):
        collect_plan_samples(b, rows, times)
    assert b.q == [0.0, 0.0]


def test_distance_selftest_refuses_wrong_units_or_sign():
    assert validate_distance_selftest(0.1, -0.05)["passed"] is True
    for separated, overlapping in [(100, -50), (0.1, 0.05), (float("nan"), -0.05)]:
        with pytest.raises(RecoveryContractError):
            validate_distance_selftest(separated, overlapping)


def test_fk_gate_refuses_wrong_world_transform_and_accepts_quaternion_sign():
    pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    assert validate_fk_agreement({"finger": pose}, {"finger": [1, 2, 3, -1, 0, 0, 0]})["passed"]
    with pytest.raises(RecoveryContractError):
        validate_fk_agreement({"finger": pose}, {"finger": [0, 0, -3, 1, 0, 0, 0]})
    with pytest.raises(RecoveryContractError):
        validate_fk_agreement({"finger": pose}, {})


def load_manipulation_probe():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[2] / "scripts/linux/probe_panthera_project_manipulation.py"
    )
    spec = importlib.util.spec_from_file_location("final_manip_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def six_limits():
    names = tuple(f"joint{i}" for i in range(1, 7))
    return JointLimits(names, (-1.6,) * 6, (1.6,) * 6, (1.0,) * 6, (2.0,) * 6, (3.0,) * 6)


def test_native_stage_rejects_out_of_bound_start_before_planner():
    module = load_manipulation_probe()
    bound = six_limits()
    state = JointState(bound.joint_names, (0, 0, 0, 1.6000008583068848, 0, 0))

    class NeverPlanner:
        def plan(self, *args, **kwargs):
            pytest.fail("planner must not run on out-of-bound state")

    result = module.execute_planned_motion(
        task=None,
        helpers=SimpleNamespace(current_arm_state=lambda _: state),
        planner=NeverPlanner(),
        goal=None,
        context=None,
        scene=None,
        limits=bound,
        timer=None,
        timing_gate={},
        stage="place",
        require_environment_collision_free=True,
        allow_robot_target_contact=False,
    )
    assert result["passed"] is False
    assert result["start_joint_bounds"]["within_bounds"] is False
    assert result["failure"]["code"] == "dynamic_limit_failed"


def test_native_stage_rejects_out_of_bound_terminal_even_if_tracking_passes(monkeypatch):
    import vla_robot_sorting.control as control
    from vla_robot_sorting.contracts.motion import ExecutionStatus

    module = load_manipulation_probe()
    bound = six_limits()
    start = JointState(bound.joint_names, (0,) * 6)
    end = JointState(bound.joint_names, (0, 0, 0, 1.6000008583068848, 0, 0))
    positions = iter([start.positions, end.positions])
    robot = SimpleNamespace(get_left_arm_real_jointState=lambda: next(positions))
    task = SimpleNamespace(
        robot=robot,
        add_physics_step_observer=lambda _: None,
        clear_physics_step_observers=lambda: None,
    )

    class Observer:
        def __init__(self, *args, **kwargs):
            pass

        def payload(self):
            return {
                "physics_steps": 1,
                "robot_target_contact_frames": 0,
                "target_pad_contact_frames": 0,
                "target_table_contact_frames": 0,
                "robot_table_contact_frames": 0,
            }

    class Executor:
        last_actual_state = end

        def __init__(self, *args, **kwargs):
            pass

        def execute(self, segment):
            return SimpleNamespace(
                status=ExecutionStatus.COMPLETED,
                replan_required=False,
                reason="synthetic completed",
            )

    trajectory = SimpleNamespace(
        duration_s=1.0,
        metadata={"controller_period_s": 0.004},
        waypoints=[
            SimpleNamespace(positions=start.positions),
            SimpleNamespace(positions=(0, 0, 0, 1.6, 0, 0)),
        ],
    )
    monkeypatch.setattr(module, "ContactObserver", Observer)
    monkeypatch.setattr(control, "PantheraRoboTwinTrajectoryExecutor", Executor)
    monkeypatch.setattr(
        control,
        "build_execution_segment",
        lambda *a, **k: SimpleNamespace(predicted_stop_state=end),
    )
    import vla_robot_sorting.safety.joint_limits as safety

    monkeypatch.setattr(
        safety,
        "validate_joint_trajectory_limits",
        lambda *a: SimpleNamespace(is_safe=True, violations=[]),
    )
    path = SimpleNamespace(positions=(start.positions, (0, 0, 0, 1.6, 0, 0)), metadata={})
    planner = SimpleNamespace(
        plan=lambda *a, **k: SimpleNamespace(
            succeeded=True, path=path, planner_id="synthetic", planning_time_s=0
        )
    )
    helpers = SimpleNamespace(
        current_arm_state=lambda _: start, collision_recheck=lambda *a: {"collision_free": True}
    )
    result = module.execute_planned_motion(
        task=task,
        helpers=helpers,
        planner=planner,
        goal=None,
        context=SimpleNamespace(scene_revision="fixture", transform_revision="fixture"),
        scene=None,
        limits=bound,
        timer=SimpleNamespace(parameterize=lambda *a: trajectory),
        timing_gate={"physics_period_s": 0.004},
        stage="transport",
        require_environment_collision_free=True,
        allow_robot_target_contact=False,
    )
    assert result["execution_status"] == "completed"
    assert result["terminal_joint_bounds"]["within_bounds"] is False
    assert result["passed"] is False


def test_independent_terminal_audit_rejects_spoofed_success_flag():
    data = {
        "stages": [
            {
                "stage": "transport",
                "execution_status": "completed",
                "qpos_after_rad": [0, 1, 1, 1.6000008583068848, 0, 0],
                "terminal_joint_bounds": {"within_bounds": True},
            }
        ]
    }
    assert audit_terminal_bounds(data)["passed"] is False


def test_independent_terminal_audit_accepts_exact_canonical_bounds():
    data = {
        "stages": [
            {
                "stage": "transport",
                "execution_status": "completed",
                "qpos_after_rad": [0, 1, 1, 1.5, 0, 0],
                "terminal_joint_bounds": {"within_bounds": True},
            }
        ]
    }
    assert audit_terminal_bounds(data)["passed"] is True
    assert audit_terminal_bounds({"stages": []})["passed"] is False


def test_plan_only_probe_preserves_source_failure_and_no_overwrite(tmp_path):
    import os
    import subprocess

    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "diagnostic"
    lock = tmp_path / "bad-lock.json"
    lock.write_text("{}", encoding="utf-8")
    cmd = [
        sys.executable,
        str(root / "scripts/linux/probe_panthera_plan_only_recovery.py"),
        "--project-root",
        str(root),
        "--robotwin-root",
        str(root),
        "--source-lock",
        str(lock),
        "--output-dir",
        str(output),
        "--seed",
        "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 2
    record = json.loads((output / "diagnostic.json").read_text(encoding="utf-8"))
    assert record["diagnostic_complete"] is False
    assert record["executed"] is False
    before = (output / "diagnostic.json").read_bytes()
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert (output / "diagnostic.json").read_bytes() == before


def test_bounded_stage_preserves_stdout_stderr_exit_and_timeout(tmp_path):
    from scripts.run_final_manipulation_session import run_stage

    result = run_stage(
        "failure",
        [
            sys.executable,
            "-c",
            "import sys;print('fixture out');print('fixture err',file=sys.stderr);sys.exit(7)",
        ],
        tmp_path,
        timeout_s=5,
    )
    assert result["exit_code"] == 7 and result["timed_out"] is False
    assert "fixture out" in (tmp_path / "failure.stdout.log").read_text()
    assert "fixture err" in (tmp_path / "failure.stderr.log").read_text()
    timeout = run_stage(
        "timeout", [sys.executable, "-c", "import time;time.sleep(10)"], tmp_path, timeout_s=0.1
    )
    assert timeout["timed_out"] is True
    assert timeout["exit_code"] == 124


def test_bounded_stage_rejects_overwrite_and_unbounded_timeout(tmp_path):
    from scripts.run_final_manipulation_session import run_stage

    for timeout in (0, -1, float("nan"), 3601):
        with pytest.raises(RecoveryContractError):
            run_stage("x", [sys.executable, "-c", "pass"], tmp_path, timeout_s=timeout)
    run_stage("x", [sys.executable, "-c", "pass"], tmp_path, timeout_s=5)
    with pytest.raises(FileExistsError):
        run_stage("x", [sys.executable, "-c", "pass"], tmp_path, timeout_s=5)


def test_source_lock_is_portable_across_git_text_line_endings(tmp_path):
    import subprocess

    root = git_repo(tmp_path)
    (root / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
    (root / "tên tệp.txt").write_text("one\ntwo\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "text",
        ],
        check=True,
        capture_output=True,
    )
    lock = create_source_lock(root)
    (root / "tên tệp.txt").write_bytes(b"one\r\ntwo\r\n")
    assert verify_source_lock(root, lock)["verified_file_count"] == 3


def test_runtime_observer_stops_after_first_forbidden_contact(monkeypatch):
    module = load_manipulation_probe()

    class Body:
        def __init__(self, identifier):
            self.identifier = identifier

        def get_global_id(self):
            return self.identifier

    bodies = [Body(i) for i in range(1, 7)]
    task = SimpleNamespace(
        pillbottle=SimpleNamespace(actor=bodies[0]),
        pad=SimpleNamespace(actor=bodies[1]),
        table=bodies[2],
        robot=SimpleNamespace(
            left_entity=SimpleNamespace(
                get_links=lambda: bodies[3:],
                find_link_by_name=lambda name: bodies[3] if name == "gripper_L_link" else bodies[4],
            ),
            get_left_arm_real_jointState=lambda: [0, 1, 1, 0, 0, 0, 0.02],
        ),
        scene=SimpleNamespace(
            get_contacts=lambda: [SimpleNamespace(bodies=(bodies[0], bodies[3]))]
        ),
    )
    monkeypatch.setattr(module, "_physical_contact", lambda _: True)
    policy = {
        "allow_robot_target_contact": False,
        "allow_target_pad_contact": False,
        "allow_target_table_contact": True,
    }
    observer = module.ContactObserver(task, strict_policy=policy, limits=six_limits())
    with pytest.raises(module.RuntimeSafetyStop, match="forbidden_contact"):
        observer.on_physics_step(task)
    assert observer.physics_steps == 1
    assert observer.payload()["robot_target_contact_frames"] == 1
    assert observer.runtime_checkpoint["first_failure"]["physics_step"] == 1


def test_runtime_observer_checks_measured_joint_limits_each_step(monkeypatch):
    module = load_manipulation_probe()

    class Body:
        def __init__(self, identifier):
            self.identifier = identifier

        def get_global_id(self):
            return self.identifier

    bodies = [Body(i) for i in range(1, 7)]
    task = SimpleNamespace(
        pillbottle=SimpleNamespace(actor=bodies[0]),
        pad=SimpleNamespace(actor=bodies[1]),
        table=bodies[2],
        robot=SimpleNamespace(
            left_entity=SimpleNamespace(
                get_links=lambda: bodies[3:],
                find_link_by_name=lambda name: bodies[3] if name == "gripper_L_link" else bodies[4],
            ),
            get_left_arm_real_jointState=lambda: [0, 1, 1, 1.6000008583068848, 0, 0, 0.02],
        ),
        scene=SimpleNamespace(get_contacts=lambda: []),
    )
    observer = module.ContactObserver(
        task, strict_policy={"allow_robot_target_contact": True}, limits=six_limits()
    )
    with pytest.raises(module.RuntimeSafetyStop, match="measured_joint_limit"):
        observer.on_physics_step(task)
    assert observer.runtime_checkpoint["joint_checks"] == 1
    assert observer.runtime_checkpoint["first_failure"]["bounds"]["within_bounds"] is False


def test_bound_report_rejects_boolean_position():
    with pytest.raises(RecoveryContractError):
        joint_bound_report(JointState(("joint1", "joint2"), (True, 0)), limits())


def test_trace_auditor_rejects_missing_and_inconsistent_native_records():
    from vla_robot_sorting.planning.native_plan_diagnostics import audit_native_plan_trace

    assert audit_native_plan_trace({})["passed"] is False
    data = {
        "schema": "s1m.plan-samples.v1",
        "source_kind": "synthetic_fixture",
        "waypoint_count": 1,
        "samples": [],
        "native_acceptance": False,
    }
    assert audit_native_plan_trace(data)["passed"] is False


def test_collector_refuses_zero_quaternion_and_missing_distance():
    class Bad(FixtureBackend):
        def sample(self, q):
            row = super().sample(q)
            row["fk_link_poses_world_wxyz"]["tool"] = [0] * 7
            row.pop("target_signed_distance_m")
            return row

    b = Bad()
    with pytest.raises(RecoveryContractError):
        collect_plan_samples(b, [[0, 0]], [0])
    assert b.q == [0, 0]


def test_session_preserves_bad_lock_before_linux_dependency_check(tmp_path):
    import subprocess

    source = Path(__file__).resolve().parents[2]
    root = git_repo(tmp_path)
    lock = tmp_path / "bad-lock.json"
    lock.write_text("{}", encoding="utf-8")
    output = tmp_path / "session"
    cmd = [
        sys.executable,
        str(source / "scripts/run_final_manipulation_session.py"),
        "--project-root",
        str(root),
        "--robotwin-root",
        str(root),
        "--source-lock",
        str(lock),
        "--output-dir",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 2
    record = json.loads((output / "session-result.json").read_text())
    assert record["status"] == "BLOCKED" and record["stages"] == []
    assert (output / "SHA256SUMS.txt").exists()
    before = (output / "session-result.json").read_bytes()
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert before == (output / "session-result.json").read_bytes()


def test_executable_waypoints_are_screened_separately_from_raw_path():
    module = load_manipulation_probe()
    checked = []

    def screen(task, path):
        checked.extend(path.positions)
        return {"collision_free": False, "collisions": [{"waypoint_index": 1}]}

    trajectory = SimpleNamespace(
        waypoints=[SimpleNamespace(positions=(0, 0)), SimpleNamespace(positions=(0.5, 0))]
    )
    result = module.screen_executable_waypoints(
        None, SimpleNamespace(collision_recheck=screen), trajectory, True
    )
    assert result["collision_free"] is False
    assert checked == [(0, 0), (0.5, 0)]


def test_runtime_checkpoint_audit_rejects_incomplete_and_nonfinite():
    from vla_robot_sorting.planning.manipulation_recovery import audit_runtime_checkpoints

    data = {
        "stage": "transport",
        "execution_status": "completed",
        "contacts": {"physics_steps": 2},
        "runtime_safety_checkpoint": {
            "joint_checks": 1,
            "first_failure": None,
            "minimum_margin_rad": 0.1,
        },
    }
    assert audit_runtime_checkpoints(data)["passed"] is False
    data["runtime_safety_checkpoint"].update(joint_checks=2, minimum_margin_rad=float("nan"))
    assert audit_runtime_checkpoints(data)["passed"] is False


def test_missing_plan_readiness_cannot_start_physical_diagnostic(tmp_path):
    from vla_robot_sorting.planning.manipulation_recovery import plan_only_readiness

    result = plan_only_readiness(tmp_path, "a" * 40)
    assert result["ready"] is False
    assert result["controller_eligible"] is False


def test_historical_qpos_is_independently_compared_with_pinned_bounds():
    from scripts.audit_final_manipulation_history import summarize_motion_records

    data = {
        "stage": "place",
        "plan_success": True,
        "execution_status": "completed",
        "passed": True,
        "qpos_after_rad": [0, 1, 1, 1.6000008583068848, 0, 0],
    }
    records = summarize_motion_records(data)
    assert len(records) == 1
    assert records[0]["terminal_bounds_recomputed"] is False
    assert records[0]["minimum_joint_margin_rad"] < 0


def test_interrupted_stage_reaps_child_and_preserves_record(tmp_path, monkeypatch):
    import scripts.run_final_manipulation_session as runner

    class Child:
        pid = 123

        def wait(self, timeout):
            raise KeyboardInterrupt()

    child = Child()
    stopped = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(runner, "_stop_process", lambda proc: stopped.append(proc))
    result = runner.run_stage("interrupted", ["synthetic-command"], tmp_path, timeout_s=5)
    assert stopped == [child]
    assert result["interrupted"] is True and result["exit_code"] == 130
    assert (tmp_path / "interrupted.exit.json").exists()


def test_gripper_command_refuses_out_of_bound_arm_before_closing():
    module = load_manipulation_probe()
    task = SimpleNamespace(
        robot=SimpleNamespace(
            get_left_arm_real_jointState=lambda: [0, 1, 1, 1.6000008583068848, 0, 0, 0.02]
        )
    )
    executor = SimpleNamespace(execute_gripper=lambda _: pytest.fail("no gripper command"))
    with pytest.raises(module.RuntimeSafetyStop, match="start"):
        module.execute_gripper_with_joint_checkpoint(task, executor, None, six_limits())


def test_strict_campaign_cannot_be_accepted_from_missing_or_plan_only_evidence(tmp_path):
    from vla_robot_sorting.planning.manipulation_recovery import audit_strict_campaign

    result = audit_strict_campaign(tmp_path, "a" * 40)
    assert result["strict_native_campaign_audit_passed"] is False
    assert result["controller_eligible"] is False


def test_independent_bound_audits_do_not_treat_placement_summaries_as_motion():
    from vla_robot_sorting.planning.manipulation_recovery import audit_runtime_checkpoints

    data = {
        "stage": "place_above_support",
        "execution_status": "completed",
        "qpos_after_rad": [0, 1, 1, 1.5, 0, 0],
        "terminal_joint_bounds": {"within_bounds": True},
        "contacts": {"physics_steps": 2},
        "runtime_safety_checkpoint": {
            "joint_checks": 2,
            "first_failure": None,
            "minimum_margin_rad": 0.1,
        },
        "placement_recovery": {
            "attempts": [{"execution_status": "completed", "passed": True, "yaw_offset_rad": 0}]
        },
    }
    assert audit_terminal_bounds(data)["passed"] is True
    assert audit_runtime_checkpoints(data)["passed"] is True
