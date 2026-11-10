from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/linux/probe_panthera_project_manipulation.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "probe_panthera_project_manipulation_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contacts(
    *,
    robot_target: int = 0,
    target_pad: int = 0,
    target_table: int = 0,
    robot_table: int = 0,
) -> dict[str, int]:
    return {
        "physics_steps": 10,
        "robot_target_contact_frames": robot_target,
        "target_pad_contact_frames": target_pad,
        "target_table_contact_frames": target_table,
        "robot_table_contact_frames": robot_table,
    }


def test_contact_policy_allows_grasp_contact_only_when_requested() -> None:
    module = _module()

    assert module.runtime_contact_policy_pass(
        _contacts(robot_target=4),
        allow_robot_target_contact=True,
    )
    assert not module.runtime_contact_policy_pass(
        _contacts(robot_target=4),
        allow_robot_target_contact=False,
    )


def test_contact_policy_rejects_target_table_contact_while_carried() -> None:
    module = _module()

    assert not module.runtime_contact_policy_pass(
        _contacts(robot_target=4, target_table=1),
        allow_robot_target_contact=True,
    )


def test_contact_policy_rejects_target_pad_contact_before_release() -> None:
    module = _module()

    assert not module.runtime_contact_policy_pass(
        _contacts(robot_target=4, target_pad=1),
        allow_robot_target_contact=True,
    )


def test_contact_policy_can_explicitly_allow_support_contact_when_needed() -> None:
    module = _module()

    assert module.runtime_contact_policy_pass(
        _contacts(robot_target=4, target_pad=2),
        allow_robot_target_contact=True,
        allow_target_pad_contact=True,
    )


def test_contact_policy_allows_table_support_before_pick() -> None:
    module = _module()

    assert module.runtime_contact_policy_pass(
        _contacts(target_table=10),
        allow_robot_target_contact=False,
        allow_target_table_contact=True,
    )


def test_contact_policy_allows_support_contact_after_release() -> None:
    module = _module()

    assert module.runtime_contact_policy_pass(
        _contacts(target_pad=10, target_table=3),
        allow_robot_target_contact=False,
        allow_target_pad_contact=True,
        allow_target_table_contact=True,
    )


def test_contact_policy_never_allows_robot_table_contact_by_default() -> None:
    module = _module()

    assert not module.runtime_contact_policy_pass(
        _contacts(robot_target=4, target_table=4, robot_table=1),
        allow_robot_target_contact=True,
        allow_target_table_contact=True,
    )


def test_support_escape_distance_is_bounded_by_geometry() -> None:
    module = _module()
    support = {
        "diagnostic_only": True,
        "conservative_overlap": True,
        "within_surface_sphere_radius": True,
        "required_lift_to_sphere_clearance_m": 0.0041,
    }

    assert module.support_escape_distance_m(support) == 0.0051

    support["required_lift_to_sphere_clearance_m"] = 0.0095
    assert module.support_escape_distance_m(support) is None


def test_support_escape_trigger_requires_exact_table_recovery_signature() -> None:
    module = _module()
    initial_lift = {
        "plan_success": False,
        "failure": {
            "details": {
                "native_status": "Start state is colliding with world",
            }
        },
    }
    support = {
        "diagnostic_only": True,
        "conservative_overlap": True,
        "within_surface_sphere_radius": True,
        "required_lift_to_sphere_clearance_m": 0.004,
    }
    table = {
        "planning_only": True,
        "executed": False,
        "obstacle_name": "table",
        "obstacle_disabled": True,
        "obstacle_reenabled": True,
        "start_state": {"available": True, "valid": True, "status": None},
        "plan_success": True,
    }

    assert module.support_escape_trigger(initial_lift, support, table)

    table["obstacle_reenabled"] = False
    assert not module.support_escape_trigger(initial_lift, support, table)


def _pose():
    from vla_robot_sorting.contracts import Pose6D

    return Pose6D(
        position_xyz=(0.13, -0.04, 0.82),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )


def _place_harness(monkeypatch, outcomes):
    from types import SimpleNamespace

    import vla_robot_sorting.orchestration as orchestration

    module = _module()
    pose = _pose()
    called = []
    task = SimpleNamespace(
        pillbottle=pose,
        robot=SimpleNamespace(get_left_ee_pose=lambda: pose),
    )
    monkeypatch.setattr(module, "actor_pose6d", lambda value: value)
    monkeypatch.setattr(module, "robotwin_pose6d", lambda value: value)
    monkeypatch.setattr(
        orchestration,
        "gripper_pose_for_desired_object_pose",
        lambda current_object, current_gripper, desired_object: desired_object,
    )

    def execute(**kwargs):
        called.append(kwargs)
        return dict(outcomes[len(called) - 1])

    monkeypatch.setattr(module, "execute_planned_motion", execute)

    def run():
        return module.execute_place_with_yaw_recovery(
            task=task,
            helpers=None,
            carried_planner=None,
            context=None,
            scene=None,
            limits=None,
            timer=None,
            timing_gate={},
            desired_object_release=pose,
        )

    return module, run, called


def _ik_failed():
    return {
        "passed": False,
        "plan_success": False,
        "execution_status": None,
        "failure": {
            "code": "optimization_failed",
            "details": {"native_status": "IK Fail"},
        },
    }


def _place_succeeded():
    return {
        "passed": True,
        "plan_success": True,
        "execution_status": "completed",
    }


def test_yaw_quaternion_rotation_preserves_xyz_and_rotates_about_world_z() -> None:
    from math import isclose, pi, sqrt

    module = _module()
    initial = _pose()
    rotated = module.yaw_rotated_pose(initial, pi / 2)
    assert rotated.position_xyz == initial.position_xyz
    assert rotated.frame_id == initial.frame_id
    assert all(
        isclose(a, b, abs_tol=1e-10)
        for a, b in zip(
            rotated.orientation_xyzw,
            (0.0, 0.0, sqrt(0.5), sqrt(0.5)),
            strict=True,
        )
    )


def test_yaw_recovery_tries_exact_order_and_stops_on_first_safe_success(
    monkeypatch,
) -> None:
    from math import pi

    _, run, called = _place_harness(
        monkeypatch,
        [_ik_failed(), _ik_failed(), _place_succeeded()],
    )
    result = run()
    recovery = result["placement_recovery"]
    assert result["passed"]
    assert recovery["selected_yaw_offset_rad"] == -pi / 2
    assert recovery["candidate_yaw_offsets_rad"] == [0.0, pi / 2, -pi / 2, pi]
    assert [v["yaw_offset_rad"] for v in recovery["attempts"]] == [0.0, pi / 2, -pi / 2]
    assert len(called) == 3
    assert recovery["bounded_attempt_count"] == 3
    assert recovery["upstream_task_source"]["check_success_orientation_constrained"] is False


def test_yaw_recovery_does_not_retry_non_ik_planning_failure(monkeypatch) -> None:
    bad = _ik_failed()
    bad["failure"]["details"]["native_status"] = "Start state colliding"
    _, run, called = _place_harness(monkeypatch, [bad, _place_succeeded()])
    result = run()
    assert not result["passed"]
    assert result["placement_recovery"]["selected_yaw_offset_rad"] is None
    assert len(called) == 1


def test_yaw_recovery_does_not_retry_unsafe_execution_failure(monkeypatch) -> None:
    unsafe = {
        "passed": False,
        "plan_success": True,
        "execution_status": "completed",
        "collision_safe": False,
        "failure": {"code": "runtime_contact_violation"},
    }
    _, run, called = _place_harness(monkeypatch, [unsafe, _place_succeeded()])
    assert not run()["passed"]
    assert len(called) == 1


def test_yaw_recovery_bounds_four_failed_candidates(monkeypatch) -> None:
    _, run, called = _place_harness(monkeypatch, [_ik_failed() for _ in range(4)])
    result = run()
    assert len(called) == 4
    assert result["placement_recovery"]["bounded_attempt_count"] == 4
    assert result["placement_recovery"]["selected_yaw_offset_rad"] is None
    assert not result["passed"]


def _separation_harness(monkeypatch, *, contact_changes=None, success=True, fingers=()):
    from types import SimpleNamespace

    module = _module()
    pose = _pose()
    calls = []
    task = SimpleNamespace(
        robot=SimpleNamespace(get_left_ee_pose=lambda: pose),
        pillbottle=pose,
        check_success=lambda: success,
    )
    contacts = {
        "physics_steps": 20,
        "target_pad_contact_frames": 20,
        "robot_table_contact_frames": 0,
        "target_table_contact_frames": 0,
    }
    contacts.update(contact_changes or {})
    monkeypatch.setattr(module, "robotwin_pose6d", lambda value: value)
    monkeypatch.setattr(module, "actor_pose6d", lambda value: value)
    monkeypatch.setattr(module, "target_finger_contacts", lambda task: list(fingers))
    monkeypatch.setattr(
        module,
        "ReleasedTargetCollisionScreenedPlanner",
        lambda task, helpers, phase: SimpleNamespace(phase=phase),
    )

    def execute(**kwargs):
        calls.append(kwargs)
        return {
            "stage": "post_release_separation",
            "passed": True,
            "plan_success": True,
            "dynamic_safe": True,
            "execution_status": "completed",
            "contacts": contacts,
        }

    monkeypatch.setattr(module, "execute_planned_motion", execute)

    def run(distance=0.01):
        return module.execute_post_release_separation(
            task=task,
            helpers=None,
            context=None,
            scene=None,
            limits=None,
            timer=None,
            timing_gate={},
            separation_m=distance,
        )

    return run, calls


def test_post_release_separation_is_bounded_and_requests_safe_execution(
    monkeypatch,
) -> None:
    run, calls = _separation_harness(monkeypatch)
    assert run(0.01)["passed"]
    assert len(calls) == 1
    assert calls[0]["stage"] == "post_release_separation"
    assert calls[0]["planner"].phase == "separation"
    assert calls[0]["require_environment_collision_free"] is True
    assert calls[0]["allow_robot_target_contact"] is True
    assert calls[0]["allow_target_pad_contact"] is True
    assert calls[0]["allow_target_table_contact"] is False
    goal = calls[0]["goal"].pose
    assert abs(goal.position_xyz[2] - (_pose().position_xyz[2] + 0.01)) < 1e-9
    for bad in (0.0, -0.001, 0.01001, float("nan"), float("inf")):
        assert not run(bad)["passed"]
    assert len(calls) == 1


def test_post_release_separation_rejects_unsafe_support_and_contacts(
    monkeypatch,
) -> None:
    for change, success, fingers in [
        ({"target_pad_contact_frames": 0}, True, ()),
        ({"physics_steps": 0}, True, ()),
        ({"robot_table_contact_frames": 1}, True, ()),
        ({"target_table_contact_frames": 1}, True, ()),
        ({}, False, ()),
        ({}, True, ("gripper_L_link",)),
    ]:
        run, _ = _separation_harness(
            monkeypatch, contact_changes=change, success=success, fingers=fingers
        )
        assert not run()["passed"]


def test_upstream_task_source_mismatch_fails_closed(tmp_path: Path) -> None:
    module = _module()
    upstream = tmp_path / "envs" / "move_pillbottle_pad.py"
    upstream.parent.mkdir()
    upstream.write_text("def check_success(self): return True", encoding="utf-8")
    try:
        module.verify_upstream_task_source(tmp_path)
    except RuntimeError as exc:
        assert "source hash mismatch" in str(exc)
    else:
        raise AssertionError("unexpectedly accepted unpinned upstream task source")


def test_missing_upstream_task_source_fails_closed(tmp_path: Path) -> None:
    module = _module()
    try:
        module.verify_upstream_task_source(tmp_path)
    except RuntimeError as exc:
        assert "source missing" in str(exc)
    else:
        raise AssertionError("missing upstream task source was not rejected")


def test_canonical_upstream_git_blob_attestation_on_audited_checkout(
    tmp_path: Path,
) -> None:
    import hashlib
    import subprocess

    import pytest

    repo = Path("D:/DAMH-Bridge/audit-current/Panthera-HT_RoboTwin-pinned")
    if not (repo / ".git").exists():
        pytest.skip("pinned upstream audit repository is not mounted here")

    pinned = "2aee31ad75cb7b0cd779540441c6025853a5b60d"
    blob = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{pinned}:RoboTwin/envs/move_pillbottle_pad.py"]
    )
    assert hashlib.sha256(blob).hexdigest() == (
        "20e7b9d954fd42c3b78e4a9431aee68b3ff10f70f20104e7dc7e41c83fc8cffe"
    )
    assert b"\r\n" not in blob

    source = tmp_path / "envs" / "move_pillbottle_pad.py"
    source.parent.mkdir()
    source.write_bytes(blob)
    module = _module()
    provenance = module.verify_upstream_task_source(tmp_path)
    assert provenance["commit"] == pinned
    assert provenance["sha256"] == hashlib.sha256(blob).hexdigest()

    # Git may materialize this pinned LF blob with CRLF on Windows.
    source.write_bytes(blob.replace(b"\n", b"\r\n"))
    crlf = module.verify_upstream_task_source(tmp_path)
    assert crlf["sha256"] == provenance["sha256"]
    assert crlf["raw_sha256"] != crlf["sha256"]
    assert crlf["line_ending_normalization"] == "CRLF_to_LF"

    # A semantic edit and malformed line ending still fail source integrity.
    source.write_bytes(blob.replace(b"check_success", b"check_FAILURE").replace(b"\n", b"\r\n"))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        module.verify_upstream_task_source(tmp_path)
    source.write_bytes(blob.replace(b"\n", b"\r"))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        module.verify_upstream_task_source(tmp_path)


def test_contact_observer_attributes_target_contact_to_finger_or_other_link(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    module = _module()

    class Body:
        def __init__(self, identifier):
            self.identifier = identifier

        def get_global_id(self):
            return self.identifier

    parts = {
        "target": Body(1),
        "pad": Body(2),
        "table": Body(3),
        "left": Body(4),
        "right": Body(5),
        "wrist": Body(6),
    }
    state = {"contacts": []}

    def contact(first, second):
        return SimpleNamespace(bodies=(parts[first], parts[second]))

    task = SimpleNamespace(
        pillbottle=SimpleNamespace(actor=parts["target"]),
        pad=SimpleNamespace(actor=parts["pad"]),
        table=parts["table"],
        robot=SimpleNamespace(
            left_entity=SimpleNamespace(
                get_links=lambda: [parts["left"], parts["right"], parts["wrist"]],
                find_link_by_name=lambda name: (
                    parts["left"] if name == "gripper_L_link" else parts["right"]
                ),
            )
        ),
        scene=SimpleNamespace(get_contacts=lambda: state["contacts"]),
    )
    monkeypatch.setattr(module, "_physical_contact", lambda _: True)
    observer = module.ContactObserver(task)
    state["contacts"] = [contact("target", "left"), contact("target", "pad")]
    observer.on_physics_step(task)
    state["contacts"] = [contact("target", "wrist"), contact("target", "pad")]
    observer.on_physics_step(task)
    state["contacts"] = [contact("target", "pad")]
    observer.on_physics_step(task)
    observed = observer.payload()
    assert observed["robot_target_contact_frames"] == 2
    assert observed["robot_target_finger_contact_frames"] == 1
    assert observed["robot_target_other_link_contact_frames"] == 1
    assert observed["first_robot_target_step"] == 1
    assert observed["last_robot_target_step"] == 2
    assert observed["target_pad_contact_frames"] == 3
    assert observed["robot_table_contact_frames"] == 0


def test_vertical_ik_diagnostic_is_three_planning_only_queries(monkeypatch) -> None:
    from types import SimpleNamespace

    module = _module()
    pose = _pose()
    task = SimpleNamespace(
        pillbottle=pose,
        robot=SimpleNamespace(get_left_ee_pose=lambda: pose),
    )
    monkeypatch.setattr(module, "actor_pose6d", lambda value: value)
    monkeypatch.setattr(module, "robotwin_pose6d", lambda value: value)
    calls = []

    class Planner:
        def plan(self, start, goal, context, scene):
            calls.append((start, goal, context, scene))
            return SimpleNamespace(
                succeeded=False,
                failure=SimpleNamespace(
                    code=SimpleNamespace(value="optimization_failed"),
                    details={"native_status": "IK Fail"},
                ),
            )

    result = module.diagnose_place_vertical_reachability(
        task=task,
        helpers=SimpleNamespace(current_arm_state=lambda _: "measured-start"),
        carried_planner=Planner(),
        context="revision",
        scene="scene",
        desired_object_release=pose,
    )
    assert result["planning_only"] is True
    assert result["executed"] is False
    assert result["acceptance_override"] is False
    assert [r["extra_height_m"] for r in result["attempts"]] == [0.01, 0.02, 0.03]
    assert all(r["plan_success"] is False for r in result["attempts"])
    assert all(r["native_status"] == "IK Fail" for r in result["attempts"])
    assert len(calls) == 3
    assert all(call[0] == "measured-start" for call in calls)
    assert all(call[2:] == ("revision", "scene") for call in calls)


def test_vertical_diagnostic_never_turns_failed_yaw_into_pass(monkeypatch) -> None:
    module, run, called = _place_harness(monkeypatch, [_ik_failed() for _ in range(4)])
    monkeypatch.setattr(
        module,
        "diagnose_place_vertical_reachability",
        lambda **_: {
            "planning_only": True,
            "executed": False,
            "acceptance_override": False,
            "attempts": [{"plan_success": True}],
        },
    )
    result = run()
    assert result["passed"] is False
    assert result["placement_recovery"]["selected_yaw_offset_rad"] is None
    diagnostic = result["placement_recovery"]["vertical_reachability_diagnostic"]
    assert diagnostic["attempts"][0]["plan_success"] is True
    assert len(called) == 4


def test_postrelease_radial_goal_is_outward_and_never_exceeds_10mm() -> None:
    from math import isclose

    from vla_robot_sorting.contracts import Pose6D

    module = _module()
    gripper = Pose6D(
        position_xyz=(0.13, -0.04, 0.82),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    bottle = Pose6D(
        position_xyz=(0.10, -0.04, 0.74),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    goal, geometry = module.bounded_postrelease_separation_goal(gripper, bottle, 0.01)
    assert geometry["strategy"] == "radial_xy_away_from_released_object_v1"
    assert isclose(goal.position_xyz[0], 0.14, abs_tol=1e-10)
    assert isclose(goal.position_xyz[1], -0.04, abs_tol=1e-10)
    assert isclose(goal.position_xyz[2], 0.82, abs_tol=1e-10)
    assert goal.orientation_xyzw == gripper.orientation_xyzw

    fallback, info = module.bounded_postrelease_separation_goal(gripper, gripper, 0.01)
    assert info["strategy"] == "vertical_fallback_when_radial_undefined_v1"
    assert isclose(fallback.position_xyz[2], 0.83, abs_tol=1e-10)
    for unsafe_distance in (0.0, -0.01, 0.01001, float("nan")):
        try:
            module.bounded_postrelease_separation_goal(gripper, bottle, unsafe_distance)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe separation must fail closed")


def test_staged_placement_requires_passed_waypoint_before_exact_canonical(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    import vla_robot_sorting.orchestration as orchestration

    module = _module()
    pose = _pose()
    task = SimpleNamespace(
        pillbottle=pose,
        robot=SimpleNamespace(get_left_ee_pose=lambda: pose),
    )
    monkeypatch.setattr(module, "actor_pose6d", lambda value: value)
    monkeypatch.setattr(module, "robotwin_pose6d", lambda value: value)
    monkeypatch.setattr(
        orchestration,
        "gripper_pose_for_desired_object_pose",
        lambda obj, grip, desired: desired,
    )
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return {
            "stage": kwargs["stage"],
            "passed": len(calls) <= 2,
            "plan_success": True,
            "execution_status": "completed",
        }

    monkeypatch.setattr(module, "execute_planned_motion", execute)
    elevated, canonical, evidence = module.execute_bounded_vertical_place_waypoint(
        task=task,
        helpers=SimpleNamespace(),
        carried_planner=object(),
        context=None,
        scene=None,
        limits=None,
        timer=None,
        timing_gate={},
        desired_object_release=pose,
    )
    assert elevated["passed"] is True
    assert canonical is not None and canonical["passed"] is True
    assert [c["stage"] for c in calls] == ["place_elevated_waypoint", "place_above_support"]
    assert (
        abs(calls[0]["goal"].pose.position_xyz[2] - calls[1]["goal"].pose.position_xyz[2] - 0.02)
        < 1e-12
    )
    assert evidence["canonical_attempted"] is True
    assert evidence["canonical_motion_passed"] is True

    calls.clear()
    monkeypatch.setattr(
        module,
        "execute_planned_motion",
        lambda **kwargs: {
            "stage": kwargs["stage"],
            "passed": False,
            "plan_success": False,
            "execution_status": None,
        },
    )
    elevated, canonical, evidence = module.execute_bounded_vertical_place_waypoint(
        task=task,
        helpers=None,
        carried_planner=None,
        context=None,
        scene=None,
        limits=None,
        timer=None,
        timing_gate={},
        desired_object_release=pose,
    )
    assert elevated["passed"] is False
    assert canonical is None
    assert evidence["canonical_attempted"] is False


def test_staged_place_recovery_never_accepts_waypoint_only(monkeypatch) -> None:
    module, run, _ = _place_harness(monkeypatch, [_ik_failed() for _ in range(4)])
    monkeypatch.setattr(
        module,
        "diagnose_place_vertical_reachability",
        lambda **kwargs: {
            "planning_only": True,
            "executed": False,
            "acceptance_override": False,
            "attempts": [
                {"extra_height_m": 0.01, "plan_success": True},
                {"extra_height_m": 0.02, "plan_success": True},
                {"extra_height_m": 0.03, "plan_success": True},
            ],
        },
    )
    monkeypatch.setattr(
        module,
        "execute_bounded_vertical_place_waypoint",
        lambda **kwargs: (
            {"passed": True, "plan_success": True},
            {
                "stage": "place_above_support",
                "passed": False,
                "plan_success": False,
                "execution_status": None,
            },
            {"canonical_attempted": True, "canonical_motion_passed": False},
        ),
    )
    result = run()
    assert result["passed"] is False
    recovery = result["placement_recovery"]
    assert recovery["strategy"] == "yaw_plus_bounded_staged_descent_v2"
    assert recovery["selected_yaw_offset_rad"] is None
    assert recovery["bounded_attempt_count"] == 4
    assert recovery["staged_descent"]["canonical_motion_passed"] is False


def test_strict_retreat_goal_is_exactly_40mm_away_and_never_relaxes_contacts() -> None:
    from math import isclose

    from vla_robot_sorting.contracts import Pose6D

    module = _module()
    ee = Pose6D(
        position_xyz=(0.14, -0.04, 0.83),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    obj = Pose6D(
        position_xyz=(0.10, -0.04, 0.74),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    goal, proof = module.bounded_contact_free_retreat_goal(ee, obj, 0.04)
    assert isclose(goal.position_xyz[0], 0.18, abs_tol=1e-12)
    assert isclose(goal.position_xyz[2], 0.83, abs_tol=1e-12)
    assert proof["strategy"] == "radial_xy_away_from_released_object_v1"
    assert proof["no_contact_override"] is True
    assert proof["requested_retreat_m"] == 0.04
    assert goal.orientation_xyzw == ee.orientation_xyzw
    fallback, info = module.bounded_contact_free_retreat_goal(ee, ee, 0.04)
    assert info["strategy"] == "vertical_fallback_when_radial_undefined_v1"
    assert isclose(fallback.position_xyz[2], 0.87, abs_tol=1e-12)
    for bad in (-0.04, 0.039, 0.05, float("nan")):
        try:
            module.bounded_contact_free_retreat_goal(ee, obj, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("retreat out of strict bound")


def test_postrelease_planner_screens_curobo_before_target_aware_fallback(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    import vla_robot_sorting.planning as planning

    module = _module()
    fallback_calls: list[str] = []
    screen_hits: list[str] = []

    class Native:
        def __init__(self, robot):
            self.robot = robot

        def plan(self, start, goal, context, scene):
            return SimpleNamespace(
                planner_id="robotwin-curobo",
                succeeded=True,
                path=SimpleNamespace(positions=((0.0,) * 6,)),
                planning_time_s=0.1,
            )

    class Fallback:
        def __init__(self, robot):
            self.robot = robot

        def plan(self, start, goal, context, scene):
            fallback_calls.append("called")
            return SimpleNamespace(
                planner_id="robotwin-mplib-rrt",
                succeeded=True,
                path=SimpleNamespace(positions=((0.0,) * 6,)),
                planning_time_s=0.2,
            )

    monkeypatch.setattr(planning, "PantheraRoboTwinMotionPlanner", Native)
    monkeypatch.setattr(planning, "PantheraRoboTwinMPlibPlanner", Fallback)
    task = SimpleNamespace(robot=object())
    for colliding in (False, True):
        fallback_calls.clear()
        helper = SimpleNamespace(
            collision_recheck=lambda task, path, colliding=colliding: (
                screen_hits.append("checked")
                or {
                    "collision_free": not colliding,
                    "collisions": [{"waypoint_index": 0}] if colliding else [],
                }
            )
        )
        planner = module.ReleasedTargetCollisionScreenedPlanner(task, helper)
        chosen = planner.plan(None, None, None, None)
        policy = planner.last_provenance
        assert screen_hits
        assert policy["strategy"] == "released_target_collision_screen_v1"
        assert policy["primary_native_success"] is True
        assert policy["primary_success"] is (not colliding)
        assert policy["primary_collision_safe"] is (not colliding)
        assert policy["fallback_attempted"] is colliding
        assert chosen.planner_id == ("robotwin-mplib-rrt" if colliding else "robotwin-curobo")
        assert len(fallback_calls) == int(colliding)
        from vla_robot_sorting.evidence.manipulation import _planner_policy_ok

        assert _planner_policy_ok(policy)
        if colliding:
            assert policy["primary_collision_check"]["collision_free"] is False
            assert policy["primary_collision_check"]["collisions"]


def test_separation_refreshes_native_target_before_screened_planning() -> None:
    """Static execution-order regression: sync must precede separation motion."""
    import ast

    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    release_sync = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "release_world_sync"
            for target in node.targets
        )
    )
    separation = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "separation" for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "execute_post_release_separation"
    )
    after_sync = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "released_world_sync"
            for target in node.targets
        )
    )
    assert release_sync.lineno < separation.lineno < after_sync.lineno
    assert "mplib_world_after_release_before_separation" in source


def test_separation_screened_planner_phase_attestation_and_bounded_policy() -> None:
    module = _module()
    planner = module.ReleasedTargetCollisionScreenedPlanner(object(), object(), phase="separation")
    assert planner._phase == "separation"
    for invalid in ("", "pregrasp", "release", "unsafe"):
        try:
            module.ReleasedTargetCollisionScreenedPlanner(object(), object(), phase=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported phase accepted")
