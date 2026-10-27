"""Windows contract tests for *observational* native retreat contact diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/linux/probe_panthera_project_manipulation.py"
)


def module_under_test():
    spec = importlib.util.spec_from_file_location("retreat_contact_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_task():
    """No simulator or fake PASS: emit a known contact stream for testing only."""

    class Body:
        def __init__(self, identifier: int) -> None:
            self.identifier = identifier

        def get_global_id(self) -> int:
            return self.identifier

        def get_pose(self):
            return SimpleNamespace(
                p=(self.identifier / 10, 0.2, 0.3), q=(1.0, 0.0, 0.0, 0.0)
            )

    bodies = {
        "target": Body(1),
        "pad": Body(2),
        "table": Body(3),
        "left": Body(4),
        "right": Body(5),
        "wrist": Body(6),
    }
    state = {"contacts": []}

    def contact(one: str, two: str):
        return SimpleNamespace(bodies=(bodies[one], bodies[two]))

    pose = SimpleNamespace(position_xyz=(0.1, 0.2, 0.3))
    robot = SimpleNamespace(
        left_entity=SimpleNamespace(
            get_links=lambda: [bodies["left"], bodies["right"], bodies["wrist"]],
            find_link_by_name=lambda name: (
                bodies["left"] if name == "gripper_L_link" else bodies["right"]
            ),
        ),
        get_left_ee_pose=lambda: pose,
        get_left_arm_real_jointState=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    task = SimpleNamespace(
        pillbottle=SimpleNamespace(actor=bodies["target"]),
        pad=SimpleNamespace(actor=bodies["pad"]),
        table=bodies["table"],
        robot=robot,
        scene=SimpleNamespace(get_contacts=lambda: state["contacts"]),
    )
    return task, state, contact, pose


def test_retreat_trace_records_contact_finger_qpos_and_physics_step(monkeypatch) -> None:
    mod = module_under_test()
    task, state, contact, pose = synthetic_task()
    monkeypatch.setattr(mod, "_physical_contact", lambda _: True)
    monkeypatch.setattr(mod, "actor_pose6d", lambda _: pose)
    monkeypatch.setattr(mod, "robotwin_pose6d", lambda _: pose)
    observer = mod.ContactObserver(task, capture_contact_trace=True)

    state["contacts"] = [contact("target", "pad")]
    observer.on_physics_step(task)
    state["contacts"] = [contact("target", "left"), contact("target", "pad")]
    observer.on_physics_step(task)
    state["contacts"] = [contact("target", "right"), contact("target", "wrist")]
    observer.on_physics_step(task)
    state["contacts"] = []
    observer.on_physics_step(task)

    assert observer.payload()["robot_target_contact_frames"] == 2
    assert observer.payload()["robot_target_finger_contact_frames"] == 2
    assert observer.payload()["first_robot_target_step"] == 2
    assert observer.payload()["last_robot_target_step"] == 3
    trace = observer.contact_trace_payload()
    assert trace["schema"] == "physx-contact-frame-observation-v1"
    assert trace["not_contact_point_or_clearance_measurement"] is True
    assert [item["physics_step"] for item in trace["frames"]] == [2, 3]
    assert [item["finger_links"] for item in trace["frames"]] == [
        ["gripper_L_link"],
        ["gripper_R_link"],
    ]
    assert trace["frames"][1]["other_robot_link_contact"] is True
    assert trace["frames"][0]["ee_xyz_m"] == [0.1, 0.2, 0.3]
    assert trace["frames"][0]["target_xyz_m"] == [0.1, 0.2, 0.3]
    assert trace["frames"][0]["arm_qpos_rad"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert not mod.runtime_contact_policy_pass(
        observer.payload(), allow_robot_target_contact=False
    )


def test_trace_is_optional_bounded_and_does_not_alter_contact_gate(monkeypatch) -> None:
    mod = module_under_test()
    task, state, contact, pose = synthetic_task()
    monkeypatch.setattr(mod, "_physical_contact", lambda _: True)
    monkeypatch.setattr(mod, "actor_pose6d", lambda _: pose)
    monkeypatch.setattr(mod, "robotwin_pose6d", lambda _: pose)
    state["contacts"] = [contact("target", "left")]
    observer = mod.ContactObserver(task, capture_contact_trace=True)
    for _ in range(260):
        observer.on_physics_step(task)
    assert observer.payload()["physics_steps"] == 260
    assert observer.payload()["robot_target_contact_frames"] == 260
    trace = observer.contact_trace_payload()
    assert len(trace["frames"]) == 256
    assert trace["truncated"] is True
    assert trace["frames"][-1]["physics_step"] == 256
    assert not mod.runtime_contact_policy_pass(
        observer.payload(), allow_robot_target_contact=False
    )
    ordinary = mod.ContactObserver(task)
    ordinary.on_physics_step(task)
    assert ordinary.contact_trace_payload()["frames"] == []
    assert ordinary.payload()["robot_target_contact_frames"] == 1


def test_missing_pose_snapshot_is_explicit_not_silently_fabricated(monkeypatch) -> None:
    mod = module_under_test()
    task, state, contact, _ = synthetic_task()
    monkeypatch.setattr(mod, "_physical_contact", lambda _: True)
    state["contacts"] = [contact("target", "left")]
    observer = mod.ContactObserver(task, capture_contact_trace=True)
    observer.on_physics_step(task)
    frame = observer.contact_trace_payload()["frames"][0]
    assert "snapshot_error" in frame
    assert "target_xyz_m" not in frame
    assert observer.payload()["robot_target_contact_frames"] == 1



def test_zero_contact_retreat_still_records_motion_for_comparison(monkeypatch) -> None:
    mod = module_under_test()
    task, state, _, pose = synthetic_task()
    monkeypatch.setattr(mod, "_physical_contact", lambda _: True)
    monkeypatch.setattr(mod, "actor_pose6d", lambda _: pose)
    monkeypatch.setattr(mod, "robotwin_pose6d", lambda _: pose)
    observer = mod.ContactObserver(task, capture_contact_trace=True)
    state["contacts"] = []
    for _ in range(139):
        observer.on_physics_step(task)
    assert observer.payload()["physics_steps"] == 139
    assert observer.payload()["robot_target_contact_frames"] == 0
    assert observer.contact_trace_payload()["frames"] == []
    trace = observer.motion_frame_trace_payload()
    assert trace["schema"] == "retreat-kinematic-frames-v1"
    assert len(trace["frames"]) == 139
    assert trace["truncated"] is False
    assert trace["frames"][0]["physics_step"] == 1
    assert trace["frames"][-1]["physics_step"] == 139
    assert all(row["robot_target_contact"] is False for row in trace["frames"])
    assert trace["frames"][0]["arm_qpos_rad"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert trace["frames"][0]["finger_link_world_pose"]["gripper_L_link"][
        "position_xyz_m"
    ] == [0.4, 0.2, 0.3]
    assert trace["frames"][0]["finger_link_world_pose"]["gripper_R_link"][
        "orientation_wxyz"
    ] == [1.0, 0.0, 0.0, 0.0]


def test_motion_trace_retains_contact_flags_and_bounded_length(monkeypatch) -> None:
    mod = module_under_test()
    task, state, contact, pose = synthetic_task()
    monkeypatch.setattr(mod, "_physical_contact", lambda _: True)
    monkeypatch.setattr(mod, "actor_pose6d", lambda _: pose)
    monkeypatch.setattr(mod, "robotwin_pose6d", lambda _: pose)
    observer = mod.ContactObserver(task, capture_contact_trace=True)
    state["contacts"] = []
    observer.on_physics_step(task)
    state["contacts"] = [contact("target", "left")]
    observer.on_physics_step(task)
    state["contacts"] = []
    for _ in range(260):
        observer.on_physics_step(task)
    assert observer.payload()["robot_target_contact_frames"] == 1
    assert len(observer.contact_trace_payload()["frames"]) == 1
    assert len(observer.motion_frame_trace_payload()["frames"]) == 256
    assert observer.motion_frame_trace_payload()["truncated"] is True
    assert [item["robot_target_contact"] for item in observer.motion_frame_trace[:3]] == [
        False, True, False
    ]
