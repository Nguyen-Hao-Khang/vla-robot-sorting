import json
from pathlib import Path

from vla_robot_sorting.evidence import audit_s1_2_motion_probe


def _payload() -> dict[str, object]:
    return {
        "schema": "panthera-project-motion-probe-v1",
        "seed": 0,
        "planner_class": "CuroboPlanner",
        "current_pose_robotwin_wxyz": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "actual_joint_positions_rad": [0.0] * 6,
        "joint_limits": {
            "lower_position_rad": [-3.0] * 6,
            "upper_position_rad": [3.0] * 6,
            "max_velocity_rad_s": [2.0] * 6,
            "max_acceleration_rad_s2": [2.0] * 6,
            "max_jerk_rad_s3": [500.0] * 6,
            "source": "runtime URDF + pinned curobo.yml cspace",
        },
        "requested_translation_m": 0.01,
        "requested_rotation_deg": 5.0,
        "probes": [
            {
                "label": "translation",
                "planner_id": "robotwin-curobo",
                "planning_time_s": 0.2,
                "succeeded": True,
                "path_samples": 10,
                "trajectory_duration_s": 0.036,
                "trajectory_samples": 10,
                "joint_limit_safe": True,
                "violations": [],
            },
            {
                "label": "orientation",
                "planner_id": "robotwin-curobo",
                "planning_time_s": 0.3,
                "succeeded": True,
                "path_samples": 12,
                "trajectory_duration_s": 0.044,
                "trajectory_samples": 12,
                "joint_limit_safe": True,
                "violations": [],
            },
        ],
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_valid_s1_2_motion_probe_passes(tmp_path: Path) -> None:
    probe = tmp_path / "project-motion.json"
    _write(probe, _payload())

    result = audit_s1_2_motion_probe(probe)

    assert result.passed
    assert result.failures == ()


def test_failed_orientation_probe_is_reported(tmp_path: Path) -> None:
    payload = _payload()
    probes = payload["probes"]
    assert isinstance(probes, list)
    orientation = probes[1]
    assert isinstance(orientation, dict)
    orientation.clear()
    orientation.update(
        {
            "label": "orientation",
            "planner_id": "robotwin-curobo",
            "planning_time_s": 0.4,
            "succeeded": False,
            "failure": {"code": "optimization_failed"},
        }
    )
    probe = tmp_path / "project-motion.json"
    _write(probe, payload)

    result = audit_s1_2_motion_probe(probe)

    assert not result.passed
    assert any(gate.name == "orientation" for gate in result.failures)


def test_wrong_backend_class_is_not_accepted_as_curobo_evidence(tmp_path: Path) -> None:
    payload = _payload()
    payload["planner_class"] = "UnexpectedPlanner"
    probe = tmp_path / "project-motion.json"
    _write(probe, payload)

    result = audit_s1_2_motion_probe(probe)

    assert not result.passed
    assert any(gate.name == "planner-class" for gate in result.failures)
