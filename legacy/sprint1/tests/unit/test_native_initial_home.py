"""Synthetic setup contracts; no SAPIEN task or native acceptance runs here."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vla_robot_sorting.planning.manipulation_recovery import RecoveryContractError
from vla_robot_sorting.planning.native_initial_home import (
    JOINT2_INTERIOR_HOME_RAD,
    JOINT3_INTERIOR_HOME_RAD,
    MIN_MEASURED_JOINT2_MARGIN_RAD,
    MIN_MEASURED_JOINT3_MARGIN_RAD,
    audit_native_initial_home,
    initialize_native_task_with_home,
    prepare_native_initial_home,
)

ARM = [f"joint{i}" for i in range(1, 7)]
NAMES = [*ARM, "gripper_L_joint", "gripper_R_joint"]


def _config():
    return {
        "homestate": [[0.0] * 6, [0.0] * 6],
        "arm_joints_name": [ARM.copy(), ARM.copy()],
        "urdf_path": "./panthera_6dof.urdf",
    }


def _args():
    return {
        "task_name": "move_pillbottle_pad",
        "arm_mode": "single",
        "robot_type": "panthera-6dof",
        "left_embodiment_config": _config(),
        "right_embodiment_config": _config(),
    }


def _entity(q):
    return SimpleNamespace(
        get_active_joints=lambda: [SimpleNamespace(get_name=lambda n=n: n) for n in NAMES],
        get_qpos=lambda: q.copy(),
    )


def test_interior_home_is_configured_before_setup_without_touching_input():
    original = _args()
    frozen = deepcopy(original)
    prepared, policy = prepare_native_initial_home(original)
    assert original == frozen
    assert prepared["left_embodiment_config"]["homestate"][0][1] == JOINT2_INTERIOR_HOME_RAD
    assert prepared["right_embodiment_config"]["homestate"][0][1] == JOINT2_INTERIOR_HOME_RAD
    assert prepared["left_embodiment_config"]["homestate"][0][2] == JOINT3_INTERIOR_HOME_RAD
    assert prepared["right_embodiment_config"]["homestate"][1][2] == JOINT3_INTERIOR_HOME_RAD
    assert prepared["left_embodiment_config"]["homestate"][0] == [
        0.0,
        JOINT2_INTERIOR_HOME_RAD,
        JOINT3_INTERIOR_HOME_RAD,
        0.0,
        0.0,
        0.0,
    ]
    assert policy["joint2_lower_rad"] == 0.0
    assert policy["joint2_upper_rad"] == 3.2
    assert policy["minimum_measured_joint2_margin_rad"] == MIN_MEASURED_JOINT2_MARGIN_RAD
    assert policy["joint3_lower_rad"] == 0.0
    assert policy["joint3_upper_rad"] == 4.0
    assert policy["joint3_configured_target_rad"] == 0.0625
    assert policy["minimum_measured_joint3_margin_rad"] == MIN_MEASURED_JOINT3_MARGIN_RAD
    assert policy["schema"] == "s1m.native-initial-home.v2"
    assert policy["controller_eligible"] is False


@pytest.mark.parametrize(
    "alter",
    [
        lambda a: a.update(arm_mode="dual"),
        lambda a: a.update(robot_type="other"),
        lambda a: a["left_embodiment_config"]["homestate"][0].__setitem__(1, 0.1),
        lambda a: a["left_embodiment_config"]["homestate"][0].__setitem__(2, 0.0625),
        lambda a: a["left_embodiment_config"].update(arm_joints_name=[ARM[::-1], ARM]),
        lambda a: a["left_embodiment_config"].update(urdf_path="./other.urdf"),
    ],
)
def test_unpinned_initial_home_configuration_refused(alter):
    args = _args()
    alter(args)
    with pytest.raises(RecoveryContractError):
        prepare_native_initial_home(args)


def test_measured_initial_state_has_exact_bounds_and_positive_joint2_joint3_reserves():
    q = [0.0, 0.049, 0.061, 0.0, 0.0, 0.0, 0.039, -0.039]
    unchanged = q.copy()
    report = audit_native_initial_home(_entity(q))
    assert report["passed"] is True
    assert report["measured_full_qpos"] == q
    assert report["joint2_margin_above_lower_rad"] == 0.049
    assert report["joint3_margin_above_lower_rad"] == 0.061
    assert report["joint3_reserve_passed"] is True
    assert report["schema"] == "s1m.native-initial-state.v2"
    assert report["native_acceptance"] is False
    assert q == unchanged


@pytest.mark.parametrize(
    "q2,q4",
    [
        (-2.52265097699933e-10, 0.0),
        (0.0, 0.0),
        (0.004, 0.0),
        (0.05, 1.6000008583068848),
    ],
)
def test_unsettled_or_out_of_bounds_initial_state_fails_closed(q2, q4):
    q = [0.0, q2, 0.061, q4, 0.0, 0.0, 0.039, -0.039]
    report = audit_native_initial_home(_entity(q))
    assert report["passed"] is False
    assert report["controller_eligible"] is False
    assert report["native_acceptance"] is False


@pytest.mark.parametrize(
    "q3",
    [-5.729727377001836e-07, 0.0, 2.0528585082502104e-05, 0.004, 4.00000001, float("nan")],
)
def test_joint3_negative_near_limit_or_nonfinite_measured_qpos_refused(q3):
    q = [0.0, 0.04971541091799736, q3, 0.0, 0.0, 0.0, 0.039, -0.039]
    if not np.isfinite(q3):
        with pytest.raises(RecoveryContractError):
            audit_native_initial_home(_entity(q))
    else:
        report = audit_native_initial_home(_entity(q))
        assert report["passed"] is False
        assert report["joint3_reserve_passed"] is False or not report["exact_urdf_bounds_passed"]
        assert report["controller_eligible"] is False
        assert report["native_acceptance"] is False


def test_joint3_measured_margin_requires_nontrivial_reserve():
    q = [0.0, 0.049, MIN_MEASURED_JOINT3_MARGIN_RAD + 0.0001, 0.0, 0.0, 0.0, 0.039, -0.039]
    assert audit_native_initial_home(_entity(q))["passed"] is True
    q[2] = MIN_MEASURED_JOINT3_MARGIN_RAD - 1e-6
    report = audit_native_initial_home(_entity(q))
    assert report["exact_urdf_bounds_passed"] is True
    assert report["joint2_reserve_passed"] is True
    assert report["joint3_reserve_passed"] is False
    assert report["passed"] is False


def test_wrong_joint_order_and_nonfinite_qpos_fail_closed():
    bad = _entity([0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.039, -0.039])
    bad.get_active_joints = lambda: [SimpleNamespace(get_name=lambda n=n: n) for n in NAMES[::-1]]
    with pytest.raises(RecoveryContractError):
        audit_native_initial_home(bad)
    with pytest.raises(RecoveryContractError):
        audit_native_initial_home(_entity([0.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.039, -0.039]))


def test_measured_native_float32_qpos_is_accepted_without_rounding():
    q = np.asarray([0.0, 0.05, 0.0625, 0.0, 0.0, 0.0, 0.039, -0.039], dtype=np.float32)
    report = audit_native_initial_home(_entity(q))
    assert report["passed"] is True
    assert report["measured_full_qpos"][1] == float(q[1])
    assert report["measured_full_qpos"][2] == float(q[2])


def test_setup_receives_interior_target_before_physics_and_records_measured_qpos():
    calls = []

    class Task:
        def setup_demo(self, **kwargs):
            calls.append(("setup", tuple(kwargs["left_embodiment_config"]["homestate"][0][1:3])))
            self.robot = SimpleNamespace(
                left_entity=_entity([0.0, 0.049, 0.061, 0.0, 0.0, 0.0, 0.039, -0.039])
            )

    task = Task()
    original = _args()
    prepared, report = initialize_native_task_with_home(task, original, seed=2)
    assert calls == [("setup", (JOINT2_INTERIOR_HOME_RAD, JOINT3_INTERIOR_HOME_RAD))]
    assert report["measured"]["passed"] is True
    assert prepared["left_embodiment_config"]["homestate"][0][1] == JOINT2_INTERIOR_HOME_RAD
    assert prepared["left_embodiment_config"]["homestate"][0][2] == JOINT3_INTERIOR_HOME_RAD
    assert original["left_embodiment_config"]["homestate"][0][1:3] == [0.0, 0.0]


def test_upstream_setup_that_keeps_joint3_near_lower_bound_is_not_authorized():
    class NonsettlingTask:
        def setup_demo(self, **kwargs):
            assert kwargs["left_embodiment_config"]["homestate"][0][2] == JOINT3_INTERIOR_HOME_RAD
            self.robot = SimpleNamespace(
                left_entity=_entity(
                    [0.0, 0.04971541091799736, 2.0528585082502104e-05, 0.0, 0.0, 0.0, 0.039, -0.039]
                )
            )

    _, evidence = initialize_native_task_with_home(NonsettlingTask(), _args(), seed=0)
    assert evidence["policy"]["controller_eligible"] is False
    assert evidence["measured"]["exact_urdf_bounds_passed"] is True
    assert evidence["measured"]["joint3_reserve_passed"] is False
    assert evidence["measured"]["passed"] is False
    assert evidence["measured"]["native_acceptance"] is False


@pytest.mark.parametrize(
    "relative,first_native_query",
    [
        (
            "scripts/linux/probe_panthera_plan_only_recovery.py",
            "backend = NativeMPlibDiagnosticBackend(task, helpers)",
        ),
        (
            "scripts/linux/probe_panthera_project_manipulation.py",
            'payload["mplib_world_initial"] = helpers.sync_scaled_target_into_mplib_world(task)',
        ),
    ],
)
def test_both_native_entrypoints_gate_initial_state_before_planning(relative, first_native_query):
    project = Path(__file__).resolve().parents[2]
    source = (project / relative).read_text(encoding="utf-8")
    setup = source.index("initialize_native_task_with_home(", source.index("def main()"))
    gate = source.index('["initial_home"]["measured"]["passed"]', setup)
    first_query = source.index(first_native_query, gate)
    assert setup < gate < first_query
    assert "task.setup_demo(" not in source
