"""Recorded seed1 joint5 failure must be rejected before physical execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vla_robot_sorting.contracts.motion import JointLimits, JointPath
from vla_robot_sorting.planning.pregrasp_joint_reserve import (
    PREGRASP_MINIMUM_MARGIN_RAD,
    audit_pregrasp_joint_reserve,
)

NAMES = tuple(f"joint{i}" for i in range(1, 7))
LO = (-2.4, 0.0, 0.0, -1.6, -1.7, -2.5)
HI = (2.4, 3.2, 4.0, 1.6, 1.7, 2.5)


def _case(peak_joint5: float, *, timed_peak: float | None = None):
    limits = JointLimits(
        joint_names=NAMES,
        lower_position_rad=LO,
        upper_position_rad=HI,
        max_velocity_rad_s=(2.0,) * 6,
        max_acceleration_rad_s2=(3.0,) * 6,
        max_jerk_rad_s3=(4.0,) * 6,
    )
    start = (0.0, 0.05, 0.0625, 0.0, 0.0, 0.0)
    end = (0.15, 2.19, 0.88, 1.295, peak_joint5, 0.01)
    path = JointPath(joint_names=NAMES, positions=(start, end), source="fixture")
    timed = SimpleNamespace(
        joint_names=NAMES,
        waypoints=(
            SimpleNamespace(positions=start),
            SimpleNamespace(
                positions=(*end[:4], timed_peak if timed_peak is not None else peak_joint5, end[5])
            ),
        ),
    )
    return path, timed, limits


def test_recorded_seed1_joint5_urdf_endpoint_is_rejected() -> None:
    path, trajectory, limits = _case(1.6999998092651367)
    report = audit_pregrasp_joint_reserve(path, trajectory, limits)
    assert not report["passed"]
    assert report["limiting_joint"] == "joint5"
    assert report["minimum_margin_rad"] < 0.000001
    assert report["required_margin_rad"] == PREGRASP_MINIMUM_MARGIN_RAD
    assert report["command_waypoints_modified"] is False
    assert report["physical_execution_performed"] is False


def test_seed0_joint5_and_all_other_interiors_are_accepted() -> None:
    path, trajectory, limits = _case(1.3983776569366455)
    assert audit_pregrasp_joint_reserve(path, trajectory, limits)["passed"]


def test_timed_waypoints_are_independently_screened() -> None:
    path, trajectory, limits = _case(1.3983776569366455, timed_peak=1.7)
    report = audit_pregrasp_joint_reserve(path, trajectory, limits)
    assert not report["passed"]
    assert report["limiting_waypoint"] == "retimed:1"


def test_joint4_boundary_is_not_mistaken_for_joint5_only_policy() -> None:
    path, trajectory, limits = _case(1.399)
    q = list(path.positions[1])
    q[3] = 1.6
    path = JointPath(joint_names=NAMES, positions=(path.positions[0], tuple(q)), source="fixture")
    report = audit_pregrasp_joint_reserve(path, trajectory, limits)
    assert not report["passed"]
    assert report["limiting_joint"] == "joint4"


def test_pregrasp_reserve_fails_closed_on_nan_order_and_missing_samples() -> None:
    path, trajectory, limits = _case(1.399)
    with pytest.raises(ValueError, match="nonfinite"):
        q = list(path.positions[1])
        q[4] = float("nan")
        bad = SimpleNamespace(joint_names=NAMES, positions=(path.positions[0], tuple(q)))
        audit_pregrasp_joint_reserve(bad, trajectory, limits)
    with pytest.raises(ValueError, match="order"):
        bad = SimpleNamespace(joint_names=NAMES[::-1], positions=path.positions)
        audit_pregrasp_joint_reserve(bad, trajectory, limits)
    with pytest.raises(ValueError, match="waypoints"):
        bad = SimpleNamespace(joint_names=NAMES, positions=())
        audit_pregrasp_joint_reserve(bad, trajectory, limits)
