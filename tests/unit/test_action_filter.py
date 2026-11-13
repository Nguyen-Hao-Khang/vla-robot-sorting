"""Runtime safety filter (src/vla_robot_sorting/safety/action_filter.py, ADR 0004)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from vla_robot_sorting.safety import ActionSafetyFilter, SafetyConfig, Status, panthera

ROOT = Path(__file__).resolve().parents[2]
LOWER = tuple(lo for lo, _ in panthera.LIMITS)
UPPER = tuple(hi for _, hi in panthera.LIMITS)
Q0 = (0.0, 1.4, 0.4, 0.6, 0.0, 0.0)  # tool 39 cm out, 10 cm above the table


def make(**overrides) -> ActionSafetyFilter:
    params = {"lower": LOWER, "upper": UPPER, "gripper_range": (0.0, 1.0)}
    params.update(overrides)
    return ActionSafetyFilter(SafetyConfig(**params))


def fast(**overrides) -> ActionSafetyFilter:
    """Rates high enough that only the rule under test can act."""
    return make(max_velocity=100.0, max_acceleration=1e4, reject_step=3.0, **overrides)


def test_small_step_passes_unchanged() -> None:
    f = fast()
    target = (*[q + 0.01 for q in Q0], 0.5)
    d = f.filter(target, Q0, 0.0)
    assert d.status is Status.PASS and d.action == pytest.approx(target)
    assert f.interventions == 0


def test_joint_limits_clamp_near_and_reject_far() -> None:
    f = fast()
    near = (*Q0[:5], 2.48, 0.5)  # joint 6 inside the margin band
    d = f.filter(near, Q0)
    assert d.status is Status.LIMITED and d.action[5] == pytest.approx(2.45)
    far = (*Q0[:5], 2.9, 0.5)
    d = f.filter(far, Q0)
    assert d.status is Status.REJECTED and d.action is None
    assert "joint 6" in d.reasons[0]


def test_gripper_is_clamped_and_malformed_actions_rejected() -> None:
    f = fast()
    d = f.filter((*Q0, 1.7), Q0)
    assert d.status is Status.LIMITED and d.action[-1] == 1.0
    assert f.filter((*Q0,), Q0).status is Status.REJECTED  # gripper missing
    assert f.filter((*Q0[:5], math.nan, 0.5), Q0).status is Status.REJECTED


def test_large_jump_is_rejected() -> None:
    f = make()
    target = (Q0[0] + 0.5, *Q0[1:], 0.5)
    d = f.filter(target, Q0)
    assert d.status is Status.REJECTED and "jump" in d.reasons[0]


def test_velocity_and_acceleration_are_limited() -> None:
    f = make(max_velocity=1.0, max_acceleration=6.0, control_dt=0.1)
    target = (Q0[0] + 0.3, *Q0[1:], 0.5)
    d = f.filter(target, Q0)
    # from rest the speed may rise by 6 rad/s^2 * 0.1 s = 0.6 rad/s, i.e. 0.06 rad this step
    assert d.status is Status.LIMITED and d.action[0] == pytest.approx(0.06)
    q1 = d.action[:6]
    d = f.filter(target, q1)
    # next step: speed capped at 1.0 rad/s -> 0.1 rad
    assert d.action[0] == pytest.approx(0.06 + 0.1)


def test_tool_below_the_table_is_rejected() -> None:
    f = fast()
    below = (0.0, 1.8, 0.4, 0.6, 0.0, 0.0)
    assert panthera.tool_position(below)[2] < 0
    d = f.filter((*below, 0.5), Q0)
    assert d.status is Status.REJECTED and "tool z" in d.reasons[0]


def test_repeated_rejections_latch_a_stop_until_reset() -> None:
    f = fast(max_consecutive_rejections=3)
    bad = (*Q0[:5], 2.9, 0.5)
    statuses = [f.filter(bad, Q0).status for _ in range(3)]
    assert statuses == [Status.REJECTED, Status.REJECTED, Status.STOPPED]
    assert f.stopped and f.filter((*Q0, 0.5), Q0).status is Status.STOPPED
    f.reset()
    assert f.filter((*Q0, 0.5), Q0).status is Status.PASS


def test_a_good_action_resets_the_rejection_count() -> None:
    f = fast(max_consecutive_rejections=2)
    bad = (*Q0[:5], 2.9, 0.5)
    assert f.filter(bad, Q0).status is Status.REJECTED
    assert f.filter((*Q0, 0.5), Q0).status is Status.PASS
    assert f.filter(bad, Q0).status is Status.REJECTED  # not a stop


def test_watchdog_and_emergency_stop() -> None:
    f = fast(watchdog_timeout_s=0.5)
    assert f.check_watchdog(10.0) is None  # nothing sent yet
    f.filter((*Q0, 0.5), Q0, time_s=1.0)
    assert f.check_watchdog(1.4) is None
    d = f.check_watchdog(1.6)
    assert d is not None and d.status is Status.STOPPED and f.stopped
    f.reset()
    f.emergency_stop("operator", 2.0)
    assert f.filter((*Q0, 0.5), Q0, 2.1).status is Status.STOPPED
    assert f.counts()["stopped"] == 3  # watchdog, emergency stop, refused action


def test_config_profiles_load_and_validate(tmp_path) -> None:
    for profile in ("simulation", "real"):
        cfg = SafetyConfig.load(ROOT / "configs" / "safety.json", profile)
        assert cfg.dof == 6 and cfg.gripper_range == (0.0, 1.0)
        # the home posture (all zeros) must be accepted, not rejected
        home = (0.0,) * 6
        assert ActionSafetyFilter(cfg).filter((*home, 0.0), home).status is not Status.REJECTED
    with pytest.raises(ValueError):
        SafetyConfig(lower=(0.0,), upper=(0.05,))
    with pytest.raises(ValueError):
        SafetyConfig(lower=(0.0,), upper=(1.0,), workspace_min=(1, 0, 0), workspace_max=(0, 1, 1))


def test_simulation_profile_admits_smooth_expert_like_motion() -> None:
    cfg = SafetyConfig.load(ROOT / "configs" / "safety.json", "simulation")
    f = ActionSafetyFilter(cfg)
    q = list(Q0)
    # 1 s of joint 1 speeding up to 1.2 rad/s at 2 rad/s^2, like the recorded expert
    v = 0.0
    for k in range(15):
        v = min(1.2, v + 2.0 * cfg.control_dt)
        target = [q[0] + v * cfg.control_dt, *q[1:]]
        d = f.filter((*target, 0.0), q, k * cfg.control_dt)
        assert d.status is Status.PASS, d.reasons
        q = list(d.action[:6])
