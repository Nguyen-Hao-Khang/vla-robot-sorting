from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "linux"))

import twin_grasp as tg
import twin_kinematics as tk


def _approach(xy, tilt_deg):
    u = np.array([xy[0] - tk.BASE_XYZ[0], xy[1] - tk.BASE_XYZ[1]])
    u /= np.linalg.norm(u)
    t = math.radians(tilt_deg)
    x = np.array([math.sin(t) * u[0], math.sin(t) * u[1], -math.cos(t)])
    y = np.array([-u[1], u[0], 0.0])
    return np.column_stack([x, y, np.cross(x, y)])


def test_home_pose_matches_the_urdf_chain() -> None:
    p, r, _, _ = tk.fk([0.0] * 6)
    assert np.allclose(p, [0.0, -0.203, 0.919], atol=1e-3)
    assert np.allclose(r[:, 0], [0.0, 1.0, 0.0], atol=1e-9)  # gripper points forward at home


def test_ik_round_trip() -> None:
    rng = np.random.default_rng(3)
    solved = 0
    for _ in range(12):
        q = np.array([rng.uniform(lo + 0.2, hi - 0.2) for lo, hi in tk.LIMITS])
        p, r, _, _ = tk.fk(q)
        sol = tk.ik(p, r, seed=q + rng.normal(0, 0.15, 6))
        if sol is not None:
            p2, r2, _, _ = tk.fk(sol)
            assert np.allclose(p2, p, atol=3e-3) and np.allclose(r2, r, atol=0.05)
            solved += 1
    assert solved >= 10


def test_vertical_gripper_cannot_lift_but_a_tilted_one_can() -> None:
    xy = (0.18, -0.04)  # far object slot; box grasp: tool 12 cm above the 0.765 m contact
    assert tk.ik([*xy, 0.885], _approach(xy, 0.0)) is None
    for z in (0.885, 0.975, 1.035):  # grasp, pre-grasp, lift
        assert tk.ik([*xy, z], _approach(xy, 45.0)) is not None
    assert tk.ik([0.0, -0.10, 1.10], _approach((0.0, -0.10), 0.0)) is None


def test_reachable_chain_uses_quaternions() -> None:
    q = [0.2, 1.3, 1.8, 0.5, 0.1, 0.3]  # mid-range (home is at the joint 2/3 limits)
    p, r, _, _ = tk.fk(q)
    quat = tg.mat_to_quat(r)
    assert tk.reachable_chain([[*p, *quat]])
    assert not tk.reachable_chain([[0.0, 0.5, 1.5, *quat]])


def _first_feasible(level, xyz, name, dest_name, slot, holder=None, up=(1.0, 0.0, 0.0, 0.0)):
    import twin_console as twin
    import twin_scene as ts

    table = 0.74
    obj = level.object(name)
    container = level.container(holder) if holder else None
    frac = ts.grasp_fraction(obj, container) if container else 0.0
    up = list(up)
    choice = tg.plan_grasp(obj, xyz[name], up, [], "y", container, frac, allow_tilt=True)
    assert choice.ok, choice.reason
    clearance = ts.transport_clearance(level, xyz, table, name)
    safe_z = ts.safe_ee_z(table, ts.transport_clearance(level, xyz, table, ""))
    min_lift = twin.minimal_lift(level, xyz, name, table, {holder} if holder else set(), slot)
    dest = level.container(dest_name)
    for i, frame in enumerate(choice.frames):
        pre, target = tg.grasp_poses(xyz[name], up, frame, [1.0, 1.0, 1.0], 0.09)
        plan = twin.plan_pick(
            pre,
            target,
            xyz[name],
            slot,
            ts.floor_top(dest, table) + choice.vertical_half + 0.01,
            ts.lift_dz(obj, xyz[name][2], clearance, table),
            min_lift,
            safe_z,
        )
        if plan is not None:
            return choice.frame_tilts[i], plan
    return None


def test_every_level6_pick_has_a_kinematically_feasible_grasp() -> None:
    import twin_scene as ts

    level = ts.LEVELS[6]
    xyz = {n: [x, y, 0.74 + level.object(n).half_height] for n, (x, y) in level.spawn_xy().items()}
    for name in xyz:
        for dest_name in ("wide tray", "deep bin"):
            dest = level.container(dest_name)
            for sx, sy in dest.slots:
                found = _first_feasible(
                    level, xyz, name, dest_name, (dest.xy[0] + sx, dest.xy[1] + sy)
                )
                assert found is not None, (name, dest_name, sx)
                tilt, _plan = found
                assert 20 <= tilt <= 55  # vertical cannot lift on this arm; tilted can


def test_swaps_out_of_the_containers_are_feasible() -> None:
    import twin_scene as ts

    level = ts.LEVELS[6]
    xyz = {n: [x, y, 0.74 + level.object(n).half_height] for n, (x, y) in level.spawn_xy().items()}
    cases = [
        ("fresh milk carton", "deep bin", (0.04, 0.0), "table", (-0.18, -0.04)),
        ("soy milk carton", "deep bin", (-0.04, 0.0), "wide tray", (-0.11, -0.20)),
        ("box 1", "wide tray", (-0.06, 0.0), "deep bin", (0.15, -0.20)),
        ("box 2", "wide tray", (0.06, 0.0), "table", (0.18, -0.04)),
    ]
    for name, holder, (sx, sy), dest_name, slot in cases:
        c = level.container(holder)
        moved = dict(xyz)
        moved[name] = [
            c.xy[0] + sx,
            c.xy[1] + sy,
            ts.floor_top(c, 0.74) + level.object(name).half_height,
        ]
        # an object placed by the twin was turned about the vertical so the tilted approach
        # leans away from the robot at that slot: its faces are aligned with that direction
        lean = math.atan2(moved[name][1] - tk.BASE_XYZ[1], moved[name][0] - tk.BASE_XYZ[0])
        yaw = lean - math.pi / 2
        q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        found = _first_feasible(level, moved, name, dest_name, slot, holder, q)
        assert found is not None, (name, holder, dest_name)


def test_yaw_to_lean_turns_only_about_the_vertical() -> None:
    xy = (0.06, -0.04)
    r = _approach(xy, 45.0)
    q = tg.mat_to_quat(r)
    turn, q2 = tg.yaw_to_lean(q, (0.23 - tk.BASE_XYZ[0], -0.20 - tk.BASE_XYZ[1]))
    r2 = tk.quat_matrix(q2)
    assert abs(r2[2, 0] - r[2, 0]) < 1e-9  # same tilt from vertical
    lean = r2[:2, 0] / np.linalg.norm(r2[:2, 0])
    u = np.array([0.23, 0.25]) / np.hypot(0.23, 0.25)
    assert float(lean @ u) > 0.9999 and 30 < abs(turn) < 60
    vertical = tg.mat_to_quat(_approach(xy, 0.0))
    assert tg.yaw_to_lean(vertical, (1.0, 0.0))[0] == 0.0


def test_lift_options_and_minimal_lift() -> None:
    import twin_console as twin
    import twin_scene as ts

    opts = twin.lift_options(0.15, 0.06)
    assert opts[0] == 0.15 and opts[-1] == 0.06 and opts == sorted(opts, reverse=True)
    level = ts.LEVELS[6]
    xyz = {n: [x, y, 0.74 + level.object(n).half_height] for n, (x, y) in level.spawn_xy().items()}
    # box 1 next to the soy carton (12 cm away is outside the 12 cm radius): the plain 5 cm
    assert twin.minimal_lift(level, xyz, "box 1", 0.74, set()) == ts.MIN_LIFT_DZ
    bin_ = level.container("deep bin")
    xyz["box 1"] = [0.15, -0.20, ts.floor_top(bin_, 0.74) + 0.025]
    xyz["fresh milk carton"] = [0.23, -0.20, ts.floor_top(bin_, 0.74) + 0.05]
    towards = twin.minimal_lift(level, xyz, "box 1", 0.74, {"deep bin"}, (0.30, -0.20))
    away = twin.minimal_lift(level, xyz, "box 1", 0.74, {"deep bin"}, (-0.23, -0.20))
    assert towards > away  # the tall carton only matters if the transit passes over it
    assert abs(away - (0.74 + 0.07 + 0.02 - (ts.floor_top(bin_, 0.74)))) < 1e-9
