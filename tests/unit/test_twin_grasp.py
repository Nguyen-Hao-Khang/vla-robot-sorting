"""Orientation-aware grasp planning (pure numpy, no SAPIEN/GPU)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/linux"))

import twin_grasp as tg  # noqa: E402
import twin_scene as ts  # noqa: E402

TABLE = 0.74
UP = (1.0, 0.0, 0.0, 0.0)
LYING_X = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # 90 deg about world x


def _q_axis(axis, angle_deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    h = math.radians(angle_deg) / 2
    return (math.cos(h), *(a * math.sin(h)))


def test_finger_axis_probe() -> None:
    down = _q_axis((0, 1, 0), 90)  # gripper x pointing down
    rot = tg.quat_to_mat(down)
    assert tg.finger_axis_from_vectors(rot[:, 1] * 0.03, down) == "y"
    assert tg.finger_axis_from_vectors(-rot[:, 2] * 0.03, down) == "z"
    assert tg.finger_axis_from_vectors(rot[:, 0] * 0.03, down) is None
    assert tg.finger_axis_from_vectors((0, 0, 0), down) is None


def test_support_function_matches_shapes() -> None:
    carton = ts.LEVELS[6].object("fresh milk carton")
    assert math.isclose(tg.support(carton, UP, (1, 0, 0)), carton.half_extents[0])
    assert math.isclose(tg.support(carton, UP, (0, 0, 1)), carton.half_height)
    assert math.isclose(tg.support(carton, LYING_X, (0, 1, 0)), carton.half_height)  # long axis
    bottle = ts.LEVELS[7].object("water bottle")
    assert math.isclose(tg.support(bottle, LYING_X, (0, 0, 1)), bottle.radius, abs_tol=1e-9)
    tetra = ts.LEVELS[3].object("tetrahedron")
    assert math.isclose(tg.support(tetra, UP, (0, 0, 1)), tetra.half_height)


def _assert_frame(frame, actor_q, finger_axis, expected_finger_world=None):
    m = np.asarray(frame, float)
    rot = m[:3, :3]
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-6) and math.isclose(np.linalg.det(rot), 1)
    up_world = tg.quat_to_mat(actor_q) @ rot[:, 1]
    assert up_world[2] > 0.99  # the gripper comes from above (approach = -column 1)
    f = tg.finger_world_axis(frame, actor_q, finger_axis)
    if expected_finger_world is not None:
        assert abs(abs(float(np.dot(f[:2], expected_finger_world))) - 1) < 1e-6
    return f


def test_upright_box_frames_follow_the_free_direction() -> None:
    box = ts.LEVELS[6].object("box 1")
    neighbour = ts.LEVELS[6].object("soy milk carton")
    near_left = [(neighbour, [-0.012, -0.054, TABLE + 0.02], LYING_X)]
    choice = tg.plan_grasp(box, [0.05, -0.05, TABLE + 0.025], UP, near_left, "y")
    assert choice.ok and abs(choice.finger_world[1]) > 0.99  # fingers along y, away from it
    assert len(choice.frames) == 2
    for frame in choice.frames:
        _assert_frame(frame, UP, "y", choice.finger_world)
    z_choice = tg.plan_grasp(box, [0.05, -0.05, TABLE + 0.025], UP, near_left, "z")
    for frame in z_choice.frames:
        _assert_frame(frame, UP, "z", z_choice.finger_world)
    boxed_in = [*near_left, (neighbour, [0.05, -0.105, TABLE + 0.05], UP)]
    refused = tg.plan_grasp(box, [0.05, -0.05, TABLE + 0.025], UP, boxed_in, "y")
    assert not refused.ok and "no room" in refused.reason


def test_lying_objects_are_grasped_across_their_short_side() -> None:
    carton = ts.LEVELS[6].object("soy milk carton")
    choice = tg.plan_grasp(carton, [0.0, -0.05, TABLE + 0.019], LYING_X, [], "y")
    assert choice.ok and choice.lying and math.isclose(choice.width, 2 * carton.half_extents[0])
    assert abs(choice.finger_world[0]) > 0.99  # across the 4.5 cm side, not the 10 cm length
    for frame in choice.frames:
        _assert_frame(frame, LYING_X, "y", choice.finger_world)
    assert math.isclose(choice.vertical_half, carton.half_extents[1], abs_tol=1e-9)
    bottle = ts.LEVELS[7].object("water bottle")
    lying = tg.plan_grasp(bottle, [0.0, -0.05, TABLE + 0.021], LYING_X, [], "y")
    assert lying.ok and lying.lying and abs(lying.finger_world[0]) > 0.99
    # the grasp point moves from the middle towards the straight body (away from the neck)
    assert lying.grasp_offset_world[1] > 0.01
    leaning = tg.plan_grasp(carton, [0, -0.05, 0.8], _q_axis((1, 0, 0), 40), [], "y")
    assert not leaning.ok and "edge" in leaning.reason


def test_prism_and_tetrahedron_grasps() -> None:
    prism = ts.LEVELS[3].object("triangular prism")
    choice = tg.plan_grasp(prism, [0, -0.05, TABLE + 0.03], UP, [], "y")
    assert choice.ok and math.isclose(choice.width, 1.5 * prism.radius)
    angle = math.degrees(math.atan2(choice.finger_world[1], choice.finger_world[0])) % 60
    assert math.isclose(angle, 30.0, abs_tol=1e-6)  # perpendicular to a side face
    lying = tg.plan_grasp(prism, [0, -0.05, 0.76], LYING_X, [], "y")
    assert not lying.ok and "too wide" in lying.reason  # 6 cm end-to-end > max opening
    tetra = ts.LEVELS[3].object("tetrahedron")
    upright = tg.plan_grasp(tetra, [0, -0.05, TABLE + tetra.half_height], UP, [], "y")
    assert upright.ok and upright.width < tg.GRIPPER_MAX_OPEN
    # low grasp: 30% of the height above the base face
    above_base = upright.vertical_half + upright.grasp_offset_world[2]
    assert math.isclose(above_base, tg.TETRA_GRASP_HEIGHT * 2 * tetra.half_height, abs_tol=1e-9)
    # resting on another face (rotated so a side face is down): still a valid grasp
    verts = np.asarray(ts.tetra_vertices(tetra.radius, tetra.half_height))
    n = np.cross(verts[1] - verts[0], verts[3] - verts[0])
    n /= np.linalg.norm(n)
    if np.dot(n, verts[0] - verts.mean(axis=0)) < 0:
        n = -n
    axis = np.cross(n, [0, 0, -1.0])
    angle = math.degrees(math.acos(float(np.dot(n, [0, 0, -1.0]))))
    q_side = _q_axis(axis, angle)
    side = tg.plan_grasp(tetra, [0, -0.05, 0.76], q_side, [], "y")
    assert side.ok and not side.lying and side.directions == 3  # one per base vertex
    for frame in side.frames:
        _assert_frame(frame, q_side, "y")


def test_deep_bin_walls_limit_the_finger_direction() -> None:
    carton = ts.LEVELS[6].object("fresh milk carton")
    bx, by = ts.DEEP_BIN.xy  # v7 log (bin at 0.16, -0.17): 6 mm right of the slot
    near_wall = [bx + 0.0463, by - 0.0017, TABLE + 0.054]
    choice = tg.plan_grasp(carton, near_wall, UP, [], "y", ts.DEEP_BIN)
    assert choice.ok and abs(choice.finger_world[1]) > 0.99  # along y, not into the side wall
    leaning_soy = (ts.LEVELS[6].object("soy milk carton"), [bx - 0.0364, by + 0.0143, 0.7935])
    with_soy = [(*leaning_soy, _q_axis((0, 1, 0), 40))]
    assert tg.plan_grasp(carton, near_wall, UP, with_soy, "y", ts.DEEP_BIN).ok


def test_unknown_finger_axis_requires_room_both_ways() -> None:
    box = ts.LEVELS[6].object("box 1")
    neighbour = (ts.LEVELS[6].object("soy milk carton"), [-0.012, -0.054, 0.76], LYING_X)
    assert not tg.plan_grasp(box, [0.05, -0.05, 0.765], UP, [neighbour], None).ok
    free = tg.plan_grasp(box, [0.05, -0.05, 0.765], UP, [], None)
    assert free.ok and len(free.frames) == 4  # the 4 usual yaws, as before v8
    assert free.frames == tg.plan_grasp(box, [0.05, -0.05, 0.765], UP, [], "y").frames


def test_tilt_changes_and_hold_vectors() -> None:
    before = {"a": list(UP), "b": list(UP)}
    after = {"a": list(_q_axis((1, 0, 0), 40)), "b": list(_q_axis((0, 0, 1), 90))}
    tipped = tg.unexpected_tilts(before, after, set())
    assert set(tipped) == {"a", "b"} and math.isclose(tipped["a"], 40, abs_tol=1e-6)
    assert tg.unexpected_tilts(before, after, {"a", "b"}) == {}
    down = _q_axis((0, 1, 0), 90)  # gripper x along world -z
    assert tg.hold_vector(down, "horizontal") == [1, 1, 1, 1, 0, 0]
    assert tg.hold_vector(down, "vertical") == [1, 1, 1, 0, 1, 1]
    assert tg.hold_vector(_q_axis((1, 1, 0), 45), "horizontal") is None


def test_free_objects_get_the_same_four_grasps_as_v7() -> None:
    """With nothing in the way the grasp set is exactly v7's (reachability unchanged)."""
    v7 = sorted(np.asarray(f, float)[:3, :3].round(6).tolist() for f in ts.contact_poses(0.0))
    for name in ("box 1", "fresh milk carton"):
        obj = ts.LEVELS[6].object(name)
        for axis in ("y", "z", None):
            choice = tg.plan_grasp(obj, [0, -0.05, 0.79], UP, [], axis)
            got = sorted(np.asarray(f, float)[:3, :3].round(6).tolist() for f in choice.frames)
            assert got == v7, (name, axis)
    cyl = ts.LEVELS[4].objects[0]
    choice = tg.plan_grasp(cyl, [0, -0.05, 0.77], UP, [], "y")
    got = sorted(np.asarray(f, float)[:3, :3].round(6).tolist() for f in choice.frames)
    assert got == v7  # diagonals are only a fallback


def test_tilted_and_leaning_grasps_need_the_obstacle_aware_planner() -> None:
    carton = ts.LEVELS[6].object("soy milk carton")
    lean = _q_axis((1, 0, 0), 40)
    refused = tg.plan_grasp(carton, [0, -0.05, 0.79], lean, [], "y")
    assert not refused.ok and "tilted approach" in refused.reason
    choice = tg.plan_grasp(carton, [0, -0.05, 0.79], lean, [], "y", allow_tilt=True)
    assert choice.ok and choice.frames and min(choice.frame_tilts) > tg.TOPDOWN_TOL_DEG
    for frame in choice.frames:
        m = np.asarray(frame, float)
        approach = tg.quat_to_mat(lean) @ m[:3, 1]
        assert approach[2] >= math.cos(math.radians(tg.MAX_TILT_DEG)) - 1e-9
    # an upright box: top-down frames first, tilted variants after them
    box = ts.LEVELS[6].object("box 1")
    up = tg.plan_grasp(box, [0, -0.05, 0.765], UP, [], "y", allow_tilt=True)
    tilts = up.frame_tilts
    assert tilts[:4] == [0.0] * 4 and max(tilts) > 25 and tilts == sorted(tilts, key=round)


def test_lying_prism_and_leaning_bottle() -> None:
    prism = ts.LEVELS[3].object("triangular prism")
    q_lying = _q_axis((1, 0, 0), 90)  # resting on its -y side face: a ridge points up
    assert np.allclose(tg.quat_to_mat(q_lying) @ [0, -1, 0], [0, 0, -1], atol=1e-9)
    choice = tg.plan_grasp(prism, [0, -0.05, 0.76], q_lying, [], "y")
    assert not choice.ok and "too wide" in choice.reason  # ends 6 cm apart
    bottle = ts.LEVELS[7].object("water bottle")
    leaning = tg.plan_grasp(bottle, [0, -0.05, 0.79], _q_axis((1, 0, 0), 60), [], "y")
    assert not leaning.ok  # 30 deg side approach needs the obstacle-aware planner
    tilted = tg.plan_grasp(
        bottle, [0, -0.05, 0.79], _q_axis((1, 0, 0), 60), [], "y", allow_tilt=True
    )
    assert tilted.ok


def test_finger_probe_from_displacements_and_contact_axis() -> None:
    down = _q_axis((0, 1, 0), 90)
    rot = tg.quat_to_mat(down)
    moves = [rot[:, 1] * 0.02, -rot[:, 1] * 0.02 + rot[:, 0] * 0.005]  # linkage: some x too
    assert tg.finger_axis_from_displacements(moves, down) == "y"
    assert tg.finger_axis_from_displacements([rot[:, 2] * 0.02], down) == "z"
    assert tg.finger_axis_from_displacements([], down) is None
    pts = [[0.02, 0.0, 0.8], [0.021, 0.003, 0.8], [-0.02, 0.001, 0.8]]
    axis = tg.contact_axis(pts)
    assert axis is not None and abs(axis[0]) > 0.99
    assert tg.contact_axis([[0.0, 0.0, 0.8], [0.002, 0.0, 0.8]]) is None  # one-sided
    assert math.isclose(tg.axis_angle_deg((1, 0), (0, -1)), 90)


def test_wrist_turn_offers_both_symmetric_yaws_smallest_turn_first() -> None:
    box = ts.LEVELS[6].object("box 1")
    down = _q_axis((0, 1, 0), 90)  # gripper pointing down, current wrist yaw
    plain = tg.plan_grasp(box, [0, -0.05, 0.765], UP, [], "y")
    ordered = tg.plan_grasp(box, [0, -0.05, 0.765], UP, [], "y", ref_quat=down)
    assert len(plain.frames) == 4 and len(ordered.frames) == 4  # 180 deg each way: all yaws
    turns = [abs(tg.wrist_turn_deg(f, UP, down)) for f in ordered.frames]
    assert turns == sorted(turns) and turns[0] < 1 and 90 < turns[-1] <= 180 + 1e-6
    assert sorted(map(str, plain.frames)) == sorted(map(str, ordered.frames))


def test_exact_grasp_poses_are_vertical_for_a_top_down_frame() -> None:
    box = ts.LEVELS[6].object("box 1")
    yawed = _q_axis((0, 0, 1), 30)
    choice = tg.plan_grasp(box, [0.06, -0.04, 0.765], yawed, [], "y")
    scale = [1.0, 1.0, 1.0]
    for frame in choice.frames:
        pre, grasp = tg.grasp_poses([0.06, -0.04, 0.765], yawed, frame, scale, 0.09)
        assert tg.approach_tilt_deg(pre) < 1e-6 and pre[3:] == grasp[3:]
        assert math.isclose(pre[2] - grasp[2], 0.09, abs_tol=1e-9)
        assert np.allclose(grasp[:2], [0.06, -0.04], atol=1e-9)
        r = tg.quat_to_mat(pre[3:7])
        assert np.allclose(r, tg.gripper_rotation(frame, yawed), atol=1e-9)
    q = tg.mat_to_quat(tg.quat_to_mat(_q_axis((1, 1, 0), 170)))
    assert np.allclose(tg.quat_to_mat(q), tg.quat_to_mat(_q_axis((1, 1, 0), 170)), atol=1e-9)


def test_tilted_variants_keep_the_finger_axis_horizontal() -> None:
    box = ts.LEVELS[6].object("box 1")
    choice = tg.plan_grasp(box, [0.06, -0.04, 0.765], UP, [], "y", allow_tilt=True)
    tilts = sorted({round(t) for t in choice.frame_tilts})
    assert tilts == [0, 20, 35, 45, 55]
    for frame in choice.frames:
        g = tg.gripper_rotation(frame, UP)
        assert abs(g[2, 1]) < 1e-6  # fingers (gripper y) stay horizontal: vertical grasp plane
    tetra = ts.LEVELS[3].object("tetrahedron")
    assert any(
        t > tg.TOPDOWN_TOL_DEG
        for t in tg.plan_grasp(tetra, [0.15, -0.04, 0.76], UP, [], "y", allow_tilt=True).frame_tilts
    )
