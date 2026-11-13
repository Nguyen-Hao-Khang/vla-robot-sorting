"""Digital twin scene levels and geometry (pure functions, no SAPIEN/GPU)."""

from __future__ import annotations

import importlib.util
import itertools
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("twin_scene", ROOT / "tools/twin/twin_scene.py")
ts = importlib.util.module_from_spec(spec)
sys.modules["twin_scene"] = ts
spec.loader.exec_module(ts)

TABLE = 0.74


def test_levels_cover_requested_difficulty_cases() -> None:
    assert sorted(ts.LEVELS) == [1, 2, 3, 4, 5, 6, 7]
    assert len(ts.LEVELS[1].objects) == 1
    assert {o.shape for o in ts.LEVELS[3].objects} == {"cube", "cylinder", "prism", "tetrahedron"}
    l4 = ts.LEVELS[4].objects
    assert {o.shape for o in l4} == {"cylinder"} and len({o.radius for o in l4}) == 3
    l5 = ts.LEVELS[5].objects
    assert len({o.color_name for o in l5}) == 2 and len({o.radius for o in l5}) == 2
    l6, l7 = ts.LEVELS[6].objects, ts.LEVELS[7].objects
    assert {o.shape for o in l6} == {"box"} and {o.shape for o in l7} == {"sphere", "bottle"}
    for objs in (l6, l7):  # look-alike pairs: same geometry, different kind
        for a, b in ((objs[0], objs[1]), (objs[2], objs[3])):
            assert (a.shape, a.radius, a.half_height) == (b.shape, b.radius, b.half_height)
            assert a.kind != b.kind and a.visual != b.visual and a.keywords and b.keywords
    assert {c.kind for c in ts.LEVELS[3].containers} == {"tray", "bin", "table"}
    for number in (3, 4, 5, 6, 7):  # no staging pad squeezed between the tray and the bin
        assert all(c.kind != "pad" for c in ts.LEVELS[number].containers)


def test_names_unique_and_tray_wide_shallow_bin_narrow_deep() -> None:
    for level in ts.LEVELS.values():
        names = level.object_names + level.destination_names
        assert len(set(names)) == len(names)
        assert len(level.objects) <= len(ts.OBJECT_SLOTS_X)
    tray, deep = ts.WIDE_TRAY, ts.DEEP_BIN
    tray_area = tray.inner_half[0] * tray.inner_half[1]
    bin_area = deep.inner_half[0] * deep.inner_half[1]
    assert tray.inner_half[0] > deep.inner_half[0] and bin_area < tray_area
    assert deep.wall_height >= 3 * tray.wall_height and deep.capacity == 2


def test_objects_fit_gripper_and_containers() -> None:
    for level in ts.LEVELS.values():
        for obj in level.objects:
            assert obj.grip_width <= 0.052, obj.name
            for c in level.containers:
                if c.kind not in ("pad", "table"):
                    assert obj.footprint < min(c.inner_half) + 0.005, (obj.name, c.name)


def _boxes(level):
    boxes = []
    for c in level.containers:
        if c.kind == "table":
            continue
        hx, hy = c.inner_half
        pad = 0 if c.kind == "pad" else ts.WALL_T
        boxes.append(
            (c.name, c.xy[0] - hx - pad, c.xy[0] + hx + pad, c.xy[1] - hy - pad, c.xy[1] + hy + pad)
        )
    for name, (x, y) in level.spawn_xy().items():
        r = level.object(name).footprint + ts.SPAWN_JITTER
        boxes.append((name, x - r, x + r, y - r, y + r))
    return boxes


def test_layout_has_no_overlaps_and_room_for_fingers() -> None:
    for n, level in ts.LEVELS.items():
        boxes = _boxes(level)
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                overlap = a[1] < b[2] and b[1] < a[2] and a[3] < b[4] and b[3] < a[4]
                assert not overlap, (n, a, b)


def test_objects_and_containers_are_spread_apart() -> None:
    xs = ts.OBJECT_SLOTS_X
    assert all(math.isclose(b - a, 0.12) for a, b in itertools.pairwise(xs))
    # free table between the object row and the far walls of the container row
    far_wall = ts.CONTAINER_ROW_Y + ts.WIDE_TRAY.inner_half[1] + ts.WALL_T
    assert (ts.OBJECT_ROW_Y - 0.025) - far_wall > 0.065
    # free table between the tray and the bin
    tray_right = ts.WIDE_TRAY.xy[0] + ts.WIDE_TRAY.inner_half[0] + ts.WALL_T
    bin_left = ts.DEEP_BIN.xy[0] - ts.DEEP_BIN.inner_half[0] - ts.WALL_T
    assert bin_left - tray_right > 0.13


def test_level_2_staging_pad_is_in_the_object_row() -> None:
    pad = ts.LEVELS[2].container("staging pad")
    assert pad.xy[1] == ts.OBJECT_ROW_Y and pad.xy[1] != ts.CONTAINER_ROW_Y


def test_grasp_point_raised_in_deep_bin_and_reachable() -> None:
    box = ts.LEVELS[6].objects[2]
    assert ts.grasp_fraction(box, None) == 0.0 and ts.grasp_fraction(box, ts.WIDE_TRAY) == 0.0
    frac = ts.grasp_fraction(box, ts.DEEP_BIN)
    grasp_z = box.half_height * (1 + frac)
    assert frac > 0 and grasp_z >= ts.DEEP_BIN.wall_height - ts.FINGER_DEPTH - 1e-9
    assert grasp_z <= 2 * box.half_height - ts.GRASP_TOP_MARGIN + 1e-9
    carton = ts.LEVELS[6].objects[0]
    assert ts.grasp_fraction(carton, ts.DEEP_BIN) == 0.0  # tall enough: grasp at the centre
    for n in (3, 4, 5, 6, 7):
        for obj in ts.LEVELS[n].objects:
            if obj.shape not in ("sphere", "tetrahedron"):
                assert ts.graspable_in(obj, ts.DEEP_BIN), obj.name


def test_tapered_or_short_objects_cannot_leave_deep_bin() -> None:
    l7 = {o.name: o for o in ts.LEVELS[7].objects}
    assert not ts.graspable_in(l7["tennis ball"], ts.DEEP_BIN)
    assert not ts.graspable_in(ts.LEVELS[3].objects[3], ts.DEEP_BIN)  # tetrahedron
    assert ts.graspable_in(l7["water bottle"], ts.DEEP_BIN)
    assert ts.graspable_in(l7["tennis ball"], ts.WIDE_TRAY)
    assert not ts.graspable_in(ts.LEVELS[2].objects[0], ts.DEEP_BIN)  # 4 cm cube, 7 cm walls


def test_release_height_keeps_fingertips_above_rim_and_drops_low() -> None:
    cube = ts.LEVELS[2].objects[0]
    floor = TABLE + ts.FLOOR_T
    gap = ts.release_gap(ts.DEEP_BIN, cube)
    assert gap + cube.half_height >= ts.DEEP_BIN.wall_height  # fingertips above the rim
    # place_actor stops PLACE_DIS above its target: the bottom ends exactly ``gap`` above the floor
    assert math.isclose(ts.release_z(ts.DEEP_BIN, cube, TABLE) + ts.PLACE_DIS, floor + gap)
    # short objects in a shallow tray / tall bottles in the deep bin: only MIN_DROP above the floor
    assert ts.release_gap(ts.WIDE_TRAY, ts.LEVELS[4].objects[2]) == ts.MIN_DROP
    bottle = ts.LEVELS[7].object("water bottle")
    assert ts.release_gap(ts.DEEP_BIN, bottle) <= 0.012  # was ~2.8 cm in v5 (bottle tipped)
    assert ts.release_gap(ts.DEEP_BIN, cube, grasp_frac=0.8) <= gap  # higher grasp, lower drop


def test_transit_waypoints_are_short_horizontal_hops() -> None:
    hops = ts.transit_waypoints((-0.15, -0.05), (0.20, -0.17))
    assert all(math.hypot(dx, dy) <= ts.TRANSIT_STEP + 1e-9 for dx, dy in hops)
    assert np.allclose(np.sum(hops, axis=0), (0.35, -0.12))
    assert len(ts.transit_waypoints((0, 0), (0, 0))) == 1


def test_grasp_blocker_flags_neighbours_too_close_for_the_fingers() -> None:
    level = ts.LEVELS[6]
    xyz = {n: [x, y, 0.8] for n, (x, y) in level.spawn_xy().items()}
    assert all(ts.grasp_blocker(level, n, xyz) is None for n in xyz)  # 10 cm row spacing
    # v5 log: the soy carton fell over at x=-0.016 and box 1 at x=0.05 was then grasped
    xyz["soy milk carton"] = [-0.0116, -0.0542, 0.76]
    tilts = {"soy milk carton": 90.0}
    name, gap = ts.grasp_blocker(level, "box 1", xyz, tilts)
    assert name == "soy milk carton" and gap < ts.GRASP_KEEPOUT
    assert ts.footprint_radius(level.object("soy milk carton"), tilted=True) == 0.05
    for number, lv in ts.LEVELS.items():  # jittered start layouts are never blocked
        for seed in range(10):
            spawn = ts.jittered_spawns(lv, random.Random(seed))
            pos = {n: [v[0], v[1], 0.8] for n, v in spawn.items()}
            assert all(ts.grasp_blocker(lv, n, pos) is None for n in pos), (number, seed)


def test_inside_location_and_occupants() -> None:
    level = ts.LEVELS[3]
    bx, by = ts.DEEP_BIN.xy
    assert ts.location_of(level, (bx + 0.06, by, TABLE + 0.03), TABLE) == "deep bin"
    assert ts.location_of(level, (bx + 0.12, by, TABLE + 0.03), TABLE) == "table"
    assert ts.location_of(level, (bx, by, TABLE + 0.4), TABLE) == "off table"
    xyz = {"cube": [bx, by, TABLE + 0.03], "cylinder": [0.0, -0.05, TABLE + 0.03]}
    assert ts.occupants(level, ts.DEEP_BIN, xyz, TABLE) == ["cube"]


def test_choose_slot_prefers_free_space() -> None:
    tray = ts.WIDE_TRAY
    first = ts.choose_slot(tray, [(0.5, 0.5)])
    assert first == (tray.xy[0] - 0.06, tray.xy[1])
    assert math.isclose(ts.choose_slot(tray, [first])[0], tray.xy[0] + 0.06)
    assert math.isclose(ts.choose_slot(ts.DEEP_BIN, [])[0], ts.DEEP_BIN.xy[0] - 0.04)


def test_align_correction_limits() -> None:
    assert ts.align_correction((0.0, 0.0), (0.0, 0.002)) is None
    dx, dy = ts.align_correction((0.0, -0.17), (0.0, -0.184))
    assert math.isclose(dx, 0.0) and math.isclose(dy, 0.014, abs_tol=1e-9)
    assert ts.align_correction((0.0, 0.0), (0.0, 0.08)) is None  # too large: do not trust it


def test_container_boxes_form_closed_walls() -> None:
    boxes = ts.container_boxes(ts.DEEP_BIN)
    assert len(boxes) == 5
    hx, _ = ts.DEEP_BIN.inner_half
    inner = sorted(abs(c[0]) - h[0] for c, h in boxes[1:3])
    assert all(math.isclose(v, hx) for v in inner)
    tops = [c[2] + h[2] for c, h in boxes[1:]]
    assert all(math.isclose(t, ts.FLOOR_T + ts.DEEP_BIN.wall_height) for t in tops)


def _check_closed_outward(verts, text, n_faces):
    faces = [
        tuple(int(i) - 1 for i in ln.split()[1:]) for ln in text.splitlines() if ln.startswith("f ")
    ]
    assert len(faces) == n_faces
    centroid = verts.mean(axis=0)
    edges: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        normal = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        assert np.dot(normal, (verts[a] + verts[b] + verts[c]) / 3 - centroid) > 0
        for e in ((a, b), (b, c), (c, a)):
            edges[e] = edges.get(e, 0) + 1
    assert all(edges.get((b, a)) == 1 for a, b in edges)


def test_prism_and_tetrahedron_meshes_closed_outward() -> None:
    pv = np.array(ts.prism_vertices(0.026, 0.03))
    _check_closed_outward(pv, ts.prism_obj_text(0.026, 0.03, "m"), 8)
    edge = 0.06
    tv = np.array(ts.tetra_vertices(ts.tetra_radius(edge), ts.tetra_half_height(edge)))
    _check_closed_outward(tv, ts.mesh_obj_text(list(map(tuple, tv)), ts.TETRA_FACES), 4)
    lengths = {
        round(float(np.linalg.norm(tv[i] - tv[j])), 6) for i in range(4) for j in range(i + 1, 4)
    }
    assert len(lengths) == 1 and math.isclose(lengths.pop(), edge, rel_tol=1e-6)


def test_grasp_data_and_contact_height() -> None:
    carton = ts.LEVELS[6].objects[0]
    data = ts.grasp_data(carton)
    assert data["scale"] == list(carton.half_extents)
    fp = np.array(data["functional_matrix"][0])
    assert math.isclose(fp[2, 3] * data["scale"][2], -carton.half_height)
    raised = ts.contact_poses(0.5)
    assert all(m[2][3] == 0.5 for m in raised) and len(raised) == 4
    assert all(np.allclose(np.array(m)[:3, 3], 0.0) for m in data["contact_points_pose"])


def test_jittered_spawns_stay_near_slots() -> None:
    level = ts.LEVELS[5]
    spawns = ts.jittered_spawns(level, random.Random(1))
    for name, (x, y, yaw) in spawns.items():
        sx, sy = level.spawn_xy()[name]
        assert (
            abs(x - sx) <= ts.SPAWN_JITTER
            and abs(y - sy) <= ts.SPAWN_JITTER
            and -math.pi <= yaw <= math.pi
        )
    w, x, y, z = ts.yaw_quat(math.pi / 2)
    assert math.isclose(w * w + z * z, 1.0) and x == y == 0.0


def test_descriptions_include_aliases() -> None:
    assert "khay sâu" in ts.DEEP_BIN.describe() and "khay nông" in ts.WIDE_TRAY.describe()
    assert "tall bin" in ts.DEEP_BIN.aliases
    assert "tứ diện" in ts.LEVELS[3].objects[3].describe()
    soy = ts.LEVELS[6].objects[1].describe()
    assert "SOY MILK" in soy and "hộp sữa đậu nành" in soy


def test_deep_bin_slots_leave_room_for_fingers() -> None:
    deep = ts.DEEP_BIN
    for ox, _ in deep.slots:
        assert deep.inner_half[0] - abs(ox) - ts.BOX_HALF[0] >= 0.02
    assert abs(deep.slots[0][0] - deep.slots[1][0]) - 2 * ts.BOX_HALF[0] >= 0.03


def test_table_destination_counts_free_objects_only() -> None:
    level = ts.LEVELS[6]
    xyz = {n: [x, y, TABLE + level.object(n).half_height] for n, (x, y) in level.spawn_xy().items()}
    table = level.container("table")
    assert ts.occupants(level, table, xyz, TABLE) == list(level.object_names)
    assert ts.free_room(level, table, xyz, TABLE) == 0
    xyz["box 1"] = [ts.DEEP_BIN.xy[0] - 0.04, ts.DEEP_BIN.xy[1], TABLE + 0.029]
    assert ts.free_room(level, table, xyz, TABLE) == 1
    assert ts.in_destination(level, ts.DEEP_BIN, xyz["box 1"], TABLE)
    assert not ts.in_destination(level, table, xyz["box 1"], TABLE)
    assert ts.location_of(level, (1.2, -0.09, 0.024), TABLE) == "off table"


def test_transport_clearance_covers_tallest_object_and_rims() -> None:
    level = ts.LEVELS[7]
    xyz = {n: [x, y, TABLE + level.object(n).half_height] for n, (x, y) in level.spawn_xy().items()}
    clearance = ts.transport_clearance(level, xyz, TABLE, carried="tennis ball")
    assert clearance >= 2 * ts.BOTTLE_HALF_H + ts.TRANSPORT_MARGIN - 1e-9  # standing bottles
    without_bottles = {k: v for k, v in xyz.items() if "bottle" not in k}
    assert (
        ts.transport_clearance(level, without_bottles, TABLE, "tennis ball")
        >= ts.DEEP_BIN.wall_height + ts.FLOOR_T
    )
    ball = level.object("tennis ball")
    dz = ts.lift_dz(ball, TABLE + ball.half_height, clearance, TABLE)
    assert abs((TABLE + ball.half_height + dz - ball.half_height) - (TABLE + clearance)) < 1e-9
    assert ts.lift_dz(ball, TABLE + 0.5, clearance, TABLE) == ts.MIN_LIFT_DZ


def test_tilt_toppled_and_unexpected_moves() -> None:
    import math as m

    assert ts.tilt_deg((1, 0, 0, 0)) == 0.0
    q90 = (m.cos(m.pi / 4), m.sin(m.pi / 4), 0, 0)  # 90 deg about x: lying on its side
    assert abs(ts.tilt_deg(q90) - 90) < 1e-6
    yaw = (m.cos(0.6), 0, 0, m.sin(0.6))
    assert ts.tilt_deg(yaw) < 1e-6
    carton = ts.LEVELS[6].objects[0]
    assert ts.toppled(carton, q90) and not ts.toppled(carton, yaw)
    assert not ts.toppled(ts.LEVELS[7].objects[0], q90)  # balls have no "up"
    before = {"a": [0, 0, 0.76], "b": [0.1, 0, 0.76], "c": [0.2, 0, 0.76]}
    after = {"a": [0.3, 0, 0.76], "b": [0.1, 0.005, 0.76], "c": [1.2, 0, 0.02]}
    moved = ts.unexpected_moves(before, after, allowed={"a"})
    assert list(moved) == ["c"] and moved["c"] > 1.0


def test_household_objects_have_realistic_masses() -> None:
    masses = {o.name: o.mass for o in ts.LEVELS[6].objects + ts.LEVELS[7].objects}
    assert masses["fresh milk carton"] == 0.10 and masses["water bottle"] == 0.12
    assert all(m >= 0.04 for m in masses.values())


def test_table_destination_builds_no_geometry() -> None:
    assert (
        ts.build_container(None, ts.TABLE_SPOT, TABLE) is None
    )  # objects must rest on the table itself


def test_table_destination_is_described_as_the_table() -> None:
    text = ts.TABLE_SPOT.describe()
    assert text.startswith("free spots on the table") and "bin" not in text.split("(")[0]


def test_tall_objects_are_grasped_higher_to_be_released_low_in_the_deep_bin() -> None:
    bottle = ts.LEVELS[7].object("water bottle")
    frac = ts.release_grasp_fraction(bottle, ts.DEEP_BIN)
    assert 0 < frac <= ts.PLACE_FRAC_MAX
    assert math.isclose(ts.release_gap(ts.DEEP_BIN, bottle, frac), ts.MIN_DROP)
    carton = ts.LEVELS[6].object("fresh milk carton")
    carton_frac = ts.release_grasp_fraction(carton, ts.DEEP_BIN)
    assert ts.release_gap(ts.DEEP_BIN, carton, carton_frac) < 0.02
    assert ts.release_grasp_fraction(ts.LEVELS[7].object("volleyball"), ts.DEEP_BIN) == 0.0
    assert ts.release_grasp_fraction(bottle, ts.WIDE_TRAY) == 0.0
    # the raised grasp stays on the straight body of the bottle (collision body top = 0.72 H)
    assert bottle.half_height * (1 + frac) < 0.72 * 2 * bottle.half_height


def test_vertical_extent_and_straight_descent() -> None:
    carton = ts.LEVELS[6].object("fresh milk carton")
    upright = (1.0, 0.0, 0.0, 0.0)
    lying = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # 90 deg about x
    assert math.isclose(ts.vertical_half_extent(carton, upright), carton.half_height)
    assert math.isclose(
        ts.vertical_half_extent(carton, lying), carton.half_extents[1], abs_tol=1e-9
    )
    bottle = ts.LEVELS[7].object("water bottle")
    assert math.isclose(ts.vertical_half_extent(bottle, lying), bottle.radius, abs_tol=1e-9)
    ball = ts.LEVELS[7].object("volleyball")
    assert ts.vertical_half_extent(ball, lying) == ball.radius
    # carried 20 cm up over the deep bin: descend until the bottom is at the release height
    frac = ts.release_grasp_fraction(bottle, ts.DEEP_BIN)
    z = TABLE + 0.20 + bottle.half_height
    dz = ts.descend_dz(bottle, z, upright, ts.DEEP_BIN, TABLE, frac)
    bottom = z + dz - bottle.half_height
    assert math.isclose(bottom, TABLE + ts.FLOOR_T + ts.MIN_DROP, abs_tol=1e-9)
    assert ts.descend_dz(bottle, TABLE + bottle.half_height, upright, ts.DEEP_BIN, TABLE) == 0.0
    assert ts.floor_top(ts.TABLE_SPOT, TABLE) == TABLE  # the table has no tray floor
    assert ts.safe_ee_z(TABLE, 0.10) == TABLE + 0.10 + ts.GRIPPER_REACH


def test_half_cylinder_mesh_closed_outward_and_sized() -> None:
    verts = ts.halfcyl_vertices(0.02, 0.015)
    text = ts.mesh_obj_text(verts, ts.halfcyl_faces())
    _check_closed_outward(np.array(verts), text, 2 * 12 + 2 * 12)
    arr = np.array(verts)
    assert np.allclose(arr.min(axis=0), (-0.02, -0.015, -0.01))
    assert np.allclose(arr.max(axis=0), (0.02, 0.015, 0.01))


def test_bottle_neck_and_cap_have_a_fixed_size() -> None:
    small = ts.bottle_collision(ts.BOTTLE_R, ts.BOTTLE_HALF_H)
    large = ts.bottle_collision(0.0325, 0.1025)
    assert math.isclose(small[2][1], 0.616 * ts.BOTTLE_R)  # unchanged for the toy bottle
    assert math.isclose(large[2][1], ts.BOTTLE_CAP_R)  # 3 cm cap on a 6.5 cm bottle
    assert math.isclose(large[2][0] + large[2][2], 0.1025)  # neck cylinder ends at the top


def test_bottles_get_a_grippier_surface() -> None:
    bottle = ts.LEVELS[7].object("water bottle")
    assert ts.grip_friction(bottle) > ts.grip_friction(ts.LEVELS[3].objects[0]) == 1.0
