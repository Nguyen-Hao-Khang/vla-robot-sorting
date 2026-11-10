"""Camera geometry and VLM detection scoring (pure numpy, no SAPIEN/GPU)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/twin"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/twin/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geo = _load("twin_geometry")
per = _load("twin_perception")
TABLE = 0.74


def test_config_file_matches_loader_and_defaults(tmp_path) -> None:
    spec = geo.load_camera_spec()
    cfg = json.loads(geo.CAMERA_CONFIG.read_text())
    assert list(spec.position) == cfg["workspace"]["position"] and spec.width == 640
    assert (
        geo.load_camera_spec(tmp_path / "missing.json").fovy_deg
        == geo.DEFAULT_WORKSPACE["fovy_deg"]
    )


def test_look_at_rotation_is_orthonormal_and_points_at_target() -> None:
    spec = geo.load_camera_spec()
    rot = spec.rotation
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-9) and math.isclose(np.linalg.det(rot), 1.0)
    u, v = geo.project(spec, spec.look_at)
    assert math.isclose(u, spec.width / 2, abs_tol=1e-6) and math.isclose(
        v, spec.height / 2, abs_tol=1e-6
    )


def test_quaternion_round_trip() -> None:
    rot = geo.load_camera_spec().rotation
    w, x, y, z = geo.mat_to_quat(rot)
    back = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    assert np.allclose(back, rot, atol=1e-9) and w >= 0


def test_straight_down_camera_is_handled() -> None:
    rot = geo.look_at_rotation((0, 0, 1.5), (0, 0, 0.74))
    assert np.allclose(rot[:, 0], [0, 0, -1])


def test_projection_and_table_ray_are_inverse() -> None:
    spec = geo.load_camera_spec()
    for point in (
        (0.1, -0.05, TABLE + 0.025),
        (-0.15, -0.17, TABLE + 0.025),
        (0.0, 0.0, TABLE + 0.025),
    ):
        u, v = geo.project(spec, point)
        assert 0 <= u <= spec.width and 0 <= v <= spec.height
        x, y = geo.pixel_to_plane(spec, u, v, point[2])
        assert math.isclose(x, point[0], abs_tol=1e-9) and math.isclose(y, point[1], abs_tol=1e-9)
    assert geo.project(spec, (0.0, 1.0, 1.3)) is None  # behind the camera


def _box_for(spec, xy, half_px=12):
    u, v = geo.project(spec, (xy[0], xy[1], TABLE + per.PLANE_ABOVE_TABLE_M))
    return [
        round((v - half_px) / spec.height * 1000),
        round((u - half_px) / spec.width * 1000),
        round((v + half_px) / spec.height * 1000),
        round((u + half_px) / spec.width * 1000),
    ]


def test_detections_project_back_onto_the_table() -> None:
    spec = geo.load_camera_spec()
    payload = {
        "objects": [
            {
                "label": "red cube",
                "color": "red",
                "shape": "cube",
                "box_2d": _box_for(spec, (-0.15, -0.05)),
            },
            {
                "label": "green cylinder",
                "color": "green",
                "shape": "cylinder",
                "box_2d": _box_for(spec, (0.05, -0.05)),
            },
            {"label": "broken", "color": "x", "shape": "y", "box_2d": [500, 500, 400, 600]},
        ]
    }
    assert per.validate_detections(payload) is None
    dets = per.parse_detections(payload, spec, TABLE)
    assert len(dets) == 2
    assert math.dist(dets[0].xy_est, (-0.15, -0.05)) < 0.005


def test_validation_rejects_bad_boxes() -> None:
    assert "objects" in per.validate_detections({})
    assert "box_2d" in per.validate_detections({"objects": [{"label": "a", "box_2d": [1, 2, 3]}]})


def test_matching_scores_errors_labels_misses_and_extras() -> None:
    dets = [
        per.Detection("red cube", "red", "cube", (0, 0, 1, 1), (-0.148, -0.052)),
        per.Detection("blue ball", "blue", "sphere", (0, 0, 1, 1), (0.05, -0.05)),
        per.Detection("ghost", "grey", "box", (0, 0, 1, 1), (0.4, 0.3)),
    ]
    truth = {
        "red cube": ((-0.15, -0.05), "red", "cube"),
        "green cube": ((0.05, -0.05), "green", "cube"),
        "blue cube": ((0.15, -0.05), "blue", "cube"),
    }
    report = per.match_to_truth(dets, truth)
    names = {m.name: m for m in report.matches}
    assert names["red cube"].label_ok and names["red cube"].error_m < 0.005
    assert not names["green cube"].label_ok
    assert report.missed == ["blue cube"] and [d.label for d in report.extra] == ["ghost"]
    lines = per.report_lines(report)
    assert lines[0].startswith("PERCEPTION: matched 2, missed 1, extra 1, labels ok 1/2")
    assert any("MISMATCH" in line for line in lines)


def test_label_agreement_uses_shape_synonyms() -> None:
    det = per.Detection("white milk carton", "white", "box", (0, 0, 1, 1))
    assert per.label_agrees(det, "white", "box")
    assert per.label_agrees(
        per.Detection("light blue bottle", "light blue", "cylinder", (0, 0, 1, 1)),
        "light blue",
        "cylinder",
    )
    assert not per.label_agrees(det, "red", "box")


def test_unique_labels_for_planning() -> None:
    dets = [
        per.Detection(lbl, "", "", (0, 0, 1, 1)) for lbl in ("Red Cube", "red cube", "", "ball")
    ]
    assert per.unique_labels(dets) == ["red cube", "red cube #2", "object", "ball"]


def test_truth_pixels_skip_points_behind_camera() -> None:
    spec = geo.load_camera_spec()
    px = per.truth_pixels(spec, {"a": [0.0, -0.05, TABLE], "b": [0.0, 2.0, 1.3]})
    assert list(px) == ["a"]


def test_look_alike_labels_need_the_kind_keyword() -> None:
    soy = per.Detection("soy milk carton", "beige", "box", (0, 0, 1, 1))
    fresh = per.Detection("fresh milk carton", "white", "box", (0, 0, 1, 1))
    assert per.label_agrees(soy, "beige and green", "box", ("soy",))
    assert not per.label_agrees(fresh, "beige and green", "box", ("soy",))
    assert per.label_agrees(
        per.Detection("box with number 2", "brown", "box", (0, 0, 1, 1)), "brown", "box", ("2",)
    )
    assert not per.label_agrees(
        per.Detection("box 12", "brown", "box", (0, 0, 1, 1)), "brown", "box", ("2",)
    )
    assert per.label_agrees(
        per.Detection("small volleyball", "white", "sphere", (0, 0, 1, 1)),
        "white",
        "sphere",
        ("volley",),
    )


def test_camera_model_matches_spec_and_gl_convention() -> None:
    spec = geo.load_camera_spec()
    model = geo.CameraModel.from_spec(spec)
    point = (0.05, -0.08, TABLE + 0.03)
    u, v = model.project(point)
    assert np.allclose((u, v), geo.project(spec, point))
    origin, d = model.ray(u, v)
    assert np.allclose(np.cross(np.asarray(point) - origin, d), 0, atol=1e-9)
    # Same camera expressed as an OpenGL model matrix (RoboTwin camera config)
    gl = np.eye(4)
    gl[:3, :3] = model.R @ geo.GL_TO_CV
    gl[:3, 3] = model.t
    again = geo.CameraModel.from_gl_matrices("w", model.K, gl, spec.width, spec.height)
    assert np.allclose(again.project(point), (u, v))
    q = geo.mat_to_quat(spec.rotation)
    sapien_pose = geo.CameraModel.from_sapien_pose(
        "w", spec.position, q, spec.width, spec.height, spec.fovy_deg
    )
    assert np.allclose(sapien_pose.project(point), (u, v))


def test_triangulation_recovers_point_and_rejects_parallel_rays() -> None:
    target = np.array([0.1, -0.05, TABLE + 0.04])
    a = np.array([0.0, 0.12, 1.12])
    b = np.array([0.2, -0.40, 1.05])
    rays = [
        (a, (target - a) / np.linalg.norm(target - a)),
        (b, (target - b) / np.linalg.norm(target - b)),
    ]
    assert np.allclose(geo.triangulate(rays), target, atol=1e-9)
    assert geo.triangulate([rays[0], (a + 0.001, rays[0][1])]) is None
    assert geo.triangulate(rays[:1]) is None


def _view(model, name, objects):
    payload = {"objects": []}
    for label, xyz in objects:
        u, v = model.project(xyz)
        payload["objects"].append(
            {
                "label": label,
                "color": "",
                "shape": "",
                "box_2d": [
                    round((v - 10) / model.height * 1000),
                    round((u - 10) / model.width * 1000),
                    round((v + 10) / model.height * 1000),
                    round((u + 10) / model.width * 1000),
                ],
            }
        )
    return per.parse_detections(payload, model, TABLE, name)


def test_fusion_triangulates_objects_seen_by_workspace_and_wrist_views() -> None:
    ws = geo.CameraModel.from_spec(geo.load_camera_spec())
    wrist_spec = geo.CameraSpec("wrist", (0.0, -0.40, 1.00), (0.0, -0.08, TABLE), 60.0, 320, 240)
    wrist = geo.CameraModel.from_spec(wrist_spec)
    bottle = (0.05, -0.05, TABLE + 0.065)  # tall: the table-plane guess is biased
    box = (-0.15, -0.05, TABLE + 0.025)
    dets = _view(ws, "workspace", [("water bottle", bottle), ("box 1", box)]) + _view(
        wrist, "wrist home", [("water bottle", bottle), ("box 1", box)]
    )
    fused = per.fuse_views(dets, TABLE)
    assert len(fused) == 2 and all(len(f.views) == 2 and f.z is not None for f in fused)
    got = {f.label: f for f in fused}
    assert math.dist(got["water bottle"].xy, bottle[:2]) < 0.01
    assert abs(got["water bottle"].z - bottle[2]) < 0.01
    single = per.fuse_views(_view(ws, "workspace", [("box 1", box)]), TABLE)
    assert single[0].z is None and math.dist(single[0].xy, box[:2]) < 0.01


def test_change_detection_ignores_masked_planned_objects() -> None:
    ref = np.zeros((120, 160, 3), np.uint8)
    cur = ref.copy()
    cur[50:60, 70:80] = 200  # a planned object moved here
    assert per.changed_fraction(ref, cur) > per.CHANGE_FRACTION
    mask = per.circle_mask((120, 160), [(75, 55, 12)])
    assert per.changed_fraction(ref, cur, mask) == 0.0
    cur[5:15, 5:15] = 200  # something else moved too
    assert per.changed_fraction(ref, cur, mask) > per.CHANGE_FRACTION
    assert per.changed_fraction(ref, np.zeros((10, 10, 3), np.uint8)) == 1.0
    circles = per.object_circles(geo.load_camera_spec(), [(0.0, -0.05, TABLE + 0.03)], 0.07)
    assert len(circles) == 1 and circles[0][2] > 5


def test_change_detection_ignores_render_noise_but_sees_objects() -> None:
    rng = np.random.RandomState(0)
    ref = (rng.rand(240, 320, 3) * 40 + 100).astype(np.uint8)
    noisy = np.clip(ref + rng.normal(0, 15, ref.shape), 0, 255).astype(np.uint8)
    noisy[rng.rand(240, 320) < 0.01] = 255  # isolated speckles (ray-tracing noise)
    assert per.changed_fraction(ref, noisy) < per.CHANGE_FRACTION
    moved = noisy.copy()
    moved[100:130, 150:180] = 20  # a real object-sized change
    assert per.changed_fraction(ref, moved) > per.CHANGE_FRACTION
    assert per.change_threshold(None) == per.CHANGE_FRACTION
    assert per.change_threshold(0.01) > 0.03  # a noisy camera raises the trigger level


def test_workspace_roi_covers_the_work_area_not_the_robot() -> None:
    spec = geo.load_camera_spec()
    roi = per.workspace_roi(spec, (spec.height, spec.width), TABLE)
    for xyz in ((0.0, -0.05, TABLE + 0.03), (-0.15, -0.17, TABLE), (0.16, -0.17, TABLE + 0.07)):
        u, v = geo.project(spec, xyz)
        assert roi[int(v), int(u)]
    wide = geo.CameraSpec("wide", (0.0, 0.5, 1.6), (0.0, -0.11, 0.74), 70.0, 640, 480)
    roi = per.workspace_roi(wide, (480, 640), TABLE)
    u, v = geo.project(wide, (0.0, -0.45, TABLE + 0.25))  # robot shoulder, behind the area
    assert not roi[int(v), int(u)] and not roi[2, 2] and not roi[477, 2]
    assert 0.05 < roi.mean() < 0.9


def test_convex_hull_and_polygon_mask() -> None:
    hull = per.convex_hull([(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (1, 3)])
    assert sorted(hull) == [(0, 0), (0, 4), (4, 0), (4, 4)]
    mask = per.polygon_mask((10, 10), hull)
    assert mask[2, 2] and not mask[6, 6]
    assert not per.polygon_mask((5, 5), [(0, 0), (1, 1)]).any()


def test_region_changes_names_the_object_that_moved() -> None:
    ref = np.full((120, 160, 3), 128, np.uint8)
    cur = ref.copy()
    cur[20:40, 20:40] = 10  # object A vanished
    circles = {"a": (30, 30, 10), "b": (120, 80, 10)}
    out = per.region_changes(ref, cur, circles)
    assert out["a"] > per.OBJECT_CHANGE and out["b"] == 0.0
    ignore = per.circle_mask((120, 160), [(30, 30, 15)])  # A was moved by the plan
    assert per.region_changes(ref, cur, circles, mask=ignore)["a"] == 0.0
    assert per.region_changes(ref, np.zeros((5, 5, 3), np.uint8), circles) == {"a": 1.0, "b": 1.0}


def test_dilate_grows_and_unstable_pixels_can_be_masked() -> None:
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    assert per.dilate(m).sum() == 9
    a = np.full((60, 80, 3), 100, np.uint8)
    b = a.copy()
    b[10:20, 10:20] = 200  # a flickering highlight seen between two reference frames
    unstable = per.dilate(per.change_map(a, b))
    assert per.changed_fraction(a, b, unstable) == 0.0
