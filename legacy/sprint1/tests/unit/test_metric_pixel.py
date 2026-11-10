"""S1-V4 cross-OS native RGB-D *artifact* tests, all synthetic offline."""

from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vla_robot_sorting.perception.metric_pixel import (
    MetricPixelError,
    project_metric_pixel,
)
from vla_robot_sorting.perception.metric_region import project_metric_region
from vla_robot_sorting.perception.native_depth_audit import (
    DepthEvidenceError,
    audit_native_depth_bundle,
)
from vla_robot_sorting.perception.native_depth_capture import NATIVE_METRIC_DEPTH_CONVENTION


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def _make_bundle(tmp_path):
    root = tmp_path / "exported-native-depth"
    root.mkdir(parents=True)
    for camera in ("head_camera", "wrist_camera"):
        stem = f"{camera}_frame_000000"
        rgb = root / f"{stem}_rgb.png"
        depth = root / f"{stem}_depth_mm.npy"
        Image.new("RGBA", (4, 3), color=(255, 255, 255, 255)).save(rgb)
        np.save(depth, np.full((3, 4), 700.0, dtype=np.float64))
        eye = np.eye(4)
        source = {
            "schema": "damh.robotwin-native-rgbd-capture.v1",
            "status": "capture_written",
            "source_classification": "recorded_sim_observation",
            "camera_id": camera,
            "frame_index": 0,
            "frame_association_source": "probe.get_obs single-frame camera refresh",
            "frame_index_verified_by_caller": True,
            "native_capture_executed": True,
            "timestamp_semantics": "probe wall-clock time; not a simulator sensor timestamp",
            "scene_revision": None,
            "controller_eligible": False,
            "pinned_sapien_version": "3.0.0b1",
            "installed_sapien_version": "3.0.0b1",
            "depth_units": "millimetres",
            "depth_convention": NATIVE_METRIC_DEPTH_CONVENTION,
            "image_size_px": [4, 3],
            "depth_shape": [3, 4],
            "depth_valid_pixel_count": 12,
            "rgb_path": f"/home/nguyen/projects/DAMH/experiments/results/{stem}_rgb.png",
            "rgb_sha256": _sha(rgb),
            "depth_path": f"/home/nguyen/projects/DAMH/experiments/results/{stem}_depth_mm.npy",
            "depth_sha256": _sha(depth),
            "intrinsic_cv": [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]],
            "extrinsic_cv_world_to_camera_cv": eye[:3].tolist(),
            "cam2world_gl": np.diag([1.0, -1.0, -1.0, 1.0]).tolist(),
            "camera_to_world_cv": eye.tolist(),
            "cv_gl_max_abs_error": 0.0,
        }
        (root / f"{stem}_depth-manifest.json").write_text(
            json.dumps(source),
            encoding="utf-8",
        )
    return root


def test_linux_absolute_source_paths_relocated_to_windows_are_sha_verified(tmp_path):
    folder = _make_bundle(tmp_path)
    result = audit_native_depth_bundle(folder)
    assert result["camera_count"] == 2
    assert result["status"] == "ARTIFACT_INTEGRITY_PASS_NOT_TASK_ACCEPTANCE"
    assert result["native_physics_or_task_success_verified"] is False


def test_metric_pixel_uses_z_depth_millimetres_and_cv_intrinsics(tmp_path):
    folder = _make_bundle(tmp_path)
    point = project_metric_pixel(folder, camera_id="head_camera", pixel_uv=(2, 1))
    assert point.depth_m == pytest.approx(0.7)
    assert point.camera_point_m == pytest.approx((0.007, 0.0, 0.7))
    assert point.world_point_m == pytest.approx((0.007, 0.0, 0.7))
    assert point.source_classification == "recorded_sim_observation"
    assert point.object_id is None
    assert point.object_pose_world is None
    assert point.object_orientation is None
    assert point.grasp_candidate_validated is False
    assert point.controller_eligible is False
    assert len(point.manifest_sha256) == 64


@pytest.mark.parametrize("pixel", [(-1, 0), (4, 0), (1, 3), (0.2, 1), (True, 2)])
def test_metric_pixel_refuses_noninteger_or_out_of_bounds(tmp_path, pixel):
    folder = _make_bundle(tmp_path)
    with pytest.raises(MetricPixelError):
        project_metric_pixel(folder, camera_id="head_camera", pixel_uv=pixel)


def test_zero_depth_pixel_rejected_even_when_rest_of_frame_is_valid(tmp_path):
    folder = _make_bundle(tmp_path)
    stem = "wrist_camera_frame_000000"
    depth_path = folder / f"{stem}_depth_mm.npy"
    depth = np.load(depth_path, allow_pickle=False)
    depth[1, 2] = 0
    np.save(depth_path, depth)
    manifest_path = folder / f"{stem}_depth-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["depth_sha256"] = _sha(depth_path)
    manifest["depth_valid_pixel_count"] = 11
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MetricPixelError, match="zero depth"):
        project_metric_pixel(folder, camera_id="wrist_camera", pixel_uv=(2, 1))


def test_corrupt_depth_or_forged_export_basename_fails_before_projection(tmp_path):
    folder = _make_bundle(tmp_path)
    stem = "head_camera_frame_000000"
    depth_path = folder / f"{stem}_depth_mm.npy"
    depth_path.write_bytes(b"tampered")
    with pytest.raises(DepthEvidenceError, match="SHA256"):
        project_metric_pixel(folder, camera_id="head_camera", pixel_uv=(1, 1))

    folder = _make_bundle(tmp_path / "different")
    manifest_path = folder / f"{stem}_depth-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rgb_path"] = "/home/nguyen/wrong-name.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DepthEvidenceError, match="basename"):
        audit_native_depth_bundle(folder)


def _replace_depth_and_manifest(folder, camera, array):
    stem = f"{camera}_frame_000000"
    path = folder / f"{stem}_depth_mm.npy"
    np.save(path, array.astype(np.float64), allow_pickle=False)
    manifest_path = folder / f"{stem}_depth-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["depth_sha256"] = _sha(path)
    manifest["depth_valid_pixel_count"] = int(np.count_nonzero(array > 0))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_audited_region_depth_returns_real_pixel_not_object_pose(tmp_path):
    folder = _make_bundle(tmp_path)
    item = project_metric_region(folder, camera_id="head_camera", bbox_xyxy_px=(0, 0, 4, 3))
    assert item.median_depth_m == pytest.approx(0.7)
    assert item.depth_iqr_m == pytest.approx(0.0)
    assert item.valid_pixels == item.total_pixels == 12
    assert item.valid_fraction == 1
    assert item.representative_point.pixel_uv == (1, 1)
    assert item.representative_point.world_point_m == pytest.approx((0, 0, 0.7))
    assert item.representative_point.depth_m == pytest.approx(item.median_depth_m)
    assert item.region_selection_kind == "operator_selected_unsegmented_rectangle"
    assert item.object_id is None
    assert item.object_pose_world is None
    assert item.object_orientation is None
    assert item.object_dimensions_m is None
    assert item.object_identity_confidence is None
    assert item.grasp_candidate_validated is False
    assert item.controller_eligible is False


def test_region_chooses_existing_median_depth_pixel_when_center_is_zero(tmp_path):
    folder = _make_bundle(tmp_path)
    depth = np.full((3, 4), 700.0)
    depth[1, 1] = 0
    _replace_depth_and_manifest(folder, "head_camera", depth)
    item = project_metric_region(folder, camera_id="head_camera", bbox_xyxy_px=(0, 0, 4, 3))
    assert item.valid_pixels == 11
    assert item.valid_fraction == pytest.approx(11 / 12)
    assert item.representative_point.pixel_uv != (1, 1)
    assert item.representative_point.depth_m == pytest.approx(0.7)


def test_region_rejects_insufficient_depth_coverage(tmp_path):
    folder = _make_bundle(tmp_path)
    depth = np.zeros((3, 4))
    depth[0, :4] = 700
    _replace_depth_and_manifest(folder, "head_camera", depth)
    with pytest.raises(MetricPixelError, match="insufficient"):
        project_metric_region(folder, camera_id="head_camera", bbox_xyxy_px=(0, 0, 4, 3))


def test_region_rejects_mixed_near_and_far_depths_not_object_segmentation(tmp_path):
    folder = _make_bundle(tmp_path)
    depth = np.full((3, 4), 700.0)
    depth[0, :] = 1200
    depth[1, :2] = 1200
    _replace_depth_and_manifest(folder, "wrist_camera", depth)
    with pytest.raises(MetricPixelError, match="excessive depth spread"):
        project_metric_region(folder, camera_id="wrist_camera", bbox_xyxy_px=(0, 0, 4, 3))


@pytest.mark.parametrize(
    "box",
    [
        (-1, 0, 2, 2),
        (0, 0, 0, 2),
        (3, 0, 2, 2),
        (0, 0, 5, 3),
        (0, 0, 4, 4),
        (0, 0, 2.5, 2),
        (False, 0, 3, 2),
        (0, 0, 1, 1),
    ],
)
def test_region_rejects_unsafe_roi_or_insufficient_area(tmp_path, box):
    folder = _make_bundle(tmp_path)
    with pytest.raises(MetricPixelError):
        project_metric_region(folder, camera_id="head_camera", bbox_xyxy_px=box)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"min_valid_fraction": float("nan")}, "min_valid_fraction"),
        ({"min_valid_fraction": 0}, "min_valid_fraction"),
        ({"min_valid_fraction": True}, "min_valid_fraction"),
        ({"max_iqr_m": float("inf")}, "max_iqr_m"),
        ({"max_iqr_m": -0.1}, "max_iqr_m"),
        ({"min_valid_pixels": 3}, "min_valid_pixels"),
        ({"min_valid_pixels": False}, "min_valid_pixels"),
        ({"min_valid_pixels": 13}, "minimum valid sample count"),
    ],
)
def test_region_fail_closed_for_bad_depth_quality_gate(tmp_path, settings, message):
    folder = _make_bundle(tmp_path)
    with pytest.raises(MetricPixelError, match=message):
        project_metric_region(
            folder,
            camera_id="head_camera",
            bbox_xyxy_px=(0, 0, 4, 3),
            **settings,
        )


def test_region_requires_two_independently_audited_camera_artifacts(tmp_path):
    folder = _make_bundle(tmp_path)
    (folder / "wrist_camera_frame_000000_depth-manifest.json").unlink()
    with pytest.raises(DepthEvidenceError, match="exactly two"):
        project_metric_region(folder, camera_id="head_camera", bbox_xyxy_px=(0, 0, 4, 3))


def test_region_rejects_tampered_depth_even_for_other_camera(tmp_path):
    folder = _make_bundle(tmp_path)
    (folder / "wrist_camera_frame_000000_depth_mm.npy").write_bytes(b"tampered")
    with pytest.raises(DepthEvidenceError, match="SHA256"):
        project_metric_region(folder, camera_id="head_camera", bbox_xyxy_px=(0, 0, 4, 3))


def _run_roi_cli(folder, *args):
    script = Path(__file__).resolve().parents[2] / "scripts/analyze_native_metric_region.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            str(folder),
            "--camera",
            "head_camera",
            "--bbox",
            "0",
            "0",
            "4",
            "3",
            *args,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_roi_cli_json_is_explicitly_not_pose_or_controller_approval(tmp_path):
    folder = _make_bundle(tmp_path)
    result = _run_roi_cli(folder)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["classification"] == "ROI_DEPTH_ONLY_NOT_OBJECT_POSE_NOT_TASK_ACCEPTANCE"
    assert data["native_execution_performed_by_this_command"] is False
    assert data["object_identity_verified"] is False
    assert data["object_pose_verified"] is False
    assert data["controller_eligible"] is False
    assert data["region"]["median_depth_m"] == pytest.approx(0.7)


def test_roi_cli_refuses_mutating_evidence_and_overwriting_analysis(tmp_path):
    folder = _make_bundle(tmp_path)
    readonly = _run_roi_cli(folder, "--output", str(folder / "bad.json"))
    assert readonly.returncode == 2
    assert "immutable native evidence" in readonly.stdout
    assert not (folder / "bad.json").exists()

    external_path = tmp_path / "analysis.json"
    external_path.write_text("preserve me", encoding="utf-8")
    overwrite = _run_roi_cli(folder, "--output", str(external_path))
    assert overwrite.returncode == 2
    assert "refuse overwrite" in overwrite.stdout
    assert external_path.read_text(encoding="utf-8") == "preserve me"


def test_legacy_native_manifest_remains_artifact_valid_but_not_metric_safe(tmp_path):
    """Historical export integrity is different from safe geometric backprojection."""
    folder = _make_bundle(tmp_path)
    for cam in ("head_camera", "wrist_camera"):
        manifest_path = folder / f"{cam}_frame_000000_depth-manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["depth_convention"] = "depth_mm = max(0, -Position[...,2] * 1000 * Color_alpha/255)"
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
    assert audit_native_depth_bundle(folder)["camera_count"] == 2
    with pytest.raises(MetricPixelError, match="not source-verified"):
        project_metric_pixel(folder, camera_id="head_camera", pixel_uv=(1, 1))
    with pytest.raises(MetricPixelError, match="not source-verified"):
        project_metric_region(folder, camera_id="head_camera", bbox_xyxy_px=(0, 0, 4, 3))


def test_new_depth_convention_still_rejects_implausibly_close_pixel(tmp_path):
    folder = _make_bundle(tmp_path)
    depth = np.full((3, 4), 8.2)
    _replace_depth_and_manifest(folder, "head_camera", depth)
    assert audit_native_depth_bundle(folder)["camera_count"] == 2
    with pytest.raises(MetricPixelError, match="50 mm"):
        project_metric_pixel(folder, camera_id="head_camera", pixel_uv=(1, 1))
    with pytest.raises(MetricPixelError, match="insufficient"):
        project_metric_region(folder, camera_id="head_camera", bbox_xyxy_px=(0, 0, 4, 3))


def test_roi_cli_rejects_legacy_manifest_even_with_foreign_editable_install(tmp_path):
    """CLI is pinned to its checkout and must not import a different worktree."""
    folder = _make_bundle(tmp_path)
    manifest = folder / "head_camera_frame_000000_depth-manifest.json"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["depth_convention"] = "legacy_bad_color_alpha_divided_by_255"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    result = _run_roi_cli(folder)
    assert result.returncode == 2, result.stderr
    data = json.loads(result.stdout)
    assert data["classification"] == "ROI_DEPTH_REJECTED"
    assert "not source-verified" in data["error"]
