"""Offline native RGB-D region-proposal provenance and claim tests."""

from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vla_robot_sorting.perception.native_depth_audit import DepthEvidenceError
from vla_robot_sorting.perception.native_depth_capture import NATIVE_METRIC_DEPTH_CONVENTION
from vla_robot_sorting.perception.native_metric_quality import NativeMetricQualityError
from vla_robot_sorting.perception.native_rgbd_proposals import (
    generate_native_rgbd_proposals,
)
from vla_robot_sorting.perception.region_proposals import RegionProposalError


def _capture(tmp_path, *, depth_mm=700.0, legacy=False):
    root = tmp_path / "capture"
    root.mkdir()
    for cam in ("head_camera", "wrist_camera"):
        stem = f"{cam}_frame_000000"
        img = Image.new("RGBA", (160, 120), (200, 200, 200, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle((25, 30, 85, 95), fill=(0, 0, 155, 255))
        draw.ellipse((96, 20, 140, 70), fill=(255, 30, 20, 255))
        rgb_path = root / f"{stem}_rgb.png"
        img.save(rgb_path)
        depth_path = root / f"{stem}_depth_mm.npy"
        np.save(depth_path, np.full((120, 160), depth_mm, dtype=np.float64))
        manifest = {
            "schema": "damh.robotwin-native-rgbd-capture.v1",
            "status": "capture_written",
            "source_classification": "recorded_sim_observation",
            "camera_id": cam,
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
            "depth_convention": (
                "legacy-alpha-div255" if legacy else NATIVE_METRIC_DEPTH_CONVENTION
            ),
            "image_size_px": [160, 120],
            "depth_shape": [120, 160],
            "depth_valid_pixel_count": 160 * 120,
            "rgb_path": str(rgb_path),
            "rgb_sha256": sha256(rgb_path.read_bytes()).hexdigest(),
            "depth_path": str(depth_path),
            "depth_sha256": sha256(depth_path.read_bytes()).hexdigest(),
            "intrinsic_cv": [[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]],
            "extrinsic_cv_world_to_camera_cv": np.eye(4)[:3].tolist(),
            "cam2world_gl": np.diag([1.0, -1.0, -1.0, 1.0]).tolist(),
            "camera_to_world_cv": np.eye(4).tolist(),
            "cv_gl_max_abs_error": 0.0,
        }
        (root / f"{stem}_depth-manifest.json").write_text(json.dumps(manifest))
    return root


def test_native_source_proposals_are_unscored_no_hdf5_pose_or_control(tmp_path):
    pytest.importorskip("cv2")
    root = _capture(tmp_path)
    first = generate_native_rgbd_proposals(root, camera_id="head_camera")
    repeat = generate_native_rgbd_proposals(root, camera_id="head_camera")
    assert first.source_kind == "native_rgbd_single_refresh_region_proposals"
    assert first.source_frame_association == "probe.get_obs single-frame camera refresh"
    assert first.verified_hdf5_episode is False
    assert first.verified_object_identity is False
    assert first.object_pose_world is None
    assert first.controller_eligible is False
    assert first.proposals
    assert first.proposals == repeat.proposals
    assert all(
        p.label is None
        and p.confidence is None
        and p.object_id is None
        and p.source_kind == "native_rgb_region_proposal"
        for p in first.proposals
    )
    assert (
        sha256((root / "head_camera_frame_000000_depth-manifest.json").read_bytes()).hexdigest()
        == first.native_depth_manifest_sha256
    )


def test_same_bundle_both_cameras_produce_independently_audited_proposals(tmp_path):
    pytest.importorskip("cv2")
    root = _capture(tmp_path)
    for name in ("head_camera", "wrist_camera"):
        assert generate_native_rgbd_proposals(root, camera_id=name).camera_id == name


@pytest.mark.parametrize("camera", ["other", "", None])
def test_unsupported_cameras_rejected_before_cv2(tmp_path, camera):
    root = _capture(tmp_path)
    with pytest.raises(RegionProposalError, match="camera_id"):
        generate_native_rgbd_proposals(root, camera_id=camera)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_proposals": 0},
        {"max_proposals": True},
        {"min_area_px": -1},
        {"max_area_fraction": 0},
        {"max_area_fraction": float("nan")},
        {"max_area_fraction": True},
    ],
)
def test_invalid_parameters_fail_before_cv2(tmp_path, kwargs):
    root = _capture(tmp_path)
    with pytest.raises(RegionProposalError):
        generate_native_rgbd_proposals(root, camera_id="head_camera", **kwargs)


def test_old_invalid_native_depth_is_rejected_not_patched(tmp_path):
    root = _capture(tmp_path, legacy=True)
    with pytest.raises(NativeMetricQualityError, match="legacy"):
        generate_native_rgbd_proposals(root, camera_id="head_camera")


def test_one_camera_tampered_depth_blocks_proposals_in_other_camera(tmp_path):
    root = _capture(tmp_path)
    (root / "wrist_camera_frame_000000_depth_mm.npy").write_bytes(b"tampered")
    with pytest.raises(DepthEvidenceError, match="SHA256"):
        generate_native_rgbd_proposals(root, camera_id="head_camera")


def test_implausible_metric_depth_rejected_before_opencv(tmp_path):
    root = _capture(tmp_path, depth_mm=3)
    with pytest.raises(NativeMetricQualityError, match="sub-50mm"):
        generate_native_rgbd_proposals(root, camera_id="head_camera")


def test_cli_writes_only_new_json_outside_immutable_native_source(tmp_path):
    cv2 = pytest.importorskip("cv2")
    # Import venv NumPy/Pillow BEFORE appending optional OpenCV user-site;
    # prepending user site via PYTHONPATH shadows NumPy with incompatible ABI.
    cv_site = str(Path(cv2.__file__).resolve().parents[1])
    root = _capture(tmp_path)
    cli = Path(__file__).resolve().parents[2] / "scripts/propose_native_rgbd_regions.py"
    out = tmp_path / "analysis.json"
    bootstrap = (
        "import sys,runpy,numpy,PIL;"
        "sys.path.append(sys.argv[1]);"
        "script=sys.argv[2];sys.argv=[script,*sys.argv[3:]];"
        "runpy.run_path(script,run_name='__main__')"
    )

    def run(*args):
        return subprocess.run(
            [sys.executable, "-c", bootstrap, cv_site, str(cli), str(root), *args],
            capture_output=True,
            text=True,
        )

    result = run("--output", str(out))
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "READ_ONLY_SOURCE_ALIGNED_REGION_PROPOSALS_NOT_OBJECT_POSE"
    assert report["controller_eligible"] is False
    assert [p["camera_id"] for p in report["cameras"]] == ["head_camera", "wrist_camera"]
    again = run("--output", str(out))
    assert again.returncode == 2 and "overwrite" in again.stdout
    assert json.loads(out.read_text()) == report
    inside = run("--output", str(root / "analysis.json"))
    assert inside.returncode == 2 and not (root / "analysis.json").exists()


def test_same_native_frame_2d_proposals_to_observed_3d_pixels(tmp_path):
    pytest.importorskip("cv2")
    from vla_robot_sorting.perception.native_region_depth_association import (
        associate_native_regions_with_observed_depth,
    )

    root = _capture(tmp_path)
    regions = generate_native_rgbd_proposals(root, camera_id="head_camera")
    points = associate_native_regions_with_observed_depth(regions, root, max_regions=10)
    assert 0 < len(points) <= 10
    for point in points:
        assert point.measured_region.median_depth_m == pytest.approx(0.7)
        assert point.measured_region.representative_point.depth_m == pytest.approx(0.7)
        assert point.source_rgb_sha256 == regions.rgb_sha256
        assert point.source_depth_sha256 == regions.depth_sha256
        assert point.object_id is None
        assert point.object_pose_world is None
        assert point.object_orientation is None
        assert point.grasp_validated is False
        assert point.controller_eligible is False


def test_native_proposal_geometry_rejects_changed_manifest_and_spoofed_identity(tmp_path):
    pytest.importorskip("cv2")
    from dataclasses import replace

    from vla_robot_sorting.perception.native_region_depth_association import (
        associate_native_regions_with_observed_depth,
    )

    root = _capture(tmp_path)
    proposals = generate_native_rgbd_proposals(root, camera_id="head_camera")
    fake = replace(proposals.proposals[0], label="fabricated object identity")
    with pytest.raises(RegionProposalError, match="identity"):
        associate_native_regions_with_observed_depth(
            replace(proposals, proposals=(fake,)),
            root,
        )

    manifest = root / "head_camera_frame_000000_depth-manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(RegionProposalError, match="manifest source hash"):
        associate_native_regions_with_observed_depth(proposals, root)


def test_native_proposal_geometry_refuses_wrong_2d_source_and_unbounded_run(tmp_path):
    pytest.importorskip("cv2")
    from dataclasses import replace

    from vla_robot_sorting.perception.native_region_depth_association import (
        associate_native_regions_with_observed_depth,
    )

    root = _capture(tmp_path)
    proposals = generate_native_rgbd_proposals(root, camera_id="head_camera")
    with pytest.raises(RegionProposalError, match="source/claim"):
        associate_native_regions_with_observed_depth(
            replace(proposals, verified_hdf5_episode=True),
            root,
        )
    for bound in (0, 33, True):
        with pytest.raises(RegionProposalError, match="max_regions"):
            associate_native_regions_with_observed_depth(
                proposals,
                root,
                max_regions=bound,
            )


def test_spatial_balance_guarantees_occupied_bottom_tiles_before_top_overflow():
    """A y-sorted cap silently lost the last quarter of the native wrist image."""
    from vla_robot_sorting.perception.native_rgbd_proposals import _spatially_balanced_boxes

    # 1,000 crowded top boxes and five lower boxes; retain at most 16.
    many_top = {(x, y, x + 9, y + 9) for y in range(0, 60, 2) for x in range(0, 240, 4)}
    bottom = {(10 + x, 300, 28 + x, 327) for x in (0, 55, 115, 180)}
    candidates = many_top | bottom
    chosen = _spatially_balanced_boxes(candidates, width=320, height=360, limit=16)
    assert len(chosen) == 16
    assert len(set(chosen)) == len(chosen)
    assert any(b[1] >= 270 for b in chosen)
    assert chosen == _spatially_balanced_boxes(candidates, width=320, height=360, limit=16)
    assert all(b in candidates for b in chosen)
    assert not any(b in chosen for b in sorted(many_top, key=lambda b: b[1])[-10:])


def _synthetic_native_proposals_json(tmp_path):
    """Test source-bound measured pixels without the optional OpenCV package."""
    from dataclasses import asdict

    from vla_robot_sorting.perception.native_rgbd_proposals import NativeRGBDProposalSet
    from vla_robot_sorting.perception.region_proposals import RegionProposal

    root = _capture(tmp_path)
    frames = []
    for name in ("head_camera", "wrist_camera"):
        stem = f"{name}_frame_000000"
        manifest_file = root / f"{stem}_depth-manifest.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        frames.append(
            asdict(
                NativeRGBDProposalSet(
                    camera_id=name,
                    frame_index=0,
                    source_classification="recorded_sim_observation",
                    source_kind="native_rgbd_single_refresh_region_proposals",
                    source_frame_association="probe.get_obs single-frame camera refresh",
                    native_depth_manifest_sha256=sha256(manifest_file.read_bytes()).hexdigest(),
                    rgb_sha256=manifest["rgb_sha256"],
                    depth_sha256=manifest["depth_sha256"],
                    calibration_manifest_sha256="e" * 64,
                    image_width_px=160,
                    image_height_px=120,
                    producer_method="offline-fixture-only",
                    producer_version="none",
                    producer_config_sha256="f" * 64,
                    proposals=(
                        RegionProposal(
                            proposal_id=f"native-{name}-000000-rp-0000",
                            camera_id=name,
                            frame_index=0,
                            bbox_xyxy_px=(25, 30, 85, 95),
                            source_kind="native_rgb_region_proposal",
                        ),
                    ),
                )
            )
        )
    payload = {
        "schema": "damh.native-rgbd-unlabelled-region-proposals.v1",
        "status": "READ_ONLY_SOURCE_ALIGNED_REGION_PROPOSALS_NOT_OBJECT_POSE",
        "native_capture_performed_by_this_command": False,
        "pretrained_weights_used": False,
        "automatic_object_detection": False,
        "object_identity_verified": False,
        "object_pose_verified": False,
        "grasp_validated": False,
        "controller_eligible": False,
        "cameras": frames,
    }
    proposals = tmp_path / "proposals.json"
    proposals.write_text(json.dumps(payload), encoding="utf-8")
    return root, proposals


def test_native_2d_to_observed_depth_cli_replays_without_opencv(tmp_path):
    root, proposals = _synthetic_native_proposals_json(tmp_path)
    out = tmp_path / "observed.json"
    cli = Path(__file__).resolve().parents[2] / "scripts/analyze_native_rgbd_regions.py"
    run = subprocess.run(
        [sys.executable, str(cli), str(root), str(proposals), "--output", str(out)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["classification"] == "NATIVE_SOURCE_BOUND_PIXEL_DEPTH_ONLY_NOT_OBJECT_POSE"
    assert report["native_acceptance"] is False
    assert len(report["cameras"]) == 2
    assert all(item["observed_depth_point_count"] == 1 for item in report["cameras"])
    assert all(item["measurements"][0]["object_id"] is None for item in report["cameras"])
    rerun = subprocess.run(
        [sys.executable, str(cli), str(root), str(proposals), "--output", str(out)],
        capture_output=True,
        text=True,
    )
    assert rerun.returncode == 2
    assert json.loads(out.read_text()) == report


def test_native_2d_depth_cli_refuses_spoofed_identity_and_changed_capture(tmp_path):
    root, proposals = _synthetic_native_proposals_json(tmp_path)
    cli = Path(__file__).resolve().parents[2] / "scripts/analyze_native_rgbd_regions.py"
    data = json.loads(proposals.read_text())
    data["object_identity_verified"] = True
    proposals.write_text(json.dumps(data))
    r = subprocess.run(
        [sys.executable, str(cli), str(root), str(proposals)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2 and "NATIVE_PIXEL_ASSOCIATION_REJECTED" in r.stderr

    data["object_identity_verified"] = False
    proposals.write_text(json.dumps(data))
    manifest = root / "head_camera_frame_000000_depth-manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    r = subprocess.run(
        [sys.executable, str(cli), str(root), str(proposals)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2
    assert "manifest source hash changed" in r.stderr


def test_native_color_landmarks_are_source_bound_pixels_not_identity(tmp_path):
    pytest.importorskip("cv2")
    from vla_robot_sorting.perception.native_color_landmarks import find_native_color_landmarks

    root = _capture(tmp_path)
    for camera in ("head_camera", "wrist_camera"):
        items = find_native_color_landmarks(root, camera_id=camera)
        assert {item.landmark_kind for item in items} == {"orange_surface", "blue_surface"}
        for item in items:
            assert item.source_frame_index == 0
            assert item.mask_pixel_count >= 30
            assert item.valid_depth_fraction >= 0.60
            assert item.median_masked_depth_m == pytest.approx(0.7)
            assert item.observed_pixel.depth_m == pytest.approx(0.7)
            assert item.object_id is None
            assert item.object_pose_world is None
            assert item.grasp_candidate_validated is False
            assert item.controller_eligible is False


def test_native_color_landmarks_refuse_tampered_opposite_camera(tmp_path):
    from vla_robot_sorting.perception.native_color_landmarks import find_native_color_landmarks

    root = _capture(tmp_path)
    (root / "wrist_camera_frame_000000_depth_mm.npy").write_bytes(b"tampered")
    with pytest.raises(DepthEvidenceError, match="SHA256"):
        find_native_color_landmarks(root, camera_id="head_camera")


def test_native_color_landmarks_invalid_camera_and_bounds_fail_closed(tmp_path):
    from vla_robot_sorting.perception.native_color_landmarks import find_native_color_landmarks

    root = _capture(tmp_path)
    with pytest.raises(RegionProposalError, match="camera"):
        find_native_color_landmarks(root, camera_id="unverified_camera")
    with pytest.raises(RegionProposalError, match="minimum"):
        find_native_color_landmarks(root, camera_id="head_camera", min_mask_px=True)
    with pytest.raises(RegionProposalError, match="minimum"):
        find_native_color_landmarks(root, camera_id="head_camera", min_mask_px=0)
