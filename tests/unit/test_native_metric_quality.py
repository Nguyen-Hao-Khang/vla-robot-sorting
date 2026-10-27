"""Offline tests for native RGB-D integrity-vs-plausibility claim boundaries."""

from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
import pytest
from PIL import Image

from vla_robot_sorting.perception.native_depth_audit import audit_native_depth_bundle
from vla_robot_sorting.perception.native_depth_capture import NATIVE_METRIC_DEPTH_CONVENTION
from vla_robot_sorting.perception.native_metric_quality import (
    NativeMetricQualityError,
    audit_native_metric_quality,
)


def _bundle(tmp_path, *, depth_mm=700.0, rgb=128, convention=None):
    folder = tmp_path / "rgbd"
    folder.mkdir()
    e = np.eye(4)
    for cam in ("head_camera", "wrist_camera"):
        stem = f"{cam}_frame_000000"
        rgba = np.full((4, 6, 4), 255, dtype=np.uint8)
        rgba[..., :3] = rgb
        rp = folder / f"{stem}_rgb.png"
        Image.fromarray(rgba, mode="RGBA").save(rp)
        dp = folder / f"{stem}_depth_mm.npy"
        np.save(dp, np.full((4, 6), depth_mm, dtype=np.float64))

        def sha(f):
            return sha256(f.read_bytes()).hexdigest()

        m = {
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
                NATIVE_METRIC_DEPTH_CONVENTION if convention is None else convention
            ),
            "image_size_px": [6, 4],
            "depth_shape": [4, 6],
            "depth_valid_pixel_count": 24,
            "rgb_path": str(rp),
            "rgb_sha256": sha(rp),
            "depth_path": str(dp),
            "depth_sha256": sha(dp),
            "intrinsic_cv": [[100.0, 0, 2], [0, 100.0, 1], [0, 0, 1]],
            "extrinsic_cv_world_to_camera_cv": e[:3].tolist(),
            "cam2world_gl": np.diag([1.0, -1.0, -1.0, 1.0]).tolist(),
            "camera_to_world_cv": e.tolist(),
            "cv_gl_max_abs_error": 0.0,
        }
        (folder / f"{stem}_depth-manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return folder


def test_native_plausibility_both_cameras_pass_but_never_task_accept(tmp_path):
    bundle = _bundle(tmp_path)
    result = audit_native_metric_quality(bundle)
    assert result["camera_count"] == 2
    assert "PASS_NOT_METRIC_ACCURACY_OR_TASK_ACCEPTANCE" in result["status"]
    assert result["metric_accuracy_verified"] is False
    assert result["object_identity_verified"] is False
    assert result["controller_eligible"] is False


def test_legacy_depth_still_passes_artifact_integrity_but_not_physical_units(tmp_path):
    bundle = _bundle(
        tmp_path,
        depth_mm=3.0,
        rgb=1,
        convention="depth_mm = max(0, -Position[...,2] * 1000 * Color_alpha/255)",
    )
    assert audit_native_depth_bundle(bundle)["camera_count"] == 2
    with pytest.raises(NativeMetricQualityError, match="legacy"):
        audit_native_metric_quality(bundle)


def test_very_small_depth_even_v2_rejected(tmp_path):
    bundle = _bundle(tmp_path, depth_mm=8.0)
    with pytest.raises(NativeMetricQualityError, match="sub-50mm"):
        audit_native_metric_quality(bundle)


def test_bad_rgb_scale_even_v2_rejected(tmp_path):
    bundle = _bundle(tmp_path, rgb=1)
    with pytest.raises(NativeMetricQualityError, match="RGB range"):
        audit_native_metric_quality(bundle)


def test_checksum_tamper_prevents_quality_pass(tmp_path):
    bundle = _bundle(tmp_path)
    (bundle / "head_camera_frame_000000_depth_mm.npy").write_bytes(b"modified")
    with pytest.raises(Exception, match="SHA256"):
        audit_native_metric_quality(bundle)
