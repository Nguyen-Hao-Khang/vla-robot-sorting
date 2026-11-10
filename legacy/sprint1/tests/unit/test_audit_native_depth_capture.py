"""Offline synthetic auditor regression tests, NOT native RGB-D capture evidence."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scripts.audit_native_depth_capture import (
    DepthEvidenceError,
    audit_native_depth_bundle,
)


def _checksum(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _bundle(folder: Path) -> None:
    folder.mkdir()
    for camera_id in ("head_camera", "wrist_camera"):
        stem = f"{camera_id}_frame_000000"
        image_path = folder / f"{stem}_rgb.png"
        depth_path = folder / f"{stem}_depth_mm.npy"
        Image.new("RGBA", (4, 3), color=(255, 255, 255, 255)).save(image_path)
        np.save(depth_path, np.full((3, 4), 700.0, dtype=np.float64))
        eye = np.eye(4)
        model = np.diag([1, -1, -1, 1])
        data = {
            "schema": "damh.robotwin-native-rgbd-capture.v1",
            "status": "capture_written",
            "source_classification": "recorded_sim_observation",
            "camera_id": camera_id,
            "frame_index": 0,
            "frame_index_verified_by_caller": True,
            "frame_association_source": "probe.get_obs single-frame camera refresh",
            "native_capture_executed": True,
            "timestamp_semantics": "probe wall-clock time; not a simulator sensor timestamp",
            "scene_revision": None,
            "controller_eligible": False,
            "pinned_sapien_version": "3.0.0b1",
            "installed_sapien_version": "3.0.0b1",
            "depth_units": "millimetres",
            "image_size_px": [4, 3],
            "depth_shape": [3, 4],
            "depth_valid_pixel_count": 12,
            "rgb_path": str(image_path.resolve()),
            "rgb_sha256": _checksum(image_path),
            "depth_path": str(depth_path.resolve()),
            "depth_sha256": _checksum(depth_path),
            "intrinsic_cv": np.eye(3).tolist(),
            "extrinsic_cv_world_to_camera_cv": eye[:3].tolist(),
            "cam2world_gl": model.tolist(),
            "camera_to_world_cv": eye.tolist(),
            "cv_gl_max_abs_error": 0.0,
        }
        (folder / f"{stem}_depth-manifest.json").write_text(
            json.dumps(data), encoding="utf-8"
        )


def test_auditor_can_verify_self_consistent_fixture_without_claiming_native_success(tmp_path):
    directory = tmp_path / "bundle"
    _bundle(directory)

    result = audit_native_depth_bundle(directory)

    assert result["camera_count"] == 2
    assert result["status"] == "ARTIFACT_INTEGRITY_PASS_NOT_TASK_ACCEPTANCE"
    assert result["native_physics_or_task_success_verified"] is False
    assert all(item["positive_depth_pixels"] == 12 for item in result["cameras"])


@pytest.mark.parametrize(("key", "value"), [
    ("frame_index_verified_by_caller", False),
    ("native_capture_executed", False),
    ("controller_eligible", True),
    ("status", "failed"),
    ("source_classification", "synthetic_fixture"),
    ("depth_units", "metres"),
    ("scene_revision", "unverified-random-revision"),
    ("frame_association_source", "unverified index"),
    ("timestamp_semantics", "simulator timestamp"),
])
def test_auditor_rejects_false_provenance_or_safety_claims(tmp_path, key, value):
    directory = tmp_path / "bundle"
    _bundle(directory)
    manifest_path = directory / "head_camera_frame_000000_depth-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[key] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DepthEvidenceError):
        audit_native_depth_bundle(directory)


def test_auditor_rejects_tampered_depth_bytes_even_if_png_is_untouched(tmp_path):
    directory = tmp_path / "bundle"
    _bundle(directory)
    depth_path = directory / "head_camera_frame_000000_depth_mm.npy"
    np.save(depth_path, np.zeros((3, 4), dtype=np.float64))

    with pytest.raises(DepthEvidenceError, match="SHA256"):
        audit_native_depth_bundle(directory)


def test_auditor_rejects_empty_depth_despite_valid_new_digest(tmp_path):
    directory = tmp_path / "bundle"
    _bundle(directory)
    stem = "head_camera_frame_000000"
    depth_path = directory / f"{stem}_depth_mm.npy"
    manifest_path = directory / f"{stem}_depth-manifest.json"
    np.save(depth_path, np.zeros((3, 4), dtype=np.float64))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["depth_sha256"] = _checksum(depth_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DepthEvidenceError, match="positive depth"):
        audit_native_depth_bundle(directory)


def test_auditor_rejects_missing_camera(tmp_path):
    directory = tmp_path / "bundle"
    _bundle(directory)
    (directory / "wrist_camera_frame_000000_depth-manifest.json").unlink()

    with pytest.raises(DepthEvidenceError, match="exactly two"):
        audit_native_depth_bundle(directory)
