from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from vla_robot_sorting.perception.native_depth_capture import (
    NativeDepthCaptureError,
    capture_sapien_depth_frame,
    depth_millimetres_from_sapien,
)


def test_native_capture_imports_are_compatible_with_pinned_python310():
    # `datetime.UTC` was introduced in 3.11; RoboTwin's Conda env is Python 3.10.
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src/vla_robot_sorting/perception/native_depth_capture.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = {
        name.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "datetime"
        for name in node.names
    }
    assert "UTC" not in imports


def test_pinned_depth_conversion_is_millimetres_and_masks_background():
    position = np.zeros((2, 2, 4), dtype=np.float32)
    position[..., 2] = -0.75
    color = np.ones((2, 2, 4), dtype=np.float32)
    color[0, 1, 3] = 0.0

    depth = depth_millimetres_from_sapien(position, color)

    np.testing.assert_allclose(depth, [[750.0, 0.0], [750.0, 750.0]])


@pytest.mark.parametrize(
    ("position", "color"),
    [
        (np.zeros((4, 4), dtype=np.float32), np.zeros((4, 4, 4), dtype=np.float32)),
        (np.zeros((2, 2, 4), dtype=np.float32), np.zeros((3, 2, 4), dtype=np.float32)),
        (np.full((2, 2, 4), np.nan, dtype=np.float32), np.ones((2, 2, 4))),
    ],
)
def test_depth_conversion_rejects_bad_or_mismatched_buffers(position, color):
    with pytest.raises(NativeDepthCaptureError):
        depth_millimetres_from_sapien(position, color)


def test_capture_probe_persists_failure_manifest_for_missing_camera(tmp_path):
    manifest_path = capture_sapien_depth_frame(
        None,
        camera_id="head_camera",
        frame_index=4,
        frame_association_source="fixture frame index",
        output_dir=tmp_path,
        frame_index_verified=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["native_capture_executed"] is False
    assert "camera handle is missing" in manifest["error"]
    timestamp = datetime.fromisoformat(manifest["capture_timestamp_utc"])
    assert timestamp.utcoffset() == timedelta(0)


def test_capture_probe_fails_closed_on_missing_depth_and_unsupported_camera(tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "3.0.0b1")

    class NoDepthCamera:
        def get_picture(self, channel):
            if channel == "Color":
                return np.ones((2, 2, 4), dtype=np.float32)
            raise KeyError("Position depth channel missing")

    missing_depth_path = capture_sapien_depth_frame(
        NoDepthCamera(),
        camera_id="head_camera",
        frame_index=2,
        frame_association_source="fixture frame index",
        output_dir=tmp_path / "missing-depth",
        frame_index_verified=True,
    )
    missing_depth = json.loads(missing_depth_path.read_text(encoding="utf-8"))
    assert missing_depth["status"] == "failed"
    assert missing_depth["native_capture_executed"] is False
    assert "Position" in missing_depth["error"]

    unsupported_path = capture_sapien_depth_frame(
        NoDepthCamera(),
        camera_id="front_camera",
        frame_index=2,
        frame_association_source="fixture frame index",
        output_dir=tmp_path / "unsupported-camera",
    )
    unsupported = json.loads(unsupported_path.read_text(encoding="utf-8"))
    assert unsupported["status"] == "failed"
    assert "camera_id" in unsupported["error"]


def test_capture_probe_rejects_unverified_frame_association_and_saves_failure(tmp_path):
    manifest_path = capture_sapien_depth_frame(
        object(),
        camera_id="wrist_camera",
        frame_index=0,
        frame_association_source="camera loop frame id",
        output_dir=tmp_path,
        frame_index_verified=False,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "verify the camera buffers" in manifest["error"]
    assert manifest["controller_eligible"] is False


def test_capture_refuses_overwrite_of_successful_native_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "3.0.0b1")

    class NativeLikeCamera:
        def get_picture(self, name):
            image = np.ones((2, 3, 4), dtype=np.float32)
            if name == "Position":
                image[..., 2] = -0.7
                image[..., 3] = 0.0
            return image

        def get_intrinsic_matrix(self):
            return np.eye(3)

        def get_extrinsic_matrix(self):
            return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

        def get_model_matrix(self):
            return np.diag([1.0, -1.0, -1.0, 1.0])

    arguments = {
        "camera": NativeLikeCamera(),
        "camera_id": "head_camera",
        "frame_index": 0,
        "frame_association_source": "verified observation loop",
        "frame_index_verified": True,
        "output_dir": tmp_path,
    }
    first_path = capture_sapien_depth_frame(**arguments)
    first_bytes = first_path.read_bytes()
    result = json.loads(first_bytes)
    assert result["status"] == "capture_written"
    assert result["depth_valid_pixel_count"] == 6
    assert result["controller_eligible"] is False

    with pytest.raises(NativeDepthCaptureError, match="refuse overwrite"):
        capture_sapien_depth_frame(**arguments)
    assert first_path.read_bytes() == first_bytes


def test_invalid_camera_name_never_becomes_output_path(tmp_path):
    path = capture_sapien_depth_frame(
        None,
        camera_id="../escaped-camera",
        frame_index=0,
        frame_association_source="bad input",
        output_dir=tmp_path / "capture",
    )
    assert path == tmp_path / "capture" / "invalid-capture_depth-manifest.json"
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "failed"
    assert not (tmp_path / "escaped-camera_frame_000000_depth-manifest.json").exists()


def test_float_sapien_color_hdr_must_not_downscale_metric_depth():
    """A Color channel >1 must not cause the 1/255 alpha-regression."""
    position = np.zeros((2, 3, 4), dtype=np.float32)
    position[..., 2] = -0.72
    position[..., 3] = 0.5
    color = np.ones((2, 3, 4), dtype=np.float32)
    color[..., 0] = 1.5  # HDR channel makes raw Color.max() > 1.
    color[0, 1, 3] = 0
    color[1, 1, 3] = 0.5  # Translucent: not reliable metric geometry.
    position[1, 2, 3] = 1.0  # SAPIEN far-plane z-buffer.
    depth = depth_millimetres_from_sapien(position, color)
    np.testing.assert_allclose(depth, [[720.0, 0.0, 720.0], [720.0, 0.0, 0.0]])


def test_float_sapien_capture_hdr_rgb_and_metric_depth_share_frame(tmp_path, monkeypatch):
    from PIL import Image

    from vla_robot_sorting.perception.native_depth_capture import (
        NATIVE_METRIC_DEPTH_CONVENTION,
    )

    monkeypatch.setattr("importlib.metadata.version", lambda _name: "3.0.0b1")

    class HDRNativeCamera:
        def get_picture(self, name):
            if name == "Color":
                image = np.full((3, 4, 4), 0.5, dtype=np.float32)
                image[..., 3] = 1.0
                image[0, 0, 0] = 1.5  # Raw image max > 1, RGB still normalized.
                return image
            image = np.zeros((3, 4, 4), dtype=np.float32)
            image[..., 2] = -0.72
            image[..., 3] = 0.5
            return image

        def get_intrinsic_matrix(self):
            return np.eye(3)

        def get_extrinsic_matrix(self):
            return np.eye(4)[:3]

        def get_model_matrix(self):
            return np.diag([1.0, -1.0, -1.0, 1.0])

    manifest_path = capture_sapien_depth_frame(
        HDRNativeCamera(),
        camera_id="head_camera",
        frame_index=0,
        frame_association_source="fixture updated render",
        frame_index_verified=True,
        output_dir=tmp_path,
    )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["status"] == "capture_written"
    assert data["depth_convention"] == NATIVE_METRIC_DEPTH_CONVENTION
    assert data["depth_valid_pixel_count"] == 12
    assert data["raw_color_rgb_float_min_max"] == [0.5, 1.5]
    assert data["raw_color_alpha_float_min_max"] == [1.0, 1.0]
    assert data["raw_position_z_m_min_max"] == pytest.approx([-0.72, -0.72], abs=1e-6)
    assert data["raw_position_zbuffer_min_max"] == [0.5, 0.5]
    depth = np.load(tmp_path / "head_camera_frame_000000_depth_mm.npy")
    np.testing.assert_allclose(depth, 720.0, atol=1e-4)
    rgba = np.asarray(Image.open(tmp_path / "head_camera_frame_000000_rgb.png"))
    assert rgba[0, 0, 0] == 255  # HDR clipped after *255
    assert rgba[1, 1, 0] in {127, 128}
    assert rgba[..., 3].min() == 255
