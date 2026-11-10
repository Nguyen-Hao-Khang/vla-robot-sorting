from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image
from scripts.replay_native_camera_episode import _verify_export_manifest

from vla_robot_sorting.perception.robotwin_hdf5 import (
    NativeEpisode,
    NativeEpisodeError,
    camera_to_world_cv_from_cam2world_gl,
)


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (8, 6), (21, 80, 140))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_episode(path: Path, *, frames: int = 2, omit: str | None = None) -> str:
    jpeg = _jpeg_bytes()
    dtype = np.dtype(f"S{len(jpeg)}")
    c2w_cv = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], frames, axis=0)
    c2w_cv[:, :3, 3] = (0.1, -0.2, 0.7)
    d_gl_to_cv = np.diag((1.0, -1.0, -1.0, 1.0)).astype(np.float32)
    c2w_gl = c2w_cv @ d_gl_to_cv
    world_to_camera_cv = np.linalg.inv(c2w_cv)[:, :3, :].astype(np.float32)
    with h5py.File(path, "w") as episode:
        episode.attrs["schema_version"] = "panthera-single-v1"
        episode.attrs["collection_fingerprint"] = "fixture-fingerprint"
        for camera in ("head_camera", "wrist_camera"):
            group = episode.create_group(f"observation/{camera}")
            datasets = {
                "rgb": np.asarray([jpeg] * frames, dtype=dtype),
                "intrinsic_cv": np.repeat(
                    np.asarray([[[100, 0, 4], [0, 100, 3], [0, 0, 1]]], dtype=np.float32),
                    frames,
                    axis=0,
                ),
                "extrinsic_cv": world_to_camera_cv,
                "cam2world_gl": c2w_gl,
            }
            for name, values in datasets.items():
                if omit != f"{camera}/{name}":
                    group.create_dataset(name, data=values)
        episode.create_dataset("joint_action/arm", data=np.zeros((frames, 6)))
        episode.create_dataset("joint_action/gripper", data=np.zeros(frames))
        episode.create_dataset("endpose/arm", data=np.zeros((frames, 7)))
        episode.create_dataset("endpose/gripper", data=np.zeros(frames))
        episode.create_dataset("joint_action/vector", data=np.zeros((frames, 7)))
        episode.create_dataset("pointcloud", data=np.empty((frames, 0)))
    return sha256(path.read_bytes()).hexdigest()


def test_native_episode_decodes_recorded_rgb_and_preserves_unavailable_scene_fields(tmp_path):
    path = tmp_path / "episode.hdf5"
    digest = _write_episode(path)
    frame = NativeEpisode(path, expected_sha256=digest).read_frame("head_camera", 1)

    assert frame.rgb_encoding == "jpeg"
    assert frame.rgb_image.size == (8, 6)
    assert frame.rgb_image.mode == "RGB"
    assert frame.camera_id == "head_camera"
    assert frame.frame_index == 1
    assert frame.frame_count == 2
    assert dict((path, shape) for path, shape, _dtype in frame.dataset_inventory)[
        "observation/head_camera/rgb"
    ] == (2,)
    assert frame.robot_fields.joint_action_arm == (0.0,) * 6
    assert frame.provenance.depth_available is False
    assert frame.provenance.depth_dataset_shape == (2, 0)
    assert len(frame.provenance.calibration_sha256) == 64
    assert frame.provenance.object_poses_available is False
    assert frame.provenance.timestamp_s is None
    assert frame.provenance.scene_revision is None
    assert frame.provenance.controller_eligible is False


def test_episode_rejects_wrong_sha256_before_reading(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_episode(path)

    with pytest.raises(NativeEpisodeError, match="SHA256"):
        NativeEpisode(path, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    ("camera", "index"), [("unknown", 0), ("head_camera", -1), ("wrist_camera", 2)]
)
def test_episode_rejects_unknown_camera_or_out_of_range_frame(tmp_path, camera, index):
    path = tmp_path / "episode.hdf5"
    digest = _write_episode(path)
    reader = NativeEpisode(path, expected_sha256=digest)

    with pytest.raises(NativeEpisodeError):
        reader.read_frame(camera, index)


def test_episode_rejects_missing_rgb_dataset(tmp_path):
    path = tmp_path / "episode.hdf5"
    digest = _write_episode(path, omit="head_camera/rgb")
    reader = NativeEpisode(path, expected_sha256=digest)

    with pytest.raises(NativeEpisodeError, match="rgb"):
        reader.read_frame("head_camera", 0)


def test_episode_rejects_non_hdf5_payload(tmp_path):
    path = tmp_path / "broken.hdf5"
    path.write_bytes(b"not an HDF5 file")
    digest = sha256(path.read_bytes()).hexdigest()

    with pytest.raises(NativeEpisodeError, match="HDF5"):
        NativeEpisode(path, expected_sha256=digest)


def test_adapter_converts_opengl_camera_to_world_to_cv_camera_to_world():
    c2w_cv = np.eye(4)
    c2w_cv[:3, :3] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    c2w_cv[:3, 3] = (0.2, -0.4, 1.3)
    cv_to_gl = np.diag((1, -1, -1, 1))
    actual = np.asarray(camera_to_world_cv_from_cam2world_gl(c2w_cv @ cv_to_gl))

    np.testing.assert_allclose(actual, c2w_cv, atol=1e-12)


def test_episode_audits_cv_gl_convention_for_both_cameras_and_every_frame(tmp_path):
    path = tmp_path / "episode.hdf5"
    digest = _write_episode(path, frames=3)
    audit = NativeEpisode(path, expected_sha256=digest).audit_camera_conventions()

    assert audit.camera_count == 2
    assert audit.frame_count_per_camera == 3
    assert audit.checked_transform_count == 6
    assert audit.max_abs_error < 1e-6


def test_episode_rejects_inconsistent_cv_and_gl_calibration(tmp_path):
    path = tmp_path / "episode.hdf5"
    digest = _write_episode(path)
    with h5py.File(path, "r+") as episode:
        matrix = episode["observation/head_camera/extrinsic_cv"]
        matrix[0, 0, 3] += 0.1
    digest = sha256(path.read_bytes()).hexdigest()
    reader = NativeEpisode(path, expected_sha256=digest)

    with pytest.raises(NativeEpisodeError, match="inconsistent"):
        reader.read_frame("head_camera", 0)



def test_episode_rejects_noncanonical_or_skewed_intrinsics(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_episode(path)
    with h5py.File(path, "r+") as episode:
        episode["observation/head_camera/intrinsic_cv"][0, 0, 1] = 0.25
    digest = sha256(path.read_bytes()).hexdigest()

    with pytest.raises(NativeEpisodeError, match="skew"):
        NativeEpisode(path, expected_sha256=digest).read_frame("head_camera", 0)


def test_episode_rejects_corrupt_jpeg_payload(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_episode(path)
    with h5py.File(path, "r+") as episode:
        episode["observation/head_camera/rgb"][0] = b"broken"
    digest = sha256(path.read_bytes()).hexdigest()

    with pytest.raises(NativeEpisodeError, match="encoding"):
        NativeEpisode(path, expected_sha256=digest).read_frame("head_camera", 0)


def test_opengl_transform_rejects_nonrigid_camera_matrix():
    invalid = np.eye(4)
    invalid[0, 0] = 1.1

    with pytest.raises(NativeEpisodeError, match="orthonormal"):
        camera_to_world_cv_from_cam2world_gl(invalid)



def test_export_manifest_binds_episode_path_size_and_checksum(tmp_path):
    export_root = tmp_path / "export"
    episode = export_root / "files" / "task" / "data" / "episode0.hdf5"
    episode.parent.mkdir(parents=True)
    episode.write_bytes(b"recorded-hdf5-payload")
    digest = sha256(episode.read_bytes()).hexdigest()
    report_path = export_root / "export-report.json"
    report_path.write_text(
        json.dumps(
            {
                "classification": "S1V_SUPPLEMENTAL_RAW_EXPORT_NOT_ACCEPTED",
                "files": [
                    {
                        "original": "/home/source/episode0.hdf5",
                        "exported": (
                            "/mnt/data/DAMH-Bridge/incoming/export/files/"
                            "task/data/episode0.hdf5"
                        ),
                        "size_bytes": episode.stat().st_size,
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report, report_sha256 = _verify_export_manifest(report_path, episode, digest)

    assert report["classification"] == "S1V_SUPPLEMENTAL_RAW_EXPORT_NOT_ACCEPTED"
    assert len(report_sha256) == 64
    with pytest.raises(NativeEpisodeError, match="export report SHA256"):
        _verify_export_manifest(report_path, episode, "0" * 64)


def test_export_manifest_rejects_a_path_outside_its_files_root(tmp_path):
    export_root = tmp_path / "export"
    files_root = export_root / "files"
    files_root.mkdir(parents=True)
    episode = tmp_path / "outside.hdf5"
    episode.write_bytes(b"not-exported")
    digest = sha256(episode.read_bytes()).hexdigest()
    report_path = export_root / "export-report.json"
    report_path.write_text('{"files": []}', encoding="utf-8")

    with pytest.raises(NativeEpisodeError, match="under the export"):
        _verify_export_manifest(report_path, episode, digest)


def test_unorganized_nonempty_pointcloud_is_not_pixel_depth(tmp_path):
    path = tmp_path / "unorganized-pointcloud.hdf5"
    _write_episode(path)
    with h5py.File(path, "r+") as episode:
        del episode["pointcloud"]
        episode.create_dataset("pointcloud", data=np.ones((2, 24, 3)))
    digest = sha256(path.read_bytes()).hexdigest()
    frame = NativeEpisode(path, expected_sha256=digest).read_frame("head_camera", 0)
    assert frame.provenance.depth_available is False
    assert frame.provenance.depth_dataset_shape == (2, 24, 3)
    assert frame.provenance.object_poses_available is False
    assert frame.provenance.controller_eligible is False


def test_replay_depth_must_match_selected_camera_resolution(tmp_path):
    path = tmp_path / "misaligned-depth.hdf5"
    _write_episode(path)
    with h5py.File(path, "r+") as episode:
        episode["observation/head_camera"].create_dataset(
            "depth", data=np.ones((2, 3, 4), dtype=np.float32)
        )
    digest = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(NativeEpisodeError, match="align"):
        NativeEpisode(path, expected_sha256=digest).read_frame("head_camera", 0)

    # Depth for another camera cannot supply head-camera pixel depth.
    frame = NativeEpisode(path, expected_sha256=digest).read_frame("wrist_camera", 0)
    assert frame.provenance.depth_available is False


def test_camera_aligned_depth_presence_does_not_make_frame_executable(tmp_path):
    path = tmp_path / "camera-aligned-depth.hdf5"
    _write_episode(path)
    with h5py.File(path, "r+") as episode:
        episode["observation/head_camera"].create_dataset(
            "depth", data=np.ones((2, 6, 8), dtype=np.float32)
        )
    digest = sha256(path.read_bytes()).hexdigest()
    frame = NativeEpisode(path, expected_sha256=digest).read_frame("head_camera", 0)
    assert frame.provenance.depth_available is True
    assert frame.provenance.depth_dataset_shape == (2, 6, 8)
    assert frame.provenance.controller_eligible is False
    assert frame.provenance.object_poses_available is False
