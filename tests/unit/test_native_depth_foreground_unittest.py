#!/usr/bin/env python3
# ruff: noqa: E501  # Scoped imported native evidence source; no semantic changes.
"""Offline non-control segmentation checks against the source-locked native RGB-D pair.

Set S1I_NATIVE_DIR to original 2026-10-13 depth-repair capture directory.
Tests copy files before applying adversarial changes; source never changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from vla_robot_sorting.perception.native_depth_foreground_proposal import (
    CONFIG,
    _segment_native_camera,
    _world_cloud,
    evaluate_native_depth_foreground,
)
from vla_robot_sorting.perception.native_multiview_surface_geometry import (
    SurfaceEvidenceError,
    _read_manifest,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class NativeDepthForegroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = os.environ.get("S1I_NATIVE_DIR")
        if not path:
            raise unittest.SkipTest("S1I_NATIVE_DIR is not set; these tests need the source-locked native RGB-D capture")
        cls.root = Path(path).resolve(strict=True)
        for camera in ("head_camera", "wrist_camera"):
            _read_manifest(cls.root, camera)
        cls.result = evaluate_native_depth_foreground(cls.root)

    def _clone(self, *, camera: str = "head_camera") -> tuple[tempfile.TemporaryDirectory[str], Path, str]:
        ctx = tempfile.TemporaryDirectory()
        temp = Path(ctx.name)
        for file in self.root.iterdir():
            if file.name.endswith(("depth-manifest.json", "_depth_mm.npy", "_rgb.png")):
                shutil.copy2(file, temp / file.name)
        return ctx, temp, f"{camera}_frame_000000"

    def _rewrite_rgb(self, root: Path, stem: str, rgb: np.ndarray) -> None:
        path = root / f"{stem}_rgb.png"
        Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)
        manifest_path = root / f"{stem}_depth-manifest.json"
        m = json.loads(manifest_path.read_text())
        m["rgb_sha256"] = digest(path.read_bytes())
        manifest_path.write_text(json.dumps(m))

    def _rewrite_depth(self, root: Path, stem: str, depth: np.ndarray) -> None:
        path = root / f"{stem}_depth_mm.npy"
        np.save(path, depth.astype(np.float64))
        manifest_path = root / f"{stem}_depth-manifest.json"
        m = json.loads(manifest_path.read_text())
        m["depth_sha256"] = digest(path.read_bytes())
        manifest_path.write_text(json.dumps(m))

    def test_two_native_camera_foreground_proposals(self) -> None:
        self.assertEqual(self.result["camera_count"], 2)
        self.assertEqual([x["camera_id"] for x in self.result["cameras"]], ["head_camera", "wrist_camera"])
        for row in self.result["cameras"]:
            self.assertGreaterEqual(row["foreground_pixel_count"], CONFIG["min_foreground_pixels"])
            self.assertGreater(row["visible_foreground_vertical_q05_q95_span_m"], .04)
            self.assertGreaterEqual(row["orange_anchor_supported_fraction"], .9)
            self.assertEqual(len(row["mask_uint8_row_major_sha256"]), 64)

    def test_expected_non_control_claim_boundaries(self) -> None:
        for key in ("cross_camera_correspondence_verified", "object_instance_segmentation_verified",
                    "object_identity_verified", "object_pose_6dof_verified", "scene_snapshot_created",
                    "semantic_grounding_authorized", "grasp_planning_authorized", "controller_eligible"):
            self.assertIs(self.result[key], False, key)
        for row in self.result["cameras"]:
            self.assertIsNone(row["object_id"])
            self.assertIsNone(row["object_class"])
            self.assertIsNone(row["object_pose_6dof"])
            self.assertIsNone(row["object_identity_confidence"])
            self.assertIsNone(row["complete_object_dimensions_m"])

    def test_mask_bbox_within_camera(self) -> None:
        for row in self.result["cameras"]:
            h, w = row["mask_shape_hw"]
            x0, y0, x1, y1 = row["bbox_xyxy_px"]
            self.assertTrue(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h)

    def test_source_hashes_preserved(self) -> None:
        for row in self.result["cameras"]:
            manifest, digest_manifest, _, _, _, _ = _read_manifest(self.root, row["camera_id"])
            self.assertEqual(row["source_manifest_sha256"], digest_manifest)
            self.assertEqual(row["source_rgb_sha256"], manifest["rgb_sha256"])
            self.assertEqual(row["source_depth_sha256"], manifest["depth_sha256"])

    def test_mask_pixel_count_and_mask_digest_recomputed(self) -> None:
        for camera, recorded in zip(("head_camera", "wrist_camera"), self.result["cameras"], strict=True):
            manifest, _, rgb, depth, transform, _ = _read_manifest(self.root, camera)
            row, mask = _segment_native_camera(rgb, depth, np.asarray(manifest["intrinsic_cv"]), transform, camera_id=camera)
            self.assertEqual(int(mask.sum()), recorded["foreground_pixel_count"])
            self.assertEqual(row["mask_uint8_row_major_sha256"], recorded["mask_uint8_row_major_sha256"])
            self.assertFalse(mask[0, 0])

    def test_cap_reference_is_only_color_heuristic(self) -> None:
        for row in self.result["cameras"]:
            self.assertIn("not_verified", row["color_anchor_role"])
            self.assertIn("not_verified", row["reference_plane_role"])
            self.assertIn("not_verified_object_instance", row["mask_semantics"])

    def test_cross_camera_wallclock_not_sensor_sync(self) -> None:
        self.assertIs(self.result["cross_camera_sensor_timestamp_synchronized"], False)
        for cam in self.result["cameras"]:
            self.assertIn("wall-clock", cam["timestamp_semantics"])
            self.assertIsNone(cam["scene_revision"])

    def test_no_physics_or_provider_calls(self) -> None:
        self.assertEqual(self.result["native_physics_steps"], 0)
        self.assertEqual(self.result["gemini_api_requests"], 0)

    def test_world_backprojection_identity(self) -> None:
        depth = np.full((2, 2), 1000., dtype=np.float64)
        k = np.array([[10., 0., 0.], [0., 10., 0.], [0., 0., 1.]])
        world = _world_cloud(depth, k, np.eye(4))
        self.assertTrue(np.allclose(world[1, 1], [.1, .1, 1.]))

    def test_missing_orange_reference_rejects(self) -> None:
        ctx, root, stem = self._clone()
        with ctx:
            rgb = np.asarray(Image.open(root / f"{stem}_rgb.png").convert("RGB")).copy()
            # All orange color removed; blue preserved.
            rgb[..., 0] = np.minimum(rgb[..., 0], 30)
            self._rewrite_rgb(root, stem, rgb)
            with self.assertRaises(SurfaceEvidenceError):
                evaluate_native_depth_foreground(root)

    def test_missing_blue_reference_rejects(self) -> None:
        ctx, root, stem = self._clone()
        with ctx:
            rgb = np.asarray(Image.open(root / f"{stem}_rgb.png").convert("RGB")).copy()
            # Change all saturated blues to grey; orange preserved.
            blue = (rgb[..., 2] > 150) & (rgb[..., 2] > rgb[..., 0] * 1.5) & (rgb[..., 2] > rgb[..., 1] * 1.5)
            rgb[blue] = 128
            self._rewrite_rgb(root, stem, rgb)
            with self.assertRaises(SurfaceEvidenceError):
                evaluate_native_depth_foreground(root)

    def test_original_rgb_tamper_rejects(self) -> None:
        ctx, root, stem = self._clone()
        with ctx:
            rgb_file = root / f"{stem}_rgb.png"
            raw = bytearray(rgb_file.read_bytes())
            raw[40] ^= 1
            rgb_file.write_bytes(raw)
            with self.assertRaises(SurfaceEvidenceError):
                evaluate_native_depth_foreground(root)

    def test_original_depth_tamper_rejects(self) -> None:
        ctx, root, stem = self._clone()
        with ctx:
            depth_file = root / f"{stem}_depth_mm.npy"
            raw = bytearray(depth_file.read_bytes())
            raw[100] ^= 1
            depth_file.write_bytes(raw)
            with self.assertRaises(SurfaceEvidenceError):
                evaluate_native_depth_foreground(root)

    def test_all_depth_nan_cannot_make_foreground(self) -> None:
        ctx, root, stem = self._clone()
        with ctx:
            depth = np.load(root / f"{stem}_depth_mm.npy")
            depth[:] = np.nan
            self._rewrite_depth(root, stem, depth)
            with self.assertRaises(SurfaceEvidenceError):
                evaluate_native_depth_foreground(root)

    def test_reference_plane_incompatible_across_cameras(self) -> None:
        ctx, root, stem = self._clone(camera="wrist_camera")
        with ctx:
            manifest_path = root / f"{stem}_depth-manifest.json"
            m = json.loads(manifest_path.read_text())
            T = np.asarray(m["camera_to_world_cv"], dtype=np.float64)
            T[2, 3] += .07
            m["camera_to_world_cv"] = T.tolist()
            m["extrinsic_cv_world_to_camera_cv"] = np.linalg.inv(T)[:3].tolist()
            manifest_path.write_text(json.dumps(m))
            with self.assertRaisesRegex(SurfaceEvidenceError, "incompatible"):
                evaluate_native_depth_foreground(root)

    def test_camera_frame_identity_tamper_rejected(self) -> None:
        ctx, root, stem = self._clone()
        with ctx:
            manifest_path = root / f"{stem}_depth-manifest.json"
            m = json.loads(manifest_path.read_text())
            m["frame_index"] = 1
            manifest_path.write_text(json.dumps(m))
            with self.assertRaises(SurfaceEvidenceError):
                evaluate_native_depth_foreground(root)

    def test_unverified_scene_revision_rejected(self) -> None:
        ctx, root, stem = self._clone()
        with ctx:
            manifest_path = root / f"{stem}_depth-manifest.json"
            m = json.loads(manifest_path.read_text())
            m["scene_revision"] = "invented-scene"
            manifest_path.write_text(json.dumps(m))
            with self.assertRaises(SurfaceEvidenceError):
                evaluate_native_depth_foreground(root)

    def test_config_too_small_rejects_cap_anchoring(self) -> None:
        with (
            patch.dict(CONFIG, {"max_horizontal_distance_from_cap_m": .001}),
            self.assertRaises(SurfaceEvidenceError),
        ):
            evaluate_native_depth_foreground(self.root)

    def test_intercamera_consistency_is_not_object_correspondence(self) -> None:
        self.assertLess(self.result["cross_view_color_anchor_xy_distance_m"], .04)
        self.assertIs(self.result["cross_camera_correspondence_verified"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
