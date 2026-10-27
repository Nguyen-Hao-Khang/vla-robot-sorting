# ruff: noqa: E501  # Scoped imported native evidence source; no semantic changes.
"""Source-verified RGB-D color *surface* geometry, never object identity/pose.

This is a read-only S1-I-V2a partial-observation measurement, NOT a
pixel-to-object recognizer, 6-DoF estimator, or motion-control interface.

Input frames are simulator RGB-D recordings, with WALL-CLOCK capture times.
Color selection is a documented HSV heuristic and may select labels/caps/etc.
Even close 3D measurements across cameras DO NOT prove synchronization,
correspondence, complete dimensions, semantic identity, or controller eligibility.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

CAMERAS = ("head_camera", "wrist_camera")
BANDS = {
    "orange_surface": ((0, 100, 70), (20, 255, 255)),
    "blue_surface": ((105, 100, 70), (135, 255, 255)),
}
CAPTURE_FRAME_ASSOCIATION = "probe.get_obs single-frame camera refresh"
CAPTURE_TIMESTAMP_SEMANTICS = "probe wall-clock time; not a simulator sensor timestamp"
PRODUCER_METHOD = "HSV(orange/blue)+3x3_OPEN+largest_component+measured_RGBD_v1"


class SurfaceEvidenceError(ValueError):
    """Source/geometry integrity, or expected sensor conventions, violated."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise SurfaceEvidenceError(message)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_ok(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _read_manifest(root: Path, camera: str) -> tuple[dict[str, Any], str, np.ndarray, np.ndarray, np.ndarray, str]:
    _require(camera in CAMERAS, "invalid native camera")
    prefix = f"{camera}_frame_000000"
    path = root / f"{prefix}_depth-manifest.json"
    raw = path.read_bytes()
    manifest = json.loads(raw)
    _require(isinstance(manifest, dict), "manifest is not an object")
    _require(manifest.get("schema") == "damh.robotwin-native-rgbd-capture.v1", "capture schema mismatch")
    _require(manifest.get("status") == "capture_written", "unwritten camera frame")
    _require(manifest.get("source_classification") == "recorded_sim_observation", "capture source not recorded native")
    _require(manifest.get("camera_id") == camera and type(manifest.get("frame_index")) is int and manifest["frame_index"] == 0, "camera/frame conflict")
    _require(manifest.get("frame_index_verified_by_caller") is True, "unverified captured frame index")
    _require(manifest.get("frame_association_source") == CAPTURE_FRAME_ASSOCIATION, "unverified same-refresh camera capture")
    _require(manifest.get("timestamp_semantics") == CAPTURE_TIMESTAMP_SEMANTICS, "sensor timestamp semantics unexpectedly changed")
    _require(manifest.get("scene_revision") is None, "scene identity claim not supported")
    _require(manifest.get("controller_eligible") is False, "source claimed control eligibility")
    _require(manifest.get("depth_units") == "millimetres" and manifest.get("depth_encoding") == "NumPy .npy float64", "unexpected depth encoding/units")
    _require(manifest.get("depth_convention") == "sapien3_position_camera_z_mm_masked_by_float_alpha_and_position_zbuffer_v2", "unexpected depth semantics")
    _require(manifest.get("native_capture_executed") is True, "capture is not native")
    utc = manifest.get("capture_timestamp_utc")
    _require(isinstance(utc, str) and utc.endswith("+00:00"), "capture wall-clock timestamp invalid")
    try:
        datetime.fromisoformat(utc)
    except ValueError as exc:
        raise SurfaceEvidenceError("unparseable capture wall-clock") from exc
    dims = manifest.get("image_size_px")
    _require(isinstance(dims, list) and len(dims) == 2 and all(type(v) is int and 0 < v <= 4096 for v in dims), "image size invalid")
    w, h = dims

    rgb_path = root / f"{prefix}_rgb.png"
    depth_path = root / f"{prefix}_depth_mm.npy"
    for kind, file in (("rgb", rgb_path), ("depth", depth_path)):
        expected = manifest.get(f"{kind}_sha256")
        _require(_digest_ok(expected) and _hash(file.read_bytes()) == expected, f"{camera} {kind} digest mismatch")
    with Image.open(rgb_path) as image:
        _require(image.size == (w, h), "RGB frame shape does not match manifest")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    depth = np.load(depth_path, allow_pickle=False)
    _require(depth.dtype == np.float64 and depth.shape == (h, w), "native depth dtype/shape mismatch")

    k = np.asarray(manifest.get("intrinsic_cv"), dtype=np.float64)
    transform = np.asarray(manifest.get("camera_to_world_cv"), dtype=np.float64)
    extrinsic = np.asarray(manifest.get("extrinsic_cv_world_to_camera_cv"), dtype=np.float64)
    _require(k.shape == (3, 3) and transform.shape == (4, 4) and extrinsic.shape == (3, 4), "invalid camera calibration shapes")
    _require(np.all(np.isfinite(k)) and np.all(np.isfinite(transform)) and np.all(np.isfinite(extrinsic)), "nonfinite calibration")
    _require(k[0, 0] > 0 and k[1, 1] > 0 and np.allclose(k[2], [0, 0, 1], atol=1e-7) and abs(k[0, 1]) < 1e-7 and abs(k[1, 0]) < 1e-7, "noncanonical pixel calibration")
    _require(np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6), "camera transform is not homogeneous")
    r = transform[:3, :3]
    _require(np.allclose(r.T @ r, np.eye(3), atol=1e-4) and abs(np.linalg.det(r) - 1) < 1e-4, "camera rotation not rigid")
    _require(np.allclose(extrinsic @ transform, np.eye(4)[:3], atol=1e-4), "camera-to-world and world-to-camera calibration disagree")
    return manifest, _hash(raw), rgb, depth, transform, utc


def _measured_color_surface(
    rgb: np.ndarray, depth_mm: np.ndarray, k: np.ndarray, transform: np.ndarray,
    kind: str, *, min_mask_px: int = 30,
) -> dict[str, Any] | None:
    """Largest contiguous HSV color patch; point cloud is only *visible patch*."""
    low, high = BANDS[kind]
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    binary = cv2.inRange(hsv, np.asarray(low, dtype=np.uint8), np.asarray(high, dtype=np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = [i for i in range(1, count) if int(stats[i, cv2.CC_STAT_AREA]) >= min_mask_px]
    if not components:
        return None
    chosen = sorted(components, key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), int(stats[i, cv2.CC_STAT_TOP]), i))[0]
    color_mask = labels == chosen
    total = int(color_mask.sum())
    valid = color_mask & np.isfinite(depth_mm) & (depth_mm >= 50.0) & (depth_mm <= 3000.0)
    valid_count = int(valid.sum())
    if valid_count < 4 or valid_count / total < 0.60:
        return None
    yy, xx = np.nonzero(valid)
    z = depth_mm[yy, xx] / 1000.0
    depth_q = np.percentile(z, [25, 50, 75])
    if depth_q[2] - depth_q[0] > 0.080:
        return None
    camera_points = np.stack(((xx - k[0, 2]) / k[0, 0] * z,
                              (yy - k[1, 2]) / k[1, 1] * z, z), axis=1)
    world_points = camera_points @ transform[:3, :3].T + transform[:3, 3]
    _require(np.all(np.isfinite(world_points)), "nonfinite projected world points")
    means = np.mean(world_points, axis=0)
    q = np.percentile(world_points, [5, 95], axis=0)
    centered = world_points - means
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    residual = np.abs(centered @ normal)
    plane_p95 = float(np.percentile(residual, 95))
    if normal[2] < 0:
        normal = -normal
    left, top, width, height = (int(x) for x in stats[chosen, :4])
    return {
        "kind": kind,
        "color_selection": "heuristic_largest_hsv_connected_component_not_object_detector",
        "hsv_lower": list(low), "hsv_upper": list(high),
        "bbox_xyxy_px": [left, top, left + width, top + height],
        "masked_pixel_count": total, "valid_depth_pixel_count": valid_count,
        "valid_depth_fraction": round(valid_count / total, 6),
        "other_same_color_components": len(components) - 1,
        "depth_m_median": float(depth_q[1]),
        "depth_m_iqr": float(depth_q[2] - depth_q[0]),
        "visible_surface_centroid_world_m": means.tolist(),
        "visible_surface_world_bounds_q05_q95_m": q.tolist(),
        "visible_surface_best_fit_plane_normal_world": normal.tolist(),
        "visible_surface_plane_residual_p95_m": plane_p95,
        "visible_surface_near_horizontal_planarity_heuristic": bool(plane_p95 <= .002 and abs(normal[2]) >= .9),
        "object_id": None, "object_type": None, "object_pose_6dof": None,
        "object_dimensions_m": None, "object_identity_confidence": None,
        "object_correspondence_verified": False, "grasp_validated": False,
        "controller_eligible": False,
    }


def evaluate_native_color_surfaces(native_dir: str | Path) -> dict[str, Any]:
    """Two native cameras, complete SHA checks, no robot/semantic/physics calls."""
    root = Path(native_dir).resolve(strict=True)
    _require(root.is_dir(), "not a native RGB-D directory")
    cameras = []
    for camera in CAMERAS:
        manifest, manifest_sha, rgb, depth_mm, transform, timestamp = _read_manifest(root, camera)
        k = np.asarray(manifest["intrinsic_cv"], dtype=np.float64)
        surfaces = []
        for kind in BANDS:
            surface = _measured_color_surface(rgb, depth_mm, k, transform, kind)
            if surface is not None:
                surfaces.append(surface)
        cameras.append({
            "camera_id": camera, "frame_index": 0,
            "manifest_sha256": manifest_sha,
            "rgb_sha256": manifest["rgb_sha256"],
            "depth_sha256": manifest["depth_sha256"],
            "capture_wallclock_utc": timestamp,
            "timestamp_semantics": CAPTURE_TIMESTAMP_SEMANTICS,
            "image_size_px": manifest["image_size_px"],
            "surfaces": surfaces,
            "object_identity_verified": False,
            "scene_snapshot_created": False,
            "controller_eligible": False,
        })
    same_color_comparisons = []
    head = {s["kind"]: s for s in cameras[0]["surfaces"]}
    wrist = {s["kind"]: s for s in cameras[1]["surfaces"]}
    for kind in sorted(set(head) & set(wrist)):
        p = np.asarray(head[kind]["visible_surface_centroid_world_m"])
        q = np.asarray(wrist[kind]["visible_surface_centroid_world_m"])
        delta = float(np.linalg.norm(p-q))
        same_color_comparisons.append({
            "kind": kind,
            "unpaired_color_centroid_distance_m": delta,
            "same_physical_surface_correspondence_verified": False,
            "sensor_timestamp_synchronized": False,
            "object_identity_verified": False,
            "claim": "diagnostic difference between two color-selected surface patches; not a matched object/pose",
        })
    wallclock_delta_s = abs((datetime.fromisoformat(cameras[1]["capture_wallclock_utc"])
                            - datetime.fromisoformat(cameras[0]["capture_wallclock_utc"])).total_seconds())
    config = {"method": PRODUCER_METHOD, "bands": BANDS, "min_mask_px": 30,
              "minimum_valid_fraction": .6, "maximum_depth_iqr_m": .08,
              "valid_depth_range_m": [.05, 3.]}
    result = {
        "schema": "s1i.native-colored-surface-geometry.v1",
        "classification": "TWO_CAMERA_MEASURED_COLOR_PATCHES_NON_CONTROL",
        "producer_method": PRODUCER_METHOD,
        "producer_config_sha256": _hash(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()),
        "source_classification": "recorded_sim_observation",
        "camera_count": len(cameras), "cameras": cameras,
        "cross_camera_wallclock_capture_delta_s": wallclock_delta_s,
        "cross_camera_sensor_timestamp_synchronized": False,
        "cross_camera_object_correspondence_verified": False,
        "same_color_diagnostics": same_color_comparisons,
        "scene_revision": None,
        "verified_object_identity": False, "object_6dof_pose_verified": False,
        "complete_object_dimensions_verified": False,
        "scene_snapshot_created": False,
        "semantic_grounding_authorized": False, "grasp_planning_authorized": False,
        "controller_eligible": False, "native_physics_steps": 0, "gemini_api_requests": 0,
        "claim_boundary": "HSV cap/pad-like colored *patches* with source-bounded RGBD metric visible-surface points. No proven physical-object labels, correspondences, full dimensions, orientations, sensor synchronization, or motion authority.",
    }
    result["report_sha256"] = _hash(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return result
