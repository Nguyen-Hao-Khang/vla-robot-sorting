# ruff: noqa: E501  # Scoped imported native evidence source; no semantic changes.
"""S1-I-V2b: color-anchored RGB-D foreground *proposal* from native captures.

The mask is a visible, depth-supported connected component near an orange
reference patch and above a blue reference plane. It is NOT a verified object
segmentation, semantic identity, 6-DoF pose, geometry/dimensions or a grasp.
This intentionally requires both color references. All thresholds are explicit
heuristics, valid only for a narrowly defined scene/camera configuration.
"""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vla_robot_sorting.perception.native_multiview_surface_geometry import (
    BANDS,
    CAMERAS,
    _measured_color_surface,
    _read_manifest,
    _require,
)

METHOD = "cap_seeded_world_xy_gating_above_measured_blue_plane_cc_v1"
CONFIG: dict[str, Any] = {
    "min_color_pixels": 30,
    "min_foreground_pixels": 200,
    "min_anchored_cap_fraction": 0.9,
    "min_visible_vertical_span_m": 0.04,
    "max_reference_plane_p95_residual_m": 0.002,
    "min_reference_plane_normal_z": 0.9,
    "min_height_above_plane_m": 0.012,
    "max_vertical_offset_above_cap_m": 0.045,
    "max_horizontal_distance_from_cap_m": 0.048,
    "max_intercamera_plane_height_difference_m": 0.015,
    "max_intercamera_cap_xy_difference_m": 0.040,
    "max_intercamera_cap_height_difference_m": 0.025,
    "max_intercamera_foreground_xy_bounds_q05q95_delta_m": 0.045,
}


def _world_cloud(
    depth_mm: np.ndarray, intrinsic: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Backproject only the native camera-Z depth convention."""
    h, w = depth_mm.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    camera_z = depth_mm / 1000.0
    points = np.stack((
        (xx - intrinsic[0, 2]) * camera_z / intrinsic[0, 0],
        (yy - intrinsic[1, 2]) * camera_z / intrinsic[1, 1],
        camera_z,
    ), axis=-1)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _largest_color_component(rgb: np.ndarray, kind: str) -> np.ndarray | None:
    low, high = BANDS[kind]
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    raw = cv2.inRange(hsv, np.asarray(low, np.uint8), np.asarray(high, np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= CONFIG["min_color_pixels"]]
    if not candidates:
        return None
    selected = sorted(candidates, key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), int(stats[i, cv2.CC_STAT_TOP]), i))[0]
    return labels == selected


def _segment_native_camera(
    rgb: np.ndarray, depth_mm: np.ndarray, intrinsic: np.ndarray,
    transform: np.ndarray,
    *, camera_id: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return one scene-constrained foreground component or reject explicitly."""
    _require(rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8,
             "RGB must be uint8 HxWx3")
    _require(depth_mm.shape == rgb.shape[:2] and depth_mm.dtype == np.float64,
             "depth must be matched float64 native mm")
    plane_patch = _measured_color_surface(rgb, depth_mm, intrinsic, transform, "blue_surface")
    cap_patch = _measured_color_surface(rgb, depth_mm, intrinsic, transform, "orange_surface")
    _require(plane_patch is not None and cap_patch is not None,
             "reference blue plane and orange cap-like patch required")
    _require(plane_patch["visible_surface_plane_residual_p95_m"] <= CONFIG["max_reference_plane_p95_residual_m"] and
             abs(plane_patch["visible_surface_best_fit_plane_normal_world"][2]) >= CONFIG["min_reference_plane_normal_z"],
             "blue reference patch not near horizontal/planar")
    plane_z = float(plane_patch["visible_surface_centroid_world_m"][2])
    cap_mask = _largest_color_component(rgb, "orange_surface")
    _require(cap_mask is not None, "orange cap anchor unavailable")

    valid_depth = np.isfinite(depth_mm) & (depth_mm >= 50) & (depth_mm <= 3000)
    _require(int((cap_mask & valid_depth).sum()) >= CONFIG["min_color_pixels"], "no anchor depth")
    world = _world_cloud(depth_mm, intrinsic, transform)
    _require(np.all(np.isfinite(world[valid_depth])), "nonfinite metric world coordinates")
    cap_points = world[cap_mask & valid_depth]
    cap_xy = np.median(cap_points[:, :2], axis=0)
    cap_height = float(np.median(cap_points[:, 2]))
    _require(cap_height > plane_z + CONFIG["min_height_above_plane_m"] + CONFIG["min_visible_vertical_span_m"],
             "cap anchor not clearly above reference surface")

    horizontal_dist = np.linalg.norm(world[..., :2] - cap_xy, axis=-1)
    potential = (
        valid_depth &
        (world[..., 2] >= plane_z + CONFIG["min_height_above_plane_m"]) &
        (world[..., 2] <= cap_height + CONFIG["max_vertical_offset_above_cap_m"]) &
        (horizontal_dist <= CONFIG["max_horizontal_distance_from_cap_m"])
    )
    total_cap = int((cap_mask & valid_depth).sum())
    n_components, labels, stats, _ = cv2.connectedComponentsWithStats(potential.astype(np.uint8), connectivity=8)
    candidate_ids = []
    for i in range(1, n_components):
        pixels = int(stats[i, cv2.CC_STAT_AREA])
        if pixels < CONFIG["min_foreground_pixels"]:
            continue
        support = int(np.count_nonzero((labels == i) & cap_mask))
        if support / total_cap >= CONFIG["min_anchored_cap_fraction"]:
            candidate_ids.append(i)
    _require(len(candidate_ids) == 1,
             "no unique measured foreground component connected to color anchor")
    selected_mask = labels == candidate_ids[0]
    points = world[selected_mask]
    _require(np.all(np.isfinite(points)) and points.shape[0] >= CONFIG["min_foreground_pixels"],
             "foreground point cloud invalid")
    q = np.percentile(points, (5, 50, 95), axis=0)
    _require(float(q[2, 2] - q[0, 2]) >= CONFIG["min_visible_vertical_span_m"],
             "foreground lacks visible vertical extent")
    yy, xx = np.nonzero(selected_mask)
    _require(len(yy) > 0, "mask unexpectedly empty")
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    mask_checksum = sha256(np.ascontiguousarray(selected_mask.astype(np.uint8)).tobytes()).hexdigest()
    report: dict[str, Any] = {
        "camera_id": camera_id,
        "bbox_xyxy_px": bbox,
        "mask_shape_hw": list(selected_mask.shape),
        "mask_uint8_row_major_sha256": mask_checksum,
        "foreground_pixel_count": len(yy),
        "orange_anchor_valid_pixel_count": total_cap,
        "orange_anchor_supported_fraction": float(np.count_nonzero(selected_mask & cap_mask) / total_cap),
        "reference_plane_height_world_z_m": plane_z,
        "reference_plane_p95_residual_m": plane_patch["visible_surface_plane_residual_p95_m"],
        "anchor_median_world_xyz_m": [float(cap_xy[0]), float(cap_xy[1]), cap_height],
        "visible_foreground_centroid_world_m": points.mean(axis=0).tolist(),
        "visible_foreground_world_q05_q50_q95_m": q.tolist(),
        "visible_foreground_vertical_q05_q95_span_m": float(q[2, 2] - q[0, 2]),
        "color_anchor_role": "heuristic_orange_cap_like_patch_not_verified_cap_or_object_identity",
        "reference_plane_role": "heuristic_blue_near_horizontal_surface_not_verified_table_or_destination",
        "proposal_kind": "color_anchored_depth_connected_foreground_component",
        "mask_semantics": "foreground_image_region_only_not_verified_object_instance",
        "object_id": None,
        "object_class": None,
        "object_identity_confidence": None,
        "object_pose_6dof": None,
        "complete_object_dimensions_m": None,
        "sensor_timestamp_synchronized": False,
        "scene_revision": None,
        "object_correspondence_verified": False,
        "semantic_grounding_authorized": False,
        "grasp_planning_authorized": False,
        "controller_eligible": False,
    }
    return report, selected_mask


def evaluate_native_depth_foreground(native_dir: str | Path) -> dict[str, Any]:
    """Source-hash checked two-camera segmentation replay, without any physics."""
    root = Path(native_dir).resolve(strict=True)
    _require(root.is_dir(), "native recording must be a directory")
    reports: list[dict[str, Any]] = []
    for camera in CAMERAS:
        manifest, manifest_digest, rgb, depth_mm, transform, utc = _read_manifest(root, camera)
        intrinsic = np.asarray(manifest["intrinsic_cv"], dtype=np.float64)
        report, _ = _segment_native_camera(rgb, depth_mm, intrinsic, transform, camera_id=camera)
        report.update({
            "source_manifest_sha256": manifest_digest,
            "source_rgb_sha256": manifest["rgb_sha256"],
            "source_depth_sha256": manifest["depth_sha256"],
            "calibration_source": "native_camera_to_world_cv",
            "capture_wallclock_utc": utc,
            "timestamp_semantics": manifest["timestamp_semantics"],
        })
        reports.append(report)

    p, q = reports
    plane_diff = abs(p["reference_plane_height_world_z_m"] - q["reference_plane_height_world_z_m"])
    cap_a = np.asarray(p["anchor_median_world_xyz_m"])
    cap_b = np.asarray(q["anchor_median_world_xyz_m"])
    cap_xy_delta = float(np.linalg.norm(cap_a[:2] - cap_b[:2]))
    cap_z_delta = float(abs(cap_a[2] - cap_b[2]))
    world_bounds = [np.asarray(row["visible_foreground_world_q05_q50_q95_m"]) for row in reports]
    bound_delta = float(np.max(np.abs(world_bounds[0][[0,2],:2] - world_bounds[1][[0,2],:2])))
    # These reject grossly incompatible geometry but NEVER certify correspondence.
    _require(plane_diff <= CONFIG["max_intercamera_plane_height_difference_m"],
             "blue reference heights incompatible across cameras")
    _require(cap_xy_delta <= CONFIG["max_intercamera_cap_xy_difference_m"] and
             cap_z_delta <= CONFIG["max_intercamera_cap_height_difference_m"],
             "orange anchor positions incompatible across cameras")
    _require(bound_delta <= CONFIG["max_intercamera_foreground_xy_bounds_q05q95_delta_m"],
             "foreground visible XY ranges incompatible across cameras")

    result: dict[str, Any] = {
        "schema": "s1i.native-depth-foreground-proposal.v1",
        "classification": "TWO_CAMERA_COLOR_ANCHORED_DEPTH_FOREGROUND_PROPOSALS_NON_CONTROL",
        "producer_method": METHOD,
        "producer_config": CONFIG,
        "producer_config_sha256": sha256(json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "camera_count": 2,
        "cameras": reports,
        "cross_view_color_anchor_xy_distance_m": cap_xy_delta,
        "cross_view_color_anchor_z_distance_m": cap_z_delta,
        "cross_view_plane_height_difference_m": plane_diff,
        "cross_view_visible_xy_quantile_bounds_max_delta_m": bound_delta,
        "cross_camera_correspondence_verified": False,
        "cross_camera_sensor_timestamp_synchronized": False,
        "object_instance_segmentation_verified": False,
        "object_identity_verified": False,
        "object_pose_6dof_verified": False,
        "complete_object_dimensions_verified": False,
        "scene_snapshot_created": False,
        "semantic_grounding_authorized": False,
        "grasp_planning_authorized": False,
        "controller_eligible": False,
        "native_physics_steps": 0,
        "gemini_api_requests": 0,
        "claim_boundary": "Measured color-anchored, depth-separated contiguous foreground for two native recordings. Heuristic scene-restricted segmentation, not an independently validated object instance, correspondence, semantic class, full geometry, 6-DoF pose, synchronized capture, or safe robot action.",
    }
    result["report_sha256"] = sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result
