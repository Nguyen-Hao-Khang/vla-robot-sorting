"""Canonical scene color landmarks from same-refresh native RGB-D.

Color landmarks are **observations of colored surfaces**, not identified
objects, segmentation ground truth, 6-DoF poses or robot commands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path

import numpy as np
from PIL import Image

from vla_robot_sorting.perception.metric_pixel import MetricPixelPoint, project_metric_pixel
from vla_robot_sorting.perception.native_metric_quality import audit_native_metric_quality
from vla_robot_sorting.perception.region_proposals import RegionProposalError, _opencv

LANDMARK_HSV_BANDS = {
    "orange_surface": ((0, 115, 65), (26, 255, 255)),
    "blue_surface": ((95, 95, 65), (145, 255, 255)),
}


@dataclass(frozen=True, slots=True)
class NativeColorLandmark:
    camera_id: str
    landmark_kind: str
    selection_method: str
    source_manifest_sha256: str
    rgb_sha256: str
    depth_sha256: str
    bbox_xyxy_px: tuple[int, int, int, int]
    mask_pixel_count: int
    valid_depth_fraction: float
    median_masked_depth_m: float
    masked_depth_iqr_m: float
    observed_pixel: MetricPixelPoint
    source_frame_index: int = 0
    object_id: None = None
    object_pose_world: None = None
    object_orientation: None = None
    grasp_candidate_validated: bool = False
    controller_eligible: bool = False


def find_native_color_landmarks(
    native_dir: str | Path, *, camera_id: str, min_mask_px: int = 30
) -> tuple[NativeColorLandmark, ...]:
    """Color-only canonical ROI discovery, masked source-v2 depth, fail closed."""
    if camera_id not in ("head_camera", "wrist_camera"):
        raise RegionProposalError("unsupported source-native camera")
    if type(min_mask_px) is not int or not 1 <= min_mask_px <= 10000:
        raise RegionProposalError("invalid minimum color mask area")
    root = Path(native_dir).resolve()
    audit_native_metric_quality(root)
    stem = f"{camera_id}_frame_000000"
    manifest_bytes = (root / f"{stem}_depth-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("camera_id") != camera_id or manifest.get("frame_index") != 0:
        raise RegionProposalError("source camera/frame identity mismatch")
    image = np.asarray(Image.open(root / f"{stem}_rgb.png").convert("RGB"), dtype=np.uint8)
    depth_mm = np.load(root / f"{stem}_depth_mm.npy", allow_pickle=False)
    if image.shape[:2] != depth_mm.shape:
        raise RegionProposalError("RGB and metric depth shapes mismatch")
    cv2 = _opencv()
    hsv = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2HSV)
    results = []
    for kind, (lower, upper) in LANDMARK_HSV_BANDS.items():
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        count, labels, stats, _centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components = [
            index for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= min_mask_px
        ]
        if not components:
            continue
        # The largest color component is a heuristic, not identity confidence.
        chosen = sorted(
            components,
            key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), int(stats[i, cv2.CC_STAT_TOP])),
        )[0]
        binary = labels == chosen
        total = int(np.count_nonzero(binary))
        valid = binary & np.isfinite(depth_mm) & (depth_mm >= 50.0)
        measured = depth_mm[valid].astype(np.float64)
        if measured.size < 4 or measured.size / total < 0.60:
            continue
        median = float(np.median(measured))
        iqr = float(np.percentile(measured, 75) - np.percentile(measured, 25))
        if not (isfinite(median) and isfinite(iqr) and median > 0 and iqr <= 80.0):
            continue
        yy, xx = np.nonzero(valid)
        selected_index = int(np.argmin(np.abs(depth_mm[yy, xx] - median)))
        uv = (int(xx[selected_index]), int(yy[selected_index]))
        point = project_metric_pixel(root, camera_id=camera_id, pixel_uv=uv)
        left, top, width, height, _area = (int(v) for v in stats[chosen, :5])
        results.append(
            NativeColorLandmark(
                camera_id=camera_id,
                landmark_kind=kind,
                selection_method="canonical_hsv_color_largest_component_not_object_detector_v1",
                source_manifest_sha256=sha256(manifest_bytes).hexdigest(),
                rgb_sha256=manifest["rgb_sha256"],
                depth_sha256=manifest["depth_sha256"],
                bbox_xyxy_px=(left, top, left + width, top + height),
                mask_pixel_count=total,
                valid_depth_fraction=float(measured.size / total),
                median_masked_depth_m=median / 1000.0,
                masked_depth_iqr_m=iqr / 1000.0,
                observed_pixel=point,
            )
        )
    return tuple(results)
