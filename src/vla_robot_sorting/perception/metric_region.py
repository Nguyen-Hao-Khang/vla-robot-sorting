"""Source-verified ROI depth summary; NOT a segmented object, pose or controller plan.

Operator-provided rectangles may include background, occlusion or more than
one object. The representative 3D point is one ACTUALLY OBSERVED depth pixel,
not a synthesized object center. No object identity or 6-DoF pose is inferred.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np

from vla_robot_sorting.perception.metric_pixel import (
    CAMERA_IDS,
    MetricPixelError,
    MetricPixelPoint,
    project_metric_pixel,
)
from vla_robot_sorting.perception.native_depth_audit import audit_native_depth_bundle
from vla_robot_sorting.perception.native_depth_capture import NATIVE_METRIC_DEPTH_CONVENTION


@dataclass(frozen=True, slots=True)
class MetricRegionEvidence:
    """Region-depth statistics from independently checked source pixels."""

    camera_id: str
    frame_index: int
    bbox_xyxy_px: tuple[int, int, int, int]
    valid_pixels: int
    total_pixels: int
    valid_fraction: float
    median_depth_m: float
    depth_iqr_m: float
    representative_point: MetricPixelPoint
    source_classification: str = "recorded_sim_observation"
    region_selection_kind: str = "operator_selected_unsegmented_rectangle"
    measurement_kind: str = "region_depth_statistics_and_observed_pixel"
    object_id: None = None
    object_pose_world: None = None
    object_dimensions_m: None = None
    object_orientation: None = None
    object_identity_confidence: None = None
    grasp_candidate_validated: bool = False
    controller_eligible: bool = False


def project_metric_region(
    evidence_dir: str | Path,
    *,
    camera_id: str,
    bbox_xyxy_px: tuple[int, int, int, int],
    min_valid_fraction: float = 0.6,
    max_iqr_m: float = 0.08,
    min_valid_pixels: int = 4,
) -> MetricRegionEvidence:
    """Validate both native source cameras; summarize a chosen unsegmented ROI.

    Quantiles describe observed depth variation, NOT depth sensor accuracy or
    object extent. Bad coverage or wide/ambiguous depth fails closed. Even good
    statistics never authorize an object hypothesis or any motor activity.
    """
    if camera_id not in CAMERA_IDS:
        raise MetricPixelError("camera must be head_camera or wrist_camera")
    if (
        not isinstance(bbox_xyxy_px, (list, tuple))
        or len(bbox_xyxy_px) != 4
        or any(type(value) is not int for value in bbox_xyxy_px)
    ):
        raise MetricPixelError("bbox_xyxy_px must contain four integers")
    if (
        isinstance(min_valid_fraction, bool)
        or not isinstance(min_valid_fraction, (int, float))
        or not isfinite(min_valid_fraction)
        or not 0 < min_valid_fraction <= 1
    ):
        raise MetricPixelError("min_valid_fraction must be finite in (0, 1]")
    if (
        isinstance(max_iqr_m, bool)
        or not isinstance(max_iqr_m, (int, float))
        or not isfinite(max_iqr_m)
        or not 0 <= max_iqr_m <= 10
    ):
        raise MetricPixelError("max_iqr_m must be finite in [0, 10]")
    if type(min_valid_pixels) is not int or min_valid_pixels < 4:
        raise MetricPixelError("min_valid_pixels must be an integer >= 4")

    x1, y1, x2, y2 = bbox_xyxy_px
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise MetricPixelError("region is empty or has negative bounds")

    directory = Path(evidence_dir).resolve()
    audit_native_depth_bundle(directory)
    manifest_path = directory / f"{camera_id}_frame_000000_depth-manifest.json"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if record.get("depth_convention") != NATIVE_METRIC_DEPTH_CONVENTION:
        raise MetricPixelError("metric depth convention is not source-verified")
    width, height = record["image_size_px"]
    if x2 > width or y2 > height:
        raise MetricPixelError("region extends beyond audited image dimensions")
    pixels = (x2 - x1) * (y2 - y1)
    if pixels > 500_000:
        raise MetricPixelError("region exceeds bounded sample count")
    if min_valid_pixels > pixels:
        raise MetricPixelError("minimum valid sample count exceeds region size")

    depth_mm = np.load(directory / f"{camera_id}_frame_000000_depth_mm.npy", allow_pickle=False)[
        y1:y2, x1:x2
    ]
    # A positive but sub-50-mm sample cannot qualify as trusted geometry.
    # The 50-mm floor is a conservative policy, not a measured camera near plane.
    valid = depth_mm >= 50.0
    count = int(np.count_nonzero(valid))
    fraction = count / pixels
    if count < min_valid_pixels or fraction < min_valid_fraction:
        raise MetricPixelError("insufficient observed positive depth coverage")

    samples = depth_mm[valid].astype(np.float64) / 1000.0
    low, median, high = np.quantile(samples, (0.25, 0.5, 0.75))
    iqr = float(high - low)
    if iqr > max_iqr_m:
        raise MetricPixelError("region has excessive depth spread; ambiguous ROI")

    # Pick one real valid pixel nearest the robust median. On ties choose
    # closest to ROI center, then row-major order for repeatability.
    ys, xs = np.nonzero(valid)
    sample_depths = depth_mm[ys, xs] / 1000.0
    center_u = (x1 + x2 - 1) / 2
    center_v = (y1 + y2 - 1) / 2
    distance_to_center = (xs + x1 - center_u) ** 2 + (ys + y1 - center_v) ** 2
    selection = np.lexsort(
        (
            xs,
            ys,
            distance_to_center,
            np.abs(sample_depths - median),
        )
    )
    index = int(selection[0])
    pixel = (int(x1 + xs[index]), int(y1 + ys[index]))
    # This call rechecks the entire two-camera source bundle before projecting.
    point = project_metric_pixel(directory, camera_id=camera_id, pixel_uv=pixel)
    return MetricRegionEvidence(
        camera_id=camera_id,
        frame_index=point.frame_index,
        bbox_xyxy_px=(x1, y1, x2, y2),
        valid_pixels=count,
        total_pixels=pixels,
        valid_fraction=fraction,
        median_depth_m=float(median),
        depth_iqr_m=iqr,
        representative_point=point,
    )
