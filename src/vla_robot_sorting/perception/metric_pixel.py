"""Audit-locked camera-pixel depth into world *point*, never object pose.

A single RGB-D pixel cannot establish object identity, full 6-DoF pose,
geometry dimensions or controller permission. The caller must separately
inspect capture provenance and validate the real native source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path

import numpy as np

from vla_robot_sorting.perception.native_depth_audit import audit_native_depth_bundle
from vla_robot_sorting.perception.native_depth_capture import NATIVE_METRIC_DEPTH_CONVENTION

CAMERA_IDS = ("head_camera", "wrist_camera")


class MetricPixelError(ValueError):
    """Native depth evidence or requested pixel cannot yield a world point."""


@dataclass(frozen=True, slots=True)
class MetricPixelPoint:
    camera_id: str
    frame_index: int
    pixel_uv: tuple[int, int]
    depth_m: float
    camera_point_m: tuple[float, float, float]
    world_point_m: tuple[float, float, float]
    source_classification: str
    manifest_sha256: str
    depth_sha256: str
    rgb_sha256: str
    calibration_max_abs_error: float
    depth_units_original: str = "millimetres"
    measurement_kind: str = "single_pixel_depth_point"
    object_id: None = None
    object_pose_world: None = None
    object_orientation: None = None
    grasp_candidate_validated: bool = False
    controller_eligible: bool = False


def project_metric_pixel(
    evidence_dir: str | Path,
    *,
    camera_id: str,
    pixel_uv: tuple[int, int],
) -> MetricPixelPoint:
    """Re-verify both captured camera files, then project one exact depth pixel.

    Intrinsics follow pixel Z-depth backprojection, not Euclidean ray range.
    This is read-only and never issues robot commands.
    """
    if camera_id not in CAMERA_IDS:
        raise MetricPixelError("camera must be head_camera or wrist_camera")
    if (
        not isinstance(pixel_uv, (tuple, list))
        or len(pixel_uv) != 2
        or any(type(value) is not int for value in pixel_uv)
    ):
        raise MetricPixelError("pixel_uv must be two integer coordinates")
    u, v = pixel_uv
    directory = Path(evidence_dir).resolve()
    # Audit ensures both manifests, source status, SHA256, RGB/depth dimensions,
    # finite positive depth count, and independent GL/CV frame conventions.
    audit = audit_native_depth_bundle(directory)
    camera_report = next(item for item in audit["cameras"] if item["camera_id"] == camera_id)
    name = f"{camera_id}_frame_000000"
    manifest_path = directory / f"{name}_depth-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    record = json.loads(manifest_bytes)
    # Old source manifests were artifact-integrity-valid, but the actual
    # Position Z was multiplied by non-uint8 Color alpha / 255. This rendered
    # old physical depths up to 255x too small. A metric consumer MUST refuse
    # those bytes rather than "repairing" them by blind scaling.
    if record.get("depth_convention") != NATIVE_METRIC_DEPTH_CONVENTION:
        raise MetricPixelError("metric depth convention is not source-verified")
    width, height = record["image_size_px"]
    if not (0 <= u < width and 0 <= v < height):
        raise MetricPixelError("pixel is outside independently checked image dimensions")
    depth_path = directory / f"{name}_depth_mm.npy"
    depth = np.load(depth_path, allow_pickle=False)
    depth_mm = float(depth[v, u])
    if not isfinite(depth_mm) or depth_mm <= 0:
        raise MetricPixelError("pixel has invalid/background zero depth")
    if depth_mm < 50.0:
        raise MetricPixelError("pixel violates conservative 50 mm metric-depth policy")
    z = depth_mm / 1000.0
    k = np.asarray(record["intrinsic_cv"], dtype=np.float64)
    c2w = np.asarray(record["camera_to_world_cv"], dtype=np.float64)
    camera_xyz = np.array(
        [
            (u - k[0, 2]) * z / k[0, 0],
            (v - k[1, 2]) * z / k[1, 1],
            z,
            1.0,
        ],
        dtype=np.float64,
    )
    world = c2w @ camera_xyz
    if not np.isfinite(world).all() or abs(float(world[3]) - 1.0) > 1e-6:
        raise MetricPixelError("calibrated homogeneous pixel projection is invalid")
    return MetricPixelPoint(
        camera_id=camera_id,
        frame_index=0,
        pixel_uv=(u, v),
        depth_m=z,
        camera_point_m=tuple(float(value) for value in camera_xyz[:3]),
        world_point_m=tuple(float(value) for value in world[:3]),
        source_classification="recorded_sim_observation",
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        depth_sha256=camera_report["depth_sha256"],
        rgb_sha256=camera_report["rgb_sha256"],
        calibration_max_abs_error=camera_report["independent_calibration_max_error"],
    )
