"""Read-only RGB-D frame capture adapter for pinned RoboTwin/SAPIEN runtime.

The adapter must be called after RoboTwin has advanced its observation and taken
the camera picture. It never steps simulation or sends robot commands.
"""

from __future__ import annotations

import importlib.metadata
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

PINNED_SAPIEN_VERSION = "3.0.0b1"
CAMERA_IDS = {"head_camera", "wrist_camera"}
CV_FROM_GL = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class NativeDepthCaptureError(RuntimeError):
    """Native depth capture failed closed."""


# This captures the SAPIEN 3 Color float buffer (not vendor uint8 RGBA).
# Conventions deliberately differ from the old, incorrectly-scaled export.
NATIVE_METRIC_DEPTH_CONVENTION = (
    "sapien3_position_camera_z_mm_masked_by_float_alpha_and_position_zbuffer_v2"
)


def depth_millimetres_from_sapien(position_rgba: Any, color_rgba: Any) -> Any:
    """Read actual Z-depth in mm; NEVER scale depth by raw/quantized RGB alpha.

    SAPIEN 3 Color is float RGBA with RGB potentially HDR (>1). The old
    max(Color)<=1 heuristic sometimes did not scale alpha, causing a catastrophic
    1/255 reduction in depth and effectively black RGB images. An alpha mask is
    a validity mask, not a metric distance multiplier. Position[...,3] is a
    z-buffer; 1 denotes pixels beyond the far plane in documented SAPIEN 3.
    """

    import numpy as np

    position = np.asarray(position_rgba)
    color = np.asarray(color_rgba)
    if (
        position.ndim != 3
        or position.shape[2] != 4
        or not np.issubdtype(position.dtype, np.floating)
    ):
        raise NativeDepthCaptureError("Position must be a floating HxWx4 SAPIEN image")
    if color.shape != position.shape or not np.issubdtype(color.dtype, np.floating):
        raise NativeDepthCaptureError("Color must be a float HxWx4 SAPIEN image")
    if not np.isfinite(position).all() or not np.isfinite(color).all():
        raise NativeDepthCaptureError("Color and Position buffers must be finite")
    # Only opaque visible surfaces can be treated as valid metric depth. Do not
    # treat a translucent render as reliable 3D sensor measurement.
    valid = (color[..., 3] >= 0.99) & (position[..., 3] < 1.0) & (position[..., 2] < 0.0)
    depth_mm = np.where(valid, -position[..., 2].astype(np.float64) * 1000.0, 0.0)
    return depth_mm


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_matrix(value: Any, name: str, shape: tuple[int, int]) -> list[list[float]]:
    import numpy as np

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise NativeDepthCaptureError(f"{name} must be finite with shape {shape}")
    return matrix.tolist()


def capture_sapien_depth_frame(
    camera: Any,
    *,
    camera_id: str,
    frame_index: int,
    frame_association_source: str,
    output_dir: str | Path,
    frame_index_verified: bool = False,
    scene_revision: str | None = None,
) -> Path:
    """Capture Position/Color and calibration to checksummed files plus JSON.

    Returns the manifest path on success or failure. The caller supplies the
    observation loop's frame index and association source; the SAPIEN camera API
    itself does not provide a simulator timestamp in the audited RoboTwin code.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Untrusted camera IDs must never become filesystem paths, even on failures.
    valid_frame = type(frame_index) is int and frame_index >= 0
    safe_stem = (
        f"{camera_id}_frame_{frame_index:06d}"
        if isinstance(camera_id, str) and camera_id in CAMERA_IDS and valid_frame
        else "invalid-capture"
    )
    manifest_path = output / f"{safe_stem}_depth-manifest.json"
    if manifest_path.exists():
        raise NativeDepthCaptureError("capture manifest already exists; refuse overwrite")
    manifest: dict[str, Any] = {
        "schema": "damh.robotwin-native-rgbd-capture.v1",
        "status": "failed",
        "source_classification": "recorded_sim_observation",
        "camera_id": camera_id,
        "frame_index": frame_index,
        "frame_association_source": frame_association_source,
        "frame_index_verified_by_caller": frame_index_verified,
        "capture_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_semantics": "probe wall-clock time; not a simulator sensor timestamp",
        "scene_revision": scene_revision,
        "depth_encoding": "NumPy .npy float64",
        "depth_units": "millimetres",
        "depth_convention": NATIVE_METRIC_DEPTH_CONVENTION,
        "pinned_sapien_version": PINNED_SAPIEN_VERSION,
        "controller_eligible": False,
    }
    try:
        if not isinstance(camera_id, str) or camera_id not in CAMERA_IDS:
            raise NativeDepthCaptureError("camera_id must be head_camera or wrist_camera")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise NativeDepthCaptureError("frame_index must be a non-negative integer")
        if not isinstance(frame_association_source, str) or not frame_association_source.strip():
            raise NativeDepthCaptureError("frame_association_source is required")
        if frame_index_verified is not True:
            raise NativeDepthCaptureError(
                "caller must verify the camera buffers match the supplied frame_index"
            )
        if camera is None:
            raise NativeDepthCaptureError("camera handle is missing")
        installed_version = importlib.metadata.version("sapien")
        manifest["installed_sapien_version"] = installed_version
        if installed_version != PINNED_SAPIEN_VERSION:
            raise NativeDepthCaptureError(
                f"expected sapien=={PINNED_SAPIEN_VERSION}, found {installed_version}"
            )

        import numpy as np
        from PIL import Image

        color = np.asarray(camera.get_picture("Color"))
        position = np.asarray(camera.get_picture("Position"))
        if color.ndim != 3 or color.shape[2] != 4 or not np.isfinite(color).all():
            raise NativeDepthCaptureError("Color must be a finite HxWx4 buffer")
        depth_mm = depth_millimetres_from_sapien(position, color)
        if depth_mm.shape != color.shape[:2] or not np.any(depth_mm > 0):
            raise NativeDepthCaptureError("camera returned no valid positive depth pixels")

        intrinsic = _json_matrix(camera.get_intrinsic_matrix(), "intrinsic_cv", (3, 3))
        extrinsic = _json_matrix(camera.get_extrinsic_matrix(), "extrinsic_cv", (3, 4))
        model = _json_matrix(camera.get_model_matrix(), "cam2world_gl", (4, 4))
        c2w_gl = np.asarray(model, dtype=np.float64)
        c2w_cv = c2w_gl @ np.asarray(CV_FROM_GL, dtype=np.float64)
        world_to_camera_cv = np.vstack((np.asarray(extrinsic), (0.0, 0.0, 0.0, 1.0)))
        transform_error = float(np.max(np.abs(np.linalg.inv(world_to_camera_cv) - c2w_cv)))
        if transform_error > 2e-4:
            raise NativeDepthCaptureError(
                f"CV/GL transform mismatch (max abs error {transform_error:.6g})"
            )
        if intrinsic[0][0] <= 0 or intrinsic[1][1] <= 0:
            raise NativeDepthCaptureError("intrinsic focal lengths must be positive")

        # SAPIEN Color is float RGBA, with possible HDR RGB > 1. Always
        # convert normalized values to uint8 by multiplying by 255 first.
        rgb = color.astype(np.float64) * 255.0
        rgb_path = output / f"{safe_stem}_rgb.png"
        depth_path = output / f"{safe_stem}_depth_mm.npy"
        if any(path.exists() for path in (manifest_path, rgb_path, depth_path)):
            raise NativeDepthCaptureError("capture evidence path already exists; refuse overwrite")
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGBA").save(rgb_path)
        np.save(depth_path, depth_mm, allow_pickle=False)
        manifest.update(
            {
                "status": "capture_written",
                "native_api": {
                    "color": 'camera.get_picture("Color")',
                    "position": 'camera.get_picture("Position")',
                    "intrinsics": "camera.get_intrinsic_matrix()",
                    "extrinsics": "camera.get_extrinsic_matrix()",
                    "camera_to_world_gl": "camera.get_model_matrix()",
                    "picture_refresh": "caller must use same updated observation frame",
                },
                "image_size_px": [int(color.shape[1]), int(color.shape[0])],
                "rgb_path": str(rgb_path.resolve()),
                "rgb_sha256": _sha256(rgb_path),
                "depth_path": str(depth_path.resolve()),
                "depth_sha256": _sha256(depth_path),
                "depth_shape": list(depth_mm.shape),
                "depth_valid_pixel_count": int(np.count_nonzero(depth_mm > 0)),
                "raw_color_rgb_float_min_max": [
                    float(np.min(color[..., :3])),
                    float(np.max(color[..., :3])),
                ],
                "raw_color_alpha_float_min_max": [
                    float(np.min(color[..., 3])),
                    float(np.max(color[..., 3])),
                ],
                "raw_position_z_m_min_max": [
                    float(np.min(position[..., 2])),
                    float(np.max(position[..., 2])),
                ],
                "raw_position_zbuffer_min_max": [
                    float(np.min(position[..., 3])),
                    float(np.max(position[..., 3])),
                ],
                "intrinsic_cv": intrinsic,
                "extrinsic_cv_world_to_camera_cv": extrinsic,
                "cam2world_gl": model,
                "camera_to_world_cv": c2w_cv.tolist(),
                "cv_gl_max_abs_error": transform_error,
                "native_capture_executed": True,
            }
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "native_capture_executed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
